"""YouTube Live Studio RTMP broadcaster with real-time Japanese text/subtitle video rendering.

Renders a 1280x720 midnight-radio dashboard frame with:
  - ON AIR badge, station name, JST clock, and permanent AI-generated content disclosure
  - Current topic title, description, and progress bar
  - Host A / Host B speaker indicators with audio-reactive level bars
  - Real-time audio-synced Japanese subtitles / scrolling dialogue log

Feeds raw RGB24 video frames to FFmpeg stdin and live 44.1kHz stereo PCM audio via a local
TCP loopback socket, encoding H.264 + AAC directly to YouTube Live Studio (RTMP/RTMPS)
with automatic reconnection.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Preferred Japanese fonts on Windows (fallback to DejaVu/Noto on Linux)
DEFAULT_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\YuGothB.ttc",
    r"C:\Windows\Fonts\meiryob.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

DEFAULT_REGULAR_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\YuGothR.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _find_font_path(configured: str, candidates: list[str]) -> Optional[str]:
    if configured:
        p = Path(configured)
        if p.exists():
            return str(p)
    for cand in candidates:
        if Path(cand).exists():
            return cand
    return None


def _load_font(path: Optional[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception as exc:
            logger.warning("Failed to load font %s (%dpx): %s", path, size, exc)
    return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    max_lines: int = 3,
) -> list[str]:
    """Wrap Japanese/mixed text by pixel width."""
    if not text:
        return []
    lines: list[str] = []
    cur = ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            if len(lines) >= max_lines:
                return lines
            continue
        test = cur + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = ch
            if len(lines) >= max_lines:
                # Truncate last line with ellipsis
                last = lines[-1]
                while len(last) > 1:
                    b = draw.textbbox((0, 0), last + "…", font=font)
                    if b[2] - b[0] <= max_width:
                        break
                    last = last[:-1]
                lines[-1] = last + "…"
                return lines
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines


class RadioFrameRenderer:
    """Renders 16:9 radio broadcast frames with live synced subtitles and AI disclosure."""

    def __init__(self, config: dict, base_dir: Path):
        yt_cfg = config.get("youtube", {})
        scfg = config.get("server", {})
        hcfg = config.get("hosts", {})

        self.width = int(yt_cfg.get("width", 1280))
        self.height = int(yt_cfg.get("height", 720))
        self.station_name = str(scfg.get("station_name", "ハルとミナの、無限ラジオ"))
        self.host_a_name = str(hcfg.get("host_a", {}).get("name", "ハル"))
        self.host_b_name = str(hcfg.get("host_b", {}).get("name", "ミナ"))
        self.ai_badge_text = str(
            yt_cfg.get("ai_disclosure_badge", "ローカルAIによる自動生成ラジオ (Haru-4B)")
        )
        self.ai_footer_text = str(
            yt_cfg.get(
                "ai_disclosure_footer",
                "※ この配信はローカルAI音声会話モデル (Haru-4B) による自動生成ラジオです。実在の人物・団体とは関係ありません。",
            )
        )

        bold_path = _find_font_path(str(yt_cfg.get("font_path", "")), DEFAULT_FONT_CANDIDATES)
        reg_path = _find_font_path(str(yt_cfg.get("font_path", "")), DEFAULT_REGULAR_FONT_CANDIDATES)
        if not bold_path:
            logger.warning("No Japanese TrueType font found; text rendering may not display Japanese properly.")
        else:
            logger.info("YouTube renderer font: bold=%s, regular=%s", bold_path, reg_path)

        scale = self.height / 720.0
        self.font_station = _load_font(bold_path, max(16, int(24 * scale)))
        self.font_badge = _load_font(bold_path, max(12, int(16 * scale)))
        self.font_clock = _load_font(bold_path, max(14, int(19 * scale)))
        self.font_label = _load_font(bold_path, max(12, int(15 * scale)))
        self.font_topic = _load_font(bold_path, max(20, int(30 * scale)))
        self.font_detail = _load_font(reg_path, max(13, int(17 * scale)))
        self.font_speaker = _load_font(bold_path, max(14, int(18 * scale)))
        self.font_sub = _load_font(reg_path, max(16, int(22 * scale)))
        self.font_sub_latest = _load_font(bold_path, max(17, int(24 * scale)))
        self.font_footer = _load_font(reg_path, max(11, int(14 * scale)))

        self._bg_base = self._build_background(yt_cfg, base_dir)

    def _build_background(self, yt_cfg: dict, base_dir: Path) -> Image.Image:
        bg_path_str = str(yt_cfg.get("background_image", "")).strip()
        if bg_path_str:
            bg_path = Path(bg_path_str)
            if not bg_path.is_absolute():
                bg_path = base_dir / bg_path
            if bg_path.exists():
                try:
                    img = Image.open(bg_path).convert("RGB").resize((self.width, self.height), Image.Resampling.LANCZOS)
                    # Dim background slightly so text stays high-contrast
                    overlay = Image.new("RGBA", (self.width, self.height), (10, 14, 24, 165))
                    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                except Exception as exc:
                    logger.warning("Failed to load background_image %s: %s", bg_path, exc)

        # Clean midnight-indigo vertical gradient
        img = Image.new("RGB", (self.width, self.height), (13, 18, 32))
        draw = ImageDraw.Draw(img)
        top = (15, 23, 42)
        bot = (9, 13, 24)
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            r = int(top[0] + (bot[0] - top[0]) * t)
            g = int(top[1] + (bot[1] - top[1]) * t)
            b = int(top[2] + (bot[2] - top[2]) * t)
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))
        return img

    def render_rgb24(self, live_state: dict[str, Any]) -> bytes:
        """Render one frame as raw RGB24 bytes."""
        img = self._bg_base.copy()
        draw = ImageDraw.Draw(img)
        w, h = self.width, self.height
        pad = int(w * 0.035)  # ~45px at 1280w

        mode = str(live_state.get("mode", "fallback"))
        topic = str(live_state.get("current_topic", "番組準備中"))
        detail = str(live_state.get("current_detail", ""))
        source_label = str(live_state.get("current_source_label", "")).strip()
        elapsed = float(live_state.get("segment_elapsed_sec", 0.0))
        duration = float(live_state.get("segment_duration_sec", 0.0))
        active_speaker = str(live_state.get("active_speaker", ""))
        audio_level = float(live_state.get("audio_level", 0.0))
        subtitles: list[dict[str, Any]] = list(live_state.get("live_subtitles", []))
        upcoming: list[str] = list(live_state.get("upcoming_topics", []))

        # --- 1. Top Header Bar ---
        header_h = int(h * 0.095)
        draw.rectangle([(0, 0), (w, header_h)], fill=(17, 24, 39))
        draw.line([(0, header_h), (w, header_h)], fill=(51, 65, 85), width=2)

        # ON AIR pill
        if mode == "talk":
            pill_color = (220, 38, 38)
            pill_label = "● ON AIR"
        elif mode == "paused":
            pill_color = (100, 116, 139)
            pill_label = "● PAUSED"
        else:
            pill_color = (217, 119, 6)
            pill_label = "● STANDBY"

        pill_x = pad
        pill_y = int(header_h * 0.24)
        pill_w = 116
        pill_h = int(header_h * 0.52)
        draw.rounded_rectangle(
            [(pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h)],
            radius=16,
            fill=pill_color,
        )
        draw.text((pill_x + 14, pill_y + 7), pill_label, font=self.font_badge, fill=(255, 255, 255))

        # Station Name
        draw.text((pill_x + pill_w + 18, pill_y + 3), self.station_name, font=self.font_station, fill=(248, 250, 252))

        # Right side: Clock + AI Disclosure Badge
        now_str = datetime.now().strftime("%H:%M:%S")
        clock_bbox = draw.textbbox((0, 0), now_str, font=self.font_clock)
        clock_w = clock_bbox[2] - clock_bbox[0]
        clock_x = w - pad - clock_w
        draw.text((clock_x, pill_y + 6), now_str, font=self.font_clock, fill=(203, 213, 225))

        badge_bbox = draw.textbbox((0, 0), self.ai_badge_text, font=self.font_badge)
        badge_w = (badge_bbox[2] - badge_bbox[0]) + 26
        badge_x = clock_x - badge_w - 18
        draw.rounded_rectangle(
            [(badge_x, pill_y), (badge_x + badge_w, pill_y + pill_h)],
            radius=14,
            fill=(30, 58, 138),
            outline=(96, 165, 250),
            width=1,
        )
        draw.text((badge_x + 13, pill_y + 7), self.ai_badge_text, font=self.font_badge, fill=(219, 234, 254))

        # --- 2. Topic Card ---
        card_top = header_h + 18
        card_h = int(h * 0.23)
        card_bot = card_top + card_h
        draw.rounded_rectangle(
            [(pad, card_top), (w - pad, card_bot)],
            radius=14,
            fill=(24, 34, 54),
            outline=(51, 65, 85),
            width=1,
        )

        label_text = "NOW TALKING — トークテーマ" if mode == "talk" else "INTERMISSION — 番組準備中"
        draw.text((pad + 22, card_top + 14), label_text, font=self.font_label, fill=(96, 165, 250))

        # RSS news source attribution badge next to the label (when topic came from RSS)
        if source_label:
            lbl_b = draw.textbbox((0, 0), label_text, font=self.font_label)
            src_x = pad + 22 + (lbl_b[2] - lbl_b[0]) + 16
            max_src_w = (w - pad - 260) - src_x
            if max_src_w > 120:
                src_shown = source_label
                while len(src_shown) > 8:
                    sb = draw.textbbox((0, 0), src_shown, font=self.font_footer)
                    if (sb[2] - sb[0]) + 20 <= max_src_w:
                        break
                    src_shown = src_shown[:-2] + "…"
                sb = draw.textbbox((0, 0), src_shown, font=self.font_footer)
                sw = (sb[2] - sb[0]) + 18
                draw.rounded_rectangle(
                    [(src_x, card_top + 11), (src_x + sw, card_top + 35)],
                    radius=10,
                    fill=(30, 41, 59),
                    outline=(251, 191, 36),
                    width=1,
                )
                draw.text((src_x + 9, card_top + 14), src_shown, font=self.font_footer, fill=(253, 224, 71))

        # Host indicator pills on top-right of topic card
        self._draw_host_pills(draw, w - pad - 22, card_top + 12, active_speaker, audio_level)

        topic_lines = _wrap_text(draw, topic, self.font_topic, max_width=w - pad * 2 - 44, max_lines=2)
        ty = card_top + 42
        for line in topic_lines:
            draw.text((pad + 22, ty), line, font=self.font_topic, fill=(255, 255, 255))
            ty += 38

        if detail:
            detail_lines = _wrap_text(draw, detail, self.font_detail, max_width=w - pad * 2 - 44, max_lines=1)
            if detail_lines:
                draw.text((pad + 22, min(ty + 4, card_bot - 42)), detail_lines[0], font=self.font_detail, fill=(148, 163, 184))

        # Progress bar + remaining/total time at bottom of topic card
        bar_y = card_bot - 16
        bar_x0 = pad + 22
        if duration > 0:
            rem_s = max(0, int(round(duration - elapsed)))
            tot_s = max(0, int(round(duration)))
            time_txt = f"残り {rem_s // 60:02d}:{rem_s % 60:02d} / {tot_s // 60:02d}:{tot_s % 60:02d}"
        else:
            time_txt = "残り --:-- / --:--"
        tb = draw.textbbox((0, 0), time_txt, font=self.font_footer)
        tw = tb[2] - tb[0]
        time_x = w - pad - 22 - tw
        draw.text((time_x, bar_y - 6), time_txt, font=self.font_footer, fill=(203, 213, 225))

        bar_x1 = time_x - 16
        draw.rounded_rectangle([(bar_x0, bar_y), (bar_x1, bar_y + 5)], radius=3, fill=(51, 65, 85))
        if duration > 0:
            ratio = max(0.0, min(1.0, elapsed / duration))
            fill_x = bar_x0 + int((bar_x1 - bar_x0) * ratio)
            if fill_x > bar_x0:
                draw.rounded_rectangle([(bar_x0, bar_y), (fill_x, bar_y + 5)], radius=3, fill=(59, 130, 246))

        # --- 3. Live Subtitles / Dialogue Log Box ---
        footer_h = 38
        sub_top = card_bot + 16
        sub_bot = h - footer_h - 10
        draw.rounded_rectangle(
            [(pad, sub_top), (w - pad, sub_bot)],
            radius=14,
            fill=(15, 23, 42),
            outline=(51, 65, 85),
            width=1,
        )

        draw.text((pad + 22, sub_top + 12), "LIVE TRANSCRIPT — リアルタイム字幕", font=self.font_label, fill=(148, 163, 184))

        # Render recent synced utterances (bottom-aligned inside the box so newest is always visible)
        content_top = sub_top + 42
        content_bot = sub_bot - 16
        text_area_w = w - pad * 2 - 140

        if not subtitles:
            placeholder = (
                "音声に合わせてリアルタイム字幕が表示されます..."
                if mode == "talk"
                else "次のトークコーナーを準備しています。しばらくお待ちください..."
            )
            draw.text((pad + 24, content_top + 16), placeholder, font=self.font_sub, fill=(100, 116, 139))
        else:
            # Take last 6 utterances and lay them out from bottom to top
            recent = subtitles[-6:]
            rendered_blocks: list[tuple[str, list[str], bool]] = []
            for idx, item in enumerate(recent):
                is_latest = idx == len(recent) - 1
                spk = str(item.get("speaker", ""))
                txt = str(item.get("text", "")).strip()
                if not txt:
                    continue
                font_use = self.font_sub_latest if is_latest else self.font_sub
                wrapped = _wrap_text(draw, txt, font_use, max_width=text_area_w, max_lines=3 if is_latest else 2)
                if wrapped:
                    rendered_blocks.append((spk, wrapped, is_latest))

            # Calculate heights from bottom up to fit inside [content_top, content_bot]
            selected: list[tuple[str, list[str], bool, int]] = []
            avail_h = content_bot - content_top
            used_h = 0
            for spk, lines, is_latest in reversed(rendered_blocks):
                line_h = 32 if is_latest else 28
                block_h = len(lines) * line_h + 18
                if used_h + block_h > avail_h and selected:
                    break
                selected.append((spk, lines, is_latest, block_h))
                used_h += block_h

            selected.reverse()
            cur_y = content_top
            for spk, lines, is_latest, block_h in selected:
                is_a = spk == self.host_a_name
                tag_bg = (37, 99, 235) if is_a else (219, 39, 119)
                row_bg = (30, 41, 59) if is_latest else (20, 29, 48)
                text_col = (255, 255, 255) if is_latest else (203, 213, 225)
                font_use = self.font_sub_latest if is_latest else self.font_sub
                line_h = 32 if is_latest else 28

                draw.rounded_rectangle(
                    [(pad + 18, cur_y), (w - pad - 18, cur_y + block_h - 8)],
                    radius=10,
                    fill=row_bg,
                    outline=(71, 85, 105) if is_latest else None,
                    width=1 if is_latest else 0,
                )
                # Speaker tag
                tag_x = pad + 28
                tag_y = cur_y + 8
                draw.rounded_rectangle([(tag_x, tag_y), (tag_x + 76, tag_y + 28)], radius=8, fill=tag_bg)
                draw.text((tag_x + 14, tag_y + 4), spk[:4], font=self.font_speaker, fill=(255, 255, 255))

                # Utterance text
                tx = tag_x + 90
                ty_line = cur_y + 7
                for ln in lines:
                    draw.text((tx, ty_line), ln, font=font_use, fill=text_col)
                    ty_line += line_h

                cur_y += block_h

        # --- 4. Footer Bar ---
        footer_y = h - footer_h
        draw.rectangle([(0, footer_y), (w, h)], fill=(11, 15, 26))
        draw.line([(0, footer_y), (w, footer_y)], fill=(30, 41, 59), width=1)
        draw.text((pad, footer_y + 10), self.ai_footer_text, font=self.font_footer, fill=(148, 163, 184))

        if upcoming:
            next_txt = f"NEXT: {upcoming[0]}"
            if len(next_txt) > 32:
                next_txt = next_txt[:31] + "…"
            nb = draw.textbbox((0, 0), next_txt, font=self.font_footer)
            nw = nb[2] - nb[0]
            draw.text((w - pad - nw, footer_y + 10), next_txt, font=self.font_footer, fill=(96, 165, 250))

        return img.tobytes()

    def _draw_host_pills(
        self,
        draw: ImageDraw.ImageDraw,
        right_x: int,
        top_y: int,
        active_speaker: str,
        audio_level: float,
    ) -> None:
        """Draw Host A / Host B speaking indicators + subtle audio level meter."""
        hosts = [
            (self.host_b_name, (219, 39, 119), (80, 24, 52)),
            (self.host_a_name, (37, 99, 235), (23, 37, 84)),
        ]
        cx = right_x
        for name, active_col, idle_col in hosts:
            is_active = active_speaker == name and audio_level > 0.01
            pw = 104
            ph = 30
            x0 = cx - pw
            draw.rounded_rectangle(
                [(x0, top_y), (cx, top_y + ph)],
                radius=15,
                fill=active_col if is_active else idle_col,
                outline=active_col,
                width=2 if is_active else 1,
            )
            draw.text((x0 + 14, top_y + 5), name[:4], font=self.font_badge, fill=(255, 255, 255))
            # Mini 3-bar level meter inside pill
            lvl = min(1.0, audio_level * 4.0) if is_active else 0.1
            for i in range(3):
                bx = x0 + 68 + i * 8
                bh = max(4, int((6 + i * 5) * lvl))
                by = top_y + ph - 7 - bh
                draw.rectangle([(bx, by), (bx + 4, top_y + ph - 7)], fill=(255, 255, 255) if is_active else (148, 163, 184))
            cx = x0 - 10


class YouTubeStreamer:
    """Streams real-time rendered video + live PCM audio to YouTube Live Studio via FFmpeg RTMP."""

    def __init__(
        self,
        config: dict,
        base_dir: Path,
        get_live_state_fn: Callable[[], dict[str, Any]],
        stream_key_override: str = "",
    ):
        self.config = config
        self.base_dir = base_dir
        self._get_live_state = get_live_state_fn

        yt_cfg = config.get("youtube", {})
        self.rtmp_url = str(yt_cfg.get("rtmp_url", "rtmp://a.rtmp.youtube.com/live2")).rstrip("/")
        self.stream_key = (
            stream_key_override.strip()
            or os.environ.get("YOUTUBE_STREAM_KEY", "").strip()
            or str(yt_cfg.get("stream_key", "")).strip()
        )
        self.width = int(yt_cfg.get("width", 1280))
        self.height = int(yt_cfg.get("height", 720))
        self.render_fps = max(1, min(15, int(yt_cfg.get("render_fps", 5))))
        self.output_fps = int(yt_cfg.get("output_fps", 30))
        self.video_bitrate = str(yt_cfg.get("video_bitrate", "1000k"))
        self.video_maxrate = str(yt_cfg.get("video_maxrate", "1200k"))
        self.video_bufsize = str(yt_cfg.get("video_bufsize", "2400k"))
        self.audio_bitrate = str(yt_cfg.get("audio_bitrate", "128k"))
        self.video_encoder = str(yt_cfg.get("video_encoder", "libx264"))
        self.preset = str(yt_cfg.get("preset", "veryfast"))
        self.reconnect_delay_sec = float(yt_cfg.get("reconnect_delay_sec", 5.0))

        self.renderer = RadioFrameRenderer(config, base_dir)

        self._stop_event = threading.Event()
        self._runner_thread: Optional[threading.Thread] = None
        self._pcm_queue: queue.Queue[bytes] = queue.Queue(maxsize=40)
        self._status = "stopped"
        self._last_error = ""

    @property
    def status_text(self) -> str:
        return self._status

    @property
    def last_error(self) -> str:
        return self._last_error

    def on_pcm_chunk(self, chunk: bytes) -> None:
        """Called by RadioStreamer for every 100ms PCM chunk."""
        if self._stop_event.is_set():
            return
        try:
            self._pcm_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._pcm_queue.get_nowait()
                self._pcm_queue.put_nowait(chunk)
            except Exception:
                pass

    def start(self) -> None:
        if not self.stream_key:
            raise ValueError(
                "YouTube stream key is empty. Set YOUTUBE_STREAM_KEY env var, "
                "youtube.stream_key in config.yaml, or pass --youtube-key."
            )
        self._stop_event.clear()
        self._status = "starting"
        self._runner_thread = threading.Thread(target=self._run_loop, name="YouTubeRTMP", daemon=True)
        self._runner_thread.start()
        logger.info(
            "YouTubeStreamer started -> %s/**** (%dx%d @ %dfps->%dfps, %s, %s)",
            self.rtmp_url,
            self.width,
            self.height,
            self.render_fps,
            self.output_fps,
            self.video_encoder,
            self.video_bitrate,
        )

    def stop(self) -> None:
        self._stop_event.set()
        self._status = "stopped"
        if self._runner_thread and self._runner_thread.is_alive():
            self._runner_thread.join(timeout=5.0)
        logger.info("YouTubeStreamer stopped.")

    def _build_ffmpeg_cmd(self, audio_port: int, output_target: str, output_format: str = "flv") -> list[str]:
        gop = max(self.output_fps, self.output_fps * 2)  # 2-second keyframe interval (YouTube requirement)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            # Video input from stdin (raw RGB24)
            "-thread_queue_size",
            "512",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.render_fps),
            "-i",
            "pipe:0",
            # Audio input from local TCP loopback socket (raw s16le 44.1kHz stereo)
            "-thread_queue_size",
            "512",
            "-f",
            "s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-i",
            f"tcp://127.0.0.1:{audio_port}",
            # Video encoding
            "-vf",
            f"fps={self.output_fps},format=yuv420p",
            "-c:v",
            self.video_encoder,
        ]
        if self.video_encoder == "libx264":
            cmd.extend([
                "-preset",
                self.preset,
                "-tune",
                "stillimage",
                "-x264-params",
                f"keyint={gop}:min-keyint={gop}:scenecut=0",
            ])
        cmd.extend([
            "-g",
            str(gop),
            "-b:v",
            self.video_bitrate,
            "-maxrate",
            self.video_maxrate,
            "-bufsize",
            self.video_bufsize,
            # Audio encoding
            "-c:a",
            "aac",
            "-b:a",
            self.audio_bitrate,
            "-ar",
            "44100",
            "-ac",
            "2",
            "-f",
            output_format,
            output_target,
        ])
        return cmd

    def _run_loop(self) -> None:
        target = f"{self.rtmp_url}/{self.stream_key}"
        while not self._stop_event.is_set():
            self._status = "connecting"
            try:
                self._stream_once(target, output_format="flv")
            except Exception as exc:
                self._last_error = str(exc)
                logger.error("YouTube RTMP session error: %s", exc)

            if self._stop_event.is_set():
                break

            self._status = f"reconnecting ({int(self.reconnect_delay_sec)}s)"
            logger.warning(
                "YouTube RTMP disconnected. Reconnecting in %.1fs...",
                self.reconnect_delay_sec,
            )
            self._stop_event.wait(self.reconnect_delay_sec)

    def _stream_once(self, output_target: str, output_format: str = "flv", max_duration_sec: Optional[float] = None) -> None:
        """Launch FFmpeg and feed synchronized video + audio until disconnect, stop, or max_duration_sec."""
        # Open ephemeral local TCP server for PCM audio
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        audio_port = srv.getsockname()[1]
        srv.listen(1)
        srv.settimeout(10.0)

        # Drain stale PCM before starting
        while not self._pcm_queue.empty():
            try:
                self._pcm_queue.get_nowait()
            except queue.Empty:
                break

        cmd = self._build_ffmpeg_cmd(audio_port, output_target, output_format=output_format)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        session_stop = threading.Event()
        stderr_lines: list[str] = []

        def _stderr_reader() -> None:
            assert proc.stderr is not None
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    # Mask stream key if FFmpeg echoes the URL
                    if self.stream_key:
                        line = line.replace(self.stream_key, "****")
                    stderr_lines.append(line)
                    if len(stderr_lines) > 20:
                        stderr_lines.pop(0)
                    logger.warning("FFmpeg[YouTube]: %s", line)

        def _audio_writer() -> None:
            conn: Optional[socket.socket] = None
            try:
                conn, _ = srv.accept()
                conn.settimeout(5.0)
                silence_chunk = b"\x00" * 17640  # 100ms @ 44.1kHz stereo s16le
                while not self._stop_event.is_set() and not session_stop.is_set():
                    try:
                        chunk = self._pcm_queue.get(timeout=0.15)
                    except queue.Empty:
                        chunk = silence_chunk
                    conn.sendall(chunk)
            except Exception as exc:
                if not self._stop_event.is_set() and not session_stop.is_set():
                    logger.debug("YouTube audio socket ended: %s", exc)
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                try:
                    srv.close()
                except Exception:
                    pass

        t_err = threading.Thread(target=_stderr_reader, name="YTFFmpegErr", daemon=True)
        t_aud = threading.Thread(target=_audio_writer, name="YTAudioFeed", daemon=True)
        t_err.start()
        t_aud.start()

        try:
            assert proc.stdin is not None
            self._status = "live"
            self._last_error = ""
            frame_interval = 1.0 / self.render_fps
            start_t = time.monotonic()
            frame_idx = 0

            while not self._stop_event.is_set() and proc.poll() is None:
                if max_duration_sec is not None and (time.monotonic() - start_t) >= max_duration_sec:
                    break
                state = self._get_live_state()
                rgb = self.renderer.render_rgb24(state)
                try:
                    proc.stdin.write(rgb)
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    break
                frame_idx += 1
                next_t = start_t + frame_idx * frame_interval
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    self._stop_event.wait(sleep_s)
        finally:
            session_stop.set()
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
            t_aud.join(timeout=2.0)
            t_err.join(timeout=2.0)
            if proc.returncode not in (0, None) and not self._stop_event.is_set() and max_duration_sec is None:
                tail = " | ".join(stderr_lines[-3:])
                self._last_error = f"FFmpeg exited {proc.returncode}: {tail}"
                logger.warning("YouTube FFmpeg exited with code %s (%s)", proc.returncode, tail)

    def render_preview_video(self, out_path: Path, duration_sec: float = 5.0) -> Path:
        """Render a short local MP4 preview file to verify visuals, Japanese fonts, and audio sync."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Feed gentle test audio chunks in background while rendering
        stop_tone = threading.Event()

        def _tone_feeder() -> None:
            import math
            import numpy as np

            sr = 44100
            n = 4410  # 100ms
            t_offset = 0
            while not stop_tone.is_set():
                t = (np.arange(n) + t_offset) / sr
                sig = 0.08 * np.sin(2.0 * math.pi * 220.0 * t)
                pcm = np.clip(sig * 32767.0, -32768, 32767).astype("<i2")
                stereo = np.repeat(pcm[:, None], 2, axis=1).tobytes()
                self.on_pcm_chunk(stereo)
                t_offset += n
                stop_tone.wait(0.1)

        t_feed = threading.Thread(target=_tone_feeder, daemon=True)
        t_feed.start()
        try:
            self._stream_once(str(out_path), output_format="mp4", max_duration_sec=duration_sec)
        finally:
            stop_tone.set()
            t_feed.join(timeout=2.0)
        return out_path


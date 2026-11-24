"""Continuous gapless radio broadcaster powered by a single persistent FFmpeg MP3 encoder.

Feeds raw s16le 44100Hz stereo PCM at strict 1.0x real-time pace (176,400 bytes/sec) into
a persistent FFmpeg process encoding CBR 128kbps MP3, and fans out the MP3 stream to all
connected HTTP clients with an instant-start ring buffer.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import random
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_WIDTH = 2  # s16le
BYTES_PER_SEC = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH  # 176,400 bytes/sec
CHUNK_MS = 100
CHUNK_BYTES = (BYTES_PER_SEC * CHUNK_MS) // 1000  # 17,640 bytes per 100ms
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass
class BroadcastState:
    started_at: float = field(default_factory=time.time)
    mode: str = "fallback"  # "talk", "fallback", or "paused"
    current_topic: str = "番組準備中（環境音・BGM）"
    current_detail: str = ""
    current_source_label: str = ""
    current_started_at: Optional[str] = None
    segments_played: int = 0
    recent_transcripts: list[dict[str, Any]] = field(default_factory=list)
    segment_elapsed_sec: float = 0.0
    segment_duration_sec: float = 0.0
    active_speaker: str = ""
    audio_level: float = 0.0
    live_subtitles: list[dict[str, Any]] = field(default_factory=list)


def _build_live_subtitles(
    transcript_raw: list[dict[str, Any]],
    elapsed_sec: float,
    max_items: int = 12,
) -> tuple[list[dict[str, Any]], str]:
    """Build audio-synced utterance list up to elapsed_sec and return (subtitles, active_speaker)."""
    utterances: list[dict[str, Any]] = []
    cur_speaker: Optional[str] = None
    cur_buf: list[str] = []
    cur_t = 0.0
    last_t = -10.0
    active_speaker = ""

    def _flush() -> None:
        nonlocal cur_speaker, cur_buf
        if cur_speaker is not None:
            txt = "".join(cur_buf).strip()
            if txt:
                utterances.append({"t": cur_t, "speaker": cur_speaker, "text": txt})
        cur_speaker = None
        cur_buf = []

    for ev in transcript_raw:
        t = float(ev.get("t", 0.0))
        if t > elapsed_sec + 0.15:
            break
        spk = str(ev.get("speaker", ""))
        txt = str(ev.get("text", ""))
        if not txt:
            continue
        if txt == " " and cur_speaker == spk:
            continue
        need_split = (
            spk != cur_speaker
            or (cur_speaker is not None and (t - last_t) > 3.0)
            or (len("".join(cur_buf)) > 55 and txt.endswith(("。", "！", "？", ".", "!", "?")))
        )
        if need_split:
            _flush()
            cur_speaker = spk
            cur_t = t
        cur_buf.append(txt)
        last_t = t
        if spk:
            active_speaker = spk

    _flush()
    # Clear active speaker if nobody spoke in the last 2.0 seconds
    if elapsed_sec - last_t > 2.0:
        active_speaker = ""
    return utterances[-max_items:], active_speaker


class RadioStreamer:
    """Manages persistent FFmpeg encoder, 1.0x PCM feeder, and multi-client MP3 fan-out."""

    def __init__(self, config: dict, base_dir: Path):
        self.config = config
        self.base_dir = base_dir

        acfg = config.get("audio", {})
        qcfg = config.get("queue", {})
        lcfg = config.get("logs", {})
        scfg = config.get("server", {})

        self.bitrate = str(acfg.get("mp3_bitrate", "128k"))
        self.bgm_dir = base_dir / acfg.get("bgm_dir", "assets/bgm")
        sid_cfg = acfg.get("station_id", {})
        self.station_id_enabled = bool(sid_cfg.get("enabled", False))
        self.station_id_text = str(sid_cfg.get("text", "ハルミナ無限ラジオ！")).strip()
        self.station_id_wav = base_dir / str(sid_cfg.get("cache_dir", "assets/cues")) / "station_id.wav"
        self.queue_dir = base_dir / qcfg.get("dir", "queue")
        self.logs_dir = base_dir / lcfg.get("dir", "logs")
        self.keep_transcripts = int(lcfg.get("keep_transcripts", 20))
        self.ring_buffer_bytes = int(scfg.get("ring_buffer_bytes", 65536))
        self.pause_when_no_listeners = bool(scfg.get("pause_when_no_listeners", True))
        self.idle_pause_grace_sec = float(scfg.get("idle_pause_grace_sec", 15.0))

        self.state = BroadcastState()
        self._state_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._has_listeners_event = threading.Event()
        self._last_listener_time: float = 0.0
        self._ffmpeg: Optional[subprocess.Popen] = None
        self._feeder_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: set[asyncio.Queue[bytes]] = set()
        self._ring: collections.deque[bytes] = collections.deque()
        self._ring_size: int = 0
        self._ring_lock = threading.Lock()

        self._ambient_pcm: Optional[bytes] = None
        self._ambient_offset: int = 0

        self._pcm_sinks: list[Callable[[bytes], None]] = []
        self._pcm_sinks_lock = threading.Lock()
        self._upcoming_cache: list[str] = []
        self._upcoming_cache_ts: float = 0.0

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()

        self._start_ffmpeg()
        self._feeder_thread = threading.Thread(target=self._feeder_loop, name="PCMFeeder", daemon=True)
        self._reader_thread = threading.Thread(target=self._reader_loop, name="MP3Reader", daemon=True)
        self._feeder_thread.start()
        self._reader_thread.start()
        logger.info("RadioStreamer started (persistent FFmpeg MP3 CBR %s @ 44.1kHz stereo)", self.bitrate)

    def stop(self) -> None:
        self._stop_event.set()
        self._has_listeners_event.set()
        if self._ffmpeg is not None:
            try:
                if self._ffmpeg.stdin:
                    self._ffmpeg.stdin.close()
            except Exception:
                pass
            try:
                self._ffmpeg.terminate()
                self._ffmpeg.wait(timeout=3)
            except Exception:
                try:
                    self._ffmpeg.kill()
                except Exception:
                    pass
            self._ffmpeg = None

        if self._feeder_thread and self._feeder_thread.is_alive():
            self._feeder_thread.join(timeout=3.0)
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3.0)
        logger.info("RadioStreamer stopped.")

    def _start_ffmpeg(self) -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(CHANNELS),
            "-i",
            "pipe:0",
            "-c:a",
            "libmp3lame",
            "-b:a",
            self.bitrate,
            "-minrate",
            self.bitrate,
            "-maxrate",
            self.bitrate,
            "-bufsize",
            self.bitrate,
            "-write_xing",
            "0",
            "-f",
            "mp3",
            "pipe:1",
        ]
        self._ffmpeg = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def subscribe(self) -> tuple[bytes, asyncio.Queue[bytes]]:
        """Register a new listener and return (initial_ring_buffer_bytes, live_queue)."""
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=128)
        with self._ring_lock:
            initial = b"".join(self._ring)
        self._subscribers.add(q)
        self._last_listener_time = time.monotonic()
        self._has_listeners_event.set()
        return initial, q

    def add_pcm_sink(self, callback: Callable[[bytes], None]) -> None:
        """Register an internal PCM consumer (e.g. YouTube RTMP streamer)."""
        with self._pcm_sinks_lock:
            if callback not in self._pcm_sinks:
                self._pcm_sinks.append(callback)
        self._last_listener_time = time.monotonic()
        self._has_listeners_event.set()

    def remove_pcm_sink(self, callback: Callable[[bytes], None]) -> None:
        with self._pcm_sinks_lock:
            if callback in self._pcm_sinks:
                self._pcm_sinks.remove(callback)
        if self.active_consumer_count == 0:
            self._last_listener_time = time.monotonic()
            self._has_listeners_event.clear()

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        self._subscribers.discard(q)
        if self.active_consumer_count == 0:
            self._last_listener_time = time.monotonic()
            self._has_listeners_event.clear()

    @property
    def listener_count(self) -> int:
        return len(self._subscribers)

    @property
    def active_consumer_count(self) -> int:
        with self._pcm_sinks_lock:
            sink_count = len(self._pcm_sinks)
        return len(self._subscribers) + sink_count

    def get_live_state(self) -> dict[str, Any]:
        """Fast lock-protected snapshot of live broadcast state & synced subtitles."""
        now = time.monotonic()
        if now - self._upcoming_cache_ts > 3.0:
            queued_files = self._list_queued_wavs()
            upcoming: list[str] = []
            for wav in queued_files[:3]:
                meta_file = wav.with_suffix(".json")
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            m = json.load(f)
                        upcoming.append(m.get("topic", {}).get("title", wav.stem))
                    except Exception:
                        upcoming.append(wav.stem)
                else:
                    upcoming.append(wav.stem)
            self._upcoming_cache = upcoming
            self._upcoming_cache_ts = now

        with self._state_lock:
            return {
                "station_name": self.config.get("server", {}).get("station_name", "ハルとミナの、無限ラジオ"),
                "mode": self.state.mode,
                "current_topic": self.state.current_topic,
                "current_detail": self.state.current_detail,
                "current_source_label": self.state.current_source_label,
                "current_started_at": self.state.current_started_at,
                "segment_elapsed_sec": round(self.state.segment_elapsed_sec, 2),
                "segment_duration_sec": round(self.state.segment_duration_sec, 2),
                "active_speaker": self.state.active_speaker,
                "audio_level": round(self.state.audio_level, 3),
                "live_subtitles": list(self.state.live_subtitles),
                "upcoming_topics": list(self._upcoming_cache),
                "listeners": self.listener_count,
            }

    def get_status(self, producer_status: str = "") -> dict[str, Any]:
        queued_files = self._list_queued_wavs()
        upcoming: list[str] = []
        for wav in queued_files:
            meta_file = wav.with_suffix(".json")
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    upcoming.append(m.get("topic", {}).get("title", wav.stem))
                except Exception:
                    upcoming.append(wav.stem)
            else:
                upcoming.append(wav.stem)

        with self._state_lock:
            return {
                "station_name": self.config.get("server", {}).get("station_name", "ハルとミナの、無限ラジオ"),
                "mode": self.state.mode,
                "current_topic": self.state.current_topic,
                "current_detail": self.state.current_detail,
                "current_source_label": self.state.current_source_label,
                "current_started_at": self.state.current_started_at,
                "segments_played": self.state.segments_played,
                "queue_count": len(queued_files),
                "upcoming_topics": upcoming,
                "listeners": self.listener_count,
                "uptime_sec": round(time.time() - self.state.started_at, 1),
                "producer_status": producer_status,
                "recent_transcripts": list(self.state.recent_transcripts),
            }

    def _list_queued_wavs(self) -> list[Path]:
        if not self.queue_dir.exists():
            return []
        return sorted(
            p
            for p in self.queue_dir.iterdir()
            if p.is_file() and p.suffix == ".wav" and not p.name.endswith(".tmp.wav")
        )

    def _ensure_ambient_pcm(self) -> bytes:
        """Generate a 20-second gentle ambient drone loop via FFmpeg if no BGM files exist."""
        if self._ambient_pcm is not None:
            return self._ambient_pcm
        logger.info("Generating fallback ambient sound loop via FFmpeg...")
        expr = (
            "0.045*(0.5+0.5*sin(2*PI*0.05*t))*"
            "(sin(2*PI*130.81*t)+0.7*sin(2*PI*196.00*t)+0.5*sin(2*PI*246.94*t)+0.35*sin(2*PI*329.63*t))"
        )
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc='{expr}':s={SAMPLE_RATE}:d=20",
            "-af",
            "aformat=channel_layouts=stereo,lowpass=f=900,afade=t=in:st=0:d=1.5,afade=t=out:st=18.5:d=1.5",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(CHANNELS),
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode == 0 and len(proc.stdout) >= CHUNK_BYTES:
            # Align to frame boundary (4 bytes per stereo s16le sample)
            aligned = len(proc.stdout) - (len(proc.stdout) % 4)
            self._ambient_pcm = proc.stdout[:aligned]
        else:
            logger.warning("Fallback ambient generation failed; using silence.")
            self._ambient_pcm = b"\x00" * BYTES_PER_SEC
        return self._ambient_pcm

    def _pick_bgm_file(self) -> Optional[Path]:
        if not self.bgm_dir.exists():
            return None
        files = [p for p in self.bgm_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        return random.choice(files) if files else None

    def _next_fallback_chunk(self) -> bytes:
        pcm = self._ensure_ambient_pcm()
        out = bytearray()
        while len(out) < CHUNK_BYTES:
            remain = CHUNK_BYTES - len(out)
            part = pcm[self._ambient_offset : self._ambient_offset + remain]
            out.extend(part)
            self._ambient_offset += len(part)
            if self._ambient_offset >= len(pcm):
                self._ambient_offset = 0
        return bytes(out)

    def _archive_and_cleanup_segment(self, wav_path: Path) -> None:
        meta_path = wav_path.with_suffix(".json")
        if meta_path.exists():
            try:
                dst = self.logs_dir / meta_path.name
                shutil.copy2(meta_path, dst)
                # Prune old transcript logs
                logs = sorted(self.logs_dir.glob("seg_*.json"))
                for old in logs[: max(0, len(logs) - self.keep_transcripts)]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            except Exception as exc:
                logger.warning("Failed to archive transcript %s: %s", meta_path, exc)
            try:
                meta_path.unlink()
            except OSError:
                pass
        try:
            wav_path.unlink()
        except OSError:
            pass

    def _wait_if_no_listeners(self) -> bool:
        """If pause_when_no_listeners is enabled and no listeners are connected past grace period,
        pause feeding until a listener connects. Returns True if paused & resumed, False if no pause."""
        if not self.pause_when_no_listeners:
            return False
        if self.active_consumer_count > 0:
            return False
        # Before any listener has ever connected, pause immediately; after disconnect, wait grace_sec
        if self._last_listener_time > 0.0:
            elapsed = time.monotonic() - self._last_listener_time
            if elapsed < self.idle_pause_grace_sec:
                return False

        with self._state_lock:
            prev_mode = self.state.mode
            prev_topic = self.state.current_topic
            prev_detail = self.state.current_detail
            self.state.mode = "paused"
            self.state.current_topic = "一時停止中（リスナー待機）"
            self.state.current_detail = "再生機器からの接続があると自動で再開します"
            self.state.active_speaker = ""
            self.state.audio_level = 0.0

        # Clear stale ring buffer so the next listener gets fresh audio
        with self._ring_lock:
            self._ring.clear()
            self._ring_size = 0

        logger.info("No active listeners -> RadioStreamer paused (0%% CPU, queue preserved).")
        while not self._stop_event.is_set() and self.active_consumer_count == 0:
            self._has_listeners_event.wait(timeout=0.5)

        if self._stop_event.is_set():
            return True

        logger.info("Listener connected -> RadioStreamer resumed.")
        with self._state_lock:
            self.state.mode = prev_mode
            self.state.current_topic = prev_topic
            self.state.current_detail = prev_detail
        return True

    def _feeder_loop(self) -> None:
        """Feed PCM to persistent FFmpeg stdin at strict 1.0x speed (176,400 bytes/sec)."""
        origin = time.monotonic()
        total_bytes_sent = 0

        def _reset_clock() -> None:
            nonlocal origin, total_bytes_sent
            origin = time.monotonic()
            total_bytes_sent = 0

        def _write_paced(chunk: bytes) -> bool:
            nonlocal total_bytes_sent
            if self._wait_if_no_listeners():
                _reset_clock()
            if self._stop_event.is_set() or self._ffmpeg is None or self._ffmpeg.stdin is None:
                return False
            try:
                self._ffmpeg.stdin.write(chunk)
                self._ffmpeg.stdin.flush()
            except (BrokenPipeError, OSError):
                if not self._stop_event.is_set():
                    logger.error("FFmpeg stdin pipe broken; stopping feeder.")
                return False

            # Fan out raw PCM to any registered internal sinks (e.g. YouTube RTMP)
            with self._pcm_sinks_lock:
                sinks = list(self._pcm_sinks)
            for sink in sinks:
                try:
                    sink(chunk)
                except Exception:
                    pass

            # Update audio level for visualizers
            if len(chunk) >= 4:
                samples = np.frombuffer(chunk, dtype="<i2")
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) / 32768.0
                with self._state_lock:
                    self.state.audio_level = min(1.0, rms * 2.5)

            total_bytes_sent += len(chunk)
            target_time = origin + (total_bytes_sent / BYTES_PER_SEC)
            sleep_sec = target_time - time.monotonic()
            if sleep_sec > 0:
                self._stop_event.wait(sleep_sec)
            return not self._stop_event.is_set()

        while not self._stop_event.is_set():
            if self._wait_if_no_listeners():
                _reset_clock()
                if self._stop_event.is_set():
                    return
            queued = self._list_queued_wavs()
            if queued:
                wav_path = queued[0]
                meta_path = wav_path.with_suffix(".json")
                topic_title = wav_path.stem
                topic_detail = ""
                topic_source_label = ""
                transcript_lines: list[str] = []
                transcript_raw: list[dict[str, Any]] = []
                seg_duration_sec = 0.0
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        tmeta = meta.get("topic", {})
                        topic_title = tmeta.get("title", topic_title)
                        topic_detail = tmeta.get("detail", "")
                        src_raw = str(tmeta.get("source", ""))
                        if src_raw.startswith("rss:"):
                            src_url = str(tmeta.get("source_url", "") or src_raw[4:]).strip()
                            src_name = str(tmeta.get("source_name", "")).strip()
                            domain = urlparse(src_url).netloc if src_url else ""
                            if src_name and domain and domain not in src_name:
                                topic_source_label = f"出典: {src_name} ({domain})"
                            elif src_name:
                                topic_source_label = f"出典: {src_name}"
                            elif domain:
                                topic_source_label = f"出典: {domain}"
                        transcript_lines = meta.get("transcript_lines", [])
                        transcript_raw = meta.get("transcript_raw", [])
                        seg_duration_sec = float(meta.get("duration_sec", 0.0))
                    except Exception:
                        pass

                # Play voice Station ID bumper ("ハルとミナの、無限ラジオ") cleanly before each topic
                if self.station_id_enabled and self.station_id_wav.exists():
                    host_a = self.config.get("hosts", {}).get("host_a", {}).get("name", "ハル")
                    host_b = self.config.get("hosts", {}).get("host_b", {}).get("name", "ミナ")
                    with self._state_lock:
                        self.state.mode = "talk"
                        self.state.current_topic = topic_title
                        self.state.current_detail = topic_detail
                        self.state.current_source_label = topic_source_label
                        self.state.segment_elapsed_sec = 0.0
                        self.state.segment_duration_sec = seg_duration_sec
                        self.state.active_speaker = host_a
                        self.state.live_subtitles = [
                            {"t": 0.0, "speaker": f"{host_a}＆{host_b}", "text": self.station_id_text}
                        ]
                    logger.info("Station ID bumper: %s", self.station_id_text)
                    try:
                        with wave.open(str(self.station_id_wav), "rb") as sid_wf:
                            frames_per_chunk = CHUNK_BYTES // (CHANNELS * SAMPLE_WIDTH)
                            while not self._stop_event.is_set():
                                sid_data = sid_wf.readframes(frames_per_chunk)
                                if not sid_data:
                                    break
                                if not _write_paced(sid_data):
                                    return
                    except Exception as exc:
                        logger.warning("Failed to play station ID %s: %s", self.station_id_wav, exc)

                with self._state_lock:
                    self.state.mode = "talk"
                    self.state.current_topic = topic_title
                    self.state.current_detail = topic_detail
                    self.state.current_source_label = topic_source_label
                    self.state.current_started_at = datetime.now().isoformat(timespec="seconds")
                    self.state.segment_elapsed_sec = 0.0
                    self.state.segment_duration_sec = seg_duration_sec
                    self.state.live_subtitles = []
                    self.state.active_speaker = ""
                    self.state.recent_transcripts.insert(
                        0,
                        {
                            "topic": topic_title,
                            "started_at": self.state.current_started_at,
                            "lines": transcript_lines[:20],
                        },
                    )
                    self.state.recent_transcripts = self.state.recent_transcripts[:5]

                logger.info("On Air [TALK]: %s (%s)", topic_title, wav_path.name)
                try:
                    with wave.open(str(wav_path), "rb") as wf:
                        if seg_duration_sec <= 0.0 and wf.getframerate() > 0:
                            seg_duration_sec = wf.getnframes() / float(wf.getframerate())
                            with self._state_lock:
                                self.state.segment_duration_sec = seg_duration_sec
                        frames_per_chunk = CHUNK_BYTES // (CHANNELS * SAMPLE_WIDTH)
                        seg_bytes_fed = 0
                        while not self._stop_event.is_set():
                            data = wf.readframes(frames_per_chunk)
                            if not data:
                                break
                            elapsed_s = seg_bytes_fed / BYTES_PER_SEC
                            subs, spk = _build_live_subtitles(transcript_raw, elapsed_s)
                            with self._state_lock:
                                self.state.segment_elapsed_sec = elapsed_s
                                self.state.live_subtitles = subs
                                self.state.active_speaker = spk
                            if not _write_paced(data):
                                return
                            seg_bytes_fed += len(data)
                except Exception as exc:
                    logger.error("Error streaming %s: %s", wav_path, exc)

                with self._state_lock:
                    self.state.segments_played += 1
                    self.state.active_speaker = ""
                self._archive_and_cleanup_segment(wav_path)
            else:
                # Queue empty -> stream fallback BGM or ambient sound
                bgm_file = self._pick_bgm_file()
                with self._state_lock:
                    self.state.mode = "fallback"
                    self.state.current_topic = (
                        f"BGMインターミッション ({bgm_file.name})" if bgm_file else "番組準備中（環境音）"
                    )
                    self.state.current_detail = "次のトークコーナーを準備中です..."
                    self.state.current_source_label = ""
                    self.state.segment_elapsed_sec = 0.0
                    self.state.segment_duration_sec = 0.0
                    self.state.active_speaker = ""
                    self.state.live_subtitles = []
                    if self.state.current_started_at is None:
                        self.state.current_started_at = datetime.now().isoformat(timespec="seconds")

                if bgm_file is not None:
                    dec = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            str(bgm_file),
                            "-f",
                            "s16le",
                            "-ar",
                            str(SAMPLE_RATE),
                            "-ac",
                            str(CHANNELS),
                            "pipe:1",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    try:
                        assert dec.stdout is not None
                        while not self._stop_event.is_set():
                            # Break smoothly once a talk segment is ready in the queue
                            if self._list_queued_wavs():
                                break
                            data = dec.stdout.read(CHUNK_BYTES)
                            if not data:
                                break
                            if not _write_paced(data):
                                return
                    finally:
                        dec.terminate()
                        try:
                            dec.wait(timeout=2)
                        except Exception:
                            dec.kill()
                else:
                    chunk = self._next_fallback_chunk()
                    if not _write_paced(chunk):
                        return

    def _reader_loop(self) -> None:
        """Read continuous MP3 chunks from FFmpeg stdout and fan out to all HTTP listeners."""
        assert self._ffmpeg is not None and self._ffmpeg.stdout is not None
        stdout = self._ffmpeg.stdout
        while not self._stop_event.is_set():
            data = stdout.read(4096)
            if not data:
                break
            with self._ring_lock:
                self._ring.append(data)
                self._ring_size += len(data)
                while self._ring_size > self.ring_buffer_bytes and len(self._ring) > 1:
                    popped = self._ring.popleft()
                    self._ring_size -= len(popped)

            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._fanout_chunk, data)

    def _fanout_chunk(self, data: bytes) -> None:
        dead: list[asyncio.Queue[bytes]] = []
        for q in list(self._subscribers):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                # Drop oldest frame for lagging listener so broadcast stays live
                try:
                    q.get_nowait()
                    q.put_nowait(data)
                except Exception:
                    dead.append(q)
        for q in dead:
            self._subscribers.discard(q)


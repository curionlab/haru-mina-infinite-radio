"""CLI entry point for Haru-4B Local Infinite AI Radio."""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import yaml

from src.haru_producer import (
    HaruDialogueEngine,
    ProducerWorker,
    ensure_station_id,
    ensure_voices,
    format_transcript_lines,
    query_nvidia_vram_mb,
)
from src.http_server import create_app
from src.radio_streamer import RadioStreamer
from src.speech_capi import SpeechLibrary, find_speech_dll, find_speech_exe, resolve_model_path
from src.topic_manager import TopicManager
from src.youtube_streamer import YouTubeStreamer

BASE_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_check(config: dict) -> int:
    """Run full environment diagnostics for speech.cpp, GPU, model, voices, and FFmpeg."""
    print("=== Haru-4B Local AI Radio: Environment Check ===")
    ok_all = True

    # 1. speech.exe & speech.dll
    exe = find_speech_exe()
    if exe:
        proc = subprocess.run([str(exe), "--version"], capture_output=True, text=True, check=False)
        print(f"[OK] speech CLI: {exe} ({proc.stdout.strip()})")
    else:
        print("[NG] speech CLI not found")
        ok_all = False

    try:
        dll = find_speech_dll(config["model"].get("speech_dll", ""))
        lib = SpeechLibrary(str(dll))
        maj, mino = lib.api_version()
        print(f"[OK] speech.dll: {dll} (version={lib.version()}, C API={maj}.{mino})")
        devices = lib.list_devices()
        print("     Detected speech.cpp devices:")
        has_gpu = False
        for d in devices:
            gb_total = d.total_bytes / (1024**3)
            gb_free = d.free_bytes / (1024**3)
            print(f"       - {d.name:8s} [{d.kind:4s}] {d.description} ({gb_free:.1f}/{gb_total:.1f} GB free)")
            if d.kind in ("gpu", "igpu"):
                has_gpu = True
        if not has_gpu:
            print("[WARN] No Vulkan GPU detected by speech.cpp; will fall back to CPU.")
    except Exception as exc:
        print(f"[NG] speech.dll / devices check failed: {exc}")
        ok_all = False

    # 2. nvidia-smi
    vram = query_nvidia_vram_mb()
    if vram is not None:
        print(f"[OK] nvidia-smi: available (current VRAM used: {vram:.0f} MB)")
    else:
        print("[WARN] nvidia-smi: unavailable (GPU driver / NVML not active)")

    # 3. Model
    model_name = config["model"]["name"]
    try:
        mpath = resolve_model_path(model_name)
        size_gb = mpath.stat().st_size / (1024**3)
        print(f"[OK] Model {model_name}: {mpath} ({size_gb:.2f} GB)")
    except Exception as exc:
        print(f"[NG] Model {model_name}: {exc}")
        ok_all = False

    # 4. Voices
    for hkey in ("host_a", "host_b"):
        hcfg = config["hosts"][hkey]
        wav_p = BASE_DIR / hcfg["wav_path"]
        gguf_p = BASE_DIR / hcfg["voice_gguf_path"]
        if gguf_p.exists():
            print(f"[OK] Voice {hkey} ({hcfg['name']}): {gguf_p}")
        elif wav_p.exists():
            print(f"[INFO] Voice {hkey} ({hcfg['name']}): WAV found at {wav_p} (will encode .voice.gguf on start)")
        else:
            print(f"[INFO] Voice {hkey} ({hcfg['name']}): not found yet (will auto-generate fallback sample on start)")

    # 5. FFmpeg / FFprobe
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        if path:
            p = subprocess.run([path, "-version"], capture_output=True, text=True, check=False)
            first = p.stdout.splitlines()[0] if p.stdout else ""
            print(f"[OK] {tool}: {first}")
        else:
            print(f"[NG] {tool}: not found in PATH")
            ok_all = False

    print("=================================================")
    return 0 if ok_all else 1


def cmd_test_segment(config: dict, duration: float) -> int:
    """Generate a single 2-speaker test corner into ./out/ and report performance metrics."""
    out_dir = BASE_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    topic_mgr = TopicManager(config, BASE_DIR)
    topic = topic_mgr.next_topic(mark=False)

    engine = HaruDialogueEngine(config, BASE_DIR)
    try:
        engine.load()
        seg = engine.generate_segment(
            topic=topic,
            out_dir=out_dir,
            duration_sec=duration,
            keep_raw=True,
        )
    finally:
        engine.close()

    m = seg.metrics
    print("\n=== Haru-4B Test Segment Complete ===")
    print(f"Topic             : {seg.topic.title}")
    print(f"Output WAV (44.1k): {seg.wav_path}")
    print(f"Raw WAV (24k)     : {seg.raw_wav_path}")
    print(f"Metadata JSON     : {seg.meta_path}")
    print(f"Device            : {m.device}")
    print(f"Duration          : {m.duration_sec:.2f} s ({m.frames_generated} frames @ 133.3ms)")
    print(f"Total Gen Time    : {m.total_gen_sec:.2f} s")
    print(f"Frame Pair Time   : avg={m.avg_pair_frame_ms:.1f} ms, p50={m.p50_pair_frame_ms:.1f} ms, p95={m.p95_pair_frame_ms:.1f} ms (per 133.3ms frame pair)")
    print(f"Real-Time Factor  : RTF={m.rtf:.3f} ({1.0 / max(m.rtf, 1e-6):.2f}x real-time)")
    print(f"Peak VRAM (NVML)  : {f'{m.peak_vram_mb:.0f} MB' if m.peak_vram_mb is not None else 'N/A (nvidia-smi unavailable)'}")
    print(f"Speech Ratio      : {m.speech_ratio:.2%}")
    print(f"Attempts          : {m.attempts}")
    print("\n--- Transcript Excerpt ---")
    lines = format_transcript_lines(seg.transcript)
    for line in lines[:20]:
        print(line)
    if not lines:
        print("(no text emitted)")
    print("=====================================")
    return 0


def cmd_test_youtube_video(config: dict, duration: float) -> int:
    """Render a short local MP4 preview of the YouTube Live stream layout with sample Japanese subtitles."""
    out_path = BASE_DIR / "out" / "youtube_preview.mp4"
    sample_state = {
        "station_name": config.get("server", {}).get("station_name", "ハルとミナの、無限ラジオ"),
        "mode": "talk",
        "current_topic": "深夜につい食べたくなる夜食とラジオの楽しみ方",
        "current_detail": "カップ麺、お茶漬け、トーストなど、夜中に食べると妙に美味しいものについて。",
        "segment_elapsed_sec": 42.0,
        "segment_duration_sec": 180.0,
        "active_speaker": config.get("hosts", {}).get("host_a", {}).get("name", "ハル"),
        "audio_level": 0.45,
        "live_subtitles": [
            {"t": 5.0, "speaker": "ハル", "text": "こんばんは、深夜のローカルAIラジオへようこそ。今日のお題は夜食ですね。"},
            {"t": 15.0, "speaker": "ミナ", "text": "夜中に食べるカップ麺とかお茶漬けって、どうしてあんなに美味しいんでしょうね！"},
            {"t": 28.0, "speaker": "ハル", "text": "わかります。ちょっと小腹が空いたときに温かいものを食べるとほっとしますよね。"},
        ],
        "upcoming_topics": ["雨の日の理想の過ごし方"],
    }
    yt = YouTubeStreamer(config, BASE_DIR, get_live_state_fn=lambda: sample_state, stream_key_override="preview")
    yt.render_preview_video(out_path, duration_sec=duration)
    size_kb = out_path.stat().st_size / 1024.0 if out_path.exists() else 0.0
    print(f"YouTube preview video saved: {out_path} ({size_kb:.1f} KB)")
    return 0


async def _run_server(config: dict, enable_youtube: bool = False, youtube_key: str = "") -> None:
    from aiohttp import web

    # Prepare voices & Station ID bumper upfront
    ensure_voices(config, BASE_DIR)
    ensure_station_id(config, BASE_DIR)

    topic_mgr = TopicManager(config, BASE_DIR)
    engine = HaruDialogueEngine(config, BASE_DIR)
    producer = ProducerWorker(engine, topic_mgr, config, BASE_DIR)
    streamer = RadioStreamer(config, BASE_DIR)

    yt_cfg = config.get("youtube", {})
    use_youtube = enable_youtube or bool(yt_cfg.get("enabled", False))
    yt_streamer: YouTubeStreamer | None = None
    if use_youtube:
        yt_streamer = YouTubeStreamer(
            config=config,
            base_dir=BASE_DIR,
            get_live_state_fn=streamer.get_live_state,
            stream_key_override=youtube_key,
        )
        streamer.add_pcm_sink(yt_streamer.on_pcm_chunk)

    loop = asyncio.get_running_loop()
    streamer.start(loop)
    producer.start()
    if yt_streamer is not None:
        yt_streamer.start()

    app = create_app(
        streamer,
        config,
        producer_status_fn=lambda: producer.status_text,
        youtube_status_fn=(lambda: yt_streamer.status_text) if yt_streamer is not None else None,
    )
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()

    scfg = config.get("server", {})
    host = str(scfg.get("host", "0.0.0.0"))
    port = int(scfg.get("port", 8000))
    site = web.TCPSite(runner, host=host, port=port)
    try:
        await site.start()
    except OSError as exc:
        logging.getLogger(__name__).error(
            "Failed to bind %s:%d (%s). Another radio server instance may already be running on port %d.",
            host,
            port,
            exc,
            port,
        )
        if yt_streamer is not None:
            streamer.remove_pcm_sink(yt_streamer.on_pcm_chunk)
            yt_streamer.stop()
        await runner.cleanup()
        producer.stop()
        streamer.stop()
        raise

    logging.getLogger(__name__).info(
        "ハルとミナの、無限ラジオ is ON AIR! Stream: http://%s:%d/radio.mp3 | Status: http://%s:%d/status | Overlay: http://%s:%d/overlay",
        host,
        port,
        host,
        port,
        host,
        port,
    )

    stop_event = asyncio.Event()

    def _handle_sig(*_args) -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_sig)
        except Exception:
            pass

    try:
        await stop_event.wait()
    finally:
        logging.getLogger(__name__).info("Shutting down radio station...")
        if yt_streamer is not None:
            streamer.remove_pcm_sink(yt_streamer.on_pcm_chunk)
            yt_streamer.stop()
        await runner.cleanup()
        producer.stop()
        streamer.stop()


def cmd_serve(config: dict, enable_youtube: bool = False, youtube_key: str = "") -> int:
    try:
        asyncio.run(_run_server(config, enable_youtube=enable_youtube, youtube_key=youtube_key))
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Haru-4B Local Infinite AI Radio")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Diagnose environment and dependencies")
    group.add_argument("--serve", action="store_true", help="Start producer and HTTP radio station")
    group.add_argument("--test-segment", action="store_true", help="Generate one test dialogue segment into ./out/")
    group.add_argument(
        "--test-youtube-video",
        action="store_true",
        help="Render a short local MP4 preview of the YouTube Live stream layout into ./out/youtube_preview.mp4",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds for test commands")
    parser.add_argument("--youtube", action="store_true", help="Enable direct RTMP streaming to YouTube Live Studio")
    parser.add_argument(
        "--youtube-key",
        default="",
        help="YouTube Live stream key (overrides YOUTUBE_STREAM_KEY env var and config.yaml)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = BASE_DIR / cfg_path
    config = load_config(cfg_path)

    if args.check:
        return cmd_check(config)
    if args.test_segment:
        return cmd_test_segment(config, duration=args.duration)
    if args.test_youtube_video:
        dur = 5.0 if args.duration == 30.0 else args.duration
        return cmd_test_youtube_video(config, duration=dur)
    if args.serve:
        return cmd_serve(config, enable_youtube=args.youtube, youtube_key=args.youtube_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())


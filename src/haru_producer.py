"""Haru-4B 2-speaker full-duplex dialogue producer, voice preparation, and audio mastering."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.moderation import check_transcript_safety
from src.speech_capi import (
    SpeechLibrary,
    SpeechModel,
    find_speech_exe,
    resolve_model_path,
)
from src.topic_manager import Topic, TopicManager

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass
class SegmentMetrics:
    frames_generated: int
    duration_sec: float
    total_gen_sec: float
    avg_pair_frame_ms: float
    p50_pair_frame_ms: float
    p95_pair_frame_ms: float
    rtf: float  # total_gen_sec / duration_sec (< 1.0 means faster than real-time)
    device: str
    peak_vram_mb: Optional[float]
    speech_ratio: float
    attempts: int


@dataclass
class GeneratedSegment:
    topic: Topic
    wav_path: Path
    meta_path: Path
    raw_wav_path: Optional[Path]
    transcript: list[dict[str, Any]]
    metrics: SegmentMetrics
    seed_a: int
    seed_b: int


def query_nvidia_vram_mb() -> Optional[float]:
    """Query current NVIDIA GPU memory usage in MB via nvidia-smi, or None if unavailable."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0].strip()
            return float(first_line)
    except Exception:
        pass
    return None


def _run_cmd(cmd: list[str], desc: str) -> subprocess.CompletedProcess:
    logger.info("Running %s: %s", desc, " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{desc} failed (exit {proc.returncode}):\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
    return proc


def ensure_voices(config: dict, base_dir: Path) -> dict[str, Path]:
    """Ensure host_a.voice.gguf and host_b.voice.gguf exist.

    1. Reuse existing .voice.gguf if present.
    2. If user's .wav exists (>= 3s), encode to .voice.gguf via `speech voice`.
    3. If .wav does not exist yet, auto-generate a 5-10s Japanese sample WAV via
       speech.cpp's lightweight TTS model (e.g. qwen3-tts-0.6b), then encode to .voice.gguf.
    """
    speech_exe = find_speech_exe()
    if not speech_exe:
        raise FileNotFoundError("speech.exe not found. Install speech.cpp first.")

    model_name = config["model"]["name"]
    device = config["model"].get("device", "auto")
    voices_cfg = config.get("voices", {})
    fallback_tts_model = voices_cfg.get("fallback_tts_model", "qwen3-tts-0.6b")
    fallback_samples = voices_cfg.get("fallback_samples", {})

    result: dict[str, Path] = {}
    for host_key in ("host_a", "host_b"):
        hcfg = config["hosts"][host_key]
        wav_path = base_dir / hcfg["wav_path"]
        gguf_path = base_dir / hcfg["voice_gguf_path"]
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)

        if gguf_path.exists() and gguf_path.stat().st_size > 0:
            logger.info("Using existing voice file for %s: %s", host_key, gguf_path)
            result[host_key] = gguf_path
            continue

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            sample_cfg = fallback_samples.get(host_key, {})
            tts_voice = sample_cfg.get(
                "tts_voice", "dylan" if host_key == "host_a" else "ono_anna"
            )
            sample_text = sample_cfg.get(
                "text",
                "こんばんは、ローカルAIラジオへようこそ。今日もゆったりとおしゃべりしていきましょう。",
            )
            logger.info(
                "Reference WAV missing for %s (%s). Generating fallback voice sample with %s (voice=%s)...",
                host_key,
                wav_path,
                fallback_tts_model,
                tts_voice,
            )
            cmd = [
                str(speech_exe),
                "tts",
                fallback_tts_model,
                "--voice",
                tts_voice,
                "--language",
                "ja",
                "--device",
                device,
                "-o",
                str(wav_path),
                sample_text,
            ]
            _run_cmd(cmd, f"fallback TTS generation for {host_key}")

        logger.info("Encoding voice file %s from %s (using CPU for maximum codec precision)...", gguf_path, wav_path)
        cmd = [
            str(speech_exe),
            "voice",
            model_name,
            "--device",
            "cpu",
            str(wav_path),
            str(gguf_path),
        ]
        _run_cmd(cmd, f"speech voice encoding for {host_key}")
        result[host_key] = gguf_path

    return result


def _load_mono_float32(wav_path: Path, target_sr: int) -> np.ndarray:
    """Decode any audio file to mono float32 PCM at target_sr via FFmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-f",
        "f32le",
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def ensure_wrapup_cues(
    config: dict,
    base_dir: Path,
    target_sr: int,
    frame_samples: int,
) -> list[tuple[str, list[np.ndarray]]]:
    """Ensure cached wrap-up cue WAVs exist and return list of (phrase_text, frame_chunks)."""
    wcfg = config.get("dialogue", {}).get("wrapup_cue", {})
    if not wcfg.get("enabled", False):
        return []

    phrases = [str(p).strip() for p in (wcfg.get("phrases") or []) if str(p).strip()]
    if not phrases:
        return []

    speech_exe = find_speech_exe()
    if not speech_exe:
        logger.warning("speech.exe not found; disabling wrap-up audio cues.")
        return []

    vcfg = config.get("voices", {})
    tts_model = str(vcfg.get("fallback_tts_model", "qwen3-tts-0.6b"))
    tts_voice = str(vcfg.get("fallback_samples", {}).get("host_b", {}).get("tts_voice", "ono_anna"))
    cache_dir = base_dir / str(wcfg.get("cache_dir", "assets/cues"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    cues: list[tuple[str, list[np.ndarray]]] = []
    for idx, phrase in enumerate(phrases):
        wav_path = cache_dir / f"wrapup_{idx}.wav"
        txt_path = cache_dir / f"wrapup_{idx}.txt"
        need_gen = not wav_path.exists() or wav_path.stat().st_size == 0
        if not need_gen and txt_path.exists():
            if txt_path.read_text(encoding="utf-8").strip() != phrase:
                need_gen = True
        if need_gen:
            logger.info("Generating cached wrap-up cue #%d (%s): %s", idx, tts_voice, phrase)
            cmd = [
                str(speech_exe),
                "tts",
                tts_model,
                "--voice",
                tts_voice,
                "--language",
                "ja",
                "--device",
                "cpu",
                "-o",
                str(wav_path),
                phrase,
            ]
            _run_cmd(cmd, f"wrap-up cue #{idx} TTS")
            txt_path.write_text(phrase, encoding="utf-8")

        audio = _load_mono_float32(wav_path, target_sr)
        if len(audio) == 0:
            continue
        chunks: list[np.ndarray] = []
        for off in range(0, len(audio), frame_samples):
            ch = audio[off : off + frame_samples]
            if len(ch) < frame_samples:
                ch = np.pad(ch, (0, frame_samples - len(ch)))
            chunks.append(ch)
        cues.append((phrase, chunks))
    return cues


def _synthesize_host_b_cue_frames(
    phrase: str,
    config: dict,
    target_sr: int,
    frame_samples: int,
    # pyrefly: ignore[invalid-syntax]
    max_sec: float = 12.0,
) -> list[np.ndarray]:
    """Synthesize a one-off Host B cue phrase on CPU via fallback TTS and return framed float32 chunks."""
    speech_exe = find_speech_exe()
    if not speech_exe:
        return []
    vcfg = config.get("voices", {})
    tts_model = str(vcfg.get("fallback_tts_model", "qwen3-tts-0.6b"))
    tts_voice = str(vcfg.get("fallback_samples", {}).get("host_b", {}).get("tts_voice", "ono_anna"))

    with tempfile.TemporaryDirectory() as td:
        raw_wav = Path(td) / "cue_raw.wav"
        cmd = [
            str(speech_exe),
            "tts",
            tts_model,
            "--voice",
            tts_voice,
            "--language",
            "ja",
            "--device",
            "cpu",
            "-o",
            str(raw_wav),
            phrase,
        ]
        _run_cmd(cmd, "opening cue TTS")
        # Trim ONLY leading and trailing silence (never cut mid-sentence on comma pauses!)
        cmd_dec = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_wav),
            "-af",
            f"silenceremove=start_periods=1:start_silence=0.05:start_threshold=-35dB,"
            f"areverse,silenceremove=start_periods=1:start_silence=0.05:start_threshold=-35dB,areverse,"
            f"atrim=0:{max_sec:.1f}",
            "-f",
            "f32le",
            "-ar",
            str(target_sr),
            "-ac",
            "1",
            "pipe:1",
        ]
        proc = subprocess.run(cmd_dec, capture_output=True, check=True)
        audio = np.frombuffer(proc.stdout, dtype="<f4").copy()

    if len(audio) == 0:
        return []
    chunks: list[np.ndarray] = []
    for off in range(0, len(audio), frame_samples):
        ch = audio[off : off + frame_samples]
        if len(ch) < frame_samples:
            ch = np.pad(ch, (0, frame_samples - len(ch)))
        chunks.append(ch)
    return chunks


def ensure_station_id(config: dict, base_dir: Path) -> Optional[Path]:
    """Generate and cache a stereo unison voice Station ID WAV (Haru + Mina) played between topics."""
    acfg = config.get("audio", {})
    scfg = acfg.get("station_id", {})
    if not scfg.get("enabled", False):
        return None

    text = str(scfg.get("text", "ハルミナ無限ラジオ！")).strip()
    if not text:
        return None

    cache_dir = base_dir / str(scfg.get("cache_dir", "assets/cues"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_wav = cache_dir / "station_id.wav"
    txt_path = cache_dir / "station_id.txt"

    if out_wav.exists() and out_wav.stat().st_size > 0 and txt_path.exists():
        if txt_path.read_text(encoding="utf-8").strip() == text:
            return out_wav

    speech_exe = find_speech_exe()
    if not speech_exe:
        logger.warning("speech.exe not found; skipping Station ID generation.")
        return None

    vcfg = config.get("voices", {})
    tts_model = str(vcfg.get("fallback_tts_model", "qwen3-tts-0.6b"))
    voice_a = str(vcfg.get("fallback_samples", {}).get("host_a", {}).get("tts_voice", "dylan"))
    voice_b = str(vcfg.get("fallback_samples", {}).get("host_b", {}).get("tts_voice", "ono_anna"))
    out_sr = int(acfg.get("output_sample_rate", 44100))
    target_i = float(acfg.get("loudnorm_target_i", -16.0))
    target_tp = float(acfg.get("loudnorm_target_tp", -1.5))
    target_lra = float(acfg.get("loudnorm_target_lra", 11.0))

    logger.info("Generating Station ID jingle (%s + %s): '%s'", voice_a, voice_b, text)
    with tempfile.TemporaryDirectory() as td:
        wav_a = Path(td) / "id_a.wav"
        wav_b = Path(td) / "id_b.wav"
        for vname, out_p, label in ((voice_a, wav_a, "host_a"), (voice_b, wav_b, "host_b")):
            cmd = [
                str(speech_exe),
                "tts",
                tts_model,
                "--voice",
                vname,
                "--language",
                "ja",
                "--device",
                "cpu",
                "-o",
                str(out_p),
                text,
            ]
            _run_cmd(cmd, f"station ID TTS ({label})")

        # Trim leading silence and stop after the first utterance (prevents short-text TTS tail babble),
        # pan slightly L/R, mix in unison, normalize, and pad gently
        tmp_out = out_wav.with_suffix(".tmp.wav")
        filter_complex = (
            f"[0:a]aresample={out_sr},"
            f"silenceremove=start_periods=1:start_silence=0.05:start_threshold=-35dB:"
            f"stop_periods=1:stop_duration=0.35:stop_threshold=-35dB,atrim=0:4.0,"
            f"pan=stereo|c0=0.80*c0|c1=0.55*c0[va];"
            f"[1:a]aresample={out_sr},"
            f"silenceremove=start_periods=1:start_silence=0.05:start_threshold=-35dB:"
            f"stop_periods=1:stop_duration=0.35:stop_threshold=-35dB,atrim=0:4.0,"
            f"pan=stereo|c0=0.55*c0|c1=0.80*c0[vb];"
            f"[va][vb]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
            f"loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA={target_lra:.1f},"
            f"adelay=250|250,apad=pad_dur=0.50[out]"
        )
        cmd_mix = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_a),
            "-i",
            str(wav_b),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ar",
            str(out_sr),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(tmp_out),
        ]
        _run_cmd(cmd_mix, "station ID stereo mix")
        os.replace(tmp_out, out_wav)
        txt_path.write_text(text, encoding="utf-8")

    logger.info("Station ID ready: %s", out_wav)
    return out_wav


def _write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray, sample_rate: int) -> None:
    """Write float32 [-1, 1] left/right channels to a 16-bit PCM stereo WAV file."""
    n = min(len(left), len(right))
    stereo = np.empty((n, 2), dtype=np.float32)
    stereo[:, 0] = left[:n]
    stereo[:, 1] = right[:n]
    peak = float(np.max(np.abs(stereo))) if n > 0 else 0.0
    if peak > 0.95:
        stereo *= 0.95 / peak
    pcm16 = np.clip(np.round(stereo * 32767.0), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _pan_mix_stereo(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    pan_a: float = -0.25,
    pan_b: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix two mono tracks into stereo using constant-power panning (-1=left, +1=right)."""
    n = min(len(audio_a), len(audio_b))
    a = audio_a[:n]
    b = audio_b[:n]

    theta_a = (max(-1.0, min(1.0, pan_a)) + 1.0) * (math.pi / 4.0)
    theta_b = (max(-1.0, min(1.0, pan_b)) + 1.0) * (math.pi / 4.0)

    left = a * math.cos(theta_a) + b * math.cos(theta_b)
    right = a * math.sin(theta_a) + b * math.sin(theta_b)
    return left.astype(np.float32), right.astype(np.float32)


def _compact_long_silences(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    transcript: list[dict[str, Any]],
    sample_rate: int,
    frame_samples: int,
    max_gap_sec: float = 2.0,
    rms_threshold: float = 0.008,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float]:
    """Compact any dead-air silence longer than max_gap_sec down to max_gap_sec,
    and shift transcript timestamps to stay in exact sync with the compacted audio."""
    if max_gap_sec <= 0.0:
        return audio_a, audio_b, transcript, 0.0

    n_frames = min(len(audio_a), len(audio_b)) // frame_samples
    if n_frames == 0:
        return audio_a, audio_b, transcript, 0.0

    frame_dur = frame_samples / float(sample_rate)
    max_silent_frames = max(2, int(round(max_gap_sec / frame_dur)))

    keep_indices: list[int] = []
    # Map each original frame index -> new output frame index (for exact subtitle remapping)
    orig_to_new: list[int] = [0] * n_frames
    silent_run = 0
    for i in range(n_frames):
        s = i * frame_samples
        e = s + frame_samples
        rms_a = float(np.sqrt(np.mean(audio_a[s:e] ** 2)))
        rms_b = float(np.sqrt(np.mean(audio_b[s:e] ** 2)))
        if max(rms_a, rms_b) <= rms_threshold:
            silent_run += 1
            if silent_run <= max_silent_frames:
                orig_to_new[i] = len(keep_indices)
                keep_indices.append(i)
            else:
                orig_to_new[i] = max(0, len(keep_indices) - 1)
        else:
            silent_run = 0
            orig_to_new[i] = len(keep_indices)
            keep_indices.append(i)

    dropped_frames = n_frames - len(keep_indices)
    if dropped_frames <= 0:
        return audio_a, audio_b, transcript, 0.0

    # Rebuild compacted arrays with a brief 10ms fade at splice boundaries to prevent clicks
    fade_n = min(int(sample_rate * 0.01), frame_samples // 4)
    chunks_a: list[np.ndarray] = []
    chunks_b: list[np.ndarray] = []
    prev_orig = -1
    for orig_i in keep_indices:
        s = orig_i * frame_samples
        e = s + frame_samples
        ca = audio_a[s:e].copy()
        cb = audio_b[s:e].copy()
        if prev_orig >= 0 and orig_i != prev_orig + 1 and fade_n > 0:
            ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            ca[:fade_n] *= ramp
            cb[:fade_n] *= ramp
            chunks_a[-1][-fade_n:] *= ramp[::-1]
            chunks_b[-1][-fade_n:] *= ramp[::-1]
        chunks_a.append(ca)
        chunks_b.append(cb)
        prev_orig = orig_i

    new_a = np.concatenate(chunks_a) if chunks_a else audio_a[:0]
    new_b = np.concatenate(chunks_b) if chunks_b else audio_b[:0]

    new_transcript: list[dict[str, Any]] = []
    for ev in transcript:
        orig_t = float(ev.get("t", 0.0))
        orig_f = min(n_frames - 1, max(0, int(round(orig_t / frame_dur))))
        new_t = round(orig_to_new[orig_f] * frame_dur, 2)
        updated = dict(ev)
        updated["t"] = new_t
        new_transcript.append(updated)

    removed_sec = dropped_frames * frame_dur
    return new_a, new_b, new_transcript, removed_sec


def _check_quality(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    transcript: list[dict[str, Any]],
    duration_sec: float,
    host_cards: dict[str, dict[str, str]],
    qcfg: dict,
    moderation_cfg: Optional[dict] = None,
) -> tuple[bool, str, float]:
    """Inspect generated dialogue audio & transcript for silence, repetition, or card leakage."""
    frame_len = 3200
    n_frames = min(len(audio_a), len(audio_b)) // frame_len
    if n_frames == 0:
        return False, "no audio frames generated", 0.0

    active_frames = 0
    cur_silent_frames = 0
    max_silent_frames = 0
    for i in range(n_frames):
        s = i * frame_len
        e = s + frame_len
        rms_a = float(np.sqrt(np.mean(audio_a[s:e] ** 2)))
        rms_b = float(np.sqrt(np.mean(audio_b[s:e] ** 2)))
        if max(rms_a, rms_b) > 0.008:
            active_frames += 1
            cur_silent_frames = 0
        else:
            cur_silent_frames += 1
            if cur_silent_frames > max_silent_frames:
                max_silent_frames = cur_silent_frames
    speech_ratio = active_frames / n_frames

    min_speech_ratio = float(qcfg.get("min_speech_ratio", 0.12))
    if speech_ratio < min_speech_ratio:
        return False, f"excessive silence (speech_ratio={speech_ratio:.2f} < {min_speech_ratio})", speech_ratio

    max_silence_sec = float(qcfg.get("max_silence_sec", 0.0))
    if max_silence_sec > 0.0 and duration_sec > max_silence_sec:
        longest_silence_sec = max_silent_frames * (frame_len / 24000.0)
        if longest_silence_sec > max_silence_sec:
            return (
                False,
                f"prolonged dead-air silence ({longest_silence_sec:.1f}s > {max_silence_sec:.0f}s)",
                speech_ratio,
            )

    full_text = "".join(ev.get("text", "") for ev in transcript)
    compact = "".join(ch for ch in full_text if not ch.isspace())
    min_chars = max(8, int(float(qcfg.get("min_chars_per_30s", 15)) * (duration_sec / 30.0)))
    if len(compact) < min_chars:
        return False, f"too little spoken text ({len(compact)} chars < {min_chars})", speech_ratio

    # N-gram repetition check (4-char sliding window)
    if len(compact) >= 24:
        ngrams = [compact[i : i + 4] for i in range(len(compact) - 3)]
        counts = Counter(ngrams)
        top_ngram, top_count = counts.most_common(1)[0]
        repeat_ratio = (top_count * 4) / max(1, len(compact))
        max_repeat = float(qcfg.get("max_repeat_ngram_ratio", 0.45))
        if repeat_ratio > max_repeat and top_count >= 5:
            return (
                False,
                f"repetition collapse detected ('{top_ngram}' x{top_count}, ratio={repeat_ratio:.2f})",
                speech_ratio,
            )

    # Check obvious verbatim character-card prompt reading
    for hkey in ("host_a", "host_b"):
        persona = host_cards.get(hkey, {}).get("persona", "")
        if len(persona) >= 16 and persona[:16] in compact:
            return False, f"character card persona leakage ({hkey})", speech_ratio

    # Check speaker balance (reject pure single-speaker monologues that tend to loop)
    min_minor_ratio = float(qcfg.get("min_minor_speaker_ratio", 0.0))
    if min_minor_ratio > 0.0 and duration_sec >= 45.0:
        spk_chars: Counter[str] = Counter()
        for ev in transcript:
            if ev.get("is_cue"):
                continue
            spk = str(ev.get("speaker", ""))
            txt = "".join(ch for ch in str(ev.get("text", "")) if not ch.isspace())
            if spk and txt:
                spk_chars[spk] += len(txt)
        total_spk_chars = sum(spk_chars.values())
        if total_spk_chars >= 40:
            name_a = host_cards.get("host_a", {}).get("name", "")
            name_b = host_cards.get("host_b", {}).get("name", "")
            minor = min(spk_chars.get(name_a, 0), spk_chars.get(name_b, 0))
            minor_ratio = minor / total_spk_chars
            if minor_ratio < min_minor_ratio:
                return (
                    False,
                    f"single-speaker monologue detected (minor_speaker_ratio={minor_ratio:.2%} < {min_minor_ratio:.0%})",
                    speech_ratio,
                )

    # Check longest continuous monologue by a single speaker
    max_mono_sec = float(qcfg.get("max_monologue_sec", 0.0))
    if max_mono_sec > 0.0 and duration_sec >= max_mono_sec:
        cur_spk = ""
        run_start = 0.0
        longest_mono = 0.0
        longest_spk = ""
        for ev in transcript:
            if ev.get("is_cue"):
                continue
            txt = str(ev.get("text", "")).strip()
            spk = str(ev.get("speaker", ""))
            t = float(ev.get("t", 0.0))
            if not txt or not spk:
                continue
            if spk != cur_spk:
                cur_spk = spk
                run_start = t
            else:
                span = t - run_start
                if span > longest_mono:
                    longest_mono = span
                    longest_spk = spk
        if longest_mono > max_mono_sec:
            return (
                False,
                f"long monologue detected ({longest_spk} spoke for {longest_mono:.1f}s > {max_mono_sec:.0f}s)",
                speech_ratio,
            )

    # Pre-broadcast content safety / NG-word check
    safe, mod_reason = check_transcript_safety(transcript, moderation_cfg)
    if not safe:
        return False, f"moderation check failed: {mod_reason}", speech_ratio

    return True, "ok", speech_ratio


def _pick_bgm_file(bgm_dir: Path) -> Optional[Path]:
    if not bgm_dir.exists():
        return None
    files = [p for p in bgm_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if not files:
        return None
    return random.choice(files)


def finish_segment_audio(
    raw_stereo_wav: Path,
    out_wav: Path,
    duration_sec: float,
    config: dict,
    base_dir: Path,
) -> None:
    """Apply resampling (44.1kHz stereo), intro/outro fade + chime, optional BGM ducking (-18dB), and loudnorm."""
    acfg = config.get("audio", {})
    out_sr = int(acfg.get("output_sample_rate", 44100))
    bgm_dir = base_dir / acfg.get("bgm_dir", "assets/bgm")
    duck_db = float(acfg.get("bgm_duck_db", -18.0))
    enable_jingle = bool(acfg.get("enable_jingle", False))
    pause_sec = max(0.0, float(acfg.get("inter_segment_pause_sec", 1.5)))
    fade_sec = min(float(acfg.get("jingle_fade_sec", 1.5)), max(0.3, duration_sec * 0.15))
    target_i = float(acfg.get("loudnorm_target_i", -16.0))
    target_tp = float(acfg.get("loudnorm_target_tp", -1.5))
    target_lra = float(acfg.get("loudnorm_target_lra", 11.0))

    bgm_file = _pick_bgm_file(bgm_dir)
    fade_out_start = max(0.0, duration_sec - fade_sec)
    pad_filter = f",apad=pad_dur={pause_sec:.2f}" if pause_sec > 0.0 else ""

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_wav.with_suffix(".tmp.wav")

    # Optional subtle 0.8s two-tone radio chime at the beginning of each corner
    jingle_expr = (
        "aevalsrc='0.06*(sin(2*PI*523.25*t)*exp(-4*t)"
        "+0.8*sin(2*PI*659.25*max(0,t-0.15))*exp(-4*max(0,t-0.15))*gt(t,0.15))'"
        f":s={out_sr}:d=0.8,aformat=channel_layouts=stereo,afade=t=out:st=0.45:d=0.35[jingle]"
    )

    if bgm_file is not None:
        logger.info("Mastering segment with BGM (%s) ducked to %.1f dB", bgm_file.name, duck_db)
        if enable_jingle:
            filter_complex = (
                f"[0:a]aresample={out_sr},aformat=channel_layouts=stereo,"
                f"afade=t=in:st=0:d=0.4,afade=t=out:st={fade_out_start:.3f}:d={fade_sec:.3f}[talk];"
                f"[1:a]aresample={out_sr},aformat=channel_layouts=stereo,atrim=0:{duration_sec:.3f},"
                f"volume={duck_db:.1f}dB,afade=t=in:st=0:d={fade_sec:.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={fade_sec:.3f}[bgm];"
                f"{jingle_expr};"
                f"[talk][bgm][jingle]amix=inputs=3:duration=first:dropout_transition=0:normalize=0,"
                f"loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA={target_lra:.1f}{pad_filter}[out]"
            )
        else:
            filter_complex = (
                f"[0:a]aresample={out_sr},aformat=channel_layouts=stereo,"
                f"afade=t=in:st=0:d=0.4,afade=t=out:st={fade_out_start:.3f}:d={fade_sec:.3f}[talk];"
                f"[1:a]aresample={out_sr},aformat=channel_layouts=stereo,atrim=0:{duration_sec:.3f},"
                f"volume={duck_db:.1f}dB,afade=t=in:st=0:d={fade_sec:.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={fade_sec:.3f}[bgm];"
                f"[talk][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                f"loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA={target_lra:.1f}{pad_filter}[out]"
            )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_stereo_wav),
            "-stream_loop",
            "-1",
            "-i",
            str(bgm_file),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ar",
            str(out_sr),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(tmp_out),
        ]
    else:
        if enable_jingle:
            filter_complex = (
                f"[0:a]aresample={out_sr},aformat=channel_layouts=stereo,"
                f"afade=t=in:st=0:d=0.4,afade=t=out:st={fade_out_start:.3f}:d={fade_sec:.3f}[talk];"
                f"{jingle_expr};"
                f"[talk][jingle]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                f"loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA={target_lra:.1f}{pad_filter}[out]"
            )
        else:
            filter_complex = (
                f"[0:a]aresample={out_sr},aformat=channel_layouts=stereo,"
                f"afade=t=in:st=0:d=0.4,afade=t=out:st={fade_out_start:.3f}:d={fade_sec:.3f},"
                f"loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA={target_lra:.1f}{pad_filter}[out]"
            )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_stereo_wav),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ar",
            str(out_sr),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(tmp_out),
        ]

    _run_cmd(cmd, "ffmpeg audio mastering")
    os.replace(tmp_out, out_wav)


class HaruDialogueEngine:
    """Holds a single loaded Haru-4B model and runs 2-host dialogue corners via libspeech C API."""

    def __init__(self, config: dict, base_dir: Path):
        self.config = config
        self.base_dir = base_dir
        self._lib: Optional[SpeechLibrary] = None
        self._model: Optional[SpeechModel] = None
        self._wrapup_cues: list[tuple[str, list[np.ndarray]]] = []
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        voice_paths = ensure_voices(self.config, self.base_dir)
        mcfg = self.config["model"]
        model_path = resolve_model_path(mcfg["name"])

        self._lib = SpeechLibrary(mcfg.get("speech_dll", ""))
        requested_device = str(mcfg.get("device", "auto"))
        selected_device = requested_device
        if requested_device.lower() == "auto":
            devices = self._lib.list_devices()
            discrete_gpus = [d for d in devices if d.kind == "gpu"]
            if discrete_gpus:
                selected_device = discrete_gpus[0].name
            else:
                # Avoid Intel iGPU Vulkan workgroup limit crashes when no discrete GPU is active
                logger.warning(
                    "No discrete GPU detected by speech.cpp (only %s). Falling back to CPU.",
                    ", ".join(f"{d.name}[{d.kind}]" for d in devices),
                )
                selected_device = "cpu"

        logger.info(
            "Loading %s from %s (device=%s, warmup=%s)...",
            mcfg["name"],
            model_path,
            selected_device,
            mcfg.get("warmup", True),
        )
        t0 = time.perf_counter()
        self._model = self._lib.load_model(
            model_path=model_path,
            device=selected_device,
            threads=int(mcfg.get("threads", 0)),
            warmup=bool(mcfg.get("warmup", True)),
        )
        logger.info(
            "Loaded Haru-4B on device=%s in %.2fs (sr=%d, frame_samples=%d)",
            self._model.device,
            time.perf_counter() - t0,
            self._model.sample_rate,
            self._model.frame_samples,
        )
        for host_key in ("host_a", "host_b"):
            vname = self.config["hosts"][host_key]["voice_name"]
            self._model.add_voice(vname, voice_paths[host_key])
            logger.info("Registered voice '%s' -> %s", vname, voice_paths[host_key])

        try:
            self._wrapup_cues = ensure_wrapup_cues(
                self.config,
                self.base_dir,
                target_sr=self._model.sample_rate,
                frame_samples=self._model.frame_samples,
            )
            if self._wrapup_cues:
                logger.info("Loaded %d wrap-up audio cue(s) for natural corner endings.", len(self._wrapup_cues))
        except Exception as exc:
            logger.warning("Failed to prepare wrap-up audio cues (will use silence-only ending): %s", exc)
            self._wrapup_cues = []

    def close(self) -> None:
        if self._model is not None:
            self._model.close()
            self._model = None

    def _build_host_options(
        self,
        host_key: str,
        topic: Topic,
        max_seconds: float,
        seed: int,
        speak_first: bool,
    ) -> dict[str, Any]:
        hcfg = self.config["hosts"][host_key]
        dcfg = self.config["dialogue"]
        situation = hcfg["situation_template"].format(
            topic=topic.title,
            detail=topic.detail,
        )
        return {
            "voice": hcfg["voice_name"],
            "language": hcfg.get("language", "ja"),
            "name": hcfg["name"],
            "persona": hcfg["persona"],
            "gender": hcfg.get("gender", "male" if host_key == "host_a" else "female"),
            "backchannel": hcfg.get("backchannel", "mid"),
            "other_name": hcfg["other_name"],
            "relationship": hcfg["relationship"],
            "situation": situation,
            "max_seconds": float(max_seconds),
            "temperature": float(dcfg.get("temperature", 0.5)),
            "top_k": int(dcfg.get("top_k", 25)),
            "top_p": float(dcfg.get("top_p", 1.0)),
            "guidance_scale": float(dcfg.get("guidance_scale", 1.5)),
            "eop_bias": float(hcfg.get("eop_bias", dcfg.get("eop_bias", 0.0))),
            "speak_first": bool(speak_first),
            "seed": int(seed),
        }

    def generate_segment(
        self,
        topic: Topic,
        out_dir: Path,
        duration_sec: Optional[float] = None,
        keep_raw: bool = False,
        stop_event: Optional[threading.Event] = None,
        progress_cb: Optional[Any] = None,
    ) -> GeneratedSegment:
        """Generate one 2-speaker corner, verify quality (with retries), and master to WAV."""
        with self._lock:
            if self._model is None:
                self.load()
            assert self._model is not None

            dcfg = self.config["dialogue"]
            qcfg = dcfg.get("quality_check", {})
            ocfg = dcfg.get("opening_cue", {})
            wcfg = dcfg.get("wrapup_cue", {})
            fixed_duration = duration_sec is not None
            if fixed_duration:
                min_dur = float(duration_sec)
                target_dur = float(duration_sec)
                max_dur = float(duration_sec)
            else:
                default_dur = float(dcfg.get("segment_duration_sec", 180))
                max_dur = float(dcfg.get("max_segment_sec", default_dur))
                min_dur = min(float(dcfg.get("min_segment_sec", 90)), max_dur)
                target_dur = min(max(min_dur, float(dcfg.get("target_segment_sec", 150))), max_dur)
            end_silence_sec = float(dcfg.get("end_silence_sec", 1.5))
            max_seconds = max_dur + float(dcfg.get("max_seconds_margin", 15))
            max_retries = int(dcfg.get("max_retries", 2))

            sr = self._model.sample_rate
            frame_samples = self._model.frame_samples
            frame_dur_sec = frame_samples / float(sr)

            # Synthesize topic-specific opening cue once per segment (reused across retries)
            opening_phrase = ""
            opening_frames: list[np.ndarray] = []
            use_opening_cue = (not fixed_duration) and bool(ocfg.get("enabled", False))
            if use_opening_cue:
                templates = [str(t).strip() for t in (ocfg.get("templates") or []) if str(t).strip()]
                if templates:
                    clean_title = (
                        topic.title.replace("「", "")
                        .replace("」", "")
                        .replace("　", "、")
                        .replace(" ", "、")
                        .strip()
                    )
                    opening_phrase = random.choice(templates).format(topic=clean_title)
                    try:
                        opening_frames = _synthesize_host_b_cue_frames(
                            opening_phrase, self.config, target_sr=sr, frame_samples=frame_samples
                        )
                        if opening_frames:
                            logger.info(
                                "Prepared opening cue (%.2fs): %s",
                                len(opening_frames) * frame_dur_sec,
                                opening_phrase,
                            )
                        else:
                            use_opening_cue = False
                    except Exception as exc:
                        logger.warning("Opening cue TTS failed (falling back to cold start): %s", exc)
                        use_opening_cue = False
                else:
                    use_opening_cue = False

            use_wrapup_cue = (not fixed_duration) and bool(wcfg.get("enabled", False)) and bool(self._wrapup_cues)
            cue_trigger_sec = min(float(wcfg.get("trigger_sec", 120.0)), max_dur - 15.0)
            cue_max_wait_sec = float(wcfg.get("max_wait_sec", 25.0))
            post_cue_silence_sec = float(wcfg.get("post_cue_silence_sec", 2.0))
            post_cue_max_sec = float(wcfg.get("post_cue_max_sec", 6.0))

            total_frames = max(1, int(round(max_dur * sr / frame_samples)))
            end_silence_frames = max(2, int(round(end_silence_sec / frame_dur_sec)))
            clean_pause_frames = max(2, int(round(0.8 / frame_dur_sec)))
            wrapup_cues = (
                "次の話題", "次のテーマ", "次のコーナー", "そろそろ", "また次回",
                "またね", "バイバイ", "おやすみ", "それでは", "ありがとう",
            )

            best_payload: Optional[tuple] = None
            attempts = 0
            last_reason = ""

            for attempt in range(1, max_retries + 2):
                attempts = attempt
                if stop_event and stop_event.is_set():
                    raise InterruptedError("Producer stopped during segment generation.")

                seed_a = random.randint(1, 2**31 - 1)
                seed_b = random.randint(1, 2**31 - 1)
                # When Mina introduces the topic via opening_cue, Haru listens first (speak_first=False)
                opts_a = self._build_host_options(
                    "host_a", topic, max_seconds, seed_a, speak_first=(not use_opening_cue)
                )
                opts_b = self._build_host_options("host_b", topic, max_seconds, seed_b, speak_first=False)

                logger.info(
                    "Starting corner '%s' (%.0f-%.0fs, open_cue=%s, wrap_cue=%s@%.0fs, attempt %d/%d, seeds=%d/%d)",
                    topic.title,
                    min_dur,
                    max_dur,
                    "on" if use_opening_cue else "off",
                    "on" if use_wrapup_cue else "off",
                    cue_trigger_sec,
                    attempt,
                    max_retries + 1,
                    seed_a,
                    seed_b,
                )

                frames_a: list[np.ndarray] = []
                frames_b: list[np.ndarray] = []
                transcript: list[dict[str, Any]] = []
                pair_times_ms: list[float] = []
                peak_vram = query_nvidia_vram_mb()

                host_a_name = self.config["hosts"]["host_a"]["name"]
                host_b_name = self.config["hosts"]["host_b"]["name"]

                silent_streak = 0
                ended_with_punct = False
                last_clean_pause_frame: Optional[int] = None
                natural_end_reason = ""
                early_abort_reason = ""
                cur_mono_spk = ""
                mono_start_sec = 0.0
                max_silence_abort_sec = float(qcfg.get("max_silence_sec", 0.0)) + 2.0 if float(qcfg.get("max_silence_sec", 0.0)) > 0.0 else 0.0
                max_mono_abort_sec = float(qcfg.get("max_monologue_sec", 0.0)) + 5.0 if float(qcfg.get("max_monologue_sec", 0.0)) > 0.0 else 0.0
                can_early_abort = (not fixed_duration) and (attempt < max_retries + 1)

                cue_phrase, cue_frames = random.choice(self._wrapup_cues) if use_wrapup_cue else ("", [])
                cue_started_frame: Optional[int] = None
                cue_idx = 0
                post_cue_spoke = False
                post_cue_ended_punct = False
                zero_frame = np.zeros(frame_samples, dtype=np.float32)

                t_gen_start = time.perf_counter()
                session_a = None
                session_b = None
                try:
                    # Start both sessions on the SINGLE loaded Haru-4B model
                    session_a = self._model.start_dialogue(opts_a, input_sample_rate=sr)
                    session_b = self._model.start_dialogue(opts_b, input_sample_rate=sr)

                    init_a = session_a.pop_frames()
                    init_b = session_b.pop_frames()
                    prev_a, txt_a = init_a[-1] if init_a else (zero_frame.copy(), "")
                    prev_b, txt_b = init_b[-1] if init_b else (zero_frame.copy(), "")
                    if use_opening_cue and opening_frames:
                        frames_a.append(zero_frame.copy())
                        frames_b.append(opening_frames[0])
                        transcript.append(
                            {"t": 0.0, "speaker": host_b_name, "text": opening_phrase, "is_cue": True}
                        )
                    else:
                        frames_a.append(prev_a)
                        frames_b.append(prev_b)
                        if txt_a.strip():
                            transcript.append({"t": 0.0, "speaker": host_a_name, "text": txt_a})
                        if txt_b.strip():
                            transcript.append({"t": 0.0, "speaker": host_b_name, "text": txt_b})

                    for idx in range(1, total_frames):
                        if stop_event and stop_event.is_set():
                            raise InterruptedError("Producer stopped during frame loop.")

                        t_sec = round(idx * frame_samples / sr, 2)
                        in_opening = use_opening_cue and idx < len(opening_frames)

                        # Trigger Mina's audio wrap-up cue at a clean sentence pause past cue_trigger_sec
                        if (not in_opening) and use_wrapup_cue and cue_started_frame is None and t_sec >= cue_trigger_sec:
                            waited = t_sec - cue_trigger_sec
                            ready_clean = ended_with_punct and silent_streak >= 3
                            ready_fallback = waited >= cue_max_wait_sec and (silent_streak >= 2 or waited >= cue_max_wait_sec + 3.0)
                            if ready_clean or ready_fallback:
                                cue_started_frame = idx
                                cue_idx = 0
                                transcript.append(
                                    {"t": t_sec, "speaker": host_b_name, "text": cue_phrase, "is_cue": True}
                                )
                                logger.info("Injecting wrap-up cue at %.1fs (%s)", t_sec, cue_phrase)

                        in_cue = cue_started_frame is not None and cue_idx < len(cue_frames)
                        if in_opening:
                            in_a = opening_frames[idx]
                            in_b = zero_frame
                        elif in_cue:
                            in_a = cue_frames[cue_idx]
                            in_b = prev_a
                            cue_idx += 1
                        else:
                            in_a = prev_b
                            in_b = prev_a

                        t_frame = time.perf_counter()
                        out_a = session_a.push_frame(in_a)
                        out_b = session_b.push_frame(in_b)
                        dt_ms = (time.perf_counter() - t_frame) * 1000.0
                        pair_times_ms.append(dt_ms)

                        cur_a, txt_a = out_a[-1] if out_a else (zero_frame.copy(), "")
                        cur_b, txt_b = out_b[-1] if out_b else (zero_frame.copy(), "")

                        frames_a.append(cur_a)
                        if in_opening:
                            frames_b.append(opening_frames[idx])
                        elif in_cue:
                            frames_b.append(cue_frames[cue_idx - 1])
                        else:
                            frames_b.append(cur_b)
                        if txt_a:
                            transcript.append({"t": t_sec, "speaker": host_a_name, "text": txt_a})
                        if txt_b and (not in_cue) and (not in_opening):
                            transcript.append({"t": t_sec, "speaker": host_b_name, "text": txt_b})

                        prev_a, prev_b = cur_a, cur_b

                        # Track natural sentence-ending pauses so topics wrap up cleanly
                        combined_txt = (txt_a + ("" if (in_cue or in_opening) else txt_b)).strip()
                        if combined_txt:
                            ended_with_punct = combined_txt.endswith(("。", "！", "？", ".", "!", "?"))

                        rms_a = float(np.sqrt(np.mean(cur_a ** 2)))
                        rms_b = float(np.sqrt(np.mean(cur_b ** 2)))
                        if max(rms_a, rms_b) <= 0.008:
                            silent_streak += 1
                        else:
                            silent_streak = 0

                        if cue_started_frame is None and silent_streak >= clean_pause_frames and ended_with_punct:
                            last_clean_pause_frame = len(frames_a)

                        # Early-abort bad attempts (prolonged dead-air silence or long solo monologue)
                        # instead of wasting 3 minutes generating a segment that will fail quality check
                        if can_early_abort and (not in_opening) and cue_started_frame is None:
                            if max_silence_abort_sec > 0.0 and (silent_streak * frame_dur_sec) > max_silence_abort_sec:
                                early_abort_reason = f"prolonged dead-air silence ({silent_streak * frame_dur_sec:.1f}s)"
                                break
                            if max_mono_abort_sec > 0.0:
                                spk_now = host_a_name if txt_a.strip() else (host_b_name if txt_b.strip() else "")
                                if spk_now:
                                    if spk_now != cur_mono_spk:
                                        cur_mono_spk = spk_now
                                        mono_start_sec = t_sec
                                    elif (t_sec - mono_start_sec) > max_mono_abort_sec:
                                        early_abort_reason = f"long monologue by {cur_mono_spk} ({t_sec - mono_start_sec:.1f}s)"
                                        break

                        # Post-cue clean stop logic (1 closing reply, or short silence, or hard cap)
                        if cue_started_frame is not None and not in_cue:
                            after_cue_s = (idx - (cue_started_frame + len(cue_frames))) * frame_dur_sec
                            if combined_txt:
                                post_cue_spoke = True
                                post_cue_ended_punct = combined_txt.endswith(("。", "！", "？", ".", "!", "?"))
                            if post_cue_spoke and post_cue_ended_punct and silent_streak >= clean_pause_frames:
                                natural_end_reason = f"wrap-up cue + 1 reply at {t_sec:.1f}s (+{after_cue_s:.1f}s)"
                                break
                            if (not post_cue_spoke) and after_cue_s >= post_cue_silence_sec:
                                natural_end_reason = f"wrap-up cue + silence at {t_sec:.1f}s (+{after_cue_s:.1f}s)"
                                break
                            if after_cue_s >= post_cue_max_sec:
                                natural_end_reason = f"wrap-up cue + cap at {t_sec:.1f}s (+{after_cue_s:.1f}s)"
                                break
                        elif not fixed_duration and not use_wrapup_cue and t_sec >= min_dur:
                            # Fallback silence-only natural end when wrap-up cue is disabled
                            if silent_streak >= clean_pause_frames:
                                recent_txt = "".join(
                                    ev["text"] for ev in transcript if t_sec - float(ev["t"]) <= 15.0
                                )
                                if any(cue in recent_txt for cue in wrapup_cues):
                                    natural_end_reason = f"wrap-up phrase at {t_sec:.1f}s"
                                    break
                            if t_sec >= target_dur and silent_streak >= end_silence_frames and ended_with_punct:
                                natural_end_reason = f"natural pause at {t_sec:.1f}s"
                                break

                        if idx % 30 == 0:
                            vram_now = query_nvidia_vram_mb()
                            if vram_now is not None:
                                peak_vram = max(peak_vram or 0.0, vram_now)

                        if idx % 10 == 0 or idx == total_frames - 1:
                            pct = (idx + 1) * 100.0 / total_frames
                            avg_ms = sum(pair_times_ms) / len(pair_times_ms)
                            eta_s = max(0.0, (total_frames - idx - 1) * (avg_ms / 1000.0))
                            eta_min, eta_rem = divmod(int(round(eta_s)), 60)
                            eta_str = (
                                f"あと約{eta_min}分{eta_rem:02d}秒"
                                if eta_min > 0
                                else f"あと約{eta_rem}秒"
                            )
                            if progress_cb is not None:
                                progress_cb(idx + 1, total_frames, pct, eta_s, eta_str)
                            if idx % 50 == 0 or idx == total_frames - 1:
                                logger.info(
                                    "Progress '%s': %d/%d frames (%.0f%%, avg %.0fms/pair, %s)",
                                    topic.title,
                                    idx + 1,
                                    total_frames,
                                    pct,
                                    avg_ms,
                                    eta_str,
                                )
                finally:
                    # ALWAYS close & free both dialogue sessions to completely discard KV caches
                    if session_b is not None:
                        session_b.close()
                    if session_a is not None:
                        session_a.close()

                if early_abort_reason:
                    last_reason = f"early abort: {early_abort_reason}"
                    logger.warning(
                        "Corner '%s' aborted early on attempt %d/%d: %s",
                        topic.title,
                        attempt,
                        max_retries + 1,
                        early_abort_reason,
                    )
                    continue

                # If we hit max_dur mid-sentence, trim back to the last clean sentence pause (within 35s)
                if not fixed_duration and not natural_end_reason and last_clean_pause_frame is not None:
                    trim_t = last_clean_pause_frame * frame_dur_sec
                    if trim_t >= min_dur and (len(frames_a) - last_clean_pause_frame) * frame_dur_sec <= 35.0:
                        frames_a = frames_a[:last_clean_pause_frame]
                        frames_b = frames_b[:last_clean_pause_frame]
                        transcript = [ev for ev in transcript if float(ev["t"]) <= trim_t]
                        natural_end_reason = f"trimmed to last sentence pause at {trim_t:.1f}s"

                actual_dur = len(frames_a) * frame_samples / float(sr)
                if natural_end_reason:
                    logger.info("Corner '%s' ended cleanly (%s)", topic.title, natural_end_reason)

                total_gen_sec = time.perf_counter() - t_gen_start
                audio_a = np.concatenate(frames_a)
                audio_b = np.concatenate(frames_b)

                ok, reason, speech_ratio = _check_quality(
                    audio_a,
                    audio_b,
                    transcript,
                    actual_dur,
                    {"host_a": opts_a, "host_b": opts_b},
                    qcfg,
                    self.config.get("moderation", {}),
                )

                arr_ms = np.array(pair_times_ms, dtype=np.float64) if pair_times_ms else np.array([0.0])
                metrics = SegmentMetrics(
                    frames_generated=len(frames_a),
                    duration_sec=actual_dur,
                    total_gen_sec=total_gen_sec,
                    avg_pair_frame_ms=float(np.mean(arr_ms)),
                    p50_pair_frame_ms=float(np.percentile(arr_ms, 50)),
                    p95_pair_frame_ms=float(np.percentile(arr_ms, 95)),
                    rtf=total_gen_sec / max(0.001, actual_dur),
                    device=self._model.device,
                    peak_vram_mb=peak_vram,
                    speech_ratio=speech_ratio,
                    attempts=attempts,
                )
                best_payload = (audio_a, audio_b, transcript, metrics, seed_a, seed_b, actual_dur)

                if ok:
                    logger.info(
                        "Corner '%s' passed quality check (dur=%.1fs, speech_ratio=%.2f, avg=%.1fms/frame-pair, RTF=%.2f)",
                        topic.title,
                        actual_dur,
                        speech_ratio,
                        metrics.avg_pair_frame_ms,
                        metrics.rtf,
                    )
                    break

                last_reason = reason
                logger.warning(
                    "Corner '%s' failed quality check on attempt %d/%d: %s",
                    topic.title,
                    attempt,
                    max_retries + 1,
                    reason,
                )

            assert best_payload is not None
            audio_a, audio_b, transcript, metrics, seed_a, seed_b, actual_dur = best_payload
            if last_reason and attempts > max_retries:
                logger.warning("Using last attempt despite quality warning: %s", last_reason)

            # Guarantee zero prolonged dead-air silence by compacting gaps > max_silence_gap_sec
            max_gap_sec = float(self.config.get("audio", {}).get("max_silence_gap_sec", 2.0))
            if max_gap_sec > 0.0:
                audio_a, audio_b, transcript, removed_sec = _compact_long_silences(
                    audio_a,
                    audio_b,
                    transcript,
                    sample_rate=sr,
                    frame_samples=frame_samples,
                    max_gap_sec=max_gap_sec,
                )
                if removed_sec > 0.0:
                    actual_dur = len(audio_a) / float(sr)
                    metrics.duration_sec = actual_dur
                    logger.info(
                        "Compacted %.1fs of dead-air silence in '%s' (final dur=%.1fs)",
                        removed_sec,
                        topic.title,
                        actual_dur,
                    )

            # Stereo panning & mastering
            pan_a = float(self.config["hosts"]["host_a"].get("pan", -0.25))
            pan_b = float(self.config["hosts"]["host_b"].get("pan", 0.25))
            left, right = _pan_mix_stereo(audio_a, audio_b, pan_a=pan_a, pan_b=pan_b)

            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seg_id = f"seg_{stamp}_{random.randint(1000, 9999)}"
            final_wav = out_dir / f"{seg_id}.wav"
            meta_path = out_dir / f"{seg_id}.json"

            raw_wav_path: Optional[Path] = None
            with tempfile.TemporaryDirectory() as tmpdir:
                raw_tmp = Path(tmpdir) / f"{seg_id}_raw.wav"
                _write_stereo_wav(raw_tmp, left, right, sr)
                if keep_raw:
                    raw_wav_path = out_dir / f"{seg_id}_raw_24k.wav"
                    shutil.copy2(raw_tmp, raw_wav_path)
                finish_segment_audio(raw_tmp, final_wav, actual_dur, self.config, self.base_dir)

            # Build readable grouped transcript lines (strip internal is_cue flag for JSON output)
            transcript_clean = [{k: v for k, v in ev.items() if k != "is_cue"} for ev in transcript]
            grouped_lines = format_transcript_lines(transcript_clean)
            meta = {
                "id": seg_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "topic": {
                    "title": topic.title,
                    "detail": topic.detail,
                    "source": topic.source,
                    "source_name": topic.source_name,
                    "source_url": topic.source_url,
                },
                "duration_sec": round(actual_dur, 2),
                "seeds": {"host_a": seed_a, "host_b": seed_b},
                "metrics": {
                    "frames_generated": metrics.frames_generated,
                    "total_gen_sec": round(metrics.total_gen_sec, 2),
                    "avg_pair_frame_ms": round(metrics.avg_pair_frame_ms, 2),
                    "p50_pair_frame_ms": round(metrics.p50_pair_frame_ms, 2),
                    "p95_pair_frame_ms": round(metrics.p95_pair_frame_ms, 2),
                    "rtf": round(metrics.rtf, 3),
                    "device": metrics.device,
                    "peak_vram_mb": metrics.peak_vram_mb,
                    "speech_ratio": round(metrics.speech_ratio, 3),
                    "attempts": metrics.attempts,
                },
                "transcript_lines": grouped_lines,
                "transcript_raw": transcript_clean,
            }
            tmp_meta = meta_path.with_suffix(".tmp.json")
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp_meta, meta_path)

            return GeneratedSegment(
                topic=topic,
                wav_path=final_wav,
                meta_path=meta_path,
                raw_wav_path=raw_wav_path,
                transcript=transcript,
                metrics=metrics,
                seed_a=seed_a,
                seed_b=seed_b,
            )


def format_transcript_lines(transcript: list[dict[str, Any]]) -> list[str]:
    """Merge consecutive token chunks from the same speaker into readable dialogue lines."""
    lines: list[str] = []
    cur_speaker: Optional[str] = None
    cur_buf: list[str] = []
    cur_t = 0.0

    def _flush() -> None:
        nonlocal cur_speaker, cur_buf
        if cur_speaker is not None:
            text = "".join(cur_buf).strip()
            if text:
                mm, ss = divmod(int(cur_t), 60)
                lines.append(f"[{mm:02d}:{ss:02d}] {cur_speaker}: {text}")
        cur_speaker = None
        cur_buf = []

    for ev in transcript:
        spk = str(ev.get("speaker", ""))
        txt = str(ev.get("text", ""))
        t = float(ev.get("t", 0.0))
        if not txt:
            continue
        if txt == " " and cur_speaker == spk:
            continue
        if spk != cur_speaker or (len("".join(cur_buf)) > 80 and txt.endswith(("。", "！", "？", ".", "!", "?"))):
            _flush()
            cur_speaker = spk
            cur_t = t
        cur_buf.append(txt)
    _flush()
    return lines


class ProducerWorker:
    """Background producer loop maintaining 2-3 pre-generated corners in ./queue/."""

    def __init__(
        self,
        engine: HaruDialogueEngine,
        topic_manager: TopicManager,
        config: dict,
        base_dir: Path,
    ):
        self.engine = engine
        self.topic_manager = topic_manager
        self.config = config
        self.base_dir = base_dir

        qcfg = config.get("queue", {})
        self.queue_dir = base_dir / qcfg.get("dir", "queue")
        self.max_segments = int(qcfg.get("max_segments", 3))
        self.poll_interval = float(qcfg.get("poll_interval_sec", 2.0))

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.status_text = "idle"
        self.last_error: Optional[str] = None

    def queued_wavs(self) -> list[Path]:
        if not self.queue_dir.exists():
            return []
        return sorted(
            p
            for p in self.queue_dir.iterdir()
            if p.is_file() and p.suffix == ".wav" and not p.name.endswith(".tmp.wav")
        )

    def start(self) -> None:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="HaruProducer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.engine.close()

    def _loop(self) -> None:
        logger.info("ProducerWorker started (target queue size: %d)", self.max_segments)
        while not self._stop_event.is_set():
            try:
                queued = self.queued_wavs()
                if len(queued) >= self.max_segments:
                    self.status_text = f"standby (queue full: {len(queued)}/{self.max_segments})"
                    self._stop_event.wait(self.poll_interval)
                    continue

                topic = self.topic_manager.next_topic(mark=True)
                self.status_text = f"generating: {topic.title} (0%)"
                self.last_error = None

                def _on_progress(
                    done: int, total: int, pct: float, _eta_s: float, eta_str: str
                ) -> None:
                    self.status_text = (
                        f"generating: {topic.title} ({pct:.0f}%, {done}/{total} frames, {eta_str})"
                    )

                self.engine.generate_segment(
                    topic=topic,
                    out_dir=self.queue_dir,
                    stop_event=self._stop_event,
                    progress_cb=_on_progress,
                )
            except InterruptedError:
                break
            except Exception as exc:
                self.last_error = str(exc)
                self.status_text = f"error: {exc}"
                logger.exception("ProducerWorker error: %s", exc)
                self._stop_event.wait(5.0)
        self.status_text = "stopped"
        logger.info("ProducerWorker stopped.")


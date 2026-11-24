# Third-Party Notices

This repository's own source code is licensed under the MIT License (see [LICENSE](LICENSE)).
The external AI models, runtime tools, Python dependencies, system fonts, and news feeds used by this project are **not** bundled in this repository and remain subject to their respective licenses and terms.

## AI Models & Inference Runtime (Not Bundled)

| Component | Purpose | Source / URL | License | Bundled? |
|---|---|---|---|---|
| **Haru-4B** (`haru-4b:q8_0`) | 2-speaker full-duplex Japanese dialogue model | https://huggingface.co/higurashi-ai/Haru-4B<br>https://huggingface.co/sakasegawa/Haru-4B-GGUF | Apache License 2.0 | No (downloaded at runtime via `speech pull`) |
| **Qwen3-TTS-12Hz-0.6B-CustomVoice** (`qwen3-tts-0.6b`) | Local synthesis of preset speaker voices (`dylan` / `ono_anna`), cues, and station ID | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | Apache License 2.0 | No (audio is generated locally on first run) |
| **speech.cpp** (v0.11.0) | GGUF model inference runtime (`speech.exe` / `speech.dll`) | https://github.com/nyosegawa/speech.cpp | MIT License | No (installed separately by user) |
| **FFmpeg / FFprobe** | Audio mastering, MP3 encoding, and YouTube RTMP H.264/AAC encoding (invoked as external subprocess) | https://ffmpeg.org | LGPL / GPL (depends on user's build) | No (installed separately by user) |

## Python Dependencies (Installed via `requirements.txt`)

| Package | Purpose | License |
|---|---|---|
| **PyYAML** | YAML configuration parsing | MIT License |
| **NumPy** | PCM audio processing | BSD-3-Clause License |
| **aiohttp** | HTTP streaming server (`/radio.mp3`, `/status`, `/overlay`) | Apache License 2.0 |
| **feedparser** | RSS/Atom feed parsing | BSD-2-Clause License |
| **Pillow** | Real-time video frame rendering for YouTube Live | MIT-CMU (HPND) License |

## System Fonts & External Feeds (Not Bundled)

- **System Fonts**: The YouTube Live frame renderer reads OS-installed Japanese fonts (e.g., Yu Gothic / Meiryo on Windows, Noto Sans CJK on Linux) at runtime and does not redistribute font files.
- **RSS Feeds**: `config.yaml` contains only public RSS feed URLs. Article titles and summaries fetched at runtime remain the copyright of their respective publishers and are not included in this repository.

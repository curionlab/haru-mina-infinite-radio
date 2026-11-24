# ハルとミナの、無限ラジオ (Haru-4B Local Infinite AI Radio)

Windows PC 単体で動作する、日本語全二重音声会話モデル **Haru-4B**（`speech.cpp` 経由）のみで構成された「ローカル無限AIラジオ局」です。
カスケード型TTS（Ollama / VOICEVOX / AivisSpeech 等）は一切使わず、`libspeech` C API（`speech_dialogue_start`）によって **1つのロード済み Haru-4B モデル上で 2セッション（Host A / Host B）を 1フレーム（133ms）ずつ相互接続** して掛け合いを生成します。

---

## 必要環境

- **OS**: Windows 10 / 11
- **Python**: 3.12
- **speech.cpp**: v0.11.0（`speech.exe` および `speech.dll` が `PATH` または `%LOCALAPPDATA%\Programs\speech.cpp\current` にあること）
- **FFmpeg / FFprobe**: `PATH` が通っていること（音声マスタリング・MP3・YouTube RTMP 配信に使用）
- **GPU**: VRAM 8GB 以上の NVIDIA GPU 推奨（RTX 3080 等・Vulkan 推論。GPU がない環境では CPU に自動フォールバックしますが低速になります）

---

## セットアップ

```powershell
git clone https://github.com/curionlab/haru-mina-infinite-radio.git
cd haru-mina-infinite-radio

python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

speech pull haru-4b:q8_0
.\.venv\Scripts\python.exe -m src.main --check
```

> **補足**: 音声ファイル（`assets/voices/`）、締めキュー（`assets/cues/`）、ステーションID音声（`station_id.wav`）はリポジトリには同梱されていません。初回 `--serve` 起動時に `Qwen3-TTS` プリセット話者を用いてローカル環境で自動生成されます。

---

## 配信URL・起動コマンド

### 1. 環境診断（初回確認）
```powershell
.\.venv\Scripts\python.exe -m src.main --check
```

### 2. 30秒テスト収録（`./out/` へ出力）
```powershell
.\.venv\Scripts\python.exe -m src.main --test-segment --duration 30
```

### 3. ラジオ局の本番起動（Producer + HTTP配信サーバー）
```powershell
.\.venv\Scripts\python.exe -m src.main --serve
```

起動すると、以下のURLで配信・ステータス確認ができます：

- **ラジオストリーム（MPD / moOde / VLC / ブラウザ用）**:
  - LAN内から: `http://<PCのIPアドレス>:8000/radio.mp3`
  - PCローカルから: `http://127.0.0.1:8000/radio.mp3`
- **ステータス＆直近ログ画面**:
  - `http://<PCのIPアドレス>:8000/status`（JSON取得: `http://<PCのIPアドレス>:8000/status?format=json`）
- **リアルタイム字幕オーバーレイ画面（ブラウザプレビュー / OBSブラウザソース用）**:
  - `http://127.0.0.1:8000/overlay`（背景透過版: `http://127.0.0.1:8000/overlay?transparent=1`）
  - リアルタイム状態JSON: `http://127.0.0.1:8000/api/live`

---

## YouTube Live Studio への直接ストリーミング（リアルタイム字幕・AI開示表示付き）

OBS を使わず、アプリから直接 YouTube Live Studio へ **映像（現在のお題・話者インジケーター・音声同期リアルタイム字幕・AI自動生成開示表示）＋音声** を RTMP 配信できます。

### 1. 映像レイアウトの事前テスト（ストリームキー不要）
```powershell
.\.venv\Scripts\python.exe -m src.main --test-youtube-video
```
`./out/youtube_preview.mp4` に 5 秒間のプレビュー動画が出力されるので、日本語フォントやレイアウトを確認できます。

### 2. YouTube Live Studio へ配信開始

#### 方法 A: バッチファイルで起動（`.env` 利用・推奨）
1. `.env.example` を `.env` にコピーし、YouTube Studio でコピーしたストリームキーを `YOUTUBE_STREAM_KEY` に貼り付けます。
2. `start_youtube.bat` をダブルクリックすると配信が開始されます（`.env` が未作成の場合は起動時にキー入力を求められます）。

> **セキュリティ注意**: `.env` ファイルは絶対に Git にコミットしたり他人へ共有したりしないでください。また、リポジトリを公開する場合は `config.yaml` の `youtube.stream_key` を空欄（`""`）のままにしてください。

#### 方法 B: PowerShell から直接起動
YouTube Studio の「ライブ配信を開始」→「ストリーム」から **ストリームキー** をコピーし、環境変数 `YOUTUBE_STREAM_KEY` または `--youtube-key` で渡して起動します：

```powershell
$env:YOUTUBE_STREAM_KEY = "xxxx-xxxx-xxxx-xxxx-xxxx"
.\.venv\Scripts\python.exe -m src.main --serve --youtube
```

> **運用上の推奨設定（YouTube Studio 側）**
> - **配信の開始方法**: PC 側で `start_youtube.bat`（または `--serve --youtube`）を起動すると YouTube Studio へ映像・音声が送信されます。プレビュー待機状態になった場合は、YouTube Studio 画面の **「ライブ配信を開始」** ボタンを押して配信を開始してください。
> - **AI開示ラベル**: YouTube Studio の詳細設定「改変されたコンテンツ / AIの使用」を **「はい」** に設定してください（配信画面の右上・下部にも「ローカルAIによる自動生成ラジオ (Haru-4B)」が常時表示されます）。
> - **自動検閲・お題フィルタ**: `config.yaml` の `moderation:` により、政治・選挙・医療・投資・事件事故などのセンシティブな RSS お題は自動除外され、収録後の文字起こしに NG ワードが含まれた場合はキュー投入前に自動で作り直します。
> - **著作権対策**: `./assets/bgm/` を空にしておけば Haru-4B の会話と環境音のみになるため、Content ID の誤検出を避けられます。
> - **OBS 併用時**: ブラウザソースに `http://127.0.0.1:8000/overlay`（1280×720）、メディアソースに `http://127.0.0.1:8000/radio.mp3` を指定すれば OBS 経由でも配信できます。

> **補足**: 起動直後はキューが空のため穏やかな環境音（または `./assets/bgm/` のBGM）を途切れず流し、裏で1本目の掛け合いが完成次第シームレスにトークへ移行します。

---

## LAN内の別機器（Raspberry Pi Zero 2 W / moOde / MPD 等）から接続するためのファイアウォール設定

管理者権限の PowerShell で以下を実行し、Private ネットワーク向けに TCP 8000 を開放してください：

```powershell
New-NetFirewallRule -DisplayName "Haru Mina Infinite Radio (TCP 8000)" `
  -Direction Inbound -Protocol TCP -LocalPort 8000 `
  -Profile Private -Action Allow
```

（削除する場合）
```powershell
Remove-NetFirewallRule -DisplayName "Haru Mina Infinite Radio (TCP 8000)"
```

---

## カスタマイズ手順

### 1. 好きな声（WAV）へ差し替える
1. 単一話者・3秒以上の日本語 WAV ファイルを以下に配置します：
   - `./assets/voices/host_a.wav`（パーソナリティA：デフォルトはハル）
   - `./assets/voices/host_b.wav`（パーソナリティB：デフォルトはミナ）
2. 既存のキャッシュファイル `./assets/voices/host_a.voice.gguf` / `host_b.voice.gguf` を削除します。
3. 次回 `--serve` または `--test-segment` 実行時に、`speech voice` で自動的に新しい `.voice.gguf` が再生成されます。

### 2. BGM を追加する
- `./assets/bgm/` フォルダに `.mp3` / `.wav` / `.flac` / `.ogg` / `.m4a` を配置するだけです。
- トーク中は自動的に **-18dB** でダッキング合成され、キュー待ちの間はインターミッションBGMとして流れます。

### 3. お題・RSS・キャラクター設定を変更する
- **お題の追加**: [topics.yaml](topics.yaml) の `topics:` リストに `title` と `detail` を追加します（既出お題は `state.db` で自動管理されます）。
- **RSSフィード・キャラクターカード・量子化モデル変更**: [config.yaml](config.yaml) を編集します。
  - VRAM が逼迫する場合は `model.name` を `"haru-4b:q8_0"`（約6.62GB）から `"haru-4b:q6_K"`（約5.61GB）へ変更してください。
  - `dialogue.segment_duration_sec`（デフォルト `180` 秒）や `dialogue.temperature`（デフォルト `0.5`）も調整可能です。

---

## GPU（RTX 3080 Vulkan）について
- `config.yaml` の `model.device: "auto"` は、`speech.cpp` が認識するディスクリートGPU（RTX 3080 等）を最優先で使用し、ディスクリートGPUが認識できない環境では自動的に CPU へフォールバックします。
- Windows デバイスマネージャーで RTX 3080 が正常稼働している状態（`nvidia-smi` / `speech devices` に表示される状態）であれば、設定変更なしで自動的に RTX 3080（Vulkan）で高速推論されます。

---

## 省エネ機能（リスナー 0 人時の自動一時停止）
- `config.yaml` の `server.pause_when_no_listeners: true`（デフォルト有効）により、リスナーが 0 人になって `idle_pause_grace_sec`（デフォルト 15 秒）経過すると配信が自動で `PAUSED` 状態になり、キューの消費と GPU 生成を停止します。
- Raspberry Pi（moOde）等から `/radio.mp3` へ再接続すると即座に再開します。

---

## 免責事項・注意

- **AI生成コンテンツ**: 本アプリのトーク内容はすべてローカルAI（Haru-4B）による自動生成であり、事実と異なる内容や誤りが含まれる場合があります。
- **YouTube のAI開示設定**: YouTube Live で配信する場合は、YouTube Studio の詳細設定で「改変されたコンテンツ / AIの使用」を **「はい」** に設定することを推奨します。
- **RSSニュースの権利**: `config.yaml` に記載された RSS フィードから取得する記事タイトル・要約の著作権は各配信元に帰属します。ヘッドラインや記事内容の放送・二次利用にあたっては、各サイトの利用規約等をご確認のうえ運用してください。
- **ガイドライン遵守**: 配信内容が YouTube コミュニティガイドライン等の規約に違反しないよう、配信者自身の責任においてトピックやモデレーション設定を管理してください。
- **音声素材の権利**: 初回自動生成される `assets/voices/host_a.wav` / `host_b.wav`（および変換後の `.voice.gguf`）は、すべて Apache-2.0 の `Qwen3-TTS` 内蔵プリセット話者（`dylan` / `ono_anna`）からローカル合成した音声です。別の声 WAV へ差し替える場合は、Haru-4B 公式の注意事項に従い **使用権・許諾のある音声のみ** を使用してください。

---

## ライセンス

- 本アプリケーションのソースコードは **MIT License** のもとで公開されています（詳細は [LICENSE](LICENSE) を参照してください）。
- 外部の AI モデル・推論ランタイム・FFmpeg・Python 依存パッケージ等のサードパーティコンポーネントは同梱されておらず、それぞれのライセンスに従います（詳細は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください）。

### 使用モデル・ツール・ライセンス

| コンポーネント | 用途 | URL | ライセンス |
|---|---|---|---|
| **Haru-4B** (`haru-4b:q8_0`) | 2人掛け合い全二重音声会話モデル | https://huggingface.co/higurashi-ai/Haru-4B<br>https://huggingface.co/sakasegawa/Haru-4B-GGUF | Apache License 2.0 |
| **Qwen3-TTS-12Hz-0.6B-CustomVoice** (`qwen3-tts-0.6b`) | 初回自動生成する仮音声（`host_a.wav`: `dylan` / `host_b.wav`: `ono_anna`）の合成 | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | Apache License 2.0 |
| **speech.cpp** (v0.11.0) | モデル推論ランタイム・音声エンコード（`speech voice`） | https://github.com/nyosegawa/speech.cpp | MIT License |



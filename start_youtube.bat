@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

title Haru-4B Local AI Radio - YouTube Live

echo ========================================================
echo   Haru-4B Local AI Radio (YouTube Live Studio 配信)
echo ========================================================
echo.

REM 1. .env があれば YOUTUBE_STREAM_KEY 等を読み込む
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "KEY=%%~A"
        set "VAL=%%~B"
        if not "!KEY!"=="" if not "!KEY:~0,1!"=="#" (
            set "!KEY!=!VAL!"
        )
    )
)

REM 2. 環境変数が未設定なら入力プロンプト（config.yaml に直書き済みの場合はそのまま Enter）
if "%YOUTUBE_STREAM_KEY%"=="" (
    echo [INFO] YOUTUBE_STREAM_KEY が未設定です。
    echo        .env ファイルに YOUTUBE_STREAM_KEY=xxxx を保存しておくと次回から自動入力されます。
    echo        ^(config.yaml の youtube.stream_key を使う場合は空のまま Enter^)
    echo.
    set /p "YOUTUBE_STREAM_KEY=YouTube ストリームキーを入力: "
)

echo.
echo [INFO] 配信サーバーを起動します...
echo        - ステータス画面 : http://127.0.0.1:8000/status
echo        - 字幕プレビュー : http://127.0.0.1:8000/overlay
echo        - LAN 音声配信   : http://127.0.0.1:8000/radio.mp3
echo        停止するには Ctrl+C を押してください。
echo.

".\.venv\Scripts\python.exe" -m src.main --serve --youtube
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] 異常終了しました ^(exit code: %EXIT_CODE%^)。
    echo         ポート 8000 が使用中の場合は、別のラジオサーバーを停止してから再実行してください。
)
pause
endlocal

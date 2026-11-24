"""HTTP Icecast-compatible streaming server (/radio.mp3) and status dashboard (/status)."""

from __future__ import annotations

import asyncio
import html
import logging
from typing import Callable, Optional

from aiohttp import web

from src.radio_streamer import RadioStreamer

logger = logging.getLogger(__name__)


def _render_overlay_html(config: dict, transparent: bool = False) -> str:
    scfg = config.get("server", {})
    yt_cfg = config.get("youtube", {})
    hcfg = config.get("hosts", {})
    station = html.escape(str(scfg.get("station_name", "ハルとミナの、無限ラジオ")))
    host_a = html.escape(str(hcfg.get("host_a", {}).get("name", "ハル")))
    host_b = html.escape(str(hcfg.get("host_b", {}).get("name", "ミナ")))
    ai_badge = html.escape(str(yt_cfg.get("ai_disclosure_badge", "ローカルAIによる自動生成ラジオ (Haru-4B)")))
    ai_footer = html.escape(
        str(
            yt_cfg.get(
                "ai_disclosure_footer",
                "※ この配信はローカルAI音声会話モデル (Haru-4B) による自動生成ラジオです。実在の人物・団体とは関係ありません。",
            )
        )
    )
    bg_css = "background: transparent;" if transparent else "background: linear-gradient(180deg, #0f172a 0%, #090d18 100%);"

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{station} — Live Overlay</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: 100%; height: 100%; overflow: hidden; font-family: "Yu Gothic UI", "Meiryo", system-ui, sans-serif; color: #f8fafc; {bg_css} }}
  .frame {{ display: flex; flex-direction: column; width: 100%; height: 100%; }}
  header {{ display: flex; align-items: center; justify-content: space-between; padding: 14px 36px; background: rgba(17, 24, 39, 0.92); border-bottom: 2px solid #334155; }}
  .left-head, .right-head {{ display: flex; align-items: center; gap: 16px; }}
  .pill {{ padding: 5px 14px; border-radius: 999px; font-weight: 700; font-size: 0.95rem; background: #d97706; color: #fff; }}
  .pill.talk {{ background: #dc2626; }}
  .pill.paused {{ background: #64748b; }}
  .station {{ font-size: 1.4rem; font-weight: 700; letter-spacing: 0.02em; }}
  .ai-badge {{ padding: 5px 14px; border-radius: 999px; font-size: 0.9rem; font-weight: 700; background: #1e3a8a; border: 1px solid #60a5fa; color: #dbeafe; }}
  .clock {{ font-size: 1.15rem; font-weight: 700; color: #cbd5e1; font-variant-numeric: tabular-nums; }}
  main {{ flex: 1; display: flex; flex-direction: column; gap: 16px; padding: 18px 36px 12px; min-height: 0; }}
  .topic-card {{ background: rgba(24, 34, 54, 0.92); border: 1px solid #334155; border-radius: 14px; padding: 16px 24px 14px; }}
  .topic-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .label {{ font-size: 0.85rem; font-weight: 700; color: #60a5fa; letter-spacing: 0.06em; }}
  .source-badge {{ display: none; margin-left: 12px; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; background: #1e293b; border: 1px solid #fbbf24; color: #fde047; }}
  .hosts {{ display: flex; gap: 10px; }}
  .host {{ padding: 4px 14px; border-radius: 999px; font-weight: 700; font-size: 0.9rem; border: 1px solid #3b82f6; background: #172554; color: #cbd5e1; transition: all 0.15s ease; }}
  .host.b {{ border-color: #db2777; background: #501834; }}
  .host.active.a {{ background: #2563eb; color: #fff; box-shadow: 0 0 14px rgba(59, 130, 246, 0.65); }}
  .host.active.b {{ background: #db2777; color: #fff; box-shadow: 0 0 14px rgba(219, 39, 119, 0.65); }}
  .topic-title {{ font-size: 1.85rem; font-weight: 700; line-height: 1.3; color: #fff; }}
  .topic-detail {{ margin-top: 6px; font-size: 1rem; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .prog-row {{ display: flex; align-items: center; gap: 14px; margin-top: 12px; }}
  .prog-track {{ flex: 1; height: 6px; border-radius: 3px; background: #334155; overflow: hidden; }}
  .prog-fill {{ height: 100%; width: 0%; background: #3b82f6; transition: width 0.25s linear; }}
  .prog-time {{ flex-shrink: 0; font-size: 0.9rem; font-weight: 700; color: #cbd5e1; font-variant-numeric: tabular-nums; }}
  .sub-box {{ flex: 1; display: flex; flex-direction: column; background: rgba(15, 23, 42, 0.9); border: 1px solid #334155; border-radius: 14px; padding: 14px 22px; min-height: 0; overflow: hidden; }}
  .sub-list {{ flex: 1; display: flex; flex-direction: column; justify-content: flex-end; gap: 10px; margin-top: 10px; overflow: hidden; }}
  .sub-row {{ display: flex; align-items: flex-start; gap: 14px; padding: 10px 14px; border-radius: 10px; background: rgba(20, 29, 48, 0.85); color: #cbd5e1; font-size: 1.25rem; line-height: 1.45; }}
  .sub-row.latest {{ background: rgba(30, 41, 59, 0.98); border: 1px solid #475569; color: #fff; font-weight: 700; font-size: 1.35rem; }}
  .spk-tag {{ flex-shrink: 0; min-width: 72px; text-align: center; padding: 3px 10px; border-radius: 8px; font-size: 0.95rem; font-weight: 700; color: #fff; background: #2563eb; }}
  .spk-tag.b {{ background: #db2777; }}
  footer {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 36px; background: rgba(11, 15, 26, 0.94); border-top: 1px solid #1e293b; font-size: 0.85rem; color: #94a3b8; }}
  .next-topic {{ color: #60a5fa; font-weight: 600; }}
</style>
</head>
<body>
<div class="frame">
  <header>
    <div class="left-head">
      <div id="modePill" class="pill">● STANDBY</div>
      <div class="station">{station}</div>
    </div>
    <div class="right-head">
      <div class="ai-badge">{ai_badge}</div>
      <div id="clock" class="clock">--:--:--</div>
    </div>
  </header>
  <main>
    <section class="topic-card">
      <div class="topic-top">
        <div style="display:flex;align-items:center;"><span id="topicLabel" class="label">NOW TALKING — トークテーマ</span><span id="sourceBadge" class="source-badge"></span></div>
        <div class="hosts">
          <div id="hostA" class="host a">{host_a}</div>
          <div id="hostB" class="host b">{host_b}</div>
        </div>
      </div>
      <div id="topicTitle" class="topic-title">番組準備中...</div>
      <div id="topicDetail" class="topic-detail"></div>
      <div class="prog-row">
        <div class="prog-track"><div id="progFill" class="prog-fill"></div></div>
        <div id="progTime" class="prog-time">残り --:-- / --:--</div>
      </div>
    </section>
    <section class="sub-box">
      <div class="label" style="color:#94a3b8;">LIVE TRANSCRIPT — リアルタイム字幕</div>
      <div id="subList" class="sub-list"></div>
    </section>
  </main>
  <footer>
    <div>{ai_footer}</div>
    <div id="nextTopic" class="next-topic"></div>
  </footer>
</div>
<script>
const HOST_A = {host_a!r};
const HOST_B = {host_b!r};
function updateClock() {{
  const d = new Date();
  document.getElementById('clock').textContent = d.toTimeString().slice(0, 8);
}}
setInterval(updateClock, 500);
updateClock();

function fmtMMSS(sec) {{
  const s = Math.max(0, Math.round(sec || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return String(m).padStart(2, '0') + ':' + String(r).padStart(2, '0');
}}

async function pollLive() {{
  try {{
    const res = await fetch('/api/live', {{ cache: 'no-store' }});
    if (!res.ok) return;
    const s = await res.json();
    const pill = document.getElementById('modePill');
    pill.className = 'pill ' + (s.mode || 'fallback');
    pill.textContent = s.mode === 'talk' ? '● ON AIR' : (s.mode === 'paused' ? '● PAUSED' : '● STANDBY');
    document.getElementById('topicLabel').textContent = s.mode === 'talk' ? 'NOW TALKING — トークテーマ' : 'INTERMISSION — 番組準備中';
    const srcEl = document.getElementById('sourceBadge');
    if (s.current_source_label) {{
      srcEl.textContent = s.current_source_label;
      srcEl.style.display = 'inline-block';
    }} else {{
      srcEl.style.display = 'none';
    }}
    document.getElementById('topicTitle').textContent = s.current_topic || '';
    document.getElementById('topicDetail').textContent = s.current_detail || '';
    const dur = Number(s.segment_duration_sec || 0);
    const elap = Number(s.segment_elapsed_sec || 0);
    const pct = dur > 0 ? Math.min(100, (elap / dur) * 100) : 0;
    document.getElementById('progFill').style.width = pct.toFixed(1) + '%';
    document.getElementById('progTime').textContent = dur > 0
      ? ('残り ' + fmtMMSS(Math.max(0, dur - elap)) + ' / ' + fmtMMSS(dur))
      : '残り --:-- / --:--';

    const aEl = document.getElementById('hostA');
    const bEl = document.getElementById('hostB');
    aEl.classList.toggle('active', s.active_speaker === HOST_A && s.audio_level > 0.01);
    bEl.classList.toggle('active', s.active_speaker === HOST_B && s.audio_level > 0.01);

    const list = document.getElementById('subList');
    const subs = (s.live_subtitles || []).slice(-6);
    if (!subs.length) {{
      list.innerHTML = '<div style="color:#64748b;font-size:1.15rem;padding:12px 4px;">音声に合わせてリアルタイム字幕が表示されます...</div>';
    }} else {{
      list.innerHTML = subs.map((item, idx) => {{
        const isLatest = idx === subs.length - 1;
        const isB = item.speaker === HOST_B;
        const spk = (item.speaker || '').replace(/</g, '&lt;');
        const txt = (item.text || '').replace(/</g, '&lt;');
        return `<div class="sub-row ${{isLatest ? 'latest' : ''}}"><span class="spk-tag ${{isB ? 'b' : ''}}">${{spk}}</span><span>${{txt}}</span></div>`;
      }}).join('');
    }}

    const nextEl = document.getElementById('nextTopic');
    nextEl.textContent = (s.upcoming_topics && s.upcoming_topics.length) ? ('NEXT: ' + s.upcoming_topics[0]) : '';
  }} catch (_e) {{}}
}}
setInterval(pollLive, 250);
pollLive();
</script>
</body>
</html>"""


def _render_status_html(status: dict) -> str:
    station = html.escape(str(status.get("station_name", "ハルとミナの、無限ラジオ")))
    mode = html.escape(str(status.get("mode", "fallback")))
    topic = html.escape(str(status.get("current_topic", "")))
    detail = html.escape(str(status.get("current_detail", "")))
    producer = html.escape(str(status.get("producer_status", "")))
    queue_count = int(status.get("queue_count", 0))
    listeners = int(status.get("listeners", 0))
    played = int(status.get("segments_played", 0))
    uptime = status.get("uptime_sec", 0)

    upcoming_items = "".join(
        f"<li>{html.escape(str(t))}</li>" for t in status.get("upcoming_topics", [])
    ) or "<li><em>キュー準備中...</em></li>"

    transcript_blocks = []
    for rec in status.get("recent_transcripts", []):
        r_topic = html.escape(str(rec.get("topic", "")))
        r_time = html.escape(str(rec.get("started_at", "")))
        lines_html = "<br>".join(html.escape(str(l)) for l in rec.get("lines", []))
        transcript_blocks.append(
            f"<div class='card'><h3>{r_topic} <small>({r_time})</small></h3>"
            f"<div class='transcript'>{lines_html or '<em>(文字起こしなし)</em>'}</div></div>"
        )
    transcripts_html = "".join(transcript_blocks) or "<p><em>まだトークログがありません。</em></p>"

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>{station}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #111827; color: #f3f4f6; margin: 0; padding: 24px; line-height: 1.6; }}
  .wrap {{ max-width: 820px; margin: 0 auto; }}
  h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.85rem; background: #2563eb; color: #fff; }}
  .badge.paused {{ background: #6b7280; }}
  .card {{ background: #1f2937; border-radius: 10px; padding: 16px 20px; margin: 16px 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 12px; }}
  .stat {{ background: #111827; padding: 10px 14px; border-radius: 8px; }}
  .stat span {{ display: block; font-size: 0.8rem; color: #9ca3af; }}
  .stat strong {{ font-size: 1.2rem; }}
  audio {{ width: 100%; margin-top: 12px; }}
  .transcript {{ font-family: ui-monospace, Consolas, monospace; font-size: 0.9rem; color: #e5e7eb; max-height: 300px; overflow-y: auto; }}
  a {{ color: #60a5fa; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{station} <span class="badge {'paused' if mode == 'paused' else ''}">{mode.upper()}</span></h1>
  <div class="card">
    <div><strong>放送中:</strong> {topic}</div>
    <div style="color:#9ca3af; font-size:0.9rem;">{detail}</div>
    <audio controls preload="none" src="/radio.mp3"></audio>
    <div style="margin-top:8px; font-size:0.85rem; color:#9ca3af;">
      ストリームURL: <code>/radio.mp3</code> · <a href="/overlay" target="_blank">ライブ字幕オーバーレイ (/overlay)</a> · <a href="/api/live">Live JSON</a> · <a href="/status?format=json">Status JSON</a>
    </div>
    <div class="grid">
      <div class="stat"><span>キュー残数</span><strong>{queue_count} 本</strong></div>
      <div class="stat"><span>リスナー数</span><strong>{listeners}</strong></div>
      <div class="stat"><span>放送済みコーナー</span><strong>{played}</strong></div>
      <div class="stat"><span>YouTube配信</span><strong>{html.escape(str(status.get("youtube_status", "off")))}</strong></div>
    </div>
    <div style="margin-top:10px; font-size:0.85rem; color:#9ca3af;">Producer: {producer} · 稼働時間: {uptime} s</div>
  </div>
  <div class="card">
    <h2 style="margin-top:0; font-size:1.1rem;">次の待機コーナー</h2>
    <ul>{upcoming_items}</ul>
  </div>
  <h2 style="font-size:1.1rem;">直近の掛け合いログ</h2>
  {transcripts_html}
</div>
</body>
</html>"""


def create_app(
    streamer: RadioStreamer,
    config: dict,
    producer_status_fn: Callable[[], str],
    youtube_status_fn: Optional[Callable[[], str]] = None,
) -> web.Application:
    """Create the aiohttp web application for /radio.mp3, /status, /overlay, and /api/live."""
    scfg = config.get("server", {})
    acfg = config.get("audio", {})
    station_name = str(scfg.get("station_name", "ハルとミナの、無限ラジオ"))
    station_genre = str(scfg.get("station_genre", "AI Talk / Japanese"))
    bitrate_num = str(acfg.get("mp3_bitrate", "128k")).lower().rstrip("k")

    async def handle_root(_request: web.Request) -> web.Response:
        raise web.HTTPFound("/status")

    async def handle_status(request: web.Request) -> web.Response:
        status = streamer.get_status(producer_status=producer_status_fn())
        status["youtube_status"] = youtube_status_fn() if youtube_status_fn else "off"
        accept = request.headers.get("Accept", "")
        if request.query.get("format") == "json" or (
            "application/json" in accept and "text/html" not in accept
        ):
            return web.json_response(status)
        return web.Response(text=_render_status_html(status), content_type="text/html")

    async def handle_live_api(_request: web.Request) -> web.Response:
        live = streamer.get_live_state()
        live["youtube_status"] = youtube_status_fn() if youtube_status_fn else "off"
        return web.json_response(live)

    async def handle_overlay(request: web.Request) -> web.Response:
        transparent = request.query.get("transparent", "0") in ("1", "true", "yes")
        return web.Response(text=_render_overlay_html(config, transparent=transparent), content_type="text/html")

    async def handle_radio_mp3(request: web.Request) -> web.StreamResponse:
        headers = {
            "Content-Type": "audio/mpeg",
            "Cache-Control": "no-cache, no-store",
            "Pragma": "no-cache",
            "Accept-Ranges": "none",
            "icy-name": station_name,
            "icy-genre": station_genre,
            "icy-br": bitrate_num,
            "icy-sr": "44100",
            "icy-pub": "0",
        }
        response = web.StreamResponse(status=200, reason="OK", headers=headers)
        await response.prepare(request)

        initial_bytes, q = streamer.subscribe()
        peer = request.remote or "unknown"
        logger.info("Listener connected: %s (total=%d)", peer, streamer.listener_count)
        try:
            if initial_bytes:
                await response.write(initial_bytes)
            while True:
                chunk = await q.get()
                await response.write(chunk)
        except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass
        except Exception as exc:
            logger.debug("Listener %s disconnected: %s", peer, exc)
        finally:
            streamer.unsubscribe(q)
            logger.info("Listener disconnected: %s (total=%d)", peer, streamer.listener_count)
        return response

    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/overlay", handle_overlay)
    app.router.add_get("/api/live", handle_live_api)
    app.router.add_get("/radio.mp3", handle_radio_mp3)
    return app


"""Topic manager with YAML fallback, RSS feed ingestion, and SQLite deduplication."""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import random
import re
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import feedparser
import yaml

from src.moderation import is_topic_safe

logger = logging.getLogger(__name__)

# Combinatorial pools for dynamic evergreen radio topics (40 themes x 12 angles x time-of-day)
DYNAMIC_THEMES: list[str] = [
    "コンビニスイーツ", "ご当地ラーメン", "喫茶店のメニュー", "朝のコーヒーとお茶", "夜食の定番",
    "街の本屋さん", "雨の日の過ごし方", "散歩で見つける景色", "お気に入りの文房具", "キッチン家電・便利グッズ",
    "イヤホンとスピーカー", "懐かしいテレビゲーム", "映画館の体験", "温泉と銭湯", "電車の旅・ローカル線",
    "商店街の雰囲気", "写真とカメラ", "デスク周りのこだわり", "スマホのホーム画面整理", "睡眠と寝具の工夫",
    "季節の旬の食べ物", "パン屋さんとベーカリー", "カレーのこだわり", "お弁当とおにぎり", "自炊の時短テク",
    "読書と積読", "ボードゲームやカードゲーム", "音楽プレイリスト作り", "星空や夜景", "公園とベンチ",
    "DIYとものづくり", "観葉植物やガーデニング", "ペットや動物の仕草", "方言や言葉の響き", "ラジオ番組の思い出",
    "ガチャガチャやカプセルトイ", "美術館や博物館", "キャンプとアウトドア", "冬の防寒・夏の涼み方", "理想の休日のタイムスケジュール",
]

DYNAMIC_ANGLES: list[tuple[str, str]] = [
    ("自分なりのこだわり・マイルール", "譲れない手順やちょっとしたマイルールについて語り合う。"),
    ("忘れられない思い出・原体験", "昔ハマったきっかけや、今でも覚えているエピソードを共有する。"),
    ("ちょっとした失敗談やあるある", "思わず笑ってしまう失敗談や、共感しやすいあるあるネタを出し合う。"),
    ("ベスト3を選ぶなら", "お気に入りや推しのベスト3を挙げて、理由を楽しく比べ合う。"),
    ("定番派 vs 変わり種派", "王道の良さと、ちょっと冒険する面白さについてゆるく討論する。"),
    ("もしも予算無制限で楽しむなら", "現実を忘れて理想の楽しみ方を自由に妄想してみる。"),
    ("昔と今で変わったこと", "時代や年齢とともに好みや楽しみ方がどう変化したか振り返る。"),
    ("初心者にひとつだけおすすめするなら", "これから試す人に向けて、最初の一歩にぴったりな楽しみ方を紹介する。"),
    ("ちょっとだけ贅沢する日の楽しみ方", "普段より少しだけ特別な選び方やご褒美の過ごし方を話す。"),
    ("意外と知られていない隠れた魅力", "地味だけど実は奥深いポイントや通な楽しみ方を掘り下げる。"),
    ("10年後にはどう進化している？", "これからの技術やライフスタイルでどう変わりそうか楽しく予想する。"),
    ("誰かと一緒に楽しむ vs ひとりで楽しむ", "ひとり時間の気楽さと、誰かと共有する楽しさの違いを語り合う。"),
]


def _current_time_mood() -> tuple[str, str]:
    hour = datetime.now().hour
    if 5 <= hour < 10:
        return "朝", "朝のさわやかな空気の中で"
    if 10 <= hour < 16:
        return "昼", "昼下がりのリラックスした気分で"
    if 16 <= hour < 20:
        return "夕方", "夕方のひと息つく時間に"
    if 20 <= hour < 24:
        return "夜", "夜のくつろぎタイムに"
    return "深夜", "静かな深夜の落ち着いた空気の中で"


@dataclass
class Topic:
    title: str
    detail: str
    source: str  # "yaml" or "rss:<url>"
    topic_hash: str
    source_name: str = ""
    source_url: str = ""


def _make_hash(title: str) -> str:
    norm = re.sub(r"\s+", " ", title.strip()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _clean_text(text: str, max_len: int = 120) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


class TopicManager:
    """Manages radio topics from topics.yaml and RSS feeds with SQLite history."""

    def __init__(self, config: dict, base_dir: Path):
        tcfg = config.get("topics", {})
        self.topics_file = base_dir / tcfg.get("file", "topics.yaml")
        self.db_path = base_dir / tcfg.get("db_path", "state.db")
        self.rss_feeds: list[str] = list(tcfg.get("rss_feeds", []))
        self.rss_refresh_sec = float(tcfg.get("rss_refresh_minutes", 45)) * 60.0
        self.rss_timeout_sec = float(tcfg.get("rss_timeout_sec", 8))
        self.rss_max_per_feed = int(tcfg.get("rss_max_per_feed", 30))
        self.rss_ratio = max(0.0, min(1.0, float(tcfg.get("rss_ratio", 0.75))))
        self.dynamic_topics_enabled = bool(tcfg.get("dynamic_topics_enabled", True))
        self.moderation_cfg = config.get("moderation", {})

        self._rss_cache: list[Topic] = []
        self._last_rss_fetch: float = 0.0
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS used_topics (
                    topic_hash TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    source TEXT NOT NULL,
                    used_at REAL NOT NULL,
                    use_count INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.commit()

    def _load_yaml_topics(self) -> list[Topic]:
        if not self.topics_file.exists():
            logger.warning("topics file not found: %s", self.topics_file)
            return [
                Topic(
                    title="最近気になるちょっとした出来事",
                    detail="日常の中でふと気づいたことや最近のマイブームについて。",
                    source="default",
                    topic_hash=_make_hash("最近気になるちょっとした出来事"),
                )
            ]
        with open(self.topics_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = data.get("topics", [])
        out: list[Topic] = []
        for item in items:
            if isinstance(item, str):
                title = _clean_text(item, 80)
                detail = ""
            elif isinstance(item, dict):
                title = _clean_text(str(item.get("title", "")), 80)
                detail = _clean_text(str(item.get("detail", "")), 140)
            else:
                continue
            if title:
                safe, reason = is_topic_safe(title, detail, self.moderation_cfg)
                if not safe:
                    logger.info("Filtered YAML topic '%s' (%s)", title, reason)
                    continue
                out.append(Topic(title=title, detail=detail, source="yaml", topic_hash=_make_hash(title)))
        return out

    def _fetch_single_rss(self, url: str) -> list[Topic]:
        out: list[Topic] = []
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "HaruLocalAIRadio/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.rss_timeout_sec) as resp:
                raw = resp.read()
            parsed = feedparser.parse(raw)
            feed_title = _clean_text(getattr(parsed.feed, "title", ""), 40) or urlparse(url).netloc
            for entry in parsed.entries[: self.rss_max_per_feed]:
                title = _clean_text(getattr(entry, "title", ""), 80)
                summary = _clean_text(getattr(entry, "summary", ""), 120)
                entry_link = str(getattr(entry, "link", "")).strip()
                if title:
                    safe, reason = is_topic_safe(title, summary, self.moderation_cfg)
                    if not safe:
                        logger.debug("Filtered RSS topic '%s' (%s)", title, reason)
                        continue
                    out.append(
                        Topic(
                            title=title,
                            detail=summary,
                            source=f"rss:{url}",
                            topic_hash=_make_hash(title),
                            source_name=feed_title,
                            source_url=entry_link or url,
                        )
                    )
        except Exception as exc:
            logger.warning("RSS fetch failed for %s: %s", url, exc)
        return out

    def _refresh_rss_if_needed(self) -> None:
        if not self.rss_feeds:
            return
        now = time.time()
        if now - self._last_rss_fetch < self.rss_refresh_sec and self._rss_cache:
            return
        self._last_rss_fetch = now
        seen_hashes: set[str] = set()
        fetched: list[Topic] = []
        max_workers = min(8, max(1, len(self.rss_feeds)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for items in ex.map(self._fetch_single_rss, self.rss_feeds):
                for item in items:
                    if item.topic_hash not in seen_hashes:
                        seen_hashes.add(item.topic_hash)
                        fetched.append(item)
        if fetched:
            self._rss_cache = fetched
            logger.info("Refreshed %d unique safe RSS topics from %d feeds", len(fetched), len(self.rss_feeds))

    def _generate_dynamic_topic(self, used: dict[str, tuple[float, int]]) -> Optional[Topic]:
        """Generate a fresh combinatorial radio topic (theme x angle x time-of-day) not recently used."""
        if not self.dynamic_topics_enabled:
            return None
        time_label, time_phrase = _current_time_mood()
        candidates: list[Topic] = []
        for theme in DYNAMIC_THEMES:
            for angle_title, angle_detail in DYNAMIC_ANGLES:
                title = f"{theme}：{angle_title}"
                detail = f"{time_phrase}、「{theme}」について{angle_detail}"
                th = _make_hash(title)
                if th not in used:
                    candidates.append(Topic(title=title, detail=detail, source="dynamic", topic_hash=th))
        if candidates:
            return random.choice(candidates)
        # If all 480 base combinations were eventually used, append time-of-day flavor
        theme = random.choice(DYNAMIC_THEMES)
        angle_title, angle_detail = random.choice(DYNAMIC_ANGLES)
        title = f"{time_label}に語る「{theme}」の{angle_title}"
        detail = f"{time_phrase}、「{theme}」について{angle_detail}"
        return Topic(title=title, detail=detail, source="dynamic", topic_hash=_make_hash(title))

    def _get_used_map(self) -> dict[str, tuple[float, int]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT topic_hash, used_at, use_count FROM used_topics"
            ).fetchall()
        return {row[0]: (float(row[1]), int(row[2])) for row in rows}

    def mark_used(self, topic: Topic) -> None:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO used_topics (topic_hash, title, detail, source, used_at, use_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(topic_hash) DO UPDATE SET
                    used_at = excluded.used_at,
                    use_count = used_topics.use_count + 1
                """,
                (topic.topic_hash, topic.title, topic.detail, topic.source, now),
            )
            conn.commit()

    def next_topic(self, mark: bool = True) -> Topic:
        """Select the next unused topic (mixing RSS and YAML), recycling oldest if exhausted."""
        self._refresh_rss_if_needed()
        yaml_topics = self._load_yaml_topics()
        rss_topics = list(self._rss_cache)

        used = self._get_used_map()

        unused_rss = [t for t in rss_topics if t.topic_hash not in used]
        unused_yaml = [t for t in yaml_topics if t.topic_hash not in used]

        chosen: Optional[Topic] = None
        if unused_rss and (unused_yaml or self.dynamic_topics_enabled):
            if random.random() < self.rss_ratio:
                chosen = random.choice(unused_rss)
            elif unused_yaml:
                chosen = random.choice(unused_yaml)
            else:
                chosen = self._generate_dynamic_topic(used) or random.choice(unused_rss)
        elif unused_rss:
            chosen = random.choice(unused_rss)
        elif unused_yaml:
            chosen = random.choice(unused_yaml)
        else:
            # Prefer a fresh dynamic combinatorial topic instead of repeating old YAML topics verbatim
            chosen = self._generate_dynamic_topic(used)
            if chosen is None:
                all_topics = yaml_topics + rss_topics
                all_topics.sort(key=lambda t: used.get(t.topic_hash, (0.0, 0))[0])
                top_k = all_topics[: max(1, min(5, len(all_topics)))]
                chosen = random.choice(top_k)

        if mark:
            self.mark_used(chosen)
        return chosen


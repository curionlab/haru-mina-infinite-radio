"""Content moderation utilities for topic filtering and pre-broadcast transcript checks.

Helps keep automated 24/7 AI radio streams compliant with YouTube Community Guidelines
by filtering out sensitive topics (politics, medical/legal/financial advice, violent news)
and rejecting generated segments containing NG words before they enter the broadcast queue.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Default sensitive-domain keywords to exclude from RSS / auto-picked topics
DEFAULT_BLOCKED_TOPIC_KEYWORDS: list[str] = [
    # Politics & elections
    "選挙", "衆院選", "参院選", "政党", "首相", "大統領", "内閣", "国会", "法案", "政治資金", "政権", "投票", "議員",
    # Medical / health advice / disease
    "診断", "処方箋", "治療法", "副作用", "ワクチン", "臨床試験", "がん治療", "うつ病", "精神疾患", "自殺", "自傷",
    # Financial / investment / legal advice
    "投資信託", "仮想通貨", "暗号資産", "FX", "株予想", "株価暴落", "確定申告", "訴訟", "裁判", "弁護士", "違法", "逮捕", "容疑者",
    # Crime, violence, disasters, tragedy
    "殺人", "殺害", "暴行", "強盗", "放火", "テロ", "爆破", "遺体", "死亡", "死傷", "重体", "虐待", "性犯罪", "戦争", "空爆", "ミサイル",
]

# Default NG words for pre-broadcast transcript inspection (hate speech, violence, self-harm, etc.)
DEFAULT_TRANSCRIPT_NG_WORDS: list[str] = [
    "死ね", "殺す", "殺せ", "自殺", "自傷", "爆破", "テロ",
    "差別", "キチガイ", "基地外", "障害者差別", "レイプ", "強姦",
    "詐欺の方法", "違法薬物", "麻薬", "覚せい剤", "大麻",
]


def is_topic_safe(
    title: str,
    detail: str,
    moderation_cfg: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return (is_safe, matched_reason) for a candidate topic."""
    cfg = moderation_cfg or {}
    if not cfg.get("enabled", True):
        return True, ""
    if not cfg.get("filter_topics", True):
        return True, ""

    keywords = list(DEFAULT_BLOCKED_TOPIC_KEYWORDS)
    extra = cfg.get("blocked_topic_keywords") or []
    if isinstance(extra, list):
        keywords.extend(str(k).strip() for k in extra if str(k).strip())

    haystack = f"{title} {detail}".lower()
    for kw in keywords:
        if kw and kw.lower() in haystack:
            return False, f"blocked topic keyword '{kw}'"
    return True, ""


def check_transcript_safety(
    transcript: Iterable[dict[str, Any]],
    moderation_cfg: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Inspect full spoken transcript text for NG words or blocked regex patterns."""
    cfg = moderation_cfg or {}
    if not cfg.get("enabled", True):
        return True, ""

    ng_words = list(DEFAULT_TRANSCRIPT_NG_WORDS)
    extra = cfg.get("ng_words") or []
    if isinstance(extra, list):
        ng_words.extend(str(w).strip() for w in extra if str(w).strip())

    full_text = "".join(str(ev.get("text", "")) for ev in transcript)
    compact = re.sub(r"\s+", "", full_text).lower()

    for word in ng_words:
        w = re.sub(r"\s+", "", word).lower()
        if w and w in compact:
            return False, f"NG word detected ('{word}')"

    patterns = cfg.get("ng_regex") or []
    if isinstance(patterns, list):
        for pat in patterns:
            pat_s = str(pat).strip()
            if not pat_s:
                continue
            try:
                if re.search(pat_s, full_text, flags=re.IGNORECASE):
                    return False, f"NG pattern matched ('{pat_s}')"
            except re.error as exc:
                logger.warning("Invalid moderation ng_regex '%s': %s", pat_s, exc)

    return True, "ok"


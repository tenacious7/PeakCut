"""Streamer clip rules: check if a streamer permits clipping, has clip incentives/tips, or forbids clipping."""
import json
import re
from pathlib import Path

RULES_FILE = Path(__file__).parent / "clip_rules.json"
_CACHE = None


def load():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        if RULES_FILE.exists():
            _CACHE = json.loads(RULES_FILE.read_text(encoding="utf-8"))
        else:
            _CACHE = {}
    except Exception:
        _CACHE = {}
    return _CACHE


def normalize_key(platform_or_url: str, handle: str = None) -> str:
    """Normalize input into platform:handle key format e.g. 'twitch:xqc' or 'youtube:@ishowspeed'."""
    if not platform_or_url:
        return ""
    s = str(platform_or_url).strip()
    if handle:
        p = s.lower().replace(".com", "")
        h = str(handle).strip().lower()
        if p == "youtube" and not h.startswith("@"):
            h = "@" + h
        return f"{p}:{h}"

    # Extract from URL
    m = re.match(r"^https?://(?:www\.)?twitch\.tv/([A-Za-z0-9_]+)", s, re.I)
    if m:
        return f"twitch:{m.group(1).lower()}"
    m = re.match(r"^https?://(?:www\.)?kick\.com/([A-Za-z0-9_]+)", s, re.I)
    if m:
        return f"kick:{m.group(1).lower()}"
    m = re.match(r"^https?://(?:www\.)?youtube\.com/(@[A-Za-z0-9_.\-]+)", s, re.I)
    if m:
        return f"youtube:{m.group(1).lower()}"
    if ":" in s:
        p, h = s.split(":", 1)
        return f"{p.lower()}:{h.lower()}"
    return s.lower()


def get(channel_or_handle: str, platform: str = None) -> dict:
    """Return clipping rule for a streamer or URL."""
    rules = load()
    key = normalize_key(platform or channel_or_handle, channel_or_handle if platform else None)
    if key in rules:
        r = rules[key]
        return {
            "key": key,
            "rule": r.get("rule", "unknown"),
            "note": r.get("note", ""),
            "source": r.get("source", "")
        }
    # Check partial key match (e.g. handle without platform)
    h_clean = key.split(":")[-1].lstrip("@").lower()
    for k, r in rules.items():
        if k.split(":")[-1].lstrip("@").lower() == h_clean:
            return {
                "key": k,
                "rule": r.get("rule", "unknown"),
                "note": r.get("note", ""),
                "source": r.get("source", "")
            }
    return {
        "key": key,
        "rule": "unknown",
        "note": "Check the creator's channel rules before publishing clips.",
        "source": ""
    }


def for_creator(c: dict) -> dict:
    """Find the most relevant rule for a creator object."""
    if not c:
        return {"rule": "unknown", "note": "", "source": ""}
    # 1. Try checking their channels dictionary
    channels = c.get("channels") or {}
    for plat, url in channels.items():
        res = get(url, platform=plat)
        if res.get("rule") != "unknown":
            return res
    # 2. Try single channel field
    if c.get("channel"):
        res = get(c["channel"])
        if res.get("rule") != "unknown":
            return res
    # 3. Try creator name
    if c.get("name"):
        res = get(c["name"])
        if res.get("rule") != "unknown":
            return res
    return {"rule": "unknown", "note": "Check creator's rules before publishing clips.", "source": ""}

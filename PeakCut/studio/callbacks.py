"""Finds BEFORE/AFTER callback moments in a transcript: an earlier claim, bet, or prediction
and its later payoff (e.g. 'I bet I will never die' -> 15 minutes later gets wasted)."""
import math
import re
from collections import Counter

CLAIM = re.compile(
    r"\b(i will never|i'll never|never again|never gonna|i('m| am) never|i guarantee|i promise|trust me|"
    r"no way i|no chance|i bet|bet you|watch (this|me|how)|easy|too easy|this time|i('m| am) (gonna|going to) win|"
    r"i can't lose|can't die|won't die|you('ll| will) see|i('ll| will) prove|i called it|prediction|mark my words)\b",
    re.I
)

PAYOFF = re.compile(
    r"\b(told you|i said|remember (when|that)|called it|knew it|again|here we go again|i lied|wasted|busted|"
    r"dead|died|die|fail|failed|crashed|lost|never mind|nevermind)\b",
    re.I
)

FAIL_WORDS = re.compile(r"\b(died|dead|die|wasted|busted|arrested|failed|fail|crashed|crash|fell|missed|lost|killed|exploded)\b", re.I)

STOP = set("""that this with have from they them were been what when your about just like yeah okay know going gonna there here then than into over some more very really want
chat bro man dude shit fuck fucking oh don't didn't can't it's i'm that's you're what's let's gotta wanna thing things right well come look need think because would could should still back
only even also much many make made good dont cant its youre thats whats lets people""".split())


def extract_words(text: str) -> list:
    return [w[:7] for w in re.findall(r"[a-z']{4,}", text.lower()) if w not in STOP and w.strip("'") not in STOP]


def find_callbacks(segments: list, min_gap: float = 60.0, max_gap: float = 3600.0, top: int = 15) -> list:
    """Finds setup/payoff candidate pairs from transcript segments.
    Each segment is expected to be a dict with 'start', 'end', 'text'."""
    if not segments or len(segments) < 2:
        return []

    lines = []
    for i, s in enumerate(segments):
        txt = s.get("text", "").strip()
        lines.append({
            "i": i,
            "start": float(s.get("start", s.get("s", 0.0))),
            "end": float(s.get("end", s.get("e", 0.0))),
            "text": txt,
            "words": extract_words(txt),
            "is_claim": bool(CLAIM.search(txt)),
            "is_payoff": bool(PAYOFF.search(txt) or FAIL_WORDS.search(txt))
        })

    df = Counter(w for l in lines for w in set(l["words"]))
    n_lines = len(lines)
    idf = lambda w: math.log(n_lines / df[w]) if df.get(w) else 0

    candidates = []
    claims = [l for l in lines if l["is_claim"]]

    for a in claims:
        wa = {w for w in a["words"] if df.get(w, 0) <= 8}
        claim_match = CLAIM.search(a["text"])
        claim_str = claim_match.group(0) if claim_match else "claim"

        for b in lines:
            gap = b["start"] - a["end"]
            if gap < min_gap or gap > max_gap:
                continue
            if not b["is_payoff"]:
                continue

            shared = sorted(wa & set(b["words"]), key=lambda w: -idf(w))
            score = 3.0 + sum(idf(w) for w in shared[:3])
            payoff_match = PAYOFF.search(b["text"]) or FAIL_WORDS.search(b["text"])
            payoff_str = payoff_match.group(0) if payoff_match else "payoff"

            candidates.append({
                "score": round(score, 1),
                "gap_min": round(gap / 60, 1),
                "setup": {
                    "start": round(a["start"], 1),
                    "end": round(a["end"], 1),
                    "text": a["text"],
                    "phrase": claim_str
                },
                "payoff": {
                    "start": round(b["start"], 1),
                    "end": round(b["end"], 1),
                    "text": b["text"],
                    "phrase": payoff_str
                },
                "shared_words": shared[:3]
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top]

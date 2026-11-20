# No-AI picker: scores windows from tags (CHAT/CLIPPED/LOUD/HEAT), exclamation words and callback payoffs; writes picks.json for batch.py.
# Usage: python nopick.py <clips_folder> [N=10] [--out picks.json] | --candidates [N=30] (writes clips/candidates.txt for the AI to read) | --selftest      (weaker than an AI pick: no sense of premise)
import json, re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent)); import callbacks
import batch
MIN_W, MAX_W, TARGET = 15, 32, 24  # batch.MAX_SPAN is 32; story length matters more than the old 22 s target
EXC = re.compile(r"\b(oh my (god|gosh)|no way|bro|what the|what\?|holy|yo+|let'?s go+|bruh|are you kidding|crazy|insane|scared|nervous)\b|!|\bwhat\b", re.I)
LAUGH = re.compile(r"\b(ha(ha)+|lol|lmao|laugh\w*|dying|dead)\b", re.I)
BAD_START = re.compile(r"^(and|but|so|he|she|it|they|that|this|then|or)\b", re.I)
END = re.compile(r"[.?!]['\")]?\s*(\[[^\]]*\]\s*)*$")
def clipped(l): return sum(float(x) for x in re.findall(r"\[CLIPPED x(\d+(?:\.\d+)?)", l["txt"]))
MOMW = dict(chase=2.0, funny=2.0, banner=2.0)  # other types (crash/fight/stunt/reaction) are hints only: 0.3
def moment(l): return sum(MOMW.get(t, 0.3) * float(v) for t, v in re.findall(r"\[MOMENT (\w+) (\d(?:\.\d+)?)\]", l["txt"]))
def lscore(l, cb=()):
    p = l["plain"]; return l["loud"] / 10 + l["chat"] / 3 + clipped(l) * 1.5 + moment(l) + l["heat"] / 40 + 0.7 * len(EXC.findall(p)) + 0.8 * len(LAUGH.findall(p)) + (1 if l["s"] in cb else 0)
def streamer(meta):
    t = meta.get("title", ""); m = re.match(r"(.+?)\s+(reacts?|plays?)\b", t, re.I)
    return re.sub(r"\s+Live$", "", (m.group(1) if m and not re.search(r"gta|vi\b", m.group(1), re.I) else meta.get("channel", "")).strip())
def hook(win, name, title=""):
    """Choose a whole local phrase; use only explicit title actions as a fallback."""
    name = " ".join(name.upper().split())
    dangling = set("a an the to and or but so yo oh what my of for with at in on from by as than because if when is are was were be been being am can could will would should must have has had do does did not very really absolutely".split())

    def phrase(text):
        text = " ".join(text.strip(" .,!?:;\"()").split()).upper()
        # Reject cut-off tokens instead of repairing them into invented words.
        if not text or re.search(r"[-–—…]|\.{2,}", text): return ""
        words = text.split()
        if words[-1].strip("'\"").lower() in dangling: return ""
        result = text if not name or text.startswith(name + " ") else f"{name} {text}"
        return result if len(result.split()) <= 6 else ""

    best, bs = "", -1
    peak = max(win, key=lscore) if win else None
    for l in win:
        if abs(l["s"] - peak["s"]) > 3: continue
        for c in re.split(r"(?<=[.?!,;])\s+", l["plain"].replace("’", "'")):
            text = c.strip(" .,!?;\"").lower()
            exclamation = re.fullmatch(r"oh my (?:god|gosh)|no way|let'?s go+|holy (?:shit|crap|cow|hell)|wow+|bruh", text)
            # shortcut: conservative English phrase patterns; extend for new transcript idioms.
            claim = re.match(
                r"(?:i|we|you|he|she|they|it|this|that)(?:'s|'re|'m|\s+(?:is|are|am|was|were|looks?|sounds?|seems?|feels?|"
                r"(?:can'?t|cannot|don't|didn't|won't|will|would|could|should)\s+\w+|"
                r"called|told|knew|found|saw|won|lost|died|crashed|added|removed|love|hate))\s+\S", text)
            if not (exclamation or claim): continue
            candidate = phrase(text)
            if not candidate: continue
            score = bool(exclamation) * 2 + bool(c.rstrip().endswith("!")) + lscore(l)
            if score > bs: best, bs = candidate, score
    if best: return best

    # Match the actual title subject, retaining negation and the stated action.
    subject = re.escape(name) + r"\s+" if name else ""
    action = re.match(subject + r"((?:(?:never|does not|doesn't)\s+)?(?:reacts?\s+to|plays?|watch(?:es)?|reviews?|tr(?:y|ies)))\s+(.+)", title, re.I)
    if action:
        verb, obj = action.groups()
        obj = re.split(r"\s+[|–—-]\s+|\s*[([]", obj)[0].strip(" .!?")
        obj = re.sub(r"\bGTA\s+(6|VI)\b", r"GTA\1", obj, flags=re.I)
        fallback = phrase(f"{verb} {obj}")
        if not fallback and re.match(r"GTA(?:6|VI)\b", obj, re.I):
            fallback = phrase(f"{verb} {obj.split()[0]}")
        if fallback: return fallback
    return phrase("WATCH THIS") or phrase("LOOK")
def windows(lines, cb):
    """One candidate per peak line: ends on the peak's sentence end, starts so the clip is ~TARGET s with the peak in the last 40%."""
    out = []
    for k, p in enumerate(lines):
        sc = lscore(p, cb)
        if sc <= 0.5: continue
        e = p["e"]; j = k + 1  # take trailing reaction lines only while still ending on a sentence end and short
        while j < len(lines) and lines[j]["s"] - e < 1.5 and lines[j]["e"] - e < 3 and not END.search(lines[j - 1]["txt"].strip()): j += 1
        e = lines[j - 1]["e"] if not END.search(p["txt"].strip()) else p["e"]
        ks = [i for i in range(k + 1) if MIN_W <= e - lines[i]["s"] <= MAX_W and e - lines[i]["s"] <= TARGET + 3]
        ks = [i for i in ks if not BAD_START.match(lines[i]["plain"])] or ks
        if not ks: continue
        i = min(ks, key=lambda i: abs(e - lines[i]["s"] - TARGET))
        if p["s"] - lines[i]["s"] < 0.4 * (e - lines[i]["s"]): continue  # peak must sit in the last 60%: payoff near the end
        out.append((sc, lines[i]["s"], e, [l for l in lines if l["s"] >= lines[i]["s"] and l["e"] <= e]))
    return out
def pick(clips, n=10):
    clips = Path(clips); cands = []
    for t in sorted((clips / "src").glob("*.txt")):
        mf = t.with_suffix(".meta.json"); meta = json.load(open(mf, encoding="utf-8")) if mf.exists() else {}
        lines = callbacks.parse(t); name = streamer(meta)
        try: cb = {r["payoff"][0] for r in callbacks.find(lines, top=10)}
        except Exception: cb = set()
        for sc, s, e, win in windows(lines, cb): cands.append(dict(score=sc, s=s, e=e, win=win, id=t.stem, meta=meta, who=name))
    cands.sort(key=lambda c: -c["score"]); res = []
    for c in cands:
        if len(res) >= n: break
        if any(r["id"] == c["id"] and min(c["e"], r["e"]) - max(c["s"], r["s"]) > 0 for r in res): continue
        if sum(r["id"] == c["id"] for r in res) >= batch.PER_SOURCE: continue
        res.append(c)
    out = []
    for c in res:
        m, w = c["meta"], c["who"]; tag = re.sub(r"\W", "", w.lower())
        out.append(dict(url=m.get("url") or f"https://www.youtube.com/watch?v={c['id']}", transcript=f"clips/src/{c['id']}.json", start=round(c["s"], 1), end=round(c["e"], 1),
                        hook=hook(c["win"], w, m.get("title", "")), name=f"{tag or c['id']}_{int(c['s'])}", score=round(c["score"], 1), title=m.get("title", "GTA 6 reaction"),
                        hashtags=" ".join(["#gta6"] + ([f"#{tag}"] if tag else []) + ["#gtavi", "#shorts"]), credit=w or m.get("channel", "")))
    return batch.validate(out)
def tags(win):
    t = []
    for k, rx in (("LOUD", r"\[LOUD"), ("CHAT", r"\[CHAT"), ("CLIP", r"\[CLIPPED"), ("REPL", r"\[REPLAYED"), ("GAME", r"\[GAME")):
        if any(re.search(rx, l["txt"]) for l in win): t.append(k)
    t += sorted({m for l in win for m in re.findall(r"\[MOMENT (\w+)", l["txt"])})
    return ",".join(t) or "-"
def trim(win, head=10, tail=26):
    w = " ".join(l["plain"] for l in win).split()
    return " ".join(w) if len(w) <= head + tail else " ".join(w[:head]) + " ... " + " ".join(w[-tail:])
def candidates(clips, n=30, per=8):
    """The n best windows over all sources as compact lines (~70 tokens each): the AI reads this instead of the full transcripts."""
    clips = Path(clips); cands = []; heads = []
    for t in sorted((clips / "src").glob("*.txt")):
        mf = t.with_suffix(".meta.json"); meta = json.load(open(mf, encoding="utf-8")) if mf.exists() else {}
        lines = callbacks.parse(t)
        try: cb = {r["payoff"][0] for r in callbacks.find(lines, top=10)}
        except Exception: cb = set()
        cs = sorted(windows(lines, cb), key=lambda w: -w[0]); keep = []
        for w in cs:
            if len(keep) >= per: break
            if all(min(w[2], k[2]) - max(w[1], k[1]) <= 0 for k in keep): keep.append(w)
        if keep: heads.append(f"# {t.stem} | {meta.get('channel', '')} | {meta.get('title', '')[:70]}")
        for sc, s, e, win in keep: cands.append((sc, t.stem, s, e, win))
    cands.sort(key=lambda c: -c[0]); top = cands[:n]; ids = {c[1] for c in top}
    out = [h for h in heads if h.split()[1] in ids]
    out.append("# id start-end score tags | words (head ... tail). Read the transcript only around a window you want to widen.")
    out += [f"{i} {s:.1f}-{e:.1f} {sc:.1f} {tags(win)} | {trim(win)}" for sc, i, s, e, win in top]
    return chr(10).join(out)
def selftest():
    import tempfile
    d = Path(tempfile.mkdtemp()); (d / "src").mkdir()
    L = ["[0.0-4.0] So we are looking at the car.", "[4.5-9.0] Here is the thing about the price.", "[9.5-14.0] It costs ten million dollars.", "[14.5-19.0] Nobody can afford this.",
         "[19.5-23.0] Oh my god, no way! [LOUD +25dB] [CHAT x4 KEKW] [HEAT 80]", "[100.0-103.0] hello there."]
    (d / "src" / "a.txt").write_text("\n".join(L), encoding="utf-8"); (d / "src" / "a.meta.json").write_text(json.dumps(dict(title="Kai Cenat Reacts To GTA 6", channel="K", url="u")))
    r = pick(d, 3); assert len(r) == 1 and 15 <= r[0]["end"] - r[0]["start"] <= 28 and r[0]["end"] == 23.0 and r[0]["hook"].startswith("KAI CENAT") and r[0]["credit"] == "Kai Cenat", r
    def line(s, text, loud=0):
        return dict(s=s, e=s + 1, plain=text, txt=text, loud=loud, chat=0, heat=0)

    cases = [
        ([line(10, "What a yo!", 30), line(11, "No way!", 20)], "IShowSpeed", "IShowSpeed Plays GTA 6", "ISHOWSPEED NO WAY"),
        ([line(10, "Oh my god this looks insane!", 30), line(11, "This looks insane!", 20)], "Kai Cenat", "Kai Cenat Plays GTA 6", "KAI CENAT THIS LOOKS INSANE"),
        ([line(0, "Oh my god, no way!", 10), line(10, "I called it.", 35)], "", "", "I CALLED IT"),
        ([line(10, "This is unbe-!", 40)], "xQc", "xQc Reacts To GTA 6", "XQC REACTS TO GTA6"),
        ([line(10, "What the? And to!", 30)], "Ned Luke", "Ned Luke Never Plays GTA 6", "NED LUKE NEVER PLAYS GTA6"),
    ]
    for win, name, title, expected in cases:
        result = hook(win, name, title)
        assert result == expected, (result, expected)
        assert len(result.split()) <= 6 and result == result.upper(), result
        assert not name or result.startswith(name.upper() + " "), result
    assert hook([], "xQc", "Kai Cenat Plays GTA 6") == "XQC WATCH THIS"
    assert hook([], "", "") == "WATCH THIS"
    assert moment(line(0, "x [MOMENT chase 0.8]")) == 1.6 and moment(line(0, "x [MOMENT crash 1.0]")) == 0.3
    c = candidates(d, 5).splitlines(); assert c[0].startswith("# a |") and any(x.startswith("a 4.5-23.0 ") and "LOUD,CHAT" in x and "Oh my god" in x for x in c), c
    print("selftest ok")
if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]: selftest(); sys.exit()
    if "--candidates" in a:
        i = a.index("--candidates"); pos = [x for x in a if not x.startswith("--")]; n = int(pos[1]) if len(pos) > 1 else 30
        txt = candidates(pos[0], n); (Path(pos[0]) / "candidates.txt").write_text(txt + chr(10), encoding="utf-8"); print(txt); sys.exit()
    o = a[a.index("--out") + 1] if "--out" in a else None; pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--out")]
    r = pick(pos[0], int(pos[1]) if len(pos) > 1 else 10); o = o or str(Path(pos[0]) / "picks.json")
    json.dump(r, open(o, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for i, p in enumerate(r, 1): print(f"{i:2} {p['score']:5} {p['name']:24} {p['start']:7}-{p['end']:<7} {p['end'] - p['start']:4.1f}s  {p['hook']}")
    print(f"wrote {len(r)} picks -> {o}")

# Learn what works from the channel's own Shorts (free, no API key, metadata only).
# Usage: python learn.py <channel url or @handle> [--out LEARN.md] [--clips clips] [--refresh] [--limit 60]
#        python learn.py --selftest
import difflib, json, re, statistics as st, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

STOP = set("the a an this that when how why what who i he she they we you it my his her their top best new first last just so and but if watch can did does is are was".split())
def med(v): return st.median(v) if v else 0

def fetch(url, limit, cache):
    # one flat request, cached 24 h so a re-run costs nothing
    if cache.exists() and time.time() - cache.stat().st_mtime < 86400: return json.loads(cache.read_text(encoding="utf-8"))
    u = url if url.startswith("http") else "https://www.youtube.com/" + url.lstrip("/")
    if not u.rstrip("/").endswith("/shorts"): u = u.rstrip("/") + "/shorts"
    r = subprocess.run([sys.executable, "-m", "yt_dlp", "-J", "--flat-playlist", "--playlist-end", str(limit), "--skip-download", u], capture_output=True, text=True, encoding="utf-8")
    if r.returncode or not r.stdout: sys.exit("yt-dlp failed: " + r.stderr[-300:])
    cache.write_text(r.stdout, encoding="utf-8"); return json.loads(r.stdout)

def local_index(clips):
    # title -> picks.json fields, for Shorts we made (picks.json + out/NN_name.txt first line)
    idx = []
    for pj in [clips / "picks.json", clips / "live" / "picks.json"]:
        if pj.exists():
            for p in json.loads(pj.read_text(encoding="utf-8")): idx.append(dict(title=p.get("title", p["hook"].capitalize()), hook=p["hook"], source=p.get("credit"), length=p["end"] - p["start"], style=p.get("style"), layout=p.get("layout")))
    for t in (clips / "out").glob("*.txt") if (clips / "out").exists() else []:
        first = t.read_text(encoding="utf-8").splitlines()[:1]
        if first and not any(i["title"] == first[0] for i in idx): idx.append(dict(title=first[0]))
    return idx

def build(entries, idx):
    rows = []
    for e in entries:
        if e.get("view_count") is None: continue
        t = e.get("title") or ""
        m = max(idx, key=lambda i: difflib.SequenceMatcher(None, i["title"].lower(), t.lower()).ratio(), default=None)
        if m and difflib.SequenceMatcher(None, m["title"].lower(), t.lower()).ratio() < .75: m = None
        m = m or {}
        w = re.findall(r"[A-Za-z0-9']+", t); first = w[0] if w else ""
        ts = e.get("timestamp")
        dur = m.get("length") or e.get("duration")
        rows.append(dict(title=t, views=e["view_count"], source=m.get("source") or (first if first.lower() not in STOP else None),
            length=None if not dur else ("under 20 s" if dur < 20 else "20-30 s" if dur <= 30 else "over 30 s"),
            style=m.get("style"), layout=m.get("layout"), hour=datetime.fromtimestamp(ts, timezone.utc).hour // 6 * 6 if ts else None,
            name_first=bool(first[:1].isupper() and first.lower() not in STOP), number=bool(re.search(r"\d", t)), question="?" in t))
    return rows

def rules(rows):
    out, base = [], med([r["views"] for r in rows])
    cats = [("source", "Sources"), ("length", "Length"), ("style", "Caption style"), ("layout", "Layout"), ("hour", "Post time (UTC 6 h block)")]
    for k, label in cats:
        for v in {r[k] for r in rows if r[k] is not None}:
            g = [r["views"] for r in rows if r[k] == v]
            if len(g) >= 3 and (med(g) / base >= 1.5 or med(g) / base <= .67): out.append((abs(med(g) / base - 1), f"{label}: {v if k != 'hour' else f'{v:02d}-{v + 6:02d}h'} gets {med(g) / base:.1f}x the median views ({len(g)} Shorts)."))
    for k, label in [("name_first", "hooks with a name first"), ("number", "titles with a number"), ("question", "titles that ask a question")]:
        a, b = [r["views"] for r in rows if r[k]], [r["views"] for r in rows if not r[k]]
        if len(a) >= 3 and len(b) >= 3 and (med(a) / max(med(b), 1) >= 1.4 or med(a) / max(med(b), 1) <= .7): out.append((abs(med(a) / max(med(b), 1) - 1), f"{label[0].upper() + label[1:]}: {med(a) / max(med(b), 1):.1f}x the views of those without ({len(a)} vs {len(b)})."))
    return [s for _, s in sorted(out, reverse=True)]

def report(rows, name, dated):
    rows = sorted(rows, key=lambda r: -r["views"]); n = len(rows); k = max(1, round(n * .2))
    L = [f"# What works on {name}", f"{n} Shorts read, median {med([r['views'] for r in rows]):,.0f} views. Matched to our own output: {sum(1 for r in rows if r['style'] or r['layout'] or r['length'] and r['source'])}."]
    if n < 8: return "\n".join(L + ["", f"Only {n} Shorts: too few to learn anything. No rules; pick by the step 2 rules only."]) + "\n"
    if not dated: L.append("Note: YouTube gives no upload date on the Shorts list, so views are raw (the 3 newest are skipped as still gaining); post time is unknown.")
    L += ["", "## Rules for the next picks (follow these when scores tie)"] + ["- " + s for s in rules(rows if dated else rows) or ["No clear pattern yet (nothing is 1.5x above or below the median)."]]
    f = lambda rs: ", ".join(f'{r["title"][:60]} ({r["views"]:,})' for r in rs)
    L += ["", f"## Top 20% ({k})", f(rows[:k]), "", f"## Bottom 20% ({k})", f(rows[-k:])]
    return "\n".join(L) + "\n"

def selftest():
    fx = Path(__file__).with_name("learn_fixture.json")
    ents = [dict(title=t, view_count=v, timestamp=1763000000 + i * 86400 * 3) for i, (t, v) in enumerate([("Speed screams at GTA 6 map", 900000), ("Speed finds KFC", 700000), ("Speed breaks car 3 times", 800000), ("Kai laughs at trailer", 120000), ("Kai reacts?", 100000), ("why gta 6 is late", 40000), ("why this matters?", 30000), ("trailer talk", 35000), ("Kai calm take", 110000), ("Speed again", 600000)])]
    fx.write_text(json.dumps(dict(channel="Fixture", entries=ents)), encoding="utf-8")
    d = json.loads(fx.read_text(encoding="utf-8")); rows = build(d["entries"], [dict(title="Speed finds KFC", source="Speed", length=22, style="word", layout="stack")])
    assert len(rows) == 10 and rows[1]["style"] == "word"
    txt = report(rows, "Fixture", True); assert "Speed" in txt and "Top 20%" in txt, txt
    assert "too few" in report(rows[:5], "x", True); print(txt); print("selftest ok")

def main(a):
    if "--selftest" in a: return selftest()
    arg = lambda f, d: a[a.index(f) + 1] if f in a else d
    if not a or a[0].startswith("--"): sys.exit(__doc__)
    clips = Path(arg("--clips", "clips")); clips.mkdir(exist_ok=True)
    d = fetch(a[0], int(arg("--limit", 60)), clips / ("learn_cache_" + re.sub(r"\W", "_", a[0]) + ".json"))
    ents = d["entries"]; dated = any(e.get("timestamp") for e in ents)
    rows = build(ents if dated else ents[3:], local_index(clips))
    Path(arg("--out", clips / "LEARN.md")).write_text(report(rows, d.get("channel") or a[0], dated), encoding="utf-8")
    print("wrote", arg("--out", clips / "LEARN.md"), "from", len(rows), "Shorts")
if __name__ == "__main__": main(sys.argv[1:])

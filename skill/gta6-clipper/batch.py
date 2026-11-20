# Render every picked moment in one go, then make one overview sheet to check them all.
# Usage: python batch.py picks.json OUT_DIR
# picks.json = [{"url": "...", "transcript": "src/<id>.json", "start": 120.5, "end": 148, "hook": "SPEED FOUND KFC IN GTA 6", "name": "speed_kfc",
#                "title": "...", "hashtags": "#gta6 ...", "credit": "Kai Cenat", "style": "hormozi", "layout": "bubble"}, ...]
#   "format": moment (default) | callback | compilation | escalation | thenvsnow | reveal | chat | waitforit: see formats.py for each one's fields   (title/hashtags/credit/style/layout optional)
import json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")  # hooks can carry emoji; a cp1252 pipe would crash the print
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import formats  # "format": callback, compilation, escalation, thenvsnow, reveal, chat, waitforit (default moment)

PER_SOURCE = 2  # h19: 3 of 6 Shorts came from one Normies video (avg 5.4); a mix scores higher
MAX_SPAN, MAX_TOTAL = 60, 64  # bench2 (18 Nov): viral Twitch clips run 6-61 s, median 25, 4/10 are 48-61 s; the old 32/36 cut setups and blocked every 35 s live clip
OPT_KEYS = ("position", "watermark", "watermark_pos", "watermark_opacity", "watermark_size", "progress", "endcard", "aspects", "emoji", "lang", "flash", "lift", "fillers", "blackfix", "denoise", "squeeze", "flip", "theme", "cap_size", "hook_text", "hook_box", "progress_color", "watermark_color", "colors")
def opts_json(p, skip=()):
    """Pick fields clip.py reads as its 10th argument (brand.json and CLIPPER_* env fill the rest)."""
    return json.dumps({k: p[k] for k in OPT_KEYS if k in p and k not in skip})

def bed_cmd(mp4, p, brand=None):
    """meme.py bed command for a pick with "music": mood (hype|chill|tense|funny) or a file name; brand.json / CLIPPER_MUSIC is the default. Writes <stem>_b.mp4; None when no music."""
    m = p.get("music") or os.environ.get("CLIPPER_MUSIC") or (brand or {}).get("music")
    if not m: return None
    mp4 = Path(mp4); return [sys.executable, str(HERE / "meme.py"), "bed", str(mp4)] + (["--mood", m] if m in ("hype", "chill", "tense", "funny") else ["--file", str(m)]) + ["--out", str(mp4.with_name(mp4.stem + "_b.mp4"))]

def dedupe(picks, moments):
    """Drop picks whose time span overlaps >50% (of the shorter span) a moment already clipped from the same source, or an earlier pick of this run."""
    keep, seen = [], [(m["id"], m["start"], m["end"], "past") for m in moments]
    for p in picks:
        i, look = Path(p["transcript"]).stem, (p.get("style"), p.get("layout"))
        a, b = p.get("start", 0), p.get("end", 1)
        # "again": true = a remake the user asked for; the same span in another style/layout within one run is a variant, not a repeat
        hit = None if p.get("again") else next((m for m in seen if m[0] == i and m[3] in ("past", look) and min(b, m[2]) - max(a, m[1]) > 0.5 * min(b - a, m[2] - m[1])), None)
        spans = [(q["start"], q["end"]) for q in p.get("parts", [])] or [(a, b)]  # a multi-part format = several moments
        hit = hit if not p.get("parts") else None if p.get("again") else next((m for a, b in spans for m in seen if m[0] == i and m[3] in ("past", look) and min(b, m[2]) - max(a, m[1]) > 0.5 * min(b - a, m[2] - m[1])), None)
        if hit: print(f"skip duplicate moment {i} {hit[1]:.0f}-{hit[2]:.0f} already clipped (add \"again\": true to remake it): {p['hook']}")
        else: keep.append(p); seen += [(i, a, b, look) for a, b in spans]
    return keep

def meme_cmd(mp4, p):
    """meme.py command for a pick with "meme": <mood> (and optional "meme_at": seconds), writing <stem>_m.mp4 next to the Short; None when the pick has no meme."""
    if not p.get("meme"): return None
    mp4 = Path(mp4); return [sys.executable, str(HERE / "meme.py"), str(mp4), "--mood", str(p["meme"]), "--out", str(mp4.with_name(mp4.stem + "_m.mp4"))] + (["--at", str(p["meme_at"])] if p.get("meme_at") is not None else [])

def validate(picks):
    """One clear line per unusable pick (skipped), sane defaults for the rest: a missing hook becomes the name in capitals, never a KeyError after the render."""
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    good = []
    for n, p in enumerate(picks, 1):
        try:
            if not isinstance(p, dict): raise ValueError("not an object")
            miss = [k for k in ("url", "transcript", "name") if not p.get(k)]
            if miss: raise ValueError("missing " + ", ".join(miss))
            fmt = formats.fmt_of(p)
            if fmt not in formats.FORMATS: raise ValueError(f"unknown format {fmt!r}")
            sp = p.get("parts") or []
            if len(sp) < formats.NEEDS_PARTS.get(fmt, 0): raise ValueError(f"{fmt} needs at least {formats.NEEDS_PARTS[fmt]} parts")
            for k, s in enumerate(sp or [p], 1):
                if not (isinstance(s, dict) and num(s.get("start")) and num(s.get("end")) and s["end"] > s["start"]): raise ValueError(("part %d" % k if sp else "pick") + " has no valid start/end")
                if s["end"] - s["start"] > MAX_SPAN: raise ValueError(f"{'part %d' % k if sp else 'window'} is {s['end'] - s['start']:.0f} s, over {MAX_SPAN} s: tighten it to one beat (SKILL Length: setup + payoff, up to 60 s) and re-run")
            if sp and sum(s["end"] - s["start"] for s in sp) > MAX_TOTAL: raise ValueError(f"parts add up to {sum(s['end'] - s['start'] for s in sp):.0f} s, over {MAX_TOTAL} s: shorten the parts and re-run")
            if not str(p.get("hook") or "").strip(): p = dict(p, hook=str(p["name"]).replace("_", " ").upper()); print(f"pick {n} ({p['name']}): no hook, using {p['hook']!r}")
            src = p.get("transcript")
            if sum(g.get("transcript") == src for g in good) >= PER_SOURCE: raise ValueError(f"already {PER_SOURCE} Shorts from this source video in this batch; pick a moment from another video")
            good.append(p)
        except Exception as e: print(f"skip bad pick {n} ({p.get('name', '?') if isinstance(p, dict) else '?'}): {e}")
    return good

def main(picks_file, out_dir):
    import shutil
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")): raise SystemExit("ffmpeg not found on PATH. Install: Windows `winget install Gyan.FFmpeg` | macOS `brew install ffmpeg` | Linux `sudo apt install ffmpeg` (then open a new terminal).")
    picks, out = validate(json.load(open(picks_file, encoding="utf-8"))), Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    mom = Path(picks[0]["transcript"]).parent / "moments.json" if picks else None  # clips/src/moments.json: every span clipped so far
    picks = dedupe(picks, formats.P.jload(mom) if mom else [])
    n0 = max((int(m.name[:2]) for m in out.glob("[0-9][0-9]_*.mp4")), default=0)  # continue after the highest number: a moved or failed Short must not cause a repeat
    bf = next((q for q in (os.environ.get("CLIPPER_BRAND"), Path.cwd() / "brand.json", out.parent / "brand.json", out.parent.parent / "brand.json") if q and Path(q).is_file()), None)  # same lookup as clip.load_brand
    try: BRAND = json.load(open(bf, encoding="utf-8")) if bf else {}
    except ValueError: BRAND = {}
    done, t0, lock = [], time.time(), threading.Lock()
    def one(i, p):
        try:
            mp4 = out / f"{n0 + i:02d}_{p['name']}.mp4"
            print(f"[{i}/{len(picks)}] start: {p['hook']}", flush=True)
            r = formats.render(p, mp4) if formats.fmt_of(p) != "moment" else subprocess.run([sys.executable, str(HERE / "clip.py"), p["url"], p["transcript"], str(p["start"]), str(p["end"]), p["hook"], str(mp4), ",".join(p.get("keywords", [])), p.get("style", ""), p.get("layout", ""), opts_json(p)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(f"[{i}/{len(picks)}] " + "\n  ".join((r.stdout + r.stderr).strip().splitlines()[-6:]), flush=True)  # each Short's log in one block, not interleaved
            if r.returncode or not mp4.exists(): print(f"[{i}/{len(picks)}] failed, skipped", flush=True); return  # one bad source must not stop the other nine
            with lock:  # shared files: one writer at a time
                done.append(mp4)
                used = Path(p["transcript"]).parent / "used.json"  # next run skips these sources (clips/src/used.json)
                formats.P.jsave(used, sorted(set(formats.P.jload(used)) | {Path(p["transcript"]).stem}))
                mom = used.with_name("moments.json")  # every clipped range: "more from that video" must never repeat a moment
                if not p.get("again"): formats.P.jsave(mom, formats.P.jload(mom) + [dict(id=Path(p["transcript"]).stem, start=q["start"], end=q["end"], hook=p["hook"]) for q in p.get("parts") or [p]])
            cmd = meme_cmd(mp4, p)  # meme gag at the payoff: never fails the Short
            if cmd:
                m = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"); mm = mp4.with_name(mp4.stem + "_m.mp4")
                if m.returncode == 0 and mm.exists(): os.replace(mm, mp4)
                else: print(f"[{i}/{len(picks)}] meme failed, Short kept as is: " + ((m.stdout + m.stderr).strip().splitlines() or ["no output"])[-1], flush=True)
            cmd = bed_cmd(mp4, p, BRAND)  # music bed under the finished file (any format): never fails the Short
            if cmd:
                m = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"); mb = mp4.with_name(mp4.stem + "_b.mp4")
                if m.returncode == 0 and mb.exists(): os.replace(mb, mp4)
                else: print(f"[{i}/{len(picks)}] music failed, Short kept as is: " + ((m.stdout + m.stderr).strip().splitlines() or ["no output"])[-1], flush=True)
            # ready-to-paste post text next to the Short
            mp4.with_suffix(".txt").write_text("\n".join(x for x in [p.get("title", p["hook"].capitalize()), p.get("hashtags", "#gta6 #gtavi #shorts"),
                                                f"Credit: {p['credit']} ({p['url']})" if p.get("credit") else f"Source: {p['url']}"] if x) + "\n", encoding="utf-8")
            # cover: the frame 1 s in (hook on screen, face in frame), for the Shorts / Reels cover picker
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", str(mp4), "-frames:v", "1", "-q:v", "2", str(out / f"{mp4.stem}_cover.jpg")])
        except Exception as e: print("  failed, skipped:", e, flush=True)
    workers = int(os.environ.get("CLIPPER_WORKERS", 3))  # 3 Shorts at once: downloads and encodes overlap (24-core test PC: see LOG); set 1 on a weak PC
    with ThreadPoolExecutor(workers) as ex: list(ex.map(lambda a: one(*a), enumerate(picks, 1)))
    done.sort()
    # overview: 4 frames per Short (the last one 0.5 s before the end, so a logo card at the end shows), one row each, so a glance shows framing, captions and hook for all of them
    try:
        for m in done:
            d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(m)]))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(m), "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{max(d - 0.6, 0.6) / 3 - 0.05})',scale=180:-1,tile=4x1", "-fps_mode", "passthrough", "-frames:v", "1", str(m.with_suffix(".jpg"))], check=True)
        allrows = [m.with_suffix(".jpg") for m in sorted(out.glob("[0-9][0-9]_*.mp4")) if m.with_suffix(".jpg").exists()]  # every Short in the folder, not just this run
        if allrows:
            rows = sum((["-i", str(j)] for j in allrows), [])
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *rows, "-filter_complex", f"vstack=inputs={len(allrows)}" if len(allrows) > 1 else "null",
                            str(out / "overview.jpg")], check=True)
    except Exception as e: print("overview failed (the Shorts are fine):", e)
    try:  # automatic quality check: PASS / WARN / FAIL per Short, qc.json next to the Shorts (FAIL = re-cut it with clip.py)
        import qc
        res = qc.run(done, picks_file=picks_file, json_out=out / "qc.json", quiet=True)
        for r in res: print(f"  {r['status']:4} {r['file']:34} {'; '.join(r['fail'] + r['warn']) or 'ok'}")
        print(f"qc: {sum(r['status'] == 'PASS' for r in res)} PASS, {sum(r['status'] == 'WARN' for r in res)} WARN, {sum(r['status'] == 'FAIL' for r in res)} FAIL -> {out / 'qc.json'}")
    except Exception as e: print("qc failed (the Shorts are fine):", e)
    print(f"ok {len(done)}/{len(picks)} Shorts in {(time.time() - t0) / 60:.1f} min -> {out}")

if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        P = lambda a, b, i="x": dict(transcript=f"src/{i}.json", start=a, end=b, hook="h")
        M = [dict(id="x", start=100, end=130)]
        assert [p["start"] for p in dedupe([P(105, 135), P(120, 150), P(125, 155), P(100, 130, "y"), P(100, 130)], M)] == [120, 100]  # 105 and 125 hit >50%; 120-150 overlaps 10/30; other id kept; same span as an earlier pick dropped
        assert len(dedupe([P(10, 30), dict(P(10, 30), style="word"), P(10, 30)], [])) == 2  # a style variant of a pick stays, an exact repeat goes
        assert len(dedupe([dict(P(100, 130), again=True)], M)) == 1  # a remake the user asked for is never dropped
        C = dict(P(0, 1), parts=[dict(start=100, end=108), dict(start=700, end=712)])
        assert dedupe([C], M) == [] and len(dedupe([dict(C, parts=[dict(start=200, end=208), dict(start=700, end=712)])], M)) == 1  # each part is checked, not the span between
        assert formats.fmt_of(C) == "callback" and formats.fmt_of(P(0, 1)) == "moment" and formats.fmt_of(dict(P(0, 1), format="chat")) == "chat"
        assert meme_cmd("a/x.mp4", dict(P(0, 1))) is None and meme_cmd("a/x.mp4", dict(P(0, 1), meme="fail", meme_at=12.5))[2:] == [str(Path("a/x.mp4")), "--mood", "fail", "--out", str(Path("a/x_m.mp4")), "--at", "12.5"]
        assert bed_cmd("a/x.mp4", P(0, 1)) is None and bed_cmd("a/x.mp4", dict(P(0, 1), music="hype"))[3:] == [str(Path("a/x.mp4")), "--mood", "hype", "--out", str(Path("a/x_b.mp4"))] and bed_cmd("a/x.mp4", P(0, 1), {"music": "my.mp3"})[4:6] == ["--file", "my.mp3"]
        assert json.loads(opts_json(dict(P(0, 1), watermark="@me", lang="es"))) == {"watermark": "@me", "lang": "es"} and opts_json(P(0, 1)) == "{}"
        ok = dict(url="u", transcript="s/x.json", name="n", start=1, end=9); cb = dict(ok, format="callback", parts=[{"start": 1, "end": 5}, {"start": 7, "end": 12}], hook="H")
        import io, contextlib, tempfile
        with contextlib.redirect_stdout(io.StringIO()) as o:
            g = validate([ok, {"url": "u"}, dict(ok, start=None), dict(ok, parts=[{"start": 1, "end": 5}, {"end": 9}]), dict(ok, format="nope"), "x", cb, dict(ok, end=40), dict(cb, parts=[{"start": 0, "end": 30}, {"start": 40, "end": 70}])])
        assert [x["hook"] for x in g] == ["N", "H"] and o.getvalue().count("skip bad pick") == 7, o.getvalue()  # missing hook defaulted; 5 bad picks + 2 too long skipped
        with contextlib.redirect_stdout(io.StringIO()) as o: g3 = validate([dict(ok, name=f"n{i}") for i in range(3)] + [dict(ok, name="o", transcript="s/y.json")])
        assert [x["name"] for x in g3] == ["n0", "n1", "o"] and "pick 3 (n2)" in o.getvalue(), o.getvalue()  # 3rd Short from one source skipped
        d = Path(tempfile.mkdtemp()); (d / "moments.json").write_text("{bad"); assert formats.P.jload(d / "moments.json") == [] and (d / "moments.json.corrupt").exists()
        formats.P.jsave(d / "used.json", ["a"]); assert formats.P.jload(d / "used.json") == ["a"]
        print("selftest ok"); sys.exit()
    main(*sys.argv[1:3])

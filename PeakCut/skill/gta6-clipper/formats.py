# Formats: one Short in one of many viral shapes. A pick says "format": name (default "moment"; a pick with "parts" and no format = "callback").
# Every format = a small spec: which parts to render with clip.py (cached, so formats can share a render), what cards and overlays ffmpeg adds, then one loudness pass.
#   moment      {start,end}                                     one clip (clip.py as is)
#   callback    {parts:[claim 3-6 s, payoff 8-15 s]}             hard cut; small EARLIER tag if gap > 10 min; "jump_card":true,"jump":"40 MINUTES LATER" = full card
#   compilation {parts:[3-5], hook:"3 TIMES KAI LOST IT"}        #3 card, clip, #2 card, clip, #1 card, best last
#   escalation  {parts:[3-6], word:"STRIKE"}                     STRIKE 1/3 ... 3/3, badge and number warm up from white to red
#   thenvsnow   {parts:[2], labels:["THEN","NOW"]}               label card + corner badge on each part (prediction vs reality)
#   reveal      {start,end, reveal:1.6, tag:"WATCH HIS FACE"}    the face blurred for `reveal` s while the game plays, then it clears
#   chat        {start,end, chat:[...optional lines]}            top chat lines pop up as bubbles after the payoff (Twitch VOD chat, or your lines)
#   waitforit   {start,end, payoff:s (optional), tag}            "WAIT FOR IT" tag + a bar that fills until the payoff, then both vanish with a flash
# Each part may carry its own start/end and "label" (small text on its card). Usage: python formats.py picks.json OUT_DIR  |  python formats.py selftest
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path
from types import SimpleNamespace
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import parts as P
FONTS = HERE / "fonts"
YEL, RED, WHITE = "0xFFD400", "0xFF3B30", "white"
BADGE_Y = 215      # corner badges / tags sit under the platform's top UI
CHAT_Y = 1330      # first chat row (three rows, 80 px apart), above the bottom UI
WHEN = {  # the one line per format for SKILL.md
    "moment": "one strong clip, the default",
    "callback": "a setup early and its payoff much later (a promise, a prediction, a bet)",
    "compilation": "the same thing happens 3-5 times in a stream (rage, fails, jokes): numbered, best last",
    "escalation": "the same thing keeps getting worse in order (strike 1, 2, 3)",
    "thenvsnow": "a claim or plan, then what really happened (prediction vs reality)",
    "reveal": "a reaction to something on screen (stack layout): the face is hidden for 1-2 s while the thing plays, then it clears",
    "chat": "the moment is funny but chat's lines are the joke: chat pops up over it",
    "waitforit": "a quiet or slow build into one big payoff: tag and bar tease it",
}

def txt(s): return P.clean_text(s)  # keeps accented / Arabic / Devanagari letters
def ffm(*a, cwd=None):
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, a)], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd)
    if r.returncode: raise RuntimeError("ffmpeg: " + r.stderr.strip()[-300:])  # cwd = fonts: drawtext wants a plain relative font path on Windows

def dt(text, size, x="(w-text_w)/2", y="200", color=WHITE, box=None, en=None, alpha=None, bw=None):
    """One drawtext (Anton, black outline). box = pill colour behind the text."""
    s0 = int(str(size).strip("'").split("+")[0])
    s = f"drawtext=fontfile={P.font_for(text)}:text='{txt(text)}':fontsize={size}:fontcolor={color}:borderw={s0 // 14 + 3 if bw is None else bw}:bordercolor=black:x={x}:y={y}"
    if box: s += f":box=1:boxcolor={box}:boxborderw={s0 // 4}"
    if alpha: s += f":alpha='{alpha}'"
    return s + (f":enable='{en}'" if en else "")

def card(frame_png, lines, out, d=0.6):
    """Punchy card: the frame blurred and dark, lines = [(text, size, colour)] stacked in the middle, each pops in (+35 % over 0.12 s), a whoosh."""
    ys, y0 = [], -sum(s * 1.15 for _, s, _ in lines) / 2
    for _, s, _ in lines: ys.append(y0 + s * 0.575); y0 += s * 1.15
    dts = "".join("," + dt(t, f"'{s}+{s * 0.35:.0f}*max(0,1-t/0.12)'", y=f"(h-text_h)/2+({y:.0f})", color=c) for (t, s, c), y in zip(lines, ys))
    vf = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:2,eq=brightness=-0.22:saturation=1.2{dts},fps=30,format=yuv420p"
    af = f"anoisesrc=d={d}:c=pink:r=48000:a=0.5,bandpass=f=1800:width_type=o:w=2.5,afade=t=in:d=0.3:curve=exp,afade=t=out:st={d - 0.15}:d=0.15,aformat=channel_layouts=stereo,volume=8dB"
    ffm("-loop", 1, "-t", d, "-i", Path(frame_png).resolve(), "-f", "lavfi", "-t", d, "-i", af, "-vf", vf, "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-shortest", Path(out).resolve(), cwd=FONTS)
    return Path(out)

def frame(mp4, out, first=True):
    ffm(*(["-ss", "0.3"] if first else ["-sseof", "-0.15"]), "-i", Path(mp4).resolve(), "-frames:v", 1, Path(out).resolve()); return Path(out)

def overlay(src, vf, out):
    """Re-encode one part with a filter chain (badges, tags, bubbles); audio untouched."""
    ffm("-i", Path(src).resolve(), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-c:a", "copy", Path(out).resolve(), cwd=FONTS); return Path(out)

def part_mp4(p, span, cache, hook="", layout=None):
    """clip.py on one span (hook only if asked). Cached by everything that shapes the picture, so two formats on the same moment download and render once."""
    a = [p["url"], p["transcript"], str(span["start"]), str(span["end"]), hook, ",".join(p.get("keywords", [])), p.get("style", ""), layout or span.get("layout") or p.get("layout", ""),
         json.dumps({k: p[k] for k in ("position", "watermark", "watermark_pos", "watermark_opacity", "watermark_size", "emoji", "lang", "flash", "lift", "fillers", "blackfix", "denoise", "squeeze") if k in p} | {"progress": 0, "endcard": "0", "aspects": []})]  # explicit off beats env/brand (clip.py opt: pick > env > brand): finish() adds progress/endcard/aspects once to the joined Short
    Path(cache).mkdir(parents=True, exist_ok=True); key = hashlib.md5("|".join(a).encode()).hexdigest()[:12]; f = Path(cache) / (key + ".mp4")
    _sweep(Path(cache))
    with _LOCKS.setdefault(key, threading.Lock()):  # workers are threads in one process: the second one waits here, then finds the finished file
        if f.exists(): _LAST[threading.get_ident()] = f; return f, "cached"
        tmp = f.with_name(f"{key}_tmp.mp4")  # clip.py writes here; os.replace publishes it whole, so a half-written mp4 is never "cached"
        r = subprocess.run([sys.executable, str(HERE / "clip.py"), *a[:5], str(tmp), *a[5:]], capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = (r.stdout + r.stderr).strip().splitlines()
        if r.returncode or not tmp.exists():
            tmp.unlink(missing_ok=True); raise RuntimeError(f"clip.py failed on {span['start']}-{span['end']}: " + (tail[-1] if tail else "no output"))
        os.replace(tmp, f); (tmp.with_name(tmp.stem + ".render.json")).exists() and os.replace(tmp.with_name(tmp.stem + ".render.json"), f.with_suffix(".render.json")); _LAST[threading.get_ident()] = f; return f, tail[-1] if tail else ""

_LAST = {}  # thread -> the last part mp4 clip.py rendered or reused; finish() reads its .render.json for the caption y
_LOCKS, _SWEPT = {}, set()  # shortcut: lock is per process (threads); two separate batch processes on one cache would still race
def _sweep(cache, days=7):
    """Once per process and cache folder: delete .partcache files older than 7 days (stale *_tmp.mp4 included)."""
    if str(cache) in _SWEPT: return
    _SWEPT.add(str(cache))
    for x in Path(cache).glob("*"):
        try:
            if x.is_file() and time.time() - x.stat().st_mtime > days * 86400: x.unlink()
        except OSError: pass

def peak_time(mp4):
    """Loudest moment in the last two thirds of a clip, seconds: where the payoff is."""
    import numpy as np; from clip import audio_db
    db = audio_db(mp4); k = len(db) // 3
    return (k + int(np.argmax(db[k:]))) / 20

# ---- the formats: each takes (pick, work dir, cache dir, log) -> list of mp4 files to join ----
def f_moment(p, w, c, log):
    f, l = part_mp4(p, p, c, p["hook"]); log.append(l); return [f]

def f_callback(p, w, c, log):
    """Claim -> HARD CUT -> payoff (what viral claim/proof Shorts do). A small EARLIER tag on part 1 when the gap is over 10 min; the full jump card only with "jump_card": true."""
    fs, sp = [], p["parts"]
    for k, s in enumerate(sp):
        f, l = part_mp4(p, s, c, p["hook"] if k == 0 else ""); log.append(f"part {k + 1}: {l}")
        if k and p.get("jump_card"): fs.append(P.card(frame(fs[-1], w / f"l{k}.png", False), p.get("jump", "LATER"), w / f"c{k}.mp4"))
        if not k and (float(sp[1]["start"]) - float(sp[0]["end"]) > 600 or p.get("tag")): f = overlay(f, dt(p.get("tag", "EARLIER"), 64, x="50", y=BADGE_Y, box="black@0.55"), w / "o0.mp4")
        fs.append(f)
    return fs

def numbered(p, w, c, log, big_of, badge_of, color_of):
    """Shared by compilation, escalation, thenvsnow: a card (big text, small label) then the part with a corner badge."""
    fs, n = [], len(p["parts"])
    for k, s in enumerate(p["parts"]):
        f, l = part_mp4(p, s, c); log.append(f"part {k + 1}: {l}")
        col, big, sub = color_of(k, n), big_of(k, n), txt(s.get("label", ""))
        lines = ([(txt(p["hook"]), 78, YEL)] if k == 0 and p.get("hook") else []) + [(big, 330 if len(big) < 5 else 190, col)] + ([(sub, 80, WHITE)] if sub else [])
        fs.append(card(frame(f, w / f"f{k}.png"), lines, w / f"c{k}.mp4", 0.9 if k == 0 else 0.6))
        fs.append(overlay(f, dt(badge_of(k, n), 100, x="50", y=BADGE_Y, color=col, box="black@0.55"), w / f"o{k}.mp4"))
    return fs

def f_compilation(p, w, c, log): return numbered(p, w, c, log, lambda k, n: f"#{n - k}", lambda k, n: f"#{n - k}", lambda k, n: YEL)

def f_escalation(p, w, c, log):
    word = txt(p.get("word", "ATTEMPT")); heat = lambda k, n: ["0xFFFFFF", "0xFFD400", "0xFF9500", "0xFF3B30"][min(3, round(3 * k / max(n - 1, 1)))]
    return numbered(p, w, c, log, lambda k, n: f"{word} {k + 1}", lambda k, n: f"{word} {k + 1}/{n}", heat)

def f_thenvsnow(p, w, c, log):
    lab = p.get("labels", ["THEN", "NOW"])
    return numbered(p, w, c, log, lambda k, n: lab[min(k, 1)], lambda k, n: lab[min(k, 1)], lambda k, n: YEL if k == 0 else RED)

def f_reveal(p, w, c, log):
    """The face is blurred and dark for `reveal` s (the game panel below is clear), then it clears with a flash. Needs the stack layout (cam on top)."""
    R = float(p.get("reveal", 1.6)); f, l = part_mp4(p, p, c, p["hook"], layout="stack"); log.append(l)
    tag = dt(p.get("tag", "WATCH HIS FACE"), 90, y=BADGE_Y, box="0xE11D48@0.9", en=f"lt(t,{R})")
    fl = f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.45:t=fill:enable='between(t,{R},{R + 0.08})'"
    try:
        from clip import has_face; face = has_face(f, 0)
    except Exception: face = True  # detector unavailable: keep the old behaviour
    if not face:
        log.append("reveal: no facecam found, rendered as a plain moment (nothing blurred)"); return [f]
    h = 620  # blur stops above the hook (y ~ 680)
    vf = f"split[a][b];[a]crop=1080:{h}:0:0,boxblur=45:3,eq=brightness=-0.35[bl];[b][bl]overlay=0:0:enable='lt(t,{R})',{tag},{fl}"
    return [overlay(f, vf, w / "reveal.mp4")]

def fetch_chat(vid, a, b):
    """Twitch VOD chat for [a, b): [(sec, user, colour, text)]. Same public GQL as chat.py, which drops the names and colours."""
    import urllib.request; from chat import HASH, CLIENT_ID
    out, seen, off = [], set(), int(a)
    while off < b:
        q = [{"operationName": "VideoCommentsByOffsetOrCursor", "variables": {"videoID": vid, "contentOffsetSeconds": off}, "extensions": {"persistedQuery": {"version": 1, "sha256Hash": HASH}}}]
        req = urllib.request.Request("https://gql.twitch.tv/gql", json.dumps(q).encode(), {"Client-ID": CLIENT_ID, "Content-Type": "application/json"})
        c = json.load(urllib.request.urlopen(req, timeout=30))[0]["data"]["video"]["comments"]; new, last = 0, off
        for e in c["edges"]:
            n = e["node"]; last = max(last, n["contentOffsetSeconds"])
            if n["id"] in seen or not a <= n["contentOffsetSeconds"] < b: continue
            seen.add(n["id"]); new += 1
            out.append((n["contentOffsetSeconds"], (n.get("commenter") or {}).get("displayName") or "", n["message"].get("userColor") or "#9146FF", "".join(f["text"] or "" for f in n["message"]["fragments"]).strip()))
        off = last if new else off + 1
    return out

def chat_rows(p, t0, d, n=5):
    """[(show_time_in_clip, user, colour, text)]: the pick's own "chat" strings, else real VOD chat from the payoff on (real spacing kept), else error."""
    if p.get("chat"): return [(t0 + 0.2 + i * 0.55, "", "#FFFFFF", txt(x)) for i, x in enumerate(p["chat"][:n])]
    from clip import clean
    m = re.search(r"videos/(\d+)", p["url"])
    if not m: raise RuntimeError('chat needs a Twitch VOD link, or give your own lines: "chat": ["KEKW", ...]')
    v0 = float(p["start"]) + t0; good, seen = [], set()
    for s, u, col, t in fetch_chat(m[1], v0 - 1, v0 + 9):
        t = " ".join(clean(x) for x in re.sub(r"[^A-Za-z0-9 ?!.*'-]", "", t).split())
        if 2 <= len(t) <= 24 and not t.startswith("!") and t.lower() not in seen and re.match(r"^[A-Za-z0-9_]{2,16}$", u): seen.add(t.lower()); good.append((s, u, col, t))
    good = sorted(sorted(good, key=lambda g: len(g[3]))[:n])  # the shortest read best; back in time order
    rows, last = [], t0
    for s, u, col, t in good:
        at = max(t0 + 0.2 + (s - v0) * 0.5, last + 0.5); last = at  # real order and rhythm, squeezed so five fit in ~4 s
        if at < d - 1.2: rows.append((at, u, col, t))
    return rows

def row_png(user, col, text, out, size=56, maxw=980):
    """One Twitch-style row: dark rounded bar (70 %), coloured name, white message."""
    from PIL import Image, ImageDraw, ImageFont
    while True:
        f = ImageFont.truetype(str(FONTS / "Poppins-Bold.ttf"), size); u = (user[:12] + ": ") if user else ""
        wu, wt = f.getlength(u), f.getlength(text); W = int(wu + wt + 56)
        if W <= maxw or size <= 44: break
        size -= 4
    H = int(size * 1.0 + 40); im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); g = ImageDraw.Draw(im)
    g.rounded_rectangle((0, 0, W - 1, H - 1), radius=H // 3, fill=(14, 14, 20, 178)); y = (H - f.size) // 2 - 6
    g.text((28, y), u, font=f, fill=col); g.text((28 + wu, y), text, font=f, fill="#FFFFFF"); im.save(out); return W, H

def f_chat(p, w, c, log):
    f, l = part_mp4(p, p, c, p["hook"]); log.append(l); d = P.dur(f)
    t0 = float(p.get("payoff") or peak_time(f)); rows = chat_rows(p, t0, d); log.append(f"chat {len(rows)} rows from {t0:.1f}s: " + "; ".join(f"{u}: {t}" for _, u, _, t in rows))
    if not rows: raise RuntimeError("no usable chat lines in this span")
    ins, fc, last, LIFE, X, YB, ROW = [], [], "[0:v]", 2.2, 40, 1440, 112  # chat scrolls up: a row moves up one slot for every newer row
    for i, (a, u, col, t) in enumerate(rows):
        png = w / f"row{i}.png"; row_png(u, col, t, png); ins += ["-loop", 1, "-framerate", 30, "-t", f"{min(LIFE, d - a):.2f}", "-i", png.resolve()]
        newer = "+".join(f"gt(t,{b:.2f})" for b, *_ in rows[i + 1:]) or "0"
        pop = [(0.85, a, a + 0.05), (0.93, a + 0.05, a + 0.1), (1.0, a + 0.1, min(d, a + LIFE))]  # small scale-up in three steps (a per-frame scale filter was 10x slower)
        fc.append(f"[{i + 1}:v]format=rgba,setpts=PTS+{a:.2f}/TB,split=3[q{i}0][q{i}1][q{i}2]")
        for k, (sc, t1, t2) in enumerate(pop):
            fc.append(f"[q{i}{k}]scale=iw*{sc}:-1[z{i}{k}]"); fc.append(f"{last}[z{i}{k}]overlay=x={X}:y='{YB}-overlay_h/2-({newer})*{ROW}':enable='between(t,{t1:.2f},{t2:.2f})':eof_action=pass[o{i}{k}]"); last = f"[o{i}{k}]"
    ffm("-i", Path(f).resolve(), *ins, "-filter_complex", ";".join(fc), "-map", last, "-map", "0:a", "-t", f"{d:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-c:a", "copy", (w / "chat.mp4").resolve(), cwd=FONTS)
    return [w / "chat.mp4"]

def f_waitforit(p, w, c, log):
    f, l = part_mp4(p | {"squeeze": p.get("squeeze", 1.5)}, p, c, p["hook"]); log.append(l)  # the wait is squeezed: talk-free stretches > 1.5 s are jump-cut (set "squeeze": 0 to keep them)
    t0 = float(p.get("payoff") or peak_time(f)); en = f"lt(t,{t0:.2f})"; log.append(f"payoff at {t0:.1f}s" + ("  WARN: payoff in the first 2.5 s, nothing to wait for; use format moment" if t0 < 2.5 else ""))
    vf = ",".join([dt(p.get("tag", "WAIT FOR IT"), 100, y=BADGE_Y, box="0xE11D48@0.9", en=en),
                   f"drawbox=x=0:y=170:w=iw:h=18:color=black@0.5:t=fill:enable='{en}'",
                   f"drawbox=x=0:y=170:w='iw*min(t/{t0:.2f},1)':h=18:color=0xFFD400:t=fill:enable='{en}':replace=0",
                   f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.45:t=fill:enable='between(t,{t0:.2f},{t0 + 0.08:.2f})'"])
    return [overlay(f, vf, w / "wait.mp4")]

FORMATS = dict(moment=f_moment, callback=f_callback, compilation=f_compilation, escalation=f_escalation, thenvsnow=f_thenvsnow, reveal=f_reveal, chat=f_chat, waitforit=f_waitforit)
NEEDS_PARTS = {"callback": 2, "compilation": 3, "escalation": 3, "thenvsnow": 2}

def fmt_of(p): return p.get("format") or ("callback" if p.get("parts") else "moment")

def finish(mp4, p, last=None):  # last: the last part mp4 of THIS render (caption y + stacked)
    """progress / endcard / aspects (pick > env > brand.json, like clip.py) applied ONCE to the joined Short, reusing clip.py's progress_bar, ass (end card) and ASPECTS.
    Extra 4:5 / 1:1 files are crops of the joined 9:16 (shortcut: captions are not re-laid out per window as in clip.py, so a caption near the window edge can be cut)."""
    import clip as C
    brand = C.load_brand(mp4); g = lambda k, d="": C.opt(k, p, brand, d)
    bar, ec = C.truthy(g("progress")), str(g("endcard") or "")
    asp = g("aspects", []); asp = asp if isinstance(asp, list) else str(asp).replace(" ", "").split(",")
    asp = [a for a in asp if a in C.ASPECTS]
    if not (bar or ec or asp): return
    d = P.dur(mp4); pcol = str(brand.get("progress_color", "FFE500")).lstrip("#"); w = mp4.parent / ".parts" / (mp4.stem + "_fin"); w.mkdir(parents=True, exist_ok=True)
    try: rj = json.load(open(last.with_suffix(".render.json"), encoding="utf-8")); cy = rj.get("caption_y"); stk = rj.get("stacked", False)
    except Exception: cy, stk = None, False  # no render.json (old cache): the old default y
    br = [("", 1920, 0)] + [("_" + a.replace(":", "x"), C.ASPECTS[a][1], 0 if stk else C.ASPECTS[a][2]) for a in asp]; n = len(br)
    shutil.copy(C.FONT_FILE, w); fg = f"[0:v]split={n}" + "".join(f"[s{k}]" for k in range(n)) + ";"; outs = []
    for k, (suf, H, y0) in enumerate(br):
        cur = f"s{k}"
        if H != 1920: fg += f"[{cur}]crop=1080:{H}:0:{y0}[x{k}];"; cur = f"x{k}"
        if ec:
            (w / f"e{k}.ass").write_text(C.ass([], "", d, (), "classic", "auto", H, y0, ec, cy=cy if stk else None), encoding="utf-8"); fg += f"[{cur}]ass=e{k}.ass:fontsdir=.[e{k}];"; cur = f"e{k}"
        if bar:
            s, o = C.progress_bar(d, pcol); fg += f"{s}[bar{k}];[{cur}]drawbox=x=0:y=ih-12:w=iw:h=12:color=black@0.35:t=fill[tr{k}];[tr{k}][bar{k}]{o}[o{k}];"; cur = f"o{k}"
        fg += f"[{cur}]null[f{k}];"; outs += ["-map", f"[f{k}]", "-map", "0:a", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "copy", (mp4.with_name(mp4.stem + suf + "_fin.mp4")).resolve()]
    try: ffm("-i", mp4.resolve(), "-filter_complex", fg.rstrip(";"), *outs, cwd=w)
    finally: shutil.rmtree(w, ignore_errors=True)
    for suf, *_ in br: os.replace(mp4.with_name(mp4.stem + suf + "_fin.mp4"), mp4.with_name(mp4.stem + suf + ".mp4"))

def render(p, mp4):
    """Render one pick of any format. Returns an object with returncode/stdout/stderr like subprocess, so batch.py treats it the same."""
    mp4 = Path(mp4); fmt = fmt_of(p); log = [f"format {fmt}"]
    try:
        if fmt not in FORMATS: raise RuntimeError(f"unknown format {fmt!r}; formats: {', '.join(FORMATS)}")
        if len(p.get("parts", [])) < NEEDS_PARTS.get(fmt, 0): raise RuntimeError(f"{fmt} needs at least {NEEDS_PARTS[fmt]} parts")
        w = mp4.parent / ".parts" / mp4.stem; w.mkdir(parents=True, exist_ok=True)
        _LAST.pop(threading.get_ident(), None)  # a format that renders no part must not see the previous pick's render.json
        try:
            fs = FORMATS[fmt](p, w, mp4.parent / ".partcache", log)
            P.join(fs, mp4); finish(mp4, p, _LAST.pop(threading.get_ident(), None)); total = P.dur(mp4); log.append(f"ok {mp4} {total:.1f}s" + ("" if 8 <= total <= 40 else "  WARN: outside 8-40 s"))
        finally: shutil.rmtree(w, ignore_errors=True)
        return SimpleNamespace(returncode=0, stdout="\n".join(log), stderr="")
    except Exception as e: return SimpleNamespace(returncode=1, stdout="\n".join(log), stderr=f"{fmt} failed: {e}")

if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        t = Path(tempfile.mkdtemp()); src = t / "s.mp4"
        ffm("-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=30:d=6", "-f", "lavfi", "-i", "sine=f=440:d=6:r=48000", "-shortest", "-c:v", "libx264", "-c:a", "aac", src)
        assert txt("Kai's #3, 2/3!") == "KAIS #3 2/3!" and txt("Café Ñandú مرحبا") == "CAFÉ ÑANDÚ مرحبا"
        d = t / "pc"; d.mkdir(); (d / "old.mp4").write_bytes(b"x"); os.utime(d / "old.mp4", (1, 1)); (d / "new.mp4").write_bytes(b"x"); _sweep(d); assert not (d / "old.mp4").exists() and (d / "new.mp4").exists()
        ca = card(frame(src, t / "f2.png"), [("Café مرحبا", 120, YEL)], t / "ca.mp4"); assert abs(P.dur(ca) - 0.6) < 0.15
        fk = t / "fake"; fk.mkdir(); (fk / "clip.py").write_text("import sys,time,shutil; time.sleep(1); shutil.copy(%r, sys.argv[6])" % str(src)); H0 = globals()["HERE"]; globals()["HERE"] = fk  # 4 threads, one key: clip.py runs once, the file is whole
        try:
            from concurrent.futures import ThreadPoolExecutor
            pp = {"url": "u", "transcript": "t", "start": 0, "end": 5}; res = list(ThreadPoolExecutor(4).map(lambda _: part_mp4(pp, pp, t / "pc2", "H"), range(4)))
        finally: globals()["HERE"] = H0
        assert [r[1] for r in res].count("cached") == 3 and P.dur(res[0][0]) > 5 and not list((t / "pc2").glob("*_tmp.mp4"))
        c = card(frame(src, t / "f.png"), [("#3", 330, YEL), ("KAI LOST IT", 80, WHITE)], t / "c.mp4"); assert abs(P.dur(c) - 0.6) < 0.15
        o = overlay(src, dt("STRIKE 1/3", 100, x="50", y=BADGE_Y, color=RED, box="black@0.55") + ",drawbox=x=0:y=170:w='iw*min(t/3,1)':h=18:color=0xFFD400:t=fill:enable='lt(t,3)':replace=0", t / "o.mp4"); assert abs(P.dur(o) - 6) < 0.2
        P.join([c, o, c, o], t / "j.mp4"); assert abs(P.dur(t / "j.mp4") - 13.2) < 0.4
        assert fmt_of({"parts": [1]}) == "callback" and fmt_of({"format": "chat"}) == "chat" and fmt_of({}) == "moment" and set(WHEN) == set(FORMATS)
        assert render({"format": "nope"}, t / "x.mp4").returncode == 1 and render({"format": "compilation", "parts": [{}]}, t / "x.mp4").returncode == 1
        j2 = t / "j2.mp4"; shutil.copy(t / "j.mp4", j2); finish(j2, {"progress": 1, "endcard": "FOLLOW", "aspects": ["4:5", "1:1"]})
        wh = lambda f: subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(f)], text=True).strip()
        assert wh(j2) == "1080,1920" and wh(t / "j2_4x5.mp4") == "1080,1350" and wh(t / "j2_1x1.mp4") == "1080,1080" and abs(P.dur(j2) - P.dur(t / "j.mp4")) < 0.2
        print("selftest ok"); shutil.rmtree(t, ignore_errors=True); sys.exit()
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(json.load(open(sys.argv[1], encoding="utf-8")), 1):
        r = render(p, out / f"{i:02d}_{p['name']}.mp4"); print(r.stdout, r.stderr)

"""Game event detection: big centred banners (WASTED, BUSTED, MISSION PASSED...), wanted stars, hard cuts.
usage: vision.py <file|url> [START-END] [--fps 1] [--grabs DIR] [--stars] [--tag clips/src/<id>.txt] [--selftest]
START-END = h:mm:ss-h:mm:ss or seconds (absolute video time; events carry absolute seconds, same base as the transcript).
Output: clips/src/<id>.events.json   Hook: tag_transcript(txt_path, events) adds [GAME X] and bumps [HEAT]."""
import sys, os, re, json, math, subprocess, difflib, tempfile
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
YTDLP = [sys.executable, "-m", "yt_dlp"]  # works on Windows and Mac (no .exe path)

# label -> (phrases, heat weight 0-1). Fuzzy matched, so OCR slips ("BUSTEO") still count.
WORDS = {
    "WASTED": (["WASTED"], .8), "BUSTED": (["BUSTED"], .7),
    "MISSION PASSED": (["MISSION PASSED", "MISSION COMPLETE"], .7), "MISSION FAILED": (["MISSION FAILED"], .7),
    "HEIST PASSED": (["HEIST PASSED", "HEIST COMPLETE"], .8), "HEIST FAILED": (["HEIST FAILED"], .7),
    "VICTORY ROYALE": (["VICTORY ROYALE"], .8), "ELIMINATED": (["ELIMINATED", "YOU ELIMINATED"], .5),
    "GAME OVER": (["GAME OVER"], .6), "YOU DIED": (["YOU DIED"], .7), "VICTORY": (["VICTORY"], .6),
    "DEFEAT": (["DEFEAT"], .5), "CHAMPION": (["CHAMPION", "YOU ARE THE CHAMPION"], .6),
}
W_STARS, W_CUT, W_TEXT = .5, .3, .35   # wanted-star rise, hard scene cut, unknown big centre text (GTA 6 / other games)
_ocr = None; n_ocr = [0]; STARS = '--stars' in sys.argv


def ocr(img):
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    n_ocr[0] += 1; r, _ = _ocr(img, use_cls=False)
    return r or []


def norm(s): return re.sub(r"[^A-Z ]", "", s.upper()).strip()


def match_word(txt):
    """fuzzy: whole string or any same-length token window vs each phrase; returns label or None"""
    t = norm(txt)
    if len(t) < 4: return None
    for lab, (phr, _) in WORDS.items():
        for p in phr:
            n = len(p.split()); tk = t.split()
            cands = [t] + [" ".join(tk[i:i + n]) for i in range(max(1, len(tk) - n + 1))]
            if any(difflib.SequenceMatcher(None, c, p).ratio() >= .8 for c in cands): return lab
    return None


def colour_cues(bgr):
    """cheap cues on the centre band: grey = desaturated frame (WASTED/BUSTED), gold = MISSION PASSED text, red = WASTED text"""
    h, w = bgr.shape[:2]; band = bgr[int(h * .25):int(h * .75)]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV); H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    full = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return {"sat": float(full[..., 1].mean()) / 255,
            "gold": float(((H > 15) & (H < 35) & (S > 110) & (V > 170)).mean()),
            "red": float((((H < 8) | (H > 172)) & (S > 140) & (V > 120)).mean())}


def banners(bgr):
    """OCR the centre band -> [(label_or_None, text, conf, rel_height)]"""
    h, w = bgr.shape[:2]; y0, x0 = int(h * .2), int(w * .08)
    out = []
    for box, txt, sc in ocr(bgr[y0:int(h * .8), x0:int(w * .92)]):
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        rh = (max(ys) - min(ys)) / h; cx = (np.mean(xs) + x0) / w
        out.append((match_word(txt) if float(sc) > .6 else None, txt, float(sc), rh, cx))
    return out


def detect_frame(bgr):
    """-> list of (kind, label, extra) for one frame"""
    ev = []; cc = colour_cues(bgr)
    for lab, txt, sc, rh, cx in banners(bgr):
        if lab: ev.append(("word", lab, {"text": txt, "conf": round(sc, 2), "cues": cc}))
        elif rh >= .06 and sc > .85 and .3 < cx < .7 and len(norm(txt).replace(" ", "")) >= 4 \
                and sum(c.isupper() for c in txt) >= .8 * sum(c.isalpha() for c in txt):
            ev.append(("text", "TEXT " + norm(txt), {"text": txt, "conf": round(sc, 2), "cues": cc}))
    return ev


def stars(bgr):
    """wanted stars: count star-sized bright-yellow blobs in the top-right HUD strip. GTA5 only; heuristic, 0 when HUD unknown."""
    h, w = bgr.shape[:2]; roi = bgr[int(h * .1):int(h * .24), int(w * .72):]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] > 15) & (hsv[..., 0] < 32) & (hsv[..., 1] > 120) & (hsv[..., 2] > 190)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    a = [s for s in st[1:] if .0004 * h * w < s[4] < .004 * h * w and .6 < s[2] / max(s[3], 1) < 1.5]
    return len(a) if len(a) <= 5 else 0


def parse_win(a):
    def t(x):
        p = [float(v) for v in x.split(":")]; s = 0
        for v in p: s = s * 60 + v
        return s
    s, e = a.split("-"); return t(s), t(e)


def fetch(url, win, h=360):
    """section download only the window at <=360p"""
    d = os.path.join(HERE, "clips", "src", "_vision"); os.makedirs(d, exist_ok=True)
    vid = re.sub(r"\W", "", url.rstrip("/").split("/")[-1].split("?")[0])
    out = os.path.join(d, f"v{vid}_{int(win[0])}_{int(win[1])}.mp4") if win else os.path.join(d, f"v{vid}.mp4")
    if not os.path.exists(out):
        cmd = [*YTDLP, "-f", f"best[height<={h}]/worst", "--force-keyframes-at-cuts", "-o", out, url]
        if win: cmd[1:1] = ["--download-sections", "*%d-%d" % win]
        subprocess.run(cmd, check=True)
    return out, vid


def scan(path, t0=0.0, fps=1.0, grabs=None, sweep=4):
    cap = cv2.VideoCapture(path); vf = cap.get(cv2.CAP_PROP_FPS) or 30; step = max(1, round(vf / fps))
    raw = []; prev_h = prev_band = None; last_n = 0; i = 0; sats = []; n_s = 0
    while True:
        if not cap.grab(): break
        if i % step == 0:
            ok, f = cap.retrieve()
            if not ok: break
            f = cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1]))) if f.shape[1] > 640 else f
            t = t0 + i / vf
            if n_s % 60 == 0: print(f'  t={t:.0f} ocr_frames={n_ocr[0]} raw={len(raw)}', file=sys.stderr, flush=True)
            hist = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]); hist = cv2.normalize(hist, hist).flatten()
            if prev_h is not None and cv2.compareHist(prev_h, hist, cv2.HISTCMP_BHATTACHARYYA) > .55: raw.append((t, "cut", "CUT", {}, f))
            prev_h = hist
            band = cv2.resize(f[int(f.shape[0] * .2):int(f.shape[0] * .8)], (80, 40)).astype(np.int16)
            cc = colour_cues(f); sats.append(cc["sat"]); sats[:] = sats[-40:]
            cue = cc["gold"] > .006 or cc["red"] > .004 or cc["sat"] < .6 * float(np.median(sats))  # banner colours / grey-out
            n_s += 1                                                                              # OCR is slow (~1 s/frame loaded): cue frames + one sweep frame per `sweep` samples
            if (cue or n_s % sweep == 0) and (prev_band is None or np.abs(band - prev_band).mean() > 6):
                for k, lab, ex in detect_frame(f): raw.append((t, k, lab, ex, f))
                prev_band = band
            n = stars(f) if STARS else 0   # off by default: test showed 6/7 false hits (stream chat/overlay in the corner)
            if n != last_n and n:
                if n > last_n: raw.append((t, "stars", f"STARS {n}", {"stars": n}, f))
                last_n = n
            elif n == 0 and last_n: last_n = 0
        i += 1
    return raw


def merge(raw, gap=15, grabs=None):
    """same label within `gap` s = one event (first time); a word seen once at conf<.8 is dropped unless a colour cue agrees"""
    ev = []; last = {}
    for t, k, lab, ex, f in sorted(raw, key=lambda r: r[0]):
        if lab in last and t - last[lab]["t_end"] <= gap: last[lab]["t_end"] = t; last[lab]["hits"] += 1; continue
        e = {"t": round(t, 1), "t_end": round(t, 1), "kind": k, "label": lab, "hits": 1, **{a: b for a, b in ex.items()}}
        if k == "word": e["weight"] = WORDS[lab][1]
        else: e["weight"] = {"stars": W_STARS, "cut": W_CUT, "text": W_TEXT}[k]
        if grabs:
            os.makedirs(grabs, exist_ok=True); fn = f"{int(t)}_{re.sub(r'[^A-Za-z0-9]+', '_', lab)}.jpg"
            cv2.imwrite(os.path.join(grabs, fn), f); e["grab"] = fn
        ev.append(e); last[lab] = e
    for e in ev: e["t_end"] = round(e["t_end"], 1)
    return ev


def tag_transcript(txt_path, events, write=True, slack=8):
    """add ` [GAME WASTED]` to the transcript line at each event's time (nearest line start within `slack` s) and bump [HEAT].
    Lines look like `[36001.3-36010.9] text [LOUD +17dB] [HEAT 40]`. Idempotent. Keeps a .bak of the original. Returns new text."""
    import chat
    L = open(txt_path, encoding="utf-8").read().split("\n")
    spans = []
    for i, l in enumerate(L):
        m = re.match(r"\[([\d.]+)-([\d.]+)\]", l)
        if m: spans.append((float(m[1]), float(m[2]), i))
    add = {}
    for e in events:
        if e["kind"] == "cut" or (e["kind"] == "text" and e["hits"] < 3) or (e["kind"] == "stars" and e["stars"] < 3): continue  # keep noisy kinds out of the transcript
        cand = [s for s in spans if s[0] - slack <= e["t"] <= s[1] + slack]
        if not cand: continue
        s = min(cand, key=lambda s: 0 if s[0] <= e["t"] <= s[1] else min(abs(e["t"] - s[0]), abs(e["t"] - s[1])))
        add.setdefault(s[2], []).append(e)
    for i, es in add.items():
        l = L[i]; n0 = l
        for e in es:
            tag = f"[GAME {e['label']}]"
            if tag in l: continue
            l = re.sub(r" \[HEAT \d+\]", "", l) + " " + tag
        if l != n0: l += f" [HEAT {chat.heat(l)}]"  # one formula (chat.heat, noisy-or over every tag on the line): a banner line never ranks below its LOUD-only neighbour
        else: l = L[i]
        L[i] = l
    txt = "\n".join(L)
    if write:
        if not os.path.exists(txt_path + ".bak"): open(txt_path + ".bak", "w", encoding="utf-8").write(open(txt_path, encoding="utf-8").read())
        open(txt_path, "w", encoding="utf-8").write(txt)
    return txt


def run(src, win=None, fps=1.0, grabs=None):
    import time; t = time.time()
    vid = re.sub(r"\W", "", os.path.splitext(os.path.basename(src))[0])
    if src.startswith("http"): src, vid = fetch(src, win)
    raw = scan(src, win[0] if win else 0.0, fps); ev = merge(raw, grabs=grabs)
    out = os.path.join(HERE, "clips", "src", f"{vid}.events.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"src": src, "window": win, "fps": fps, "events": ev, "scan_s": round(time.time() - t, 1)}, open(out, "w", encoding="utf-8"), indent=1)
    print(out, len(ev), "events", round(time.time() - t), "s"); return ev


def selftest():
    def card(txt, col, grey, y=.5):
        f = np.full((360, 640, 3), 90, np.uint8); cv2.randn(f, 90, 20)
        if grey: f = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 2.2, 5)[0]
        cv2.putText(f, txt, ((640 - sz[0]) // 2, int(360 * y)), cv2.FONT_HERSHEY_DUPLEX, 2.2, col, 5); return f
    assert match_word("BUSTEO") == "BUSTED" and match_word("hello world") is None and match_word("MISSION PASSEO!") == "MISSION PASSED"
    for txt, col, grey, want in [("WASTED", (30, 30, 220), True, "WASTED"), ("MISSION PASSED", (40, 190, 230), False, "MISSION PASSED"),
                                 ("BUSTED", (230, 200, 120), True, "BUSTED")]:
        got = [l for k, l, _ in detect_frame(card(txt, col, grey)) if k == "word"]
        assert want in got, (txt, got)
    assert colour_cues(card("X", (40, 190, 230), True))["sat"] < .05
    p = os.path.join(tempfile.mkdtemp(), "t.txt")
    open(p, "w", encoding="utf-8").write("[10.0-14.0] oh no [LOUD +9dB] [HEAT 36]\n[30.0-31.0] fine\n")
    t = tag_transcript(p, [{"t": 12.0, "kind": "word", "label": "WASTED", "weight": .8}])
    assert "[GAME WASTED]" in t.split("\n")[0] and "[HEAT 74]" in t and int(re.search(r"\[HEAT (\d+)", t)[1]) >= __import__("chat").heat("x [LOUD +9dB]") and "GAME" not in t.split("\n")[1], t
    assert tag_transcript(p, [{"t": 12.0, "kind": "word", "label": "WASTED", "weight": .8}]).count("[GAME WASTED]") == 1
    print("vision selftest OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--selftest": selftest(); sys.exit()
    fps = float(a[a.index("--fps") + 1]) if "--fps" in a else 1.0
    grabs = a[a.index("--grabs") + 1] if "--grabs" in a else None
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--fps", "--grabs", "--tag"))]
    ev = run(pos[0], parse_win(pos[1]) if len(pos) > 1 else None, fps, grabs)
    if "--tag" in a: tag_transcript(a[a.index("--tag") + 1], ev); print("tagged", a[a.index("--tag") + 1])

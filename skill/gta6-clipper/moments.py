"""Moment types without AI: label stretches of a stream chase / crash / fight / stunt / reaction / funny (+ banner events).
usage: moments.py <video-or-audio> [START-END] [--txt clips/src/<id>.txt] [--events clips/src/<id>.events.json] [--cam x0,y0,x1,y1 (640x360 px)]
                  [--sheet DIR] [--out file.json] [--selftest]
START-END (h:mm:ss or seconds) only gives the absolute time of the file's first second, same base as the transcript (as vision.py).
Cheap: ffmpeg pipes 2 fps 160 px gray frames + 16 kHz mono audio in chunks (never loads the file); frame diff + Farneback flow
(motion, radial outflow = speed, rotation = stunt) and audio bands (low 60-400 Hz engine, voice 300-3 kHz, high 3-8 kHz), 20 Hz envelope.
Output: <src>.moments.json (or --out) =
  {"src","t0","dur","has_video","cam":[x0,y0,x1,y1]|null (160x90 units),"scan_s","x_realtime",
   "moments":[{"t0","t1","type","score"[,"label"]}]}   absolute seconds, score 0-1, sorted by t0, types may overlap.
  type in chase|crash|fight|stunt|reaction|funny|banner (banner = vision.py WASTED/BUSTED..., label set).
Hook: tag_transcript(txt_path, moments) adds ` [MOMENT chase 0.8]` to the line (before the trailing [HEAT n]), score>=.55, max 2 per line. Idempotent."""
import sys, os, re, json, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
FPS, W, H = 2, 160, 90          # video sampling
AR = 16000; FR = 20             # audio rate, envelope frames/s
MIN_LEN = {"chase": 5, "crash": 1.0, "fight": 4, "stunt": 3, "reaction": 2, "funny": 2}   # seconds
THR = {"chase": .5, "crash": .7, "fight": .6, "stunt": .7, "reaction": .55, "funny": .5}
HYPE = re.compile(r"\b(oh my (god|gosh)|no way|what the|holy|bruh|let'?s go+|are you kidding|crazy|insane|dude+|bro+)\b|!", re.I)
LAUGH = re.compile(r"\b(ha(ha)+|haha\w*|lol|lmao|lmfao|laugh\w*|dying|dead|cry(ing)?)\b", re.I)
CHATLOL = re.compile(r"LUL|KEKW|OMEGALUL|LMAO|ICANT", re.I)


def pct(x, lo=50, hi=98):
    """scale-free 0..1: median -> 0, 98th percentile -> 1 (a quiet stream and a loud one rank alike)"""
    x = np.asarray(x, float); a, b = np.percentile(x, [lo, hi]); return np.clip((x - a) / (b - a + 1e-6), 0, 1)


def sm(x, n):   # centred moving average, n bins
    x = np.asarray(x, float); n = max(1, n); return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), np.ones(n) / n, "valid")


def prev_med(x, n=6):  # median of the n bins BEFORE each bin (causal baseline for spikes)
    x = np.asarray(x, float); o = np.zeros_like(x)
    for i in range(len(x)): o[i] = np.median(x[max(0, i - n):i]) if i else x[0]
    return o


def audio_feats(pcm):
    """pcm float32 16 kHz -> dict of per-20Hz-frame dB arrays low/voice/high/all"""
    k = AR // FR; n = len(pcm) // k; f = pcm[: n * k].reshape(n, k) * np.hanning(k)
    S = np.abs(np.fft.rfft(f, axis=1)) ** 2; fr = np.fft.rfftfreq(k, 1 / AR)
    band = lambda a, b: 10 * np.log10(S[:, (fr >= a) & (fr < b)].sum(1) + 1e-9)
    return {"low": band(60, 400), "voice": band(300, 3000), "high": band(3000, 8000), "all": band(60, 8000)}


def laugh_mod(env, win=60):
    """periodic 3.5-8 Hz amplitude modulation of the voice envelope (laughter 'ha-ha-ha'): max normalised autocorr at lag 2-5 frames of
    a detrended 3 s window. shortcut: steady fast speech rings a little too, so text/chat/loudness must agree (audio-only funny capped .75)."""
    e = np.asarray(env, float); out = np.zeros(len(e)); d = e - sm(e, 20); st = FR // FPS
    for i in range(0, len(e), st):
        w = d[max(0, i - win // 2): i + win // 2]
        if len(w) < 20: continue
        v = (w * w).sum() + 1e-9; out[i: i + st] = max((w[:-l] * w[l:]).sum() / v for l in range(2, 6))
    return out


def to_bins(a, n, fn=np.mean):
    k = FR // FPS; m = len(a) // k; b = fn(np.asarray(a[: m * k]).reshape(m, k), axis=1)
    return np.pad(b, (0, max(0, n - len(b))), mode="edge")[:n]


def score(F, has_video=True, has_cam=False):
    """F = per-bin (2/s) features: m motion, out radial outflow, rot rotation, cam cam-tile motion, low/voice/high/all (dB), mod laughter periodicity,
    txt_laugh, txt_hype, chat_lol (0/1) -> {type: score array 0-1}"""
    n = len(F["voice"]); z = np.zeros(n); k = lambda s: int(s * FPS)
    L, V, A = pct(sm(F["low"], k(6))), pct(F["voice"]), pct(F["all"])
    tr = lambda x: pct(np.maximum(0, x - prev_med(x)), 75, 99)           # positive jump above the last 3 s
    Ta, Tv, flux = tr(F["all"]), tr(F["voice"]), tr(F["high"])
    dens = np.minimum(1, sm((Ta > .45).astype(float), k(8)) * k(8) / 4)   # >=4 transients in 8 s
    if has_video:
        m = pct(F["m"]); ms = pct(sm(F["m"], k(6))); out = pct(sm(F["out"], k(6)), 50, 97); rot = pct(sm(np.abs(F["rot"]), k(4)), 50, 97)
        mj = pct(np.maximum(0, F["m"] - prev_med(F["m"])), 75, 99)
    else:   # audio proxies, capped .7: motion unknown
        m = ms = A * .8; out = L; rot = z; mj = Tv
    cap = 1 if has_video else .7
    S = {}
    S["chase"] = cap * np.minimum(sm(np.minimum(ms, np.maximum(out, L)), k(2)), sm(np.maximum(L, A * .8), k(4)))
    S["crash"] = np.where(Ta > .35, cap * np.minimum(np.maximum(mj, Ta * .8), np.maximum(Ta, flux)) * np.minimum(1, .4 + Ta), 0)
    S["fight"] = cap * np.where(dens > .3, (dens + sm(m, k(3)) + sm(V, k(3))) / 3 * 1.25, 0)
    S["stunt"] = np.minimum(rot, np.maximum(ms, .5)) * np.where(L < .85, 1, .8) if has_video else z
    camm = pct(sm(F["cam"], k(2))) if (has_video and has_cam) else sm(Tv, k(2))
    r = np.minimum(1, (camm * .45 + sm(V, k(2)) * .55) * 1.2) * (.5 + .5 * np.maximum(Tv, V)) * (V > .5)
    S["reaction"] = np.minimum(1, r + .2 * sm(F["txt_hype"], k(3)).clip(0, 1) * (V > .4))
    mod = np.clip((sm(F["mod"], k(2)) - .35) / .35, 0, 1) * (V > .4)       # periodic voice-band bursts
    txt = np.minimum(1, sm(F["txt_laugh"], k(4)) * 2.8 + sm(F["chat_lol"], k(4)) * 2)
    f = np.minimum(1, mod * .55 + txt * .6 + np.minimum(mod, txt) * .3)
    S["funny"] = np.where((mod > .3) | (txt > .3), f, 0) * np.where(txt > .3, 1, .75)
    return {t: np.clip(np.nan_to_num(s), 0, 1) for t, s in S.items()}


def segments(S, t0=0.0, gap=3):
    """score arrays -> [{t0,t1,type,score}] (threshold, merge gaps <=3 s, drop under the type's minimum length; score = 90th pct inside)"""
    out = []
    for t, s in S.items():
        on = s >= THR[t]; i = 0; n = len(s)
        while i < n:
            if not on[i]: i += 1; continue
            j = i
            while j < n and (on[j] or on[j + 1: j + 1 + gap * FPS].any()): j += 1
            a, b = i / FPS, j / FPS
            if b - a >= MIN_LEN[t]: out.append({"t0": round(t0 + a, 1), "t1": round(t0 + b, 1), "type": t, "score": round(float(np.percentile(s[i:j], 90)), 2)})
            i = j + 1
    return sorted(out, key=lambda r: (r["t0"], r["type"]))


def parse_win(a):
    s = 0
    for v in a.split("-")[0].split(":"): s = s * 60 + float(v)
    return s


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json", path], capture_output=True, text=True)
    j = json.loads(r.stdout or "{}"); st = j.get("streams", [])
    v = [s for s in st if s["codec_type"] == "video" and s.get("height", 0) > 16]
    return float(j.get("format", {}).get("duration", 0)), (v[0] if v else None), any(s["codec_type"] == "audio" for s in st)


def read_audio(path, chunk=60):
    """stream 16 kHz mono in 60 s chunks -> concatenated per-frame band dB (small); the pcm itself is never held whole"""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(AR), "-f", "s16le", "-"], stdout=subprocess.PIPE)
    acc = {}; step = chunk * AR * 2
    while True:
        b = p.stdout.read(step)
        if not b: break
        a = np.frombuffer(b[: len(b) // 2 * 2], np.int16).astype(np.float32) / 32768
        for k, v in audio_feats(a).items(): acc.setdefault(k, []).append(v)
    p.wait(); return {k: np.concatenate(v) for k, v in acc.items()} if acc else None


def find_cam(path, dur):
    """face cam box: YuNet face in a corner on ~12 frames at 640x360 -> (x0,y0,x1,y1) in 160x90 units, or None (= no cam, reaction from audio)"""
    import cv2
    yn = cv2.FaceDetectorYN.create("onnx", np.frombuffer(open(os.path.join(HERE, "models", "yunet.onnx"), "rb").read(), np.uint8), np.empty(0, np.uint8), (640, 360), 0.6)
    cap = cv2.VideoCapture(path); hits = []
    for k in range(12):
        cap.set(cv2.CAP_PROP_POS_MSEC, (k + .5) / 12 * dur * 1000); ok, f = cap.read()
        if not ok: continue
        _, fs = yn.detect(cv2.resize(f, (640, 360)))
        for (x, y, w, h) in ([] if fs is None else fs[:, :4]):
            cx, cy = x + w / 2, y + h / 2
            if (cx < 160 or cx > 480) and (cy < 90 or cy > 270): hits.append((cx, cy, w))
    if len(hits) < 4: return None
    cx, cy, w = np.median(np.array(hits), axis=0); r = w * 1.6
    return tuple(int(v / 4) for v in (max(0, cx - r), max(0, cy - r), min(640, cx + r), min(360, cy + r)))


def read_video(path, cam=None):
    """2 fps 160x90 gray via ffmpeg pipe -> motion m (game area), radial outflow, rotation, cam-tile motion per bin"""
    import cv2
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", f"fps={FPS},scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); xc, yc = xx - W / 2, yy - H / 2; r = np.sqrt(xc ** 2 + yc ** 2) + 1
    mask = np.ones((H, W), bool)
    if cam: mask[cam[1]:cam[3], cam[0]:cam[2]] = False
    R = {k: [] for k in ("m", "out", "rot", "cam")}; prev = None; sz = W * H
    while True:
        b = p.stdout.read(sz)
        if len(b) < sz: break
        f = np.frombuffer(b, np.uint8).reshape(H, W)
        if prev is None: v = (0, 0, 0, 0)
        else:
            fl = cv2.calcOpticalFlowFarneback(prev, f, None, .5, 2, 9, 2, 5, 1.1, 0); vx, vy = fl[..., 0], fl[..., 1]
            cm = float(np.abs(f[cam[1]:cam[3], cam[0]:cam[2]].astype(np.int16) - prev[cam[1]:cam[3], cam[0]:cam[2]]).mean()) if cam else 0
            v = (float(np.hypot(vx, vy)[mask].mean()), float(((xc * vx + yc * vy) / r)[mask].mean()), float(((xc * vy - yc * vx) / r)[mask].mean()), cm)
        for k, x in zip(R, v): R[k].append(x)
        prev = f
    p.wait(); return {k: np.array(v) for k, v in R.items()}


def text_feats(txt_path, n, t0):
    """per-bin 0/1 arrays: laugh words, hype words, chat LUL/KEKW (from [CHAT ...] tags in the transcript lines)"""
    a = {k: np.zeros(n) for k in ("txt_laugh", "txt_hype", "chat_lol")}
    if not txt_path or not os.path.exists(txt_path): return a
    for l in open(txt_path, encoding="utf-8").read().split("\n"):
        m = re.match(r"\[([\d.]+)-([\d.]+)\]\s*(.*)", l)
        if not m: continue
        i, j = max(0, int((float(m[1]) - t0) * FPS)), min(n, int((float(m[2]) - t0) * FPS) + 1)
        if j <= i: continue
        body = m[3]; plain = re.sub(r"\[[^\]]*\]", "", body)
        if LAUGH.search(plain): a["txt_laugh"][i:j] = 1
        if HYPE.search(plain): a["txt_hype"][i:j] = 1
        c = re.search(r"\[CHAT[^\]]*\]", body)
        if c and CHATLOL.search(c[0]): a["chat_lol"][i:j] = 1
    return a


def banner_moments(events_path):
    if not events_path or not os.path.exists(events_path): return []
    ev = json.load(open(events_path, encoding="utf-8")).get("events", [])
    return [{"t0": e["t"], "t1": max(e["t_end"], e["t"] + 3), "type": "banner", "label": e["label"], "score": round(e["weight"], 2)}
            for e in ev if e["kind"] == "word" or (e["kind"] == "text" and e.get("hits", 0) >= 3)]


def tag_transcript(txt_path, moments, write=True, min_score=.55, slack=2):
    """append ` [MOMENT chase 0.8]` (best 2 types, score>=.55) to every transcript line that overlaps the moment; before the final [HEAT n]. Idempotent, .bak once."""
    L = open(txt_path, encoding="utf-8").read().split("\n"); spans = []
    for i, l in enumerate(L):
        m = re.match(r"\[([\d.]+)-([\d.]+)\]", l)
        if m: spans.append((float(m[1]), float(m[2]), i))
    add = {}
    for mo in moments:
        if mo["type"] == "banner" or mo["score"] < min_score: continue
        for s in spans:
            if s[1] + slack >= mo["t0"] and s[0] - slack <= mo["t1"]:
                d = add.setdefault(s[2], {}); d[mo["type"]] = max(mo["score"], d.get(mo["type"], 0))
    for i, d in add.items():
        l = L[i]; h = re.search(r" \[HEAT \d+\]\s*$", l); tail = h[0] if h else ""; base = l[: len(l) - len(tail)]
        for t, sc in sorted(d.items(), key=lambda kv: -kv[1])[:2]:
            if f"[MOMENT {t} " not in base: base += f" [MOMENT {t} {sc:.1f}]"
        L[i] = base + tail
    txt = "\n".join(L)
    if write:
        if not os.path.exists(txt_path + ".bak"): open(txt_path + ".bak", "w", encoding="utf-8").write(open(txt_path, encoding="utf-8").read())
        open(txt_path, "w", encoding="utf-8").write(txt)
    return txt


def sheet(path, moments, outdir, t0=0.0):
    """contact sheet: best moment per type, 4 frames each (start..end), labelled"""
    import cv2
    os.makedirs(outdir, exist_ok=True); cap = cv2.VideoCapture(path); rows = []
    for t in ("chase", "crash", "fight", "stunt", "reaction", "funny", "banner"):
        c = [m for m in moments if m["type"] == t]
        if not c: continue
        m = max(c, key=lambda m: m["score"]); fr = []
        for q in np.linspace(m["t0"], m["t1"], 4):
            cap.set(cv2.CAP_PROP_POS_MSEC, (q - t0) * 1000); ok, f = cap.read()
            fr.append(cv2.resize(f, (320, 180)) if ok else np.zeros((180, 320, 3), np.uint8))
        row = np.hstack(fr); cv2.rectangle(row, (0, 0), (520, 22), (0, 0, 0), -1)
        cv2.putText(row, f"{t} {m['score']} {m['t0']:.0f}-{m['t1']:.0f}s" + (f" {m.get('label', '')}" if t == "banner" else ""), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        rows.append(row)
    if rows: cv2.imwrite(os.path.join(outdir, "moments_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(rows)


def run(src, t0=0.0, txt=None, events=None, cam=None, sheet_dir=None, out=None):
    t = time.time(); dur, v, has_a = probe(src); has_v = v is not None
    if not dur or not has_a: raise SystemExit("need a readable file with audio: " + src)
    A = read_audio(src); n = int(dur * FPS); cam_box = None
    if has_v:
        cam_box = tuple(int(int(x) / 4) for x in cam.split(",")) if cam else find_cam(src, dur)
        Vd = read_video(src, cam_box)
    F = {k: to_bins(a, n) for k, a in A.items()}; F["mod"] = to_bins(laugh_mod(10 ** (A["voice"] / 20)), n, np.max)
    for k in ("m", "out", "rot", "cam"): F[k] = np.pad(Vd[k], (0, max(0, n - len(Vd[k]))), mode="edge")[:n] if has_v else np.zeros(n)
    F.update(text_feats(txt, n, t0))
    mo = segments(score(F, has_v, bool(cam_box)), t0) + banner_moments(events)
    mo.sort(key=lambda r: (r["t0"], r["type"])); sc = time.time() - t
    o = out or os.path.splitext(src)[0] + ".moments.json"
    json.dump({"src": src, "t0": t0, "dur": round(dur, 1), "has_video": has_v, "cam": list(cam_box) if cam_box else None, "scan_s": round(sc, 1),
               "x_realtime": round(dur / max(sc, .01), 1), "moments": mo}, open(o, "w", encoding="utf-8"), indent=1)
    if txt and os.path.exists(txt): tag_transcript(txt, mo)
    if sheet_dir and has_v: sheet(src, mo, sheet_dir, t0)
    print(o, len(mo), "moments", round(sc), "s", round(dur / max(sc, .01)), "x realtime"); return mo


def selftest():
    rng = np.random.default_rng(1); n = 600   # 5 min of bins
    t = np.arange(AR * 20) / AR; noise = rng.normal(0, .02, len(t)).astype(np.float32)
    steady = noise + .2 * np.sin(2 * np.pi * 800 * t)
    laugh = noise + .4 * np.sin(2 * np.pi * 800 * t) * (np.sin(2 * np.pi * 5.5 * t) > 0)
    mod = lambda x: laugh_mod(10 ** (audio_feats(x)["voice"] / 20))[100:300].max()
    assert mod(laugh) > .6 > mod(steady) + .1, (mod(laugh), mod(steady))
    F = {k: np.abs(rng.normal(0, .1, n)) for k in ("m", "out", "rot", "cam", "mod")}
    for k in ("low", "voice", "high", "all"): F[k] = -50 + rng.normal(0, .4, n)
    F["mod"] *= .3
    for k in ("txt_laugh", "txt_hype", "chat_lol"): F[k] = np.zeros(n)
    def put(a, b, **kw):
        for k, v in kw.items(): F[k][a:b] += v
    put(100, 160, m=1.5, out=1.0, low=14, all=10)                       # chase: 30 s sustained motion + speed + engine
    put(250, 253, m=3.5, all=22, high=18, voice=10)                     # crash: spike + bang
    for a in range(330, 370, 4): put(a, a + 1, all=18, high=14, voice=14, m=2)   # fight: repeated hits
    put(330, 370, m=1.2)
    put(440, 456, rot=3.0, m=2.0)                                       # stunt: rotation
    put(520, 540, voice=14, all=8, cam=3.0)                             # reaction
    put(570, 590, voice=8, mod=.9, txt_laugh=1, chat_lol=1)             # funny
    ms = segments(score(F, True, True))
    for ty, a, b in [("chase", 100, 160), ("crash", 250, 253), ("fight", 330, 370), ("stunt", 440, 456), ("reaction", 520, 540), ("funny", 570, 590)]:
        assert any(m["type"] == ty and m["t0"] < b / 2 and m["t1"] > a / 2 for m in ms), (ty, ms)
    assert not any(m["t1"] < 45 for m in ms), "false hit in the quiet start: " + str([m for m in ms if m["t1"] < 45])
    mo2 = segments(score(F, False, False)); assert any(m["type"] == "crash" for m in mo2) and not any(m["type"] == "stunt" for m in mo2)
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "a.txt")
    open(p, "w", encoding="utf-8").write("[40.0-45.0] go [LOUD +9dB] [HEAT 30]\n[300.0-305.0] hello\n")
    mm = [{"t0": 41, "t1": 50, "type": "chase", "score": .8}, {"t0": 41, "t1": 50, "type": "funny", "score": .3}]
    x = tag_transcript(p, mm, write=False)
    assert x.split("\n")[0].endswith("[MOMENT chase 0.8] [HEAT 30]") and "funny" not in x
    open(p, "w", encoding="utf-8").write(x); assert tag_transcript(p, mm, write=False) == x   # idempotent
    print("moments selftest ok", len(ms), "moments")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a: selftest(); sys.exit()
    g = lambda f: a[a.index(f) + 1] if f in a else None
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
    run(pos[0], parse_win(pos[1]) if len(pos) > 1 else 0.0, g("--txt"), g("--events"), g("--cam"), g("--sheet"), g("--out"))

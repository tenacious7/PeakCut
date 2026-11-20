# Free auto-reframe 16:9 -> 9:16 that follows the subject (people/faces), per shot.
# Usage: python reframe.py in.mp4 out.mp4 [--debug plan.json]
# How: one pass finds cuts + dissolves (colour histograms) -> MediaPipe finds people/faces on sampled frames ->
# smoothed crop centre per frame -> shots whose subjects are too spread out get a blurred-fill frame.
# Several people far apart (couch podcast): follow whoever talks (mouth motion per face x speech loudness), hard cut on change,
# split screen when two trade lines fast; if the signal is unsure the shot stays blurred-fill.
import json, subprocess, sys
from functools import lru_cache
from pathlib import Path
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

M = Path(__file__).resolve().parent / "models"
YUNET = np.frombuffer((M / "yunet.onnx").read_bytes(), np.uint8)  # bytes, not a path: OpenCV cannot open paths with non-ASCII usernames
OW, OH = 1080, 1920
SAMPLE = 4  # detect every Nth frame
TINY_FACE = 0.022  # cam face narrower than this share of the frame width (42 px at 1920) = too small to fill the top panel (measured usable cams 0.03-0.07 W; the Kai trailer-reaction cam 0.018 W)

def detectors():
    obj = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
        base_options=mpt.BaseOptions(model_asset_buffer=(M / "object.tflite").read_bytes()), score_threshold=0.35, category_allowlist=["person", "car", "motorcycle"]))
    face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mpt.BaseOptions(model_asset_buffer=(M / "face.tflite").read_bytes()), min_detection_confidence=0.5))
    return obj, face

def subject_span(frame, obj, face):
    """Return (centre_x, left, right) in pixels of the main subject, or None."""
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes = [(d.bounding_box, 3.0) for d in face.detect(img).detections]  # faces weigh most
    has_face = bool(boxes)
    for d in obj.detect(img).detections:
        person = d.categories[0].category_name == "person"
        if has_face and not person: continue  # a face is on screen: parked cars behind a streamer are not the subject
        boxes.append((d.bounding_box, 1.5 if person else 1.0))
    if not boxes: return None
    # weight by area x type, keep boxes that matter (>= 25% of the biggest weight)
    scored = [(b, wt * b.width * b.height) for b, wt in boxes]
    top = max(s for _, s in scored)
    keep = [b for b, s in scored if s >= 0.25 * top]
    main = max(scored, key=lambda t: t[1])[0]  # follow the main subject, never the gap between two people
    l, r = min(b.origin_x for b in keep), max(b.origin_x + b.width for b in keep)
    fb = [b for b, wt in boxes if wt == 3.0]
    if len(fb) >= 2:  # dialogue shot: span every face, even small ones
        l, r = min(l, min(b.origin_x for b in fb)), max(r, max(b.origin_x + b.width for b in fb))
    return main.origin_x + main.width / 2, l, r, float(len(fb) >= 2), [(b.origin_x, b.origin_y, b.width, b.height) for b, wt in boxes if wt == 3.0]

LM = M / "face_landmarker.task"  # Google's public FaceLandmarker (3.6 MB, blendshapes incl. jawOpen); missing file = speaker mode off
SPEECH_MOTION = 0.01  # mean jawOpen change per sample that counts as a talking mouth (measured, see LOG)
HOLD = 1.5  # a speaker stays on screen at least this long before the cut to the next one

def runs(lab):
    """Consecutive equal labels -> [label, start, end) runs."""
    out = []
    for i, v in enumerate(lab):
        if out and out[-1][0] == v: out[-1][2] = i + 1
        else: out.append([v, i, i + 1])
    return out

def hold(lab, n):
    """Give runs shorter than n samples to their longer neighbour, shortest first, so nobody flashes by."""
    lab = list(lab); rs = runs(lab)
    while len(rs) > 1:
        i = min(range(len(rs)), key=lambda j: rs[j][2] - rs[j][1])
        if rs[i][2] - rs[i][1] >= n: break
        nb = max((j for j in (i - 1, i + 1) if 0 <= j < len(rs)), key=lambda j: rs[j][2] - rs[j][1])
        lab[rs[i][1]:rs[i][2]] = [rs[nb][0]] * (rs[i][2] - rs[i][1]); rs = runs(lab)
    return lab

def trades(lab, n):
    """Spans where two people swap lines fast: >= 3 runs in a row, each shorter than n samples, only two people -> (start, end, (a, b))."""
    rs, out, i = runs(lab), [], 0
    while i < len(rs):
        j = i
        while j < len(rs) and rs[j][2] - rs[j][1] < n: j += 1
        who = sorted({r[0] for r in rs[i:j]})
        if j - i >= 3 and len(who) == 2: out.append((rs[i][1], rs[j - 1][2], tuple(who)))
        i = max(j, i + 1)
    return out

@lru_cache(maxsize=2)
def loudness(path, N, fps):
    """True per frame where the clip is louder than its quietest third: somebody is talking (a listener's grin must not steal the crop)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) ** 2; step = 8000 / fps
    if len(x) < step * N: x = np.pad(x, (0, int(step * N) - len(x)))
    e = 10 * np.log10(np.add.reduceat(x, (np.arange(N) * step).astype(int)) / step + 1)
    return e > np.percentile(e, 33)

def rect(box, W, H, k):
    """Source crop around a face: k=0 full 9:16 screen, 1/2 = top/bottom half of a split (face a little low on top, lower still on the bottom, so captions miss it)."""
    x, y, w, h = box; ch = float(np.clip(h * (5 if k == 0 else 4.2), (0.6 if k == 0 else 0.3) * H, H)); cw = ch * (9 / 16 if k == 0 else OW / (OH / 2))
    x0 = np.clip(x + w / 2 - cw / 2, 0, W - cw); y0 = np.clip(y + h / 2 - ch * (0.38, 0.6, 0.7)[k], 0, H - ch)
    return int(x0), int(y0), int(x0 + cw), int(y0 + ch)

def panels(f, rs):
    """One crop = the whole 9:16 frame; two crops = split screen, first on top."""
    if len(rs) == 1: x0, y0, x1, y1 = rs[0]; return cv2.resize(f[y0:y1, x0:x1], (OW, OH), interpolation=cv2.INTER_CUBIC)
    out = np.zeros((OH, OW, 3), np.uint8)
    for n, (x0, y0, x1, y1) in enumerate(rs): out[n * OH // 2:(n + 1) * OH // 2] = cv2.resize(f[y0:y1, x0:x1], (OW, OH // 2), interpolation=cv2.INTER_CUBIC)
    out[OH // 2 - 3:OH // 2 + 3] = 0; return out  # thin seam

def speakers(path, a, b, W, H, fps, N):
    """People sitting apart (couch podcast): follow whoever talks. Lip motion per face (jawOpen changes) while the clip is loud;
    returns (per-frame crop rects, why) for frames a..b, or None when unsure so the caller keeps the blurred-fill frame."""
    if not LM.exists(): return None
    cap = cv2.VideoCapture(str(path)); yn = cv2.FaceDetectorYN.create("onnx", YUNET, np.empty(0, np.uint8), (960, int(960 * H / W)), 0.6); sc_ = 960 / W
    cl, ks = [], np.linspace(a, b - 1, 6).astype(int)  # people = clusters of face boxes by x (a couch hardly moves); found at 960 px: couch faces are ~40 px in the 640 px detector copy
    for k in ks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, f = cap.read()
        _, fs = yn.detect(cv2.resize(f, (960, int(960 * H / W)))) if ok else (0, None)
        for bx in ([] if fs is None else fs[:, :4] / sc_):
            c = next((c for c in cl if abs(np.median([q[0] + q[2] / 2 for q in c]) - bx[0] - bx[2] / 2) < 0.6 * bx[2]), None)
            c.append(bx) if c else cl.append([bx])
    cl = sorted((c for c in cl if len(c) >= 0.5 * len(ks)), key=lambda c: np.median([q[0] for q in c]))[:5]
    if len(cl) == 1 and np.median(cl[0], axis=0)[2] > 0.1 * W:  # one big face (close-up with a stray shoulder at the edge): just crop on it
        return [[rect(tuple(np.median(cl[0], axis=0)), W, H, 0)]] * (b - a), "close-up"  # shortcut: fixed per shot, a head that leans far out leaves the crop
    if len(cl) < 2: return None
    boxes = [tuple(np.median(c, axis=0)) for c in cl]; ns = (b - a + 2) // 3  # landmarks on every 3rd frame (10 Hz is plenty for lips)
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_buffer=LM.read_bytes()), output_face_blendshapes=True, num_faces=1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a); jaw = np.zeros((len(cl), ns))
    for k in range(b - a):
        ok, f = cap.read()
        if not ok: break
        if k % 3: continue
        for c, (x, y, w, h) in enumerate(boxes):  # landmarks on a crop of full-size pixels: a couch face is ~40 px in the 640 px detector copy
            x0, y0, x1, y1 = (int(v) for v in (max(x - w / 3, 0), max(y - h / 3, 0), min(x + 4 * w / 3, W), min(y + 4 * h / 3, H)))
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))
            if r.face_blendshapes: jaw[c, k // 3] = next(s.score for s in r.face_blendshapes[0] if s.category_name == "jawOpen")
    cap.release(); hz = fps / 3; kn = min(max(3, int(hz * 0.8)), ns)  # a shot shorter than the window: "same" would return the window's length
    sc = np.array([np.convolve(np.abs(np.diff(j, prepend=j[0])), np.ones(kn) / kn, "same") for j in jaw])
    sc *= loudness(str(path), N, fps)[a:b:3][:ns]
    if (sc.max(0) > SPEECH_MOTION).mean() < 0.3: return None  # lips barely move anywhere: not a talking shot, keep the fill
    lab = [int(sc[:, 0].argmax())]
    for t in range(1, ns):  # keep the speaker through silence and ties; switch only to a clearly more active mouth
        top = np.sort(sc[:, t]); lab.append(int(sc[:, t].argmax()) if top[-1] > 1.3 * top[-2] and top[-1] > SPEECH_MOTION else lab[-1])
    fast = hold(lab, int(0.5 * hz)); slow = hold(fast, int(HOLD * hz)); tr = trades(fast, int(2.5 * hz))
    for s0, s1, (p, q) in tr: slow[s0:s1] = [(p, q)] * (s1 - s0)  # fast trades become a split
    out = [[rect(boxes[v], W, H, 0)] if isinstance(v, int) else [rect(boxes[v[0]], W, H, 1), rect(boxes[v[1]], W, H, 2)] for v in slow]
    return [out[min(k // 3, ns - 1)] for k in range(b - a)], f"{len(cl)} people, {len(runs(slow)) - 1} cuts, {len(tr)} split"

def plan(path):
    cap = cv2.VideoCapture(str(path)); W = int(cap.get(3)); H = int(cap.get(4)); N = int(cap.get(7)); fps = cap.get(5)
    cw = int(round(H * 9 / 16)) // 2 * 2
    obj, face = detectors(); hits = {}; hists = []; cuts = [0]
    i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        # own cut detector in the same pass: hard cuts (vs previous frame) and dissolves (vs 10 frames back)
        sm = cv2.cvtColor(cv2.resize(f, (96, 54)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([sm], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]); cv2.normalize(hist, hist)
        hists.append(hist)
        if i - cuts[-1] > fps * 0.4:
            d1 = cv2.compareHist(hists[-2], hist, cv2.HISTCMP_BHATTACHARYYA) if i else 0
            d10 = cv2.compareHist(hists[-11], hist, cv2.HISTCMP_BHATTACHARYYA) if i >= 10 else 0
            if d1 > 0.35 or d10 > 0.55: cuts.append(i)
        if i % SAMPLE == 0:  # detect on a 640px copy (fast), scale boxes back to full size
            s = 640 / W; r = subject_span(cv2.resize(f, (640, int(H * s))), obj, face)
            hits[i] = (*(v / s for v in r[:4]), [tuple(c / s for c in b) for b in r[4]]) if r else None
        i += 1
    N = i; cap.release()
    shots = list(zip(cuts, cuts[1:] + [N]))
    xs = np.full(N, W / 2.0); mode = ["crop"] * N; shotlog = []
    for a, b in shots:
        b = min(b, N); samp = [(k, v) for k, v in hits.items() if a <= k < b and v]
        nsamp = len([k for k in hits if a <= k < b]) or 1
        if len(samp) < 0.5 * nsamp:  # logos, titles, landscapes: show the whole frame instead of cutting it
            for k in range(a, b): mode[k] = "fill"
            shotlog.append(dict(start=a, end=b, mode="fill", why=f"subject in {len(samp)}/{nsamp} samples")); continue
        spread = np.median([r - l for _, (_, l, r, *_) in samp]); dialogue = np.mean([v[3] for _, v in samp]) >= 0.5
        if (dialogue and spread > cw * 1.1) or spread > cw * 2.2:  # two faces apart, or a crowd -> keep everything visible
            sp = speakers(path, a, b, W, H, fps, N)
            if sp:
                mode[a:b] = sp[0]; shotlog.append(dict(start=a, end=b, mode="speaker", why=sp[1])); continue
            for k in range(a, b): mode[k] = "fill"
            shotlog.append(dict(start=a, end=b, mode="fill", why=f"subject spread {spread:.0f}px > window {cw}px")); continue
        ks = np.array([k for k, _ in samp]); cs = np.array([v[0] for _, v in samp])
        track = np.interp(np.arange(a, b), ks, cs)
        # heavy smoothing inside the shot so the camera glides, never jitters
        win = max(3, int(fps * 0.8)) | 1
        pad = np.pad(track, win // 2, mode="edge"); track = np.convolve(pad, np.ones(win) / win, mode="valid")
        xs[a:b] = track
        shotlog.append(dict(start=a, end=b, mode="crop", why=f"{len(samp)} detections"))
    xs = np.clip(xs, cw / 2, W - cw / 2)
    return dict(W=W, H=H, fps=fps, cw=cw, N=N, shots=shotlog), xs, mode

def encoder():
    """NVIDIA GPU encoder if it really works on this PC, else CPU x264 (works everywhere)."""
    probe = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "nullsrc=s=256x256", "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
    return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"] if probe.returncode == 0 else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]

TOP_CUT = 0.08  # stream overlays (viewer counters, alerts) sit in the top strip: crop it off, which also zooms the face a little
BAND = (.15, .10, .15, .20)  # left/top/right/bottom share of the source frame where stream widgets live; a blob centred there is overlay, deeper in is game

def overlay_boxes(frames):
    """Static stream overlays (chat, counters, cam frame, ad fine print, footers) = dense edges at the same pixels in consecutive samples
    while the game moves, centred in the border band. Big or sparse blobs are game (logos, minimaps). Boxes are (x0,y0,x1,y1) frame fractions."""
    if len(frames) < 8: return []
    E = []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (960, 960 * f.shape[0] // f.shape[1]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        E.append((cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) > 100).astype(np.uint8))
    h, w = E[0].shape; k = np.ones((3, 3), np.uint8)
    P = [E[i + 1] & cv2.dilate(E[i], k) for i in range(len(E) - 1)]  # edge here and in the previous sample; an overlay that shows only part of the clip still counts: still all through one third
    keep = ((np.mean(P, axis=0) >= 0.5) | (np.max([np.mean(c, axis=0) for c in np.array_split(P, 3)], axis=0) >= 0.8)).astype(np.uint8)
    keep &= 1 - cv2.dilate(cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((1, 151), np.uint8)), k)  # long horizontal rules (letterbox bars, panel borders) are not text and glue real blobs together
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 25), np.uint8)))  # letters -> one blob
    out = []
    for i, (x, y, bw, bh, a) in enumerate(st[1:n], 1):
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        if 0.0035 * w * h < a < 0.06 * w * h and (keep[y:y + bh, x:x + bw].mean() >= 0.2 or keep[lab == i].mean() >= 0.5) and (cx < BAND[0] or cy < BAND[1] or cx > 1 - BAND[2] or cy > 1 - BAND[3]):
            out.append((x / w, y / h, (x + bw) / w, (y + bh) / h))
    return out

def clutter(src, idx, K=24):
    """overlay_boxes over K frames spread through the frame numbers idx (one layout at a time: a layout switch moves every overlay)."""
    if len(idx) < 8: return []
    cap = cv2.VideoCapture(str(src)); fr = []
    for k in np.array(idx)[np.linspace(0, len(idx) - 1, min(K, len(idx))).astype(int)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, f = cap.read()
        if ok: fr.append(f)
    cap.release(); return overlay_boxes(fr)

def fit(boxes, win, W, H, xlim=None, ylim=None):
    """Window (x0,y0,w,h) moved inside xlim/ylim and shrunk (same aspect, at most 22 %) to dodge the clutter boxes.
    Cost = share of each box left inside (padded a little: no half-cut digits) + a price per zoom and per move; no boxes = the same window."""
    x0, y0, w, h = win; xlim = xlim or (0, W); ylim = ylim or (0, H); pad = 0.005 * W
    bx = np.array([(a * W - pad, b * H - pad, c * W + pad, d * H + pad) for a, b, c, d in boxes]).reshape(-1, 4)
    best = (1e9, win)
    for s in np.linspace(1, .78, 8):
        cw, ch = w * s, h * s
        X, Y = np.meshgrid(np.arange(x0 - 8 * ((x0 - xlim[0]) // 8), max(xlim[1] - cw, xlim[0]) + 1, 8), np.arange(y0 - 8 * ((y0 - ylim[0]) // 8), max(ylim[1] - ch, ylim[0]) + 1, 8))  # grid anchored on the start window
        cost = 1.0 * (1 - s) + 0.3 * (abs(X - x0) / W + abs(Y - y0) / H)
        for a, b, c, d in bx:
            ix = np.clip(np.minimum(X + cw, c) - np.maximum(X, a), 0, None); iy = np.clip(np.minimum(Y + ch, d) - np.maximum(Y, b), 0, None)
            cost = cost + (ix * iy > 0) * (0.3 + 0.2 * ix * iy / ((c - a) * (d - b)))  # any touch costs: a half-cut counter looks worse than a whole one
        i = np.unravel_index(cost.argmin(), cost.shape)
        if cost[i] < best[0]: best = (cost[i], (int(X[i]), int(Y[i]), int(cw), int(ch)))
    return best[1]

def crop916(f, x, win=None):
    """9:16 crop centred on x, minus the top strip. win=(y0, height, xoff) from trim916() trims overlays that sit in the crop (counters, footers)."""
    H, W = f.shape[:2]; y0, ch, xo = win or (int(H * TOP_CUT), H - int(H * TOP_CUT), 0); cw = int(ch * 9 / 16)
    x0 = int(np.clip(round(x - cw / 2 + xo), 0, W - cw)); return cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (OW, OH), interpolation=cv2.INTER_CUBIC)

def trim916(boxes, xs, W, H):
    """fit() for crop916 around subject columns xs (the crop pans over them: the tightest answer of the min/median/max column wins). Top strip stays cut. -> (y0, height, xoff)."""
    y0 = int(H * TOP_CUT); cw = int((H - y0) * 9 / 16); best = (y0, H - y0, 0)
    for x in {min(xs), float(np.median(xs)), max(xs)}:
        x0 = int(np.clip(round(x - cw / 2), 0, W - cw)); fx, fy, fw, fh = fit(boxes, (x0, y0, cw, H - y0), W, H, (x0, x0 + cw), (y0, H))
        if fh < best[1]: best = (fy, fh, fx - (x0 + (cw - fw) // 2))
    return best

SPEAKER_FILL = False  # speaker tile crops on a Zoom grid mis-aim and cut tiles in half (tested 2026-11-18 on Speed/Obama): off until the tile grid itself is found
FILL_A = 15 / 16  # crop aspect (w/h) of a smart fill: 1080 wide -> 1152 tall = 60 % of the 1920 frame, against 42 % for the old 4:3 trim

def peel(c):
    """(l,t,r,b) of mask c after peeling edge rows/columns that are mostly empty: a webcam, logo or counter touching a window is a thin ragged edge."""
    l, t, r, b = 0, 0, c.shape[1], c.shape[0]
    while r - l > 4 and b - t > 4:
        if c[t:b, l:r][0].mean() < 0.5: t += 1
        elif c[t:b, l:r][-1].mean() < 0.5: b -= 1
        elif c[t:b, l:r][:, 0].mean() < 0.5: l += 1
        elif c[t:b, l:r][:, -1].mean() < 0.5: r -= 1
        else: break
    return l, t, r, b

def content_box(f):
    """(x0,y0,x1,y1) of the biggest lit region: black pillars/letterbox (a call window inside a stream canvas) are cut, small widgets fall away in the opening."""
    H, W = f.shape[:2]; g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, max(int(160 * H / W), 1)), interpolation=cv2.INTER_AREA)
    m = cv2.morphologyEx(cv2.morphologyEx((g > 10).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return 0, 0, W, H
    i = 1 + int(np.argmax(st[1:, 4])); x, y, w, h, a = st[i]
    if a < 0.12 * m.size: return 0, 0, W, H
    l, t, r, b = peel((lab == i)[y:y + h, x:x + w])
    kx, ky = W / 160, H / g.shape[0]; return int((x + l) * kx), int((y + t) * ky), int((x + r) * kx), int((y + b) * ky)

def bar_box(frames, thr=24, frac=0.4, vote=0.75):
    """(x0,y0,x1,y1) of the picture inside black bars baked into the source (a 16:9 trailer or cutscene letterboxed in the stream frame), or None.
    A row/column is bar when under `frac` of its pixels are lit (chat text or a logo inside a bar stays bar; a very dark night scene edge may be cropped, capped at half the frame) in >= `vote` of the frames; all-black frames are skipped, <1.5 % trim is ignored."""
    rows, cols, grays, H, W = [], [], [], 0, 0
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); H, W = g.shape
        if g.mean() < 6: continue
        grays.append(g)
        lit = g > thr; rows.append(lit.mean(1) >= frac); cols.append(lit.mean(0) >= frac)
    if not rows: return None
    r = np.mean(rows, 0) > 1 - vote; c = np.mean(cols, 0) > 1 - vote; ys, xs = np.flatnonzero(r), np.flatnonzero(c)
    if not len(ys) or not len(xs): return None
    box = (int(xs[0]) // 2 * 2, int(ys[0]) // 2 * 2, (int(xs[-1]) + 1) // 2 * 2, (int(ys[-1]) + 1) // 2 * 2)
    if box[1] < 0.015 * H and box[0] < 0.015 * W and box[3] > 0.985 * H and box[2] > 0.985 * W: return None
    if (box[2] - box[0]) * (box[3] - box[1]) <= 0.5 * W * H: return None
    ar = (box[2] - box[0]) / (box[3] - box[1])
    if not any(abs(ar / s - 1) < 0.05 for s in (16 / 9, 1.85, 2.0, 2.39, 4 / 3)): return None  # a dark game area is not a bar: the picture left must be a cinema/TV aspect
    outside = np.concatenate([g[:max(box[1], 0)].ravel() for g in grays] + [g[box[3]:].ravel() for g in grays] + [g[:, :box[0]].ravel() for g in grays] + [g[:, box[2]:].ravel() for g in grays])
    return box if np.median(outside) <= 20 else None  # real bars are near-black

def crop_bars(path, n=8):
    """Crop baked-in black bars off the source file in place (re-encode) so every layout frames the real picture; returns the box or None."""
    cap = cv2.VideoCapture(str(path)); N = cap.get(cv2.CAP_PROP_FRAME_COUNT); fr = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(N * (k + .5) / n)); ok, f = cap.read()
        if ok: fr.append(f)
    cap.release(); b = bar_box(fr)
    if not b: return None
    tmp = Path(path).with_suffix(".bars.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vf", f"crop={b[2] - b[0]}:{b[3] - b[1]}:{b[0]}:{b[1]}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "copy", str(tmp)], check=True)
    tmp.replace(path); print(f"  baked-in black bars cropped off the source: picture box {b}"); return b

def tiles(f, cb):
    """Boxes (x0,y0,x1,y1, frame px) of the tiles inside a content box: pixels that differ from the panel colour (its median), grouped. Zoom/Meet grids -> one box per tile."""
    x0, y0, x1, y1 = cb; c = f[y0:y1, x0:x1]; k = 0.25; g = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    m = (np.abs(g.astype(np.int16) - int(np.median(g))) > 12).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # no closing: the 8 px gaps between tiles must stay open
    n, lab, st, _ = cv2.connectedComponentsWithStats(m); out = []
    for i, (x, y, w, h, a) in enumerate(st[1:n], 1):
        if a < 0.01 * m.size or w <= 12 or h <= 12: continue
        l, t, r, b = peel((lab == i)[y:y + h, x:x + w]); out.append((int(x0 + (x + l) / k), int(y0 + (y + t) / k), int(x0 + (x + r) / k), int(y0 + (y + b) / k)))
    return out

def real_tiles(tl, W, H):
    """Tiles without overlays: a box under 8 % of the frame (cam inset, chat, viewer counter) is not a grid tile; all small = keep as is (small-window grid)."""
    return [t for t in tl if (t[2] - t[0]) * (t[3] - t[1]) >= 0.08 * W * H] or tl

def trim_to_tiles(f, cb, t=None):
    """cb shrunk to the union of its tiles when the tiles cover clearly less (an empty dark panel around a grid); several tiles only."""
    t = tiles(f, cb) if t is None else t
    if len(t) < 2: return cb
    u = (min(b[0] for b in t), min(b[1] for b in t), max(b[2] for b in t), max(b[3] for b in t))
    return u if (u[2] - u[0]) * (u[3] - u[1]) < 0.85 * (cb[2] - cb[0]) * (cb[3] - cb[1]) else cb

def grid_cells(tl, fc, need_face=True):
    """4-6 tiles (a call grid) -> 2-column stack filling the 9:16 frame: [(src rect, dst rect)]. Each tile is cut out by its box; a tile whose faces fit a cell-shaped crop is
    cropped on them (big faces), otherwise shown whole inside its cell (nobody cut). None when it is not a grid worth stacking."""
    if len(tl) == 2:  # two call tiles side by side (Omegle, 1-to-1 call): left on top, right below, each full width = faces ~2x bigger than the whole-picture layout
        (a, b), = [sorted(tl)]; ya, yb = max(a[1], b[1]), min(a[3], b[3])
        if a[2] > b[0] or yb - ya < 0.7 * min(a[3] - a[1], b[3] - b[1]) or not 0.7 < (a[2] - a[0]) / (b[2] - b[0]) < 1 / 0.7 or (need_face and not any(t[0] <= q[0] + q[2] / 2 <= t[2] and t[1] <= q[1] + q[3] / 2 <= t[3] for t in tl for q in fc)): return None
    elif not 4 <= len(tl) <= 6: return None
    cols = 1 if len(tl) == 2 else 2
    hs = [b[3] - b[1] for b in tl]; own = [[b for b in fc if t[0] <= b[0] + b[2] / 2 <= t[2] and t[1] <= b[1] + b[3] / 2 <= t[3]] for t in tl]
    ws = [b[2] - b[0] for b in tl]
    if max(hs) > 1.6 * min(hs) or min(ws) < 0.7 * max(ws) or (len(tl) > 2 and sum(1 for o in own if o) < 3): return None
    tl = sorted(zip(tl, own), key=lambda q: (round(q[0][1] / 200), q[0][0])); n = len(tl); ch = OH // ((n + cols - 1) // cols); cells = []
    for k, (t, o) in enumerate(tl):
        r, c = divmod(k, cols); alone = cols == 1 or (n % 2 and k == n - 1); cw = OW if alone else OW // 2; cx, cy = 0 if alone else c * (OW // 2), r * ch
        tw, th = t[2] - t[0], t[3] - t[1]; need = th * cw / ch
        if o and need < tw and max(b[0] + b[2] for b in o) - min(b[0] for b in o) < 0.9 * need:
            a = np.clip((min(b[0] for b in o) + max(b[0] + b[2] for b in o)) / 2 - need / 2, t[0], t[2] - need); cells.append(((int(a), t[1], int(a + need), t[3]), (cx, cy, cx + cw, cy + ch)))
        elif min(need / tw, tw / need) >= 0.7:  # no face to aim at: centre crop that fills the cell (a fitted tile left black bands; r7 critics), when it keeps >= 70 % of the tile
            if need < tw: cells.append(((int(t[0] + (tw - need) / 2), t[1], int(t[0] + (tw + need) / 2), t[3]), (cx, cy, cx + cw, cy + ch)))
            else: nh = tw * ch / cw; cells.append(((t[0], int(t[1] + (th - nh) / 2), t[2], int(t[1] + (th + nh) / 2)), (cx, cy, cx + cw, cy + ch)))
        else:
            sc = min(cw / tw, ch / th); dw, dh = int(tw * sc), int(th * sc); cells.append((t, (cx + (cw - dw) // 2, cy + (ch - dh) // 2, cx + (cw - dw) // 2 + dw, cy + (ch - dh) // 2 + dh)))
    return cells

def smart_window(cb, faces, spk=None):
    """Crop window (x0,y0,x1,y1) inside content box cb: aspect FILL_A on the faces when they fit (or on the active speaker spk=(cx,cy,h)), else the whole box.
    Boxes narrower than 9:16 are cut to 9:16 around the faces. faces = (x,y,w,h) boxes in frame pixels, already inside cb."""
    x0, y0, x1, y1 = cb; bw, bh = x1 - x0, y1 - y0
    fy = float(np.mean([y + h / 2 for x, y, w, h in faces])) if faces else (y0 + y1) / 2
    if bw / bh < 9 / 16: ch = bw * 16 / 9; ya = np.clip(fy - ch / 2, y0, y1 - ch); return x0, int(ya), x1, int(ya + ch)
    if bw / bh <= FILL_A: return cb
    if spk: ch = min(bh, spk[2]); ww = ch * FILL_A; cx, cy = spk[0], spk[1]
    elif faces:
        l, r = min(x for x, _, _, _ in faces), max(x + w for x, _, w, _ in faces); fw = max(w for _, _, w, _ in faces); ww = bh * FILL_A
        if r - l + fw > ww: return cb  # faces spread wider than the window: everybody stays in, the picture stays small
        ch = bh; cx = (l + r) / 2; cy = fy
    else:  # no faces (trailer, gameplay): centre 4:3 so the picture fills more of the panel than a whole 16:9 would; centre action stays
        if bw / bh <= 4 / 3 + 0.01: return cb  # shortcut: centre 4:3 drops 25 % of the width of a face-less 16:9 panel (a corner minimap/HUD or off-centre action is cut); upgrade = pick the window on motion/saliency
        ww = bh * 4 / 3; ch = bh; cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
    ww = min(ww, bw); xa = np.clip(cx - ww / 2, x0, x1 - ww); ya = np.clip(cy - ch / 2, y0, y1 - ch); return int(xa), int(ya), int(xa + ww), int(ya + ch)

def fill_rects(src, idx, W, H, fps, N, every=3):
    """Smart crop window per 'fill' frame (dict frame -> rect). Content box + faces every 3rd frame, smoothed, snapped at jumps; wide shots with several faces ask speakers() for the talker."""
    idx = sorted(idx)
    if not idx: return {}
    want = set(idx); cap = cv2.VideoCapture(str(src)); yn = cv2.FaceDetectorYN.create("onnx", YUNET, np.empty(0, np.uint8), (960, int(960 * H / W)), 0.6); s = 960 / W
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx[0]); ks, cbs, fcs, gcs, nts, tls, fas = [], [], [], [], [], [], []
    for i in range(idx[0], idx[-1] + 1):
        ok, f = cap.read()
        if not ok: break
        if i not in want or (i - idx[0]) % every: continue
        cb0 = content_box(f); tl = real_tiles(tiles(f, cb0), W, H); cb = trim_to_tiles(f, cb0, tl); _, fs = yn.detect(cv2.resize(f, (960, int(960 * H / W))))
        fc = [tuple(b) for b in ([] if fs is None else fs[:, :4] / s) if cb[0] <= b[0] + b[2] / 2 <= cb[2] and cb[1] <= b[1] + b[3] / 2 <= cb[3] and b[2] > TINY_FACE * W]
        ks.append(i); nts.append(len(tl)); tls.append(tl); fas.append([tuple(b) for b in ([] if fs is None else fs[:, :4] / s)]); cbs.append(cb); fcs.append(fc); gcs.append(grid_cells(tl, [tuple(b) for b in ([] if fs is None else fs[:, :4] / s)]))
    gj = [j for j, g in enumerate(gcs) if g and len(g) >= 4]
    for j in range(len(ks)):  # a dark tile (Speed's black pillars) melts into the panel: the finder sees one big tile while 4+ faces are on screen = same grid as the nearest sample that found it
        if not gcs[j] and nts[j] < 2 and len(fas[j]) >= 4 and gj: gcs[j] = grid_cells(tls[min(gj, key=lambda q: abs(q - j))], fas[j])
    ct = call_tiles(src, W, H)  # a stream-embedded call: the panel-colour tile finder sees the whole frame as one tile, so the seam finder gives the tiles
    if ct:
        for j in range(len(ks)): gcs[j] = grid_cells(ct, fas[j], False); nts[j] = 2
    cap.release(); win = [g or (cb if nt >= 2 else smart_window(cb, fc)) for g, cb, fc, nt in zip(gcs, cbs, fcs, nts)]  # a grid that is not stacked stays whole: never crop between its tiles
    wide = [len(fc) >= 2 and not g and w == cb and (cb[2] - cb[0]) / (cb[3] - cb[1]) > FILL_A for g, cb, fc, w in zip(gcs, cbs, fcs, win)]  # several faces that did not fit
    for _, j0, j1 in [r for r in runs(wide) if r[0]]:  # runs of such samples >= 1 s: who talks?
        if not SPEAKER_FILL or (ks[j1 - 1] - ks[j0]) / fps < 1: continue
        sp = speakers(src, ks[j0], ks[j1 - 1] + every, W, H, fps, N)
        if not sp: continue
        for j in range(j0, j1):
            r = sp[0][min(ks[j] - ks[j0], len(sp[0]) - 1)]
            if len(r) == 1 and not gcs[j]: x0, y0, x1, y1 = r[0]; win[j] = smart_window(cbs[j], fcs[j], ((x0 + x1) / 2, (y0 + y1) / 2, y1 - y0))
    cur = None; sm, snap = [], []
    for j, w in enumerate(win):
        if gcs[j]:  # a stacked grid: hold the tile layout until it changes by more than jitter (no EMA, no blending of two layouts)
            if sm and gcs[j - 1] and len(w) == len(sm[-1]) and all(max(abs(a - b) for a, b in zip(x[0], y[0])) < 16 for x, y in zip(w, sm[-1])): w = sm[-1]
            snap.append(not (j and gcs[j - 1])); sm.append(w); cur = None; continue
        w = np.array(w, float); jump = cur is None or abs((w[0] + w[2]) - (cur[0] + cur[2])) / 2 > 0.25 * (cur[2] - cur[0]) or abs((w[3] - w[1]) / (cur[3] - cur[1]) - 1) > 0.25
        cur = w if jump else cur + 0.4 * (w - cur); snap.append(bool(jump and j)); sm.append(cur.copy())
    out = {}
    for i in idx:
        j = max(int(np.searchsorted(ks, i, "right")) - 1, 0); r = sm[j]
        if not gcs[j] and j + 1 < len(ks) and not gcs[j + 1] and not snap[j + 1]: t = (i - ks[j]) / (ks[j + 1] - ks[j]); r = sm[j] * (1 - t) + sm[j + 1] * t
        out[i] = r if gcs[j] else tuple(int(v) for v in r)
    return out

def call_tiles(path, W, H, n=8):
    """Two call tiles [(l,t,r,b)] x2 of a 1-to-1 call (Omegle) shown in a stream frame, or None: a hard vertical seam near the centre (two unrelated videos meet) and a hard horizontal edge below it
    (tiles end, button bar starts), both far stronger than the rest of the frame in the median of n frames, and a face on each side in >= 30 % of them."""
    cap = cv2.VideoCapture(str(path)); N = cap.get(7); yn = cv2.FaceDetectorYN.create("onnx", YUNET, np.empty(0, np.uint8), (960, int(960 * H / W)), 0.6); gs, L, R = [], 0, 0
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(N * (k + .5) / n)); ok, f = cap.read()
        if not ok: continue
        gs.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (480, 270)).astype(np.float32)); _, fs = yn.detect(cv2.resize(f, (960, int(960 * H / W))))
        cx = [] if fs is None else (fs[:, 0] + fs[:, 2] / 2) / 960; L += any(c < .5 for c in cx); R += any(c > .5 for c in cx)
    cap.release()
    if len(gs) < n / 2 or min(L, R) < 0.3 * len(gs): return None
    col = np.median([np.abs(np.diff(g[:189], axis=1)).mean(0) for g in gs], 0); row = np.median([np.abs(np.diff(g, axis=0)).mean(1) for g in gs], 0)
    x = 144 + int(np.argmax(col[144:336])); y = 148 + int(np.argmax(row[148:256]))
    if col[x] < 6 * np.median(col) or row[y] < 6 * np.median(row): return None
    sx, sy = int((x + 1) / 480 * W), int((y + 1) / 270 * H); m, mb = int(.006 * W), int(.02 * H)  # edges are found on a 480x270 grid (±4 px) and the bar starts with dark rows
    out = []
    for c0, c1, X0, X1 in ((0, x + 1, 0, sx - m), (x + 1, 480, sx + m, W)):  # a tile's own video may be letterboxed inside it: drop its dark top/bottom rows (r7 critics saw a black band between the panels)
        r = np.median([g[:y + 1, c0:c1].mean(1) for g in gs], 0); lit = np.nonzero(r > 14)[0]
        t0, t1 = (int(lit[0]), int(lit[-1]) + 1) if len(lit) and lit[-1] - lit[0] > 0.5 * (y + 1) else (0, y + 1)
        out.append((X0, int(t0 / 270 * H), X1, min(int(t1 / 270 * H), sy - mb)))
    return out

def is_call(path):
    """True for a 1-to-1 video call: such a frame is stacked, never run through the facecam finder (the other tile's big face reads as a full-screen cam and Speed was cut out)."""
    cap = cv2.VideoCapture(str(path)); W, H = int(cap.get(3)), int(cap.get(4)); cap.release(); return call_tiles(path, W, H) is not None

def fill(f, W, H, cw, rc=None):
    """Blurred fill: the crop rc (smart_window; none = frame sides trimmed to 4:3) shown full width over a blurred, darkened copy of itself."""
    if isinstance(rc, list):  # stacked call grid: tiles cut out and laid in two columns on black
        out = np.zeros((OH, OW, 3), np.uint8)
        for (x0, y0, x1, y1), (a, b, c, d) in rc:
            if FLIP and len(rc) == 2: b, d = OH - d, OH - b  # flip: two call tiles swap places
            out[b:d, a:c] = cv2.resize(f[y0:y1, x0:x1], (c - a, d - b), interpolation=cv2.INTER_CUBIC)
        return out
    if rc is None:
        rows = np.flatnonzero(f[:, ::8].max(axis=(1, 2)) > 30)  # letterbox bars (cinematic uploads) are pure black: trim them, equal top and bottom
        t = min(rows[0], H - 1 - rows[-1]) if len(rows) and rows[-1] - rows[0] > H // 2 else 0
        fx = min(W, int((H - 2 * t) * 4 / 3)); rc = ((W - fx) // 2, t, (W + fx) // 2, H - t)
    c = f[rc[1]:rc[3], rc[0]:rc[2]]; fh = min(OH, int(OW * c.shape[0] / c.shape[1]))
    small = cv2.GaussianBlur(cv2.resize(c, (OW // 8, OH // 8)), (0, 0), 4)  # blur small then upscale: same look, ~50x cheaper
    out = cv2.convertScaleAbs(cv2.resize(small, (OW, OH), interpolation=cv2.INTER_LINEAR), alpha=0.55)  # integer darken, numpy float math was ~60% of render time
    y = (OH - fh) // 2; out[y:y + fh] = cv2.resize(c, (OW, fh), interpolation=cv2.INTER_AREA); return out

def bubble_ok(cam, full, W, H, share=0.6):
    """A round-bubble cam must be a real webcam overlay: a corner/edge box that looks like itself in most frames. A face on a couch or in a trailer sits mid-frame or comes and goes."""
    cx, cy = (cam[0] + cam[2]) / 2 / W, (cam[1] + cam[3]) / 2 / H
    return (cx < .35 or cx > .65 or cy < .3 or cy > .7) and sum(v is None for v in full) >= share * len(full)

def render(src, dst, info, xs, mode, smart=True):
    W, H, cw = info["W"], info["H"], info["cw"]
    tmp = Path(dst).with_suffix(".noaudio.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{OW}x{OH}", "-r", str(info["fps"]),
        "-i", "-", *encoder(), "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    ks = [k for k in range(info["N"]) if mode[k] == "crop"]  # shortcut: one trim for the whole clip, cut shots of other scenes share it
    win = trim916(clutter(src, ks), xs[ks], W, H) if ks else None
    fr = fill_rects(src, [k for k in range(info["N"]) if mode[k] == "fill"], W, H, info["fps"], info["N"]) if smart else {}
    cap = cv2.VideoCapture(str(src)); i = 0
    while i < info["N"]:
        ok, f = cap.read()
        if not ok: break
        if mode[i] == "crop":
            out = crop916(f, xs[i], win)
        elif mode[i] == "fill": out = fill(f, W, H, cw, fr.get(i))
        else: out = panels(f, mode[i])
        ff.stdin.write(cv2.cvtColor(out, cv2.COLOR_BGR2YUV_I420).tobytes()); i += 1  # half the bytes of BGR through the pipe
    cap.release(); ff.stdin.close(); ff.wait()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(src), "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)], check=True)
    tmp.unlink()

if __name__ == "__main__":
    L = [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # a 2-sample blip is absorbed, a real turn stays
    assert hold(L, 3) == [0] * 8 + [1] * 8, hold(L, 3)
    T = [0] * 4 + [1] * 2 + [0] * 2 + [1] * 2 + [0] * 4  # fast ping-pong between two people = split screen, slow turns are not
    assert trades(T, 5) == [(0, 14, (0, 1))] and trades([0] * 9 + [1] * 9, 5) == [], trades(T, 5)
    rng = np.random.default_rng(0); g = cv2.GaussianBlur(rng.integers(0, 255, (540, 960), dtype=np.uint8), (0, 0), 6); g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    fr = []  # sharp-edged "game" sliding 37 px per frame + a still text-like box bottom right + a still dark sky top left
    for t in range(10):
        f = cv2.cvtColor(np.roll(g, 37 * t, axis=1) // 2 + 100, cv2.COLOR_GRAY2BGR); f[:200, :300] = 10; f[480:520, 600:900] = 255; fr.append(f)
        for x in range(610, 890, 6): f[484:516, x:x + 3] = 0  # letters
    bx = overlay_boxes(fr); assert len(bx) == 1 and bx[0][0] > .6 and bx[0][1] > .85, bx  # the box is found, the sky and the moving game are not
    assert overlay_boxes([np.roll(f, 37 * i, axis=0) for i, f in enumerate(fr)]) == []  # now it moves too: nothing stays put = no overlay
    assert bubble_ok((1500, 700, 1900, 1050), [None] * 10, 1920, 1080) and not bubble_ok((700, 300, 1100, 700), [None] * 10, 1920, 1080) and not bubble_ok((1500, 700, 1900, 1050), [None] * 3 + ["fill"] * 7, 1920, 1080)  # corner cam in every frame = bubble; mid-frame or mostly gone = not
    cb = content_box(np.pad(np.full((600, 800, 3), 120, np.uint8), ((0, 0), (560, 560), (0, 0)))); assert cb[0] in range(540, 580) and cb[2] in range(1340, 1380), cb  # black pillars cut off
    lb = np.full((540, 960, 3), 90, np.uint8); lb[:30] = 0; lb[510:] = 0; lb[:30, 760:] = 200  # letterboxed 16:9 with chat text in the top bar
    assert bar_box([lb] * 4) == (0, 30, 960, 510), bar_box([lb] * 4); assert bar_box([np.full((540, 960, 3), 90, np.uint8)] * 4) is None  # real picture untouched
    dk = np.full((540, 960, 3), 90, np.uint8); dk[300:] = 10; assert bar_box([dk] * 4) is None  # dark gameplay below a lit sky is not a letterbox (review finding 1)
    assert smart_window((0, 0, 1600, 900), [(700, 300, 100, 120)]) == (int(750 - 450 * FILL_A), 0, int(750 + 450 * FILL_A), 900)  # one face in a wide window: tall FILL_A crop on it
    assert len(real_tiles([(0, 0, 400, 250), (1200, 0, 1900, 500), (0, 540, 900, 1040), (0, 700, 100, 800)], 1920, 1080)) == 2  # cam inset (tiny) dropped; real panels stay
    assert len(real_tiles([(0, 0, 900, 500), (960, 0, 1900, 500), (0, 540, 900, 1040)], 1920, 1080)) == 3  # 3-panel grid untouched
    assert len(real_tiles([(0, 0, 200, 150), (300, 0, 500, 150)], 1920, 1080)) == 2  # all small: unchanged
    assert smart_window((0, 0, 1600, 900), []) == (200, 0, 1400, 900)  # no faces, 16:9: centre 4:3
    assert smart_window((0, 0, 1600, 900), [(100, 300, 100, 120), (1300, 300, 100, 120)]) == (0, 0, 1600, 900)  # two faces far apart: whole window
    assert smart_window((600, 0, 1200, 1080), []) == (600, 0, 1200, 1080) or smart_window((600, 0, 1200, 1080), [])[3] - 0 <= 1080  # portrait window stays
    T5 = [(0, 0, 450, 300), (460, 0, 910, 300), (920, 0, 1370, 300), (230, 310, 680, 610), (690, 310, 1140, 610)]; F5 = [(150, 100, 60, 80), (600, 100, 60, 80), (1100, 100, 60, 80), (400, 400, 60, 80), (900, 400, 60, 80)]
    gc = grid_cells(T5, F5); assert gc and len(gc) == 5 and all(0 <= d[0] < d[2] <= OW and 0 <= d[1] < d[3] <= OH for _, d in gc) and grid_cells(T5[:3], F5) is None and grid_cells(T5[:4] + [(0, 310, 100, 610)], F5) is None and grid_cells(T5, []) is None  # 5 tiles stack inside the frame; 3 tiles or no faces do not
    c2 = grid_cells([(0, 0, 640, 540), (650, 0, 1290, 540)], [(300, 200, 90, 110), (900, 200, 90, 110)]); assert c2 and c2[0][1] == (0, 0, OW, OH // 2) and c2[1][1] == (0, OH // 2, OW, OH) and grid_cells([(0, 0, 640, 540), (650, 0, 1290, 200)], [(300, 200, 90, 110)]) is None and grid_cells([(0, 0, 640, 540), (650, 0, 1290, 540)], []) is None and grid_cells([(0, 0, 640, 540), (650, 0, 1290, 540)], [], False)  # 2 call tiles = left on top, right below, full width
    import reframe as _m; mm = np.zeros((20, 30), np.uint8); mm[5:15, 8:25] = 1; mm[2, 2] = 1; assert _m.maxrect(mm) == (170, 8, 5, 25, 15)  # the solid block, not the stray pixel (game window finder)
    _nor = Path(__file__).resolve().parents[1] / "test" / "h19b" / "src" / "gym.mp4"
    if _nor.exists(): g, c = _m.reaction(_nor); assert g[2] - g[0] > 600 and 1.3 < (g[2] - g[0]) / (g[3] - g[1]) < 2.3 and c[3] < g[1] and c[2] - c[0] < 600  # group reaction: game window found, one face box above it
    assert _m.reaction(Path(__file__)) is None  # not a video: nothing
    assert fit([], (100, 0, 900, 1000), 1920, 1080) == (100, 0, 900, 1000)  # no clutter, same window
    wy = fit([(.3, .85, .7, .9)], (500, 30, 900, 1000), 1920, 1080)  # fine print low in the window: shrink up, same aspect
    assert wy[1] + wy[3] <= .85 * 1080 and wy[2] >= .8 * 900 and abs(wy[2] / wy[3] - .9) < .01, wy
    wc = fit([(.9, .3, 1., .6)], (900, 30, 900, 1000), 1920, 1080, (900, 1920)); assert wc[0] + wc[2] <= .9 * 1920, wc  # chat column at the right edge: sliding left is blocked (cam there), so shrink off it
    wc = fit([(.9, .3, 1., .6)], (900, 30, 900, 1000), 1920, 1080, (700, 1920)); assert wc[0] < 900 and wc[2] == 900, wc  # room to slide: slide, no zoom
    assert 0.018 < TINY_FACE < 0.03, TINY_FACE  # tiny Kai cam (0.016-0.024 W median 0.018) is rejected, the smallest measured good cams (0.03 W) are kept
    _kai = Path(__file__).resolve().parents[1] / "test" / "tinycam25" / "seg.mp4"
    if _kai.exists(): import reframe as _r; assert _r.find_facecam(_kai) is None  # that real tiny-cam clip must never produce a stack panel
    import reframe as _m; g0, g1 = _m.stack_geom(False), _m.stack_geom(True); assert g0["cam"][0] == 0 and g1["game"][0] == 0 and g1["cam"] == (_m.OH - _m.TOP_H, _m.TOP_H) and g1["seam"] == g1["cam"][0] == g1["game"][1], (g0, g1)  # flip: game on top, cam below, same sizes
    assert g1["game"][1] + g1["cam"][1] == _m.OH
    if sys.argv[1:] == ["selftest"]: print("selftest ok"); raise SystemExit
    if len(sys.argv) < 3: print("usage: reframe.py <src.mp4> <dst.mp4> | reframe.py selftest"); raise SystemExit(2)
    src, dst = sys.argv[1], sys.argv[2]
    info, xs, mode = plan(src)
    assert len(xs) == info["N"] and all(info["cw"] / 2 <= x <= info["W"] - info["cw"] / 2 for x in xs), "crop window out of frame"
    render(src, dst, info, xs, mode)
    if "--debug" in sys.argv: Path(sys.argv[sys.argv.index("--debug") + 1]).write_text(json.dumps(info, indent=1))
    print("ok", dst, info["N"], "frames,", sum(1 for s in info["shots"] if s["mode"] == "fill"), "fill shots of", len(info["shots"]))


# ---------- Stacked layout for streamer reactions: facecam on top, game below ----------
TOP_H = 768  # 40% of 1920
FLIP = False  # flip option: stacked layouts put the cam BELOW the game (clip.py sets it per Short)
def stack_geom(flip=False):
    """Row layout of the stacked 9:16: cam panel (y, h), game panel (y, h), seam centre y. flip = cam below the game."""
    bh = OH - TOP_H
    return dict(cam=(bh, TOP_H), game=(0, bh), seam=bh) if flip else dict(cam=(0, TOP_H), game=(TOP_H, bh), seam=TOP_H)

def find_facecam(path, samples=40):
    """Facecam overlay = a face that stays in one spot in most frames AND jitters a little (a live person).
    Faces in a paused trailer sit still with zero jitter; faces in a playing video move around. Returns (x0,y0,x1,y1) or None."""
    cap = cv2.VideoCapture(str(path)); W = int(cap.get(3)); H = int(cap.get(4)); N = int(cap.get(7))
    yn = cv2.FaceDetectorYN.create("onnx", YUNET, np.empty(0, np.uint8), (960, int(960 * H / W)), 0.6); s = 960 / W
    cells = {}  # grid cell -> list of (sample index, box)
    for si, k in enumerate(np.linspace(0, max(N - 2, 0), samples).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, f = cap.read()
        if not ok: continue
        _, fs = yn.detect(cv2.resize(f, (960, int(960 * H / W))))
        for x, y, w, h in ([] if fs is None else fs[:, :4] / s):
            c = (int((x + w / 2) / W * 8), int((y + h / 2) / H * 6))
            cells.setdefault(c, {})[si] = (x, y, w, h)  # one face per cell per sample
    cap.release()
    best = None
    for c, hits in cells.items():
        if len(hits) < samples * 0.4: continue
        b = np.array(list(hits.values())); ctr = b[:, :2] + b[:, 2:] / 2
        jitter = float(np.std(ctr, axis=0).mean()); fw = float(np.median(b[:, 2]))
        if jitter < 1.0 or fw > W * 0.1: continue  # frozen (paused video) or a full-screen webcam (measured: overlay cam faces 0.03-0.07 W, full-screen cam 0.12 W)
        score = len(hits) + c[1] * 2 + (3 if c[0] in (0, 1, 6, 7) else 0)  # cams sit low and in corners
        if not best or score > best[0]: best = (score, np.median(ctr, axis=0), fw)
    if not best or best[2] < W * TINY_FACE: return None  # tiny cam over a full-screen video: its box would be blown up 7x+ into a blurry panel while the video behind it goes black; no cam = the caller frames the whole screen instead
    r = reaction(path)
    if r: return r[1]  # group reaction (a game window + 3 or more faces): the person who reacts most
    return cam_box(*best[1], best[2], W, H)

def cam_box(fx, fy, fw, W, H, k=4.2):
    bw = min(W, fw * k); bh = min(H, bw * TOP_H / OW); bw = bh * OW / TOP_H  # head-and-shoulders box, top-panel shape
    x0 = np.clip(fx - bw / 2, 0, W - bw); y0 = np.clip(fy - bh * 0.45, 0, H - bh)
    return int(x0), int(y0), int(x0 + bw), int(y0 + bh)

_REACT = {}
def maxrect(m):
    """Largest all-ones rectangle of a 0/1 matrix: (area, x0, y0, x1, y1). The trailer window changes in every pixel, people only in blobs."""
    h = np.zeros(m.shape[1], int); best = (0, 0, 0, 0, 0)
    for y in range(m.shape[0]):
        h = np.where(m[y], h + 1, 0); st = []
        for x in range(len(h) + 1):
            cur = h[x] if x < len(h) else 0; s = x
            while st and st[-1][1] >= cur:
                s, hh = st.pop()
                if hh * (x - s) > best[0]: best = (hh * (x - s), s, y - hh + 1, x, y + 1)
            st.append((s, cur))
    return best

def reaction(path, n=30):
    """Group reaction (3+ people on a couch / in cams + the trailer in its own window): (game window rect, cam box of the person who moves most) or None. The game window = the biggest
    block of strong frame-to-frame change (>= 8 % of the frame, 1.3-2.3 wide, no face in it); the reacting person = the face cluster outside it with the most motion around the face and mouth."""
    k = str(path)
    if k in _REACT: return _REACT[k]
    cap = cv2.VideoCapture(k); W, H, N = int(cap.get(3)), int(cap.get(4)), int(cap.get(7)); yn = cv2.FaceDetectorYN.create("onnx", YUNET, np.empty(0, np.uint8), (960, int(960 * H / W)), 0.6); s = 960 / W
    gs, fb = [], []
    for j in np.linspace(0, max(N - 3, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(j)); ok, f = cap.read()
        if not ok: continue
        gs.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (480, 270)).astype(np.float32)); _, fs = yn.detect(cv2.resize(f, (960, int(960 * H / W)))); fb.append([] if fs is None else [tuple(b) for b in fs[:, :4] / s])
    cap.release(); _REACT[k] = None
    if len(gs) < 8: return None
    d = np.mean([np.abs(a - b) for a, b in zip(gs[1:], gs[:-1])], 0); m = cv2.morphologyEx((d > 3).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    _, x0, y0, x1, y1 = maxrect(m); k4 = W / 480; g = (int(x0 * k4), int(y0 * k4), int(x1 * k4), int(y1 * k4))
    if (g[2] - g[0]) * (g[3] - g[1]) < 0.08 * W * H or not 1.3 < (g[2] - g[0]) / (g[3] - g[1]) < 2.3: return None
    cl = []  # face clusters outside the game window: [centre x, centre y, width, [(sample, box)]]
    for i, bs in enumerate(fb):
        for b in bs:
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            if g[0] <= cx <= g[2] and g[1] <= cy <= g[3] or b[2] < TINY_FACE * W: continue
            c = next((c for c in cl if abs(c[0] - cx) < 0.6 * c[2] and abs(c[1] - cy) < 0.6 * c[2]), None)
            if c: c[3].append((i, b))
            else: cl.append([cx, cy, b[2], [(i, b)]])
    cl = [c for c in cl if len(c[3]) >= 0.3 * len(gs)]
    if len(cl) < 3: return None
    def move(c):  # mean frame-to-frame change over the head and chest/mouth area of this person
        v = []
        for i, b in c[3]:
            if i == 0: continue
            x0, y0, x1, y1 = [int(q * 480 / W) for q in (b[0] - 0.5 * b[2], b[1] - 0.2 * b[3], b[0] + 1.5 * b[2], b[1] + 1.8 * b[3])]; v.append(np.abs(gs[i][max(y0, 0):y1, max(x0, 0):x1] - gs[i - 1][max(y0, 0):y1, max(x0, 0):x1]).mean())
        return np.mean(v) if v else 0
    c = max(cl, key=move); r = (g, cam_box(c[0], c[1], float(np.median([b[2] for _, b in c[3]])), W, H, 3.4)); _REACT[k] = r; return r  # 3.4 face widths: tighter than a streamer cam, the face is the panel

def stack_runs(src, cam, step=4):
    """Edited uploads cut between 'cam over game', a full-screen face, and game with no cam at all.
    Per frame: None = stack, a float = x centre of a full-screen face, "fill" = the cam is gone (no face in its box)."""
    cap = cv2.VideoCapture(str(src)); W = int(cap.get(3)); H = int(cap.get(4)); N = int(cap.get(7)); fps = cap.get(5) or 30
    yn = cv2.FaceDetectorYN.create("onnx", YUNET, np.empty(0, np.uint8), (640, int(640 * H / W)), 0.6); s = 640 / W
    big_min = 1.8 * (cam[2] - cam[0]) / 4.2  # find_facecam's box is 4.2 face widths wide: a face twice the cam face = full screen
    lab, xs, th = [], [], []  # per sample: 0 stack, 1 full-screen face, 2 cam gone; th = small grey thumb of the cam box
    for k in range(0, N, step):
        for _ in range(step - 1 if k else 0): cap.grab()  # read straight through: seeking re-decodes from a keyframe each time (2x slower)
        ok, f = cap.read()
        if not ok: break
        _, fs = yn.detect(cv2.resize(f, (640, int(640 * H / W))))
        b = np.zeros((0, 4)) if fs is None else fs[:, :4] / s
        big = [x + w / 2 for x, y, w, h in b if w >= big_min]
        in_cam = any(cam[0] <= x + w / 2 <= cam[2] and cam[1] <= y + h / 2 <= cam[3] for x, y, w, h in b)
        lab.append(1 if big else 0 if in_cam else 2); xs.append(max(big) if big else None)
        th.append(cv2.resize(cv2.cvtColor(f[cam[1]:cam[3], cam[0]:cam[2]], cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32).ravel())
    cap.release()
    # a face in the cam box is not enough: game cutscenes and loading art have faces too. The box must also still look like the streamer's room
    # (median of the cam frames); when the stream goes full-screen game, it doesn't, and the frame becomes "fill" instead of a stack with a seam
    camk = [k for k, l in enumerate(lab) if l == 0]
    if len(camk) >= 5:
        ref = np.median(np.array([th[k] for k in camk]), axis=0)
        for k in camk:
            if np.corrcoef(th[k], ref)[0, 1] < 0.4: lab[k] = 2
    lab = np.array(lab); win = max(5, int(fps / step) | 1)  # ~1 s majority vote: a glance away or one missed detection never flips the layout
    pad = np.pad(lab, win // 2, mode="edge")
    sm = np.array([np.bincount(pad[i:i + win], minlength=3).argmax() for i in range(len(lab))])
    out, i = [None] * N, 0
    while i < len(sm):  # one fixed layout (and crop) per run, so the frame never wobbles
        j = i
        while j < len(sm) and sm[j] == sm[i]: j += 1
        v = None if sm[i] == 0 else "fill" if sm[i] == 2 else float(np.median([x for x in xs[i:j] if x is not None] or [W / 2]))
        for k in range(i * step, min(j * step, N)): out[k] = v
        i = j
    return out

BUB, BUB_Y = 380, 170  # "bubble" layout: round cam this wide, its top edge (top centre: clear of the Shorts buttons on the right and the bottom UI)
def bubble_mask():
    m = np.zeros((BUB, BUB), np.float32); cv2.circle(m, (BUB // 2, BUB // 2), BUB // 2 - 6, 1, -1, cv2.LINE_AA); return m[..., None]

def render_stack(src, dst, cam, layout="stack"):
    cap = cv2.VideoCapture(str(src)); W = int(cap.get(3)); H = int(cap.get(4)); N = int(cap.get(7)); fps = cap.get(5)
    full = stack_runs(src, cam); cw = int(round(H * 9 / 16)) // 2 * 2
    if layout == "bubble" and not bubble_ok(cam, full, W, H): return None  # no real webcam box (couch shot, trailer face): caller frames it normally
    bh = OH - TOP_H; gy0, gy1 = int(H * 0.03), int(H * 0.96); bw = int((gy1 - gy0) * OW / bh)  # game panel: panel aspect, minus thin top/bottom strips (social footers)
    cx0, _, cx1, _ = cam
    sharpen = OW / (cam[2] - cam[0]) >= 3
    # place the game crop beside the cam so the cam isn't shown twice; centre if the cam is mid-frame
    m = int(0.2 * (cx1 - cx0))  # cam overlays are wider than the face box: keep a margin so no slivers show
    left = (cx0 + cx1) / 2 < W / 2
    gx = min(cx1 + m, W - bw) if left else max(cx0 - m - bw, 0)
    # stream overlays inside the game panel (chat column, counters, cam frame, fine print): slide / shrink the window (<= 22 %) off them
    rg = reaction(src); rg = rg[0] if rg and layout != "bubble" else None  # group reaction: the game window is the game panel, shown whole
    if not rg: gx, gy0, bw, gh = fit(clutter(src, [k for k, v in enumerate(full) if v is None]), (gx, gy0, bw, gy1 - gy0), W, H, (gx, W) if left else (0, gx + bw), (0, H)); gy1 = gy0 + gh
    wins = {}  # full-screen face runs: same trim per run (key = its x centre)
    fb = clutter(src, [k for k, v in enumerate(full) if isinstance(v, float)])
    tmp = Path(dst).with_suffix(".noaudio.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{OW}x{OH}", "-r", str(fps),
        "-i", "-", *encoder(), "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    out = np.zeros((OH, OW, 3), np.uint8); i = 0; G = stack_geom(FLIP); (cy, _), (gy, _), sm = G["cam"], G["game"], G["seam"]
    if layout == "bubble":  # whole frame is game (a 9:16 slice of the game window), the cam a round bubble on top
        fw = int((gy1 - gy0) * OW / OH) // 2 * 2; fx = int(np.clip(gx + (bw - fw) // 2, 0, W - fw)); mask = bubble_mask()
        s = min(cx1 - cx0, cam[3] - cam[1]); sx, sy = (cx0 + cx1 - s) // 2, (cam[1] + cam[3] - s) // 2  # square around the cam centre
    while True:
        ok, f = cap.read()
        if not ok: break
        v = full[i] if i < N else None
        if v == "fill": frame = fill(f, W, H, cw)  # cam gone (edited upload): whole frame, never an empty panel
        elif v is not None:  # full-screen face: plain 9:16 crop on it
            if v not in wins: wins[v] = trim916(fb, [v], W, H)
            frame = crop916(f, v, wins[v])
        elif layout == "bubble":
            out[:] = cv2.resize(f[gy0:gy1, fx:fx + fw], (OW, OH), interpolation=cv2.INTER_CUBIC)
            b = cv2.resize(f[sy:sy + s, sx:sx + s], (BUB, BUB), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            x0, y0 = (OW - BUB) // 2, BUB_Y
            cv2.circle(out, (x0 + BUB // 2, y0 + BUB // 2), BUB // 2 - 1, (255, 255, 255), -1, cv2.LINE_AA)  # white ring = the 6 px the mask leaves
            roi = out[y0:y0 + BUB, x0:x0 + BUB].astype(np.float32)
            out[y0:y0 + BUB, x0:x0 + BUB] = (b * mask + roi * (1 - mask)).astype(np.uint8)
            frame = out
        else:
            top = cv2.resize(f[cam[1]:cam[3], cam[0]:cam[2]], (OW, TOP_H), interpolation=cv2.INTER_CUBIC)
            out[cy:cy + TOP_H] = cv2.addWeighted(top, 1.6, cv2.GaussianBlur(top, (0, 0), 3), -0.6, 0) if sharpen else top  # small cams scaled 3x+ look soft: unsharp mask
            if rg:  # game window full width over a blurred copy of itself (inset 2 % = its frame)
                ix, iy = int(.02 * (rg[2] - rg[0])), int(.02 * (rg[3] - rg[1])); c = f[rg[1] + iy:rg[3] - iy, rg[0] + ix:rg[2] - ix]; fh = min(bh, int(OW * c.shape[0] / c.shape[1]))
                bg = cv2.convertScaleAbs(cv2.resize(cv2.GaussianBlur(cv2.resize(c, (OW // 8, bh // 8)), (0, 0), 4), (OW, bh)), alpha=0.55); y = (bh - fh) // 2; bg[y:y + fh] = cv2.resize(c, (OW, fh), interpolation=cv2.INTER_AREA); out[gy:gy + bh] = bg
            else: out[gy:gy + bh] = cv2.resize(f[gy0:gy1, gx:gx + bw], (OW, bh), interpolation=cv2.INTER_CUBIC)
            out[sm - 3:sm + 3] = 0  # thin seam
            frame = out
        ff.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).tobytes()); i += 1
    cap.release(); ff.stdin.close(); ff.wait()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(src), "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)], check=True)
    tmp.unlink()
    return dict(layout=layout, flip=FLIP and layout != "bubble", cam=cam, game_x=int(gx), full_cam_frames=sum(isinstance(x, float) for x in full), fill_frames=full.count("fill"))

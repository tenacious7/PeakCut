"""Where the facecam is.

A gaming stream is a game with a small webcam picture in one corner of it. The
short that comes out of it is better when the game fills the screen and the
webcam sits in the corner or top split panel, instead of falling back to gameplay.

Detection uses a robust 4-layer pipeline so real streamer webcams are reliably
found (including dark gaming rooms, rounded corners, small corner cams, slow
gameplay cutscenes, and borderless/green-screen cams):
  1. Multi-frame Vision AI (`ask` sampled across 3 spread timestamps) with safe
     verification that never rejects dark-room or active-streamer webcams.
  2. Full-frame + 4-quadrant border & stillness detection (isolates corner
     borders without getting drowned out by 80% full-frame game scenery).
  3. Multi-frame OpenCV Haar face clustering (finds stationary human faces in
     webcam regions even on borderless, rounded, or green-screen overlays).
  4. Corner stillness-blob fallback when gameplay moves around a quiet corner.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

W, H = 320, 180            # the picture is looked at small: only the lines matter
MAX_FRAMES = 24            # frames spread through the video
MIN_SIZE = 0.10            # the smallest a webcam may be, as a part of the width or height
MIN_AI_SIZE = 0.08         # AI-spotted corner webcams can be slightly smaller than 10%
MAX_SIZE = 0.62            # and the largest
ASPECT_MIN = 0.50          # the narrowest: a phone held upright is about 0.56
ASPECT_MAX = 3.20          # the widest: a banner or a ticker is wider than this
STILL_FRAC = 0.42          # how much of the game's own movement a webcam picture may have
STILL_MIN = 8.0            # and the floor, for a game that itself is almost still
STILL_SHARE = 0.40         # at most this much of a webcam picture may be moving, or the game is showing through
MIN_LIGHT = 6.0            # only reject pitch-black bars (< 6/255), never dark gaming rooms
EDGE_FLUSH = 0.02          # a side this close to the frame's own edge is the edge
MATCH = 0.35               # how much a box the AI points at must cover a measured box to be that box
EDGE = 2.0                 # how much brighter than the average line a border must be


def _frames(path: str | Path, count: int = MAX_FRAMES, window: tuple[float, float] | None = None) -> np.ndarray:
    """The video as a few grey frames, spread through it (or inside `window=(start, end)`)."""
    try:
        dur = _duration(path)
        cmd = ["ffmpeg", "-v", "error"]
        span = dur
        if window and len(window) == 2:
            w0, w1 = max(0.0, float(window[0])), float(window[1])
            if dur > 0:
                w1 = min(dur, w1)
            if w1 - w0 >= 1.5:
                cmd += ["-ss", f"{w0:.2f}", "-t", f"{w1 - w0:.2f}"]
                span = w1 - w0
        rate = max(count / span, 0.02) if span > 0 else 0.2
        cmd += ["-i", str(path), "-vf", f"fps={rate:.4f},scale={W}:{H},format=gray",
                "-frames:v", str(count), "-f", "rawvideo", "-"]
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
    except Exception:  # noqa: BLE001 - no ffmpeg or an unreadable file: no facecam, not a failure
        return np.zeros((0, H, W), dtype=np.uint8)
    raw = proc.stdout or b""
    size = W * H
    n = len(raw) // size
    if n < 4:
        if window is not None:
            return _frames(path, count=count, window=None)
        return np.zeros((0, H, W), dtype=np.uint8)
    return np.frombuffer(raw[:n * size], dtype=np.uint8).reshape(n, H, W)


def _duration(path: str | Path) -> float:
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "default=nw=1:nk=1", str(path)],
                              capture_output=True, text=True, timeout=60)
        return float((proc.stdout or "").strip() or 0)
    except Exception:  # noqa: BLE001
        return 0.0


def _profiles(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vertical lines, horizontal lines): how much changes at each column and each row.

    The median over the frames is used, so one busy moment cannot invent a line."""
    if len(frames) < 4:
        return np.zeros(W), np.zeros(H)
    f = frames.astype(np.int16)
    gx = np.median(np.abs(np.diff(f, axis=2)).sum(axis=1), axis=0)   # change across each column boundary
    gy = np.median(np.abs(np.diff(f, axis=1)).sum(axis=2), axis=0)   # change across each row boundary
    return gx, gy


def _peaks(profile: np.ndarray, limit: int, edge_mult: float = EDGE) -> list[tuple[int, float]]:
    """The lines that stand out: (where, how much)."""
    if len(profile) < 3:
        return []
    med = float(np.median(profile)) or 1.0
    out = []
    for i in range(1, len(profile) - 1):
        if profile[i] >= profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > edge_mult * med:
            out.append((i, float(profile[i] / med)))
    return sorted(out, key=lambda p: -p[1])[:limit]


def _size(path: str | Path) -> tuple[int, int]:
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                           "-show_entries", "stream=width,height", "-of", "default=nw=1", str(path)],
                          capture_output=True, text=True, timeout=60)
    w = h = 0
    for line in (proc.stdout or "").splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        try:
            if k.strip() == "width":
                w = int(v.strip())
            elif k.strip() == "height":
                h = int(v.strip())
        except ValueError:
            continue
    return w, h


def _stillness(frames: np.ndarray) -> np.ndarray:
    """How much each pixel changes over the frames.

    A webcam picture barely changes: a person sits in a room and shifts a little. The game changes every
    frame. This tells a webcam apart from game HUDs where the game shows through."""
    return frames.astype(np.float32).std(axis=0)


def _plausible(x0, y0, x1, y1, still, light, game, share=STILL_SHARE, min_size=MIN_SIZE, strict_still=True):
    """Could this rectangle be a webcam picture? Judged on its shape, its stillness and its light.

    Returns how much it moves, or None."""
    bw, bh = (x1 - x0) / W, (y1 - y0) / H
    if not (min_size <= bw <= MAX_SIZE and min_size <= bh <= MAX_SIZE):
        return None
    aspect = (bw * W) / max(1e-6, bh * H)
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return None
    a, b, c, d = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if c <= a or d <= b:
        return None
    inside = still[b:d, a:c]
    if not inside.size:
        return None
    mean_light = float(light[b:d, a:c].mean())
    if mean_light < MIN_LIGHT:
        return None                    # solid black bar / blank screen, not a picture
    moving = float(np.median(inside))
    if strict_still:
        cut = max(STILL_MIN, STILL_FRAC * game)
        if moving > cut:
            return None                # it changes with the game: it is the game
        if float((inside > cut).mean()) > share:
            return None                # a good part of it moves: the game is showing through
    else:
        # Relaxed check for AI-confirmed facecams: only reject pure high-motion game noise
        if game >= 12.0 and moving > max(12.0, 0.78 * game):
            return None
    return moving


def _local(vpeaks, hpeaks, still, light):
    """Every rectangle the strong straight lines can make, that still looks like a camera.

    Returns them best first: the one whose weakest border is the strongest, and of those the bigger."""
    game = float(np.median(still))
    xs = sorted({0.0, float(W)} | {float(x) for x, _ in vpeaks[:10]})
    ys = sorted({0.0, float(H)} | {float(y) for y, _ in hpeaks[:10]})
    vs = {float(x): s for x, s in vpeaks}
    hs = {float(y): s for y, s in hpeaks}
    out = []
    for i, x0 in enumerate(xs):
        wx = [vs[v] for v in (x0,) if v in vs]
        for x1 in xs[i + 1:]:
            wscore = wx + [vs[v] for v in (x1,) if v in vs]
            for j, y0 in enumerate(ys):
                hy0 = [hs[v] for v in (y0,) if v in hs]
                for y1 in ys[j + 1:]:
                    score = wscore + hy0 + [hs[v] for v in (y1,) if v in hs]
                    if len(score) < 2:
                        continue       # a box needs two borders of its own, or it is just the frame
                    moving = _plausible(x0, y0, x1, y1, still, light, game)
                    if moving is None:
                        continue
                    key = (min(score), (x1 - x0) * (y1 - y0))
                    out.append({"key": key, "x": x0, "y": y0, "x1": x1, "y1": y1,
                                "score": round(min(score), 2), "moving": moving, "source": "borders"})
    return sorted(out, key=lambda c: c["key"], reverse=True)


def _quadrant_local(frames: np.ndarray, still: np.ndarray, light: np.ndarray) -> list[dict]:
    """Run border-profile detection inside each of the 4 corner quadrants separately.

    A small or rounded corner webcam only covers ~20% of the full frame height, so full-frame
    gradient sums dilute its borders 5x. Scanning each 55%x55% quadrant boosts corner border SNR."""
    if len(frames) < 4:
        return []
    game = float(np.median(still))
    qw = int(round(W * 0.55))
    qh = int(round(H * 0.55))
    quads = [
        (0, 0, qw, qh),
        (W - qw, 0, W, qh),
        (0, H - qh, qw, H),
        (W - qw, H - qh, W, H),
    ]
    out = []
    f_all = frames.astype(np.int16)
    for qx0, qy0, qx1, qy1 in quads:
        sub = f_all[:, qy0:qy1, qx0:qx1]
        gx = np.median(np.abs(np.diff(sub, axis=2)).sum(axis=1), axis=0)
        gy = np.median(np.abs(np.diff(sub, axis=1)).sum(axis=2), axis=0)
        vp = [(x + qx0, s) for x, s in _peaks(gx, 12, edge_mult=1.8)]
        hp = [(y + qy0, s) for y, s in _peaks(gy, 12, edge_mult=1.8)]
        if not vp or not hp:
            continue
        xs = sorted({float(qx0) if qx0 == 0 else float(qx1)} | {float(x) for x, _ in vp[:8]})
        ys = sorted({float(qy0) if qy0 == 0 else float(qy1)} | {float(y) for y, _ in hp[:8]})
        vs = {float(x): s for x, s in vp}
        hs = {float(y): s for y, s in hp}
        for i, x0 in enumerate(xs):
            wx = [vs[v] for v in (x0,) if v in vs]
            for x1 in xs[i + 1:]:
                wscore = wx + [vs[v] for v in (x1,) if v in vs]
                for j, y0 in enumerate(ys):
                    hy0 = [hs[v] for v in (y0,) if v in hs]
                    for y1 in ys[j + 1:]:
                        score = wscore + hy0 + [hs[v] for v in (y1,) if v in hs]
                        if len(score) < 2:
                            continue
                        moving = _plausible(x0, y0, x1, y1, still, light, game)
                        if moving is None:
                            continue
                        key = (min(score), (x1 - x0) * (y1 - y0))
                        out.append({"key": key, "x": x0, "y": y0, "x1": x1, "y1": y1,
                                    "score": round(min(score), 2), "moving": moving, "source": "quadrant"})
    return sorted(out, key=lambda c: c["key"], reverse=True)


def _merge_candidates(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Combine full-frame and quadrant candidates without near-duplicates."""
    merged = list(primary)
    for cand in secondary:
        cb = (cand["x"] / W, cand["y"] / H, (cand["x1"] - cand["x"]) / W, (cand["y1"] - cand["y"]) / H)
        if not any(_overlap(existing, cb) > 0.70 for existing in merged):
            merged.append(cand)
    return sorted(merged, key=lambda c: c["key"], reverse=True)


def _overlap(a, box) -> float:
    """How much two boxes cover the same ground: 1 when they are the same, 0 when they do not touch."""
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x1"], a["y1"]
    bx, by, bw, bh = box
    bx0, by0, bx1, by1 = bx * W, by * H, (bx + bw) * W, (by + bh) * H
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _full_frame(path: str | Path, t: float | None = None) -> np.ndarray | None:
    """One grey frame at full size, at timestamp `t` (or the middle of the video): where borders are measured."""
    w, h = _size(path)
    if not w or not h:
        return None
    ts = max(0.0, float(t) if t is not None else _duration(path) / 2.0)
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{ts:.2f}", "-i", str(path),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, timeout=120).stdout or b""
    except Exception:  # noqa: BLE001
        return None
    if len(raw) < w * h:
        return None
    return np.frombuffer(raw[:w * h], dtype=np.uint8).reshape(h, w).astype(np.float32)


def _best(profile: np.ndarray, approx: float, span: float, min_ratio: float = 0.0) -> float:
    """The strongest change near a guessed line (optionally requiring a real peak above local median)."""
    lo, hi = max(0, int(approx - span)), min(len(profile), int(approx + span) + 1)
    seg = profile[lo:hi]
    if not len(seg):
        return float(approx)
    idx = int(seg.argmax())
    if min_ratio > 0.0:
        med = float(np.median(seg)) or 1.0
        if float(seg[idx]) < min_ratio * med:
            return float(approx)
    return float(lo + idx)


def _trim_moving_margins(box: tuple[float, float, float, float], still: np.ndarray) -> tuple[float, float, float, float]:
    """Shrink outer rows/columns of a candidate box that move like high-motion gameplay rather than a webcam.

    Guards against collapsing active facecams in slow cutscenes by requiring moving gameplay (`game >= 10`)
    and refusing to shrink below 55% of the original box size."""
    x0, y0, x1, y1 = int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))
    orig_w, orig_h = x1 - x0, y1 - y0
    min_w = max(4, int(round(W * MIN_SIZE)), int(round(orig_w * 0.55)))
    min_h = max(4, int(round(H * MIN_SIZE)), int(round(orig_h * 0.55)))
    game = float(np.median(still))
    if game < 10.0:
        return (float(x0), float(y0), float(x1), float(y1))
    cut = max(10.0, 0.55 * game)

    changed = True
    while changed:
        changed = False
        if y1 - y0 > min_h and float((still[y0, x0:x1] > cut).mean()) > 0.55:
            y0 += 1
            changed = True
        if y1 - y0 > min_h and float((still[y1 - 1, x0:x1] > cut).mean()) > 0.55:
            y1 -= 1
            changed = True
        if x1 - x0 > min_w and float((still[y0:y1, x0] > cut).mean()) > 0.55:
            x0 += 1
            changed = True
        if x1 - x0 > min_w and float((still[y0:y1, x1 - 1] > cut).mean()) > 0.55:
            x1 -= 1
            changed = True
    return (float(x0), float(y0), float(x1), float(y1))


def _snap(path: str | Path, box, w: int, h: int, span: float = 0.0, t: float | None = None, require_peak: bool = False):
    """A box on the small grid, put onto the borders that are really in the picture.

    Gradients are summed ONLY along the box's own local row/column span so distant game scenery
    (trees, sky, HUD bars outside the webcam) cannot pull a webcam edge outward."""
    frame = _full_frame(path, t=t)
    if frame is None:
        return None
    fh, fw = frame.shape
    x0_s, y0_s, x1_s, y1_s = box
    ax0, ay0 = x0_s * fw / W, y0_s * fh / H
    ax1, ay1 = x1_s * fw / W, y1_s * fh / H

    pad_x = max(2, int(round((ax1 - ax0) * 0.10)))
    pad_y = max(2, int(round((ay1 - ay0) * 0.10)))
    ry0 = max(0, min(fh - 2, int(round(ay0)) + pad_y))
    ry1 = max(ry0 + 2, min(fh, int(round(ay1)) - pad_y))
    rx0 = max(0, min(fw - 2, int(round(ax0)) + pad_x))
    rx1 = max(rx0 + 2, min(fw, int(round(ax1)) - pad_x))

    col = np.abs(np.diff(frame[ry0:ry1, :], axis=1)).sum(axis=0)
    row = np.abs(np.diff(frame[:, rx0:rx1], axis=0)).sum(axis=1)
    spread = max(4.0, span * max(fw, fh), 1.5 * max(fw / W, fh / H))
    peak_ratio = 1.6 if require_peak else 0.0

    def line(profile, approx, edge, extent):
        if abs(approx - edge) <= EDGE_FLUSH * extent:
            return float(edge)
        return _best(profile, approx, spread, min_ratio=peak_ratio)

    return (line(col, ax0, 0, fw), line(row, ay0, 0, fh),
            line(col, ax1, fw, fw), line(row, ay1, fh, fh))


def _corner(x0, y0, x1, y1, w, h) -> str:
    """What part of the frame the box is on: its corner, its edge, or the middle."""
    near = EDGE_FLUSH
    left, right = x0 <= w * near, x1 >= w * (1 - near)
    top, bottom = y0 <= h * near, y1 >= h * (1 - near)
    if left and top:
        return "top-left"
    if right and top:
        return "top-right"
    if left and bottom:
        return "bottom-left"
    if right and bottom:
        return "bottom-right"
    if left or right:
        return "right" if right else "left"
    if top or bottom:
        return "bottom" if bottom else "top"
    return "middle"


def _finish(path: str | Path, found: dict, w: int, h: int, still: np.ndarray,
            trim_moving: bool = False, t: float | None = None, require_peak: bool = False,
            min_size: float = MIN_SIZE, fallback_to_raw: bool = False) -> dict | None:
    """The box in source pixels, with how sure it is that this is really a camera."""
    raw_box = (found["x"], found["y"], found["x1"], found["y1"])
    if trim_moving:
        raw_box = _trim_moving_margins(raw_box, still)
    snapped = _snap(path, raw_box, w, h, t=t, require_peak=require_peak)
    if snapped:
        x0, y0, x1, y1 = snapped
    elif fallback_to_raw:
        x0, y0, x1, y1 = raw_box[0] * w / W, raw_box[1] * h / H, raw_box[2] * w / W, raw_box[3] * h / H
    else:
        return None
    if x1 - x0 < w * min_size or y1 - y0 < h * min_size:
        if fallback_to_raw:
            x0, y0, x1, y1 = raw_box[0] * w / W, raw_box[1] * h / H, raw_box[2] * w / W, raw_box[3] * h / H
            if x1 - x0 < w * min_size or y1 - y0 < h * min_size:
                return None
        else:
            return None
    x0 = max(0.0, min(float(w - 20), x0))
    y0 = max(0.0, min(float(h - 20), y0))
    x1 = max(x0 + 20.0, min(float(w), x1))
    y1 = max(y0 + 20.0, min(float(h), y1))
    inside = still[max(0, int(raw_box[1])):min(H, int(raw_box[3])), max(0, int(raw_box[0])):min(W, int(raw_box[2]))]
    game = float(np.median(still))
    moving = float(np.median(inside)) if inside.size else 99.0
    conf = "high" if (moving <= max(4.0, 0.30 * game) or found.get("source") in ("ai", "opencv_face")) else "low"
    return {"x": int(round(x0)), "y": int(round(y0)), "w": int(round(x1 - x0)), "h": int(round(y1 - y0)),
            "corner": _corner(x0, y0, x1, y1, w, h), "score": found.get("score", 0.0), "conf": conf,
            "source": found.get("source", "borders")}


def _frame_jpeg(path: str | Path, width: int = 960, t: float | None = None) -> bytes:
    """One frame of the stream as a crisp JPEG: what a vision model is shown."""
    ts = max(0.0, float(t) if t is not None else _duration(path) / 2.0)
    try:
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{ts:.2f}", "-i", str(path),
             "-frames:v", "1", "-vf", f"scale={width}:-2", "-pix_fmt", "yuvj420p", "-q:v", "3",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True, timeout=120).stdout or b""
    except Exception:  # noqa: BLE001
        return b""


def _sample_times(path: str | Path, count: int = 3, window: tuple[float, float] | None = None) -> list[float]:
    """Spread sample timestamps inside `window` or across the full video (midpoint first)."""
    dur = _duration(path)
    t0, t1 = 0.0, max(0.0, dur)
    if window and len(window) == 2:
        w0, w1 = max(0.0, float(window[0])), float(window[1])
        if dur > 0:
            w1 = min(dur, w1)
        if w1 - w0 >= 1.0:
            t0, t1 = w0, w1
    span = max(0.0, t1 - t0)
    if span < 2.0 or count <= 1:
        return [round(t0 + span * 0.5, 2)]
    if count == 3:
        return [round(t0 + span * f, 2) for f in (0.50, 0.25, 0.75)]
    return [round(t0 + span * (i + 1) / (count + 1), 2) for i in range(count)]


def _asked(path: str | Path, ask, window: tuple[float, float] | None = None):
    """Show up to 3 spread frames to the vision model.

    Stops on the first frame where a webcam is found, or immediately if the model cannot look.
    Sampling 3 timestamps prevents a single cutscene/loading screen at midpoint from killing detection."""
    times = _sample_times(path, count=3, window=window)
    ever_looked = False
    for ts in times:
        jpeg = _frame_jpeg(path, t=ts)
        if not jpeg:
            continue
        try:
            answer = ask(jpeg) or {}
        except Exception:  # noqa: BLE001 - a model that cannot look is not a failure
            return False, None, None
        looked = bool(answer.get("looked"))
        box = answer.get("box")
        if not looked:
            return False, None, None
        ever_looked = True
        if box:
            return True, box, ts
    return ever_looked, None, None


def _seen(path: str | Path, box, w: int, h: int, still: np.ndarray, light: np.ndarray,
          t: float | None = None) -> dict | None:
    """Verify and refine a box the vision model proposed.

    Unlike blind border detection, the model already confirmed a real person is here — so we allow
    dark gaming rooms and talking/moving streamers, only rejecting obviously bogus tiny/noise boxes."""
    fx, fy, fw, fh = (float(v) for v in box)
    if not (MIN_AI_SIZE <= fw <= MAX_SIZE and MIN_AI_SIZE <= fh <= MAX_SIZE):
        return None
    raw_small = (max(0.0, fx * W), max(0.0, fy * H), min(float(W), (fx + fw) * W), min(float(H), (fy + fh) * H))
    game = float(np.median(still))
    # Reject bogus tiny/noise boxes (e.g. pointing at pure moving gameplay with no stillness)
    if _plausible(*raw_small, still, light, game, share=1.0, min_size=MIN_AI_SIZE, strict_still=False) is None:
        return None
    trimmed = _trim_moving_margins(raw_small, still)
    snapped = _snap(path, trimmed, w, h, span=0.06, t=t, require_peak=True)
    if snapped:
        x0, y0, x1, y1 = snapped
        # If snapping collapsed or distorted the box wildly, fall back to the trimmed AI box
        if (x1 - x0) < w * MIN_AI_SIZE or (y1 - y0) < h * MIN_AI_SIZE:
            x0, y0, x1, y1 = trimmed[0] * w / W, trimmed[1] * h / H, trimmed[2] * w / W, trimmed[3] * h / H
    else:
        x0, y0, x1, y1 = trimmed[0] * w / W, trimmed[1] * h / H, trimmed[2] * w / W, trimmed[3] * h / H
    small = (x0 * W / w, y0 * H / h, x1 * W / w, y1 * H / h)
    if _plausible(*small, still, light, game, share=1.0, min_size=MIN_AI_SIZE, strict_still=False) is None:
        return None
    found = {"x": small[0], "y": small[1], "x1": small[2], "y1": small[3], "score": 0.0, "source": "ai"}
    return _finish(path, found, w, h, still, trim_moving=False, t=t, require_peak=True,
                   min_size=MIN_AI_SIZE, fallback_to_raw=True)


def _detect_still_corner(path: str | Path, w: int, h: int, still: np.ndarray, light: np.ndarray,
                         corner: str | None = None, t: float | None = None) -> dict | None:
    """Find a quiet rectangular webcam region in a corner when gameplay is moving around it.

    Works even on rounded, dark-room, or soft-bordered webcams that lack sharp straight edge peaks."""
    game = float(np.median(still))
    if game < 10.0:
        return None
    cut = max(STILL_MIN, 0.45 * game)
    quiet = (still <= cut) & (light >= MIN_LIGHT)

    qw, qh = int(round(W * 0.46)), int(round(H * 0.48))
    regions = [
        ("top-left", 0, 0, qw, qh),
        ("top-right", W - qw, 0, W, qh),
        ("bottom-left", 0, H - qh, qw, H),
        ("bottom-right", W - qw, H - qh, W, H),
    ]
    if corner:
        regions = [r for r in regions if r[0] == corner]

    best_cand = None
    best_score = 0.0
    for _name, rx0, ry0, rx1, ry1 in regions:
        sub = quiet[ry0:ry1, rx0:rx1]
        if float(sub.mean()) < 0.08:
            continue
        # Find rows and columns that are predominantly quiet
        row_ok = np.where(sub.mean(axis=1) > 0.30)[0]
        col_ok = np.where(sub.mean(axis=0) > 0.30)[0]
        if len(row_ok) < int(H * MIN_SIZE) or len(col_ok) < int(W * MIN_SIZE):
            continue
        y0, y1 = float(ry0 + row_ok[0]), float(ry0 + row_ok[-1] + 1)
        x0, x1 = float(rx0 + col_ok[0]), float(rx0 + col_ok[-1] + 1)
        trimmed = _trim_moving_margins((x0, y0, x1, y1), still)
        tx0, ty0, tx1, ty1 = trimmed
        moving = _plausible(tx0, ty0, tx1, ty1, still, light, game, share=0.45, min_size=MIN_SIZE, strict_still=True)
        if moving is None:
            continue
        quiet_frac = float(quiet[int(ty0):int(ty1), int(tx0):int(tx1)].mean())
        if quiet_frac < 0.55:
            continue
        area = (tx1 - tx0) * (ty1 - ty0)
        sc = quiet_frac * area / max(1.0, moving + 1.0)
        if sc > best_score:
            best_score = sc
            best_cand = {"x": tx0, "y": ty0, "x1": tx1, "y1": ty1, "score": round(quiet_frac * 5.0, 2),
                         "moving": moving, "source": "still_corner"}
    if not best_cand:
        return None
    return _finish(path, best_cand, w, h, still, trim_moving=False, t=t, require_peak=True, fallback_to_raw=True)


def _detect_by_faces(path: str | Path, w: int, h: int, still: np.ndarray,
                     window: tuple[float, float] | None = None,
                     corner: str | None = None) -> dict | None:
    """Multi-frame OpenCV face clustering: locates a stationary streamer face across spread frames.

    Real streamer webcams stay in the same corner/edge across frames, whereas in-game character faces
    move around or disappear. Works even on borderless, rounded, dark-room, or green-screen webcams."""
    if cv2 is None:
        return None
    det = _get_cascade()
    if det is None:
        return None
    prof_det = _get_profile_cascade()
    times = _sample_times(path, count=8, window=window)
    if not times:
        return None
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        detections = []  # (cx_norm, cy_norm, fw_norm, fh_norm, ts)
        for ts in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts * 1000.0))
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            fh0, fw0 = frame.shape[:2]
            scale = min(1.0, 960.0 / max(1, fw0))
            if scale < 1.0:
                small = cv2.resize(frame, (int(round(fw0 * scale)), int(round(fh0 * scale))))
            else:
                small = frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
            gray = cv2.equalizeHist(gray)
            sh0, sw0 = gray.shape[:2]
            min_f = max(20, int(round(sw0 * 0.025)))
            max_f = max(min_f + 10, int(round(sw0 * 0.30)))
            faces = list(det.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                              minSize=(min_f, min_f), maxSize=(max_f, max_f)))
            if not faces and prof_det is not None:
                faces = list(prof_det.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                                       minSize=(min_f, min_f), maxSize=(max_f, max_f)))
            for fx, fy, fw_p, fh_p in faces:
                cx = (fx + fw_p / 2.0) / sw0
                cy = (fy + fh_p / 2.0) / sh0
                wn = fw_p / sw0
                hn = fh_p / sh0
                if corner:
                    want_left = "left" in corner
                    want_top = "top" in corner
                    if (want_left and cx > 0.55) or (not want_left and cx < 0.45):
                        continue
                    if (want_top and cy > 0.55) or (not want_top and cy < 0.45):
                        continue
                detections.append((cx, cy, wn, hn, ts))
    except Exception:
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    if len(detections) < 2:
        return None

    # Cluster stationary face detections (within 6% of frame width/height)
    clusters: list[list[tuple]] = []
    for d in detections:
        placed = False
        for cl in clusters:
            mcx = float(np.median([p[0] for p in cl]))
            mcy = float(np.median([p[1] for p in cl]))
            if abs(d[0] - mcx) <= 0.065 and abs(d[1] - mcy) <= 0.08:
                cl.append(d)
                placed = True
                break
        if not placed:
            clusters.append([d])

    # Keep clusters seen in at least 2 distinct frames
    valid = []
    for cl in clusters:
        uniq_t = {p[4] for p in cl}
        if len(uniq_t) < 2:
            continue
        mcx = float(np.median([p[0] for p in cl]))
        mcy = float(np.median([p[1] for p in cl]))
        mfw = float(np.median([p[2] for p in cl]))
        mfh = float(np.median([p[3] for p in cl]))
        # Prefer faces located in corners or along outer edges (typical streamer webcam positions)
        in_edge = (mcx <= 0.36 or mcx >= 0.64 or mcy <= 0.36 or mcy >= 0.64)
        if not in_edge and len(uniq_t) < 4:
            continue
        in_corner = (mcx <= 0.38 or mcx >= 0.62) and (mcy <= 0.40 or mcy >= 0.60)
        score = len(uniq_t) * (2.0 if in_corner else 1.25 if in_edge else 0.7)
        valid.append((score, len(uniq_t), mcx, mcy, mfw, mfh, cl[0][4]))

    if not valid:
        return None
    valid.sort(key=lambda item: (-item[0], -item[4] * item[5]))
    best_score, hits, cx, cy, fw_n, fh_n, sample_t = valid[0]

    # Expand natural webcam box around the stationary face (head + shoulders + webcam framing)
    cam_w = max(MIN_SIZE, min(0.42, fw_n * 3.0))
    cam_h = max(MIN_SIZE, min(0.45, fh_n * 2.5))
    bx0 = max(0.0, cx - cam_w * 0.50)
    by0 = max(0.0, cy - cam_h * 0.42)
    bx1 = min(1.0, bx0 + cam_w)
    by1 = min(1.0, by0 + cam_h)
    bx0 = max(0.0, bx1 - cam_w)
    by0 = max(0.0, by1 - cam_h)

    # Flush-snap to frame edge when the expanded cam sits very close to a border
    if bx0 < 0.04:
        bx0 = 0.0
    if bx1 > 0.96:
        bx1 = 1.0
    if by0 < 0.04:
        by0 = 0.0
    if by1 > 0.96:
        by1 = 1.0

    raw_small = (bx0 * W, by0 * H, bx1 * W, by1 * H)
    trimmed = _trim_moving_margins(raw_small, still)
    snapped = _snap(path, trimmed, w, h, span=0.06, t=sample_t, require_peak=True)
    if snapped:
        x0, y0, x1, y1 = snapped
        # Ensure the snapped box still contains the detected face center!
        fcx_px, fcy_px = cx * w, cy * h
        if not (x0 <= fcx_px <= x1 and y0 <= fcy_px <= y1) or (x1 - x0) < w * MIN_SIZE or (y1 - y0) < h * MIN_SIZE:
            x0, y0, x1, y1 = trimmed[0] * w / W, trimmed[1] * h / H, trimmed[2] * w / W, trimmed[3] * h / H
    else:
        x0, y0, x1, y1 = trimmed[0] * w / W, trimmed[1] * h / H, trimmed[2] * w / W, trimmed[3] * h / H

    found = {"x": x0 * W / w, "y": y0 * H / h, "x1": x1 * W / w, "y1": y1 * H / h,
             "score": round(best_score, 2), "source": "opencv_face"}
    res = _finish(path, found, w, h, still, trim_moving=False, t=sample_t, require_peak=True, fallback_to_raw=True)
    if res and hits >= 3:
        res["conf"] = "high"
    return res


def detect_in_corner(path: str | Path, corner: str, window: tuple[float, float] | None = None) -> dict | None:
    """Find the tight webcam box inside a user-specified corner (top-left, top-right, bottom-left, bottom-right).

    Prevents static corner presets from grabbing huge strips of surrounding gameplay."""
    frames = _frames(path, window=window)
    if len(frames) < 4:
        return None
    w, h = _size(path)
    if not w or not h:
        return None
    still = _stillness(frames)
    light = frames.astype(np.float32).mean(axis=0)
    vx, hy = _profiles(frames)
    vpeaks, hpeaks = _peaks(vx, 40), _peaks(hy, 40)
    found = _merge_candidates(
        _local(vpeaks, hpeaks, still, light) if (vpeaks and hpeaks) else [],
        _quadrant_local(frames, still, light),
    )

    want_left = "left" in corner
    want_top = "top" in corner

    def _in_quadrant(c: dict) -> bool:
        cx = (c["x"] + c["x1"]) / 2.0
        cy = (c["y"] + c["y1"]) / 2.0
        ok_x = cx < W * 0.55 if want_left else cx > W * 0.45
        ok_y = cy < H * 0.55 if want_top else cy > H * 0.45
        return ok_x and ok_y

    in_q = [c for c in found if _in_quadrant(c)]
    if in_q:
        res = _finish(path, in_q[0], w, h, still, trim_moving=True)
        if res:
            return res

    # Try OpenCV face clustering inside the requested corner
    face_box = _detect_by_faces(path, w, h, still, window=window, corner=corner)
    if face_box:
        return face_box

    # Try corner stillness blob
    blob = _detect_still_corner(path, w, h, still, light, corner=corner)
    if blob:
        return blob

    # Fallback: start from a generous corner quadrant and peel away moving gameplay pixels
    qx0, qx1 = (0.0, W * 0.40) if want_left else (W * 0.60, float(W))
    qy0, qy1 = (0.0, H * 0.45) if want_top else (H * 0.55, float(H))
    trimmed = _trim_moving_margins((qx0, qy0, qx1, qy1), still)
    if (trimmed[2] - trimmed[0]) >= W * MIN_SIZE and (trimmed[3] - trimmed[1]) >= H * MIN_SIZE:
        res = _finish(path, {"x": trimmed[0], "y": trimmed[1], "x1": trimmed[2], "y1": trimmed[3], "score": 0.0},
                      w, h, still, trim_moving=False, require_peak=True, fallback_to_raw=True)
        if res:
            return res
    return None


def detect(path: str | Path, ask=None, window: tuple[float, float] | None = None) -> dict | None:
    """The facecam's box in the source, in source pixels, or None.

    The box is given as x, y, width, height, with the confidence in "score" and
    how sure we are that it is a camera in "conf".

    Uses 4 robust layers:
      1. Multi-frame Vision AI (`ask` sampled across up to 3 timestamps)
      2. Full-frame + 4-quadrant border & stillness detection
      3. Multi-frame OpenCV face clustering (handles borderless / rounded / dark / green-screen cams)
      4. Corner stillness-blob fallback
    """
    frames = _frames(path, window=window)
    if len(frames) < 4:
        return None
    w, h = _size(path)
    if not w or not h:
        return None
    still = _stillness(frames)
    light = frames.astype(np.float32).mean(axis=0)
    mid_t = _sample_times(path, count=1, window=window)[0]

    looked, hinted, t_found = (False, None, None)
    if ask is not None:
        looked, hinted, t_found = _asked(path, ask, window=window)
        if looked and not hinted:
            # AI saw NO CAM across all sampled frames. Only override if OpenCV confirms a
            # stationary real human face across multiple frames (synthetic non-face videos return None).
            face_override = _detect_by_faces(path, w, h, still, window=window)
            if face_override and face_override.get("conf") == "high":
                return face_override
            return None

    vx, hy = _profiles(frames)
    vpeaks, hpeaks = _peaks(vx, 40), _peaks(hy, 40)
    found = _merge_candidates(
        _local(vpeaks, hpeaks, still, light) if (vpeaks and hpeaks) else [],
        _quadrant_local(frames, still, light),
    )

    if looked and hinted:
        match = max(found, key=lambda c: _overlap(c, hinted)) if found else None
        if match and _overlap(match, hinted) >= MATCH:
            res = _finish(path, match, w, h, still, trim_moving=True, t=t_found or mid_t)
            if res:
                return res
        seen = _seen(path, hinted, w, h, still, light, t=t_found or mid_t)
        if seen:
            return seen

    # If multiple border candidates exist, prefer one backed by an OpenCV-detected stationary face
    face_box = None
    if found:
        if len(found) > 1 and cv2 is not None:
            face_box = _detect_by_faces(path, w, h, still, window=window)
            if face_box:
                fb_norm = (face_box["x"] / w, face_box["y"] / h, face_box["w"] / w, face_box["h"] / h)
                best_face_match = max(found, key=lambda c: _overlap(c, fb_norm))
                if _overlap(best_face_match, fb_norm) >= 0.25:
                    res = _finish(path, best_face_match, w, h, still, trim_moving=True, t=mid_t)
                    if res:
                        return res
        res = _finish(path, found[0], w, h, still, trim_moving=True, t=mid_t)
        if res:
            return res

    # Layer 3: Multi-frame OpenCV stationary face clustering (catches borderless / rounded / dark / green-screen cams)
    if face_box is None:
        face_box = _detect_by_faces(path, w, h, still, window=window)
    if face_box:
        return face_box

    # Layer 4: Corner stillness blob fallback (when gameplay moves around a rounded/dark quiet corner)
    return _detect_still_corner(path, w, h, still, light, t=mid_t)


try:
    import cv2
except ImportError:
    cv2 = None

_face_cascade = None
_profile_cascade = None


def _get_cascade():
    global _face_cascade
    if _face_cascade is None and cv2 is not None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            det = cv2.CascadeClassifier(cascade_path)
            _face_cascade = det if not det.empty() else False
        except Exception:
            _face_cascade = False
    return _face_cascade if _face_cascade is not False else None


def _get_profile_cascade():
    global _profile_cascade
    if _profile_cascade is None and cv2 is not None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
            det = cv2.CascadeClassifier(cascade_path)
            _profile_cascade = det if not det.empty() else False
        except Exception:
            _profile_cascade = False
    return _profile_cascade if _profile_cascade is not False else None


def face_center(frame: np.ndarray, cam: tuple | list) -> tuple[int, int] | None:
    """Detect dominant face center (cx, cy) inside webcam box in source pixels.
    Returns (cx, cy) or None if no face is detected."""
    if cv2 is None or frame is None or not cam or len(cam) != 4:
        return None
    det = _get_cascade()
    if det is None:
        return None
    try:
        x, y, w, h = (int(v) for v in cam)
        fh_max, fw_max = frame.shape[:2]
        x0 = max(0, min(fw_max - 1, x))
        y0 = max(0, min(fh_max - 1, y))
        w0 = max(1, min(fw_max - x0, w))
        h0 = max(1, min(fh_max - y0, h))
        sub = frame[y0:y0+h0, x0:x0+w0]
        if sub.size == 0 or w0 < 20 or h0 < 20:
            return None
        if len(sub.shape) == 3:
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        else:
            gray = sub
        min_dim = max(20, int(min(w0, h0) * 0.15))
        faces = det.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(min_dim, min_dim))
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        return (int(round(x0 + fx + fw / 2)), int(round(y0 + fy + fh / 2)))
    except Exception:
        return None


def find_clip_face_focus(src_path: str | Path, cam: tuple | list, start: float, end: float, count: int = 4) -> list[int] | None:
    """Sample count frames in [start, end], detect face center, and return median [cx, cy] or None."""
    if cv2 is None or not cam or len(cam) != 4:
        return None
    try:
        dur = max(0.5, float(end - start))
        pts = [start + dur * (i + 1) / (count + 1) for i in range(count)]
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            return None
        centers = []
        for t in pts:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            fc = face_center(frame, cam)
            if fc:
                centers.append(fc)
        cap.release()
        if not centers:
            return None
        med_x = int(round(float(np.median([c[0] for c in centers]))))
        med_y = int(round(float(np.median([c[1] for c in centers]))))
        return [med_x, med_y]
    except Exception:
        return None

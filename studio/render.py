"""Turns a time range into a vertical clip: 9:16 layout, bouncy word captions, hook title, cover."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import config, edit, roman
from .media import NO_WINDOW, run

W, H = config.OUT_W, config.OUT_H
DEFAULT_SPLIT_RATIO = 0.35
DEFAULT_SPLIT_DIVIDER = True
STACK_CAM_INSET = 0.035   # trims ~3.5% off webcam edges to remove rounded bezels / corner bleed

# ASS colours are &HBBGGRR. `size` is the ASS font size (line height); _Font turns it into real pixels.
STYLES = {
    "bounce": {"label": "Bounce", "font": "Poppins ExtraBold", "size": 132, "base": "FFFFFF", "active": "00F0FF",
               "strong": "FFFFFF", "outline": 7, "pop": 112, "upper": True, "per": 3},
    "hype": {"label": "Hype", "font": "Anton", "size": 152, "base": "FFFFFF", "active": "3C3CFF",
             "strong": "00D7FF", "outline": 7, "pop": 114, "upper": True, "per": 2},
    "clean": {"label": "Clean", "font": "Poppins ExtraBold", "size": 100, "base": "FFFFFF", "active": "6BD7FF",
              "strong": "FFFFFF", "outline": 5, "pop": 106, "upper": False, "per": 4},
    "beast": {"label": "Beast (Viral)", "font": "Anton", "size": 160, "base": "FFFFFF", "active": "00D4FF",
              "strong": "FFD400", "outline": 8, "pop": 115, "upper": True, "per": 2},
    "word": {"label": "One Word (Speed)", "font": "Anton", "size": 190, "base": "FFFFFF", "active": "00E5FF",
             "strong": "FFE500", "outline": 9, "pop": 120, "upper": True, "per": 1},
    "pop": {"label": "Gaming Pop", "font": "Lilita One", "size": 130, "base": "FFFFFF", "active": "0AD6FF",
            "strong": "FFD60A", "outline": 7, "pop": 110, "upper": True, "per": 3},
    "neon": {"label": "Neon Cyan", "font": "Poppins ExtraBold", "size": 130, "base": "FFFFFF", "active": "EED322",
             "strong": "22D3EE", "outline": 7, "pop": 110, "upper": False, "per": 3},
}
HOOK = {"font": "Montserrat ExtraBold", "size": 96, "ink": "000000", "box": "FFFFFF", "pad_x": 42, "pad_y": 24, "radius": 28,
        "max_w": 980, "tracking": 0.032}
FONT_FILES = {
    "Poppins ExtraBold": "Poppins-ExtraBold.ttf",
    "Anton": "Anton-Regular.ttf",
    "Archivo Black": "ArchivoBlack-Regular.ttf",
    "Lilita One": "LilitaOne-Regular.ttf",
    "Montserrat": "Montserrat-ExtraBold.ttf",
    "Montserrat ExtraBold": "Montserrat-ExtraBold.ttf",
    "Bebas Neue": "BebasNeue-Regular.ttf",
    "Righteous": "Righteous-Regular.ttf"
}
CAP_MAX_W = 960   # captions never run wider than this, whatever is said


class _Font:
    """Measures text exactly as the renderer will draw it, so every word can be placed by hand: the highlighted
    word can grow without pushing its neighbours, and the heading box fits its text."""
    _cache: dict = {}

    def __init__(self, name):
        import struct
        path = Path(__file__).resolve().parent / "fonts" / FONT_FILES.get(name, FONT_FILES["Poppins ExtraBold"])
        d = path.read_bytes()
        tables = {}
        for i in range(struct.unpack(">H", d[4:6])[0]):
            tag, _, off, _ = struct.unpack(">4sIII", d[12 + 16 * i: 28 + 16 * i])
            tables[tag] = off
        self.upm = struct.unpack(">H", d[tables[b"head"] + 18: tables[b"head"] + 20])[0]
        o = tables[b"OS/2"]
        self.asc, self.desc = struct.unpack(">HH", d[o + 74: o + 78])      # what the renderer sizes the font by
        self.cap = struct.unpack(">h", d[o + 88: o + 90])[0] or int(self.upm * 0.7)
        self.path = str(path)
        self._pil = {}

    @classmethod
    def get(cls, name):
        if name not in cls._cache:
            cls._cache[name] = cls(name)
        return cls._cache[name]

    def em(self, size):
        return size * self.upm / (self.asc + self.desc)

    def width(self, text, size):
        """Pixels wide at this ASS size."""
        try:
            from PIL import ImageFont
            if "f" not in self._pil:
                self._pil["f"] = ImageFont.truetype(self.path, 200)
            return self._pil["f"].getlength(text) * self.em(size) / 200
        except Exception:  # noqa: BLE001 - no measuring library: a fair estimate keeps clips rendering
            return len(text) * self.em(size) * 0.62

    def cap_shift(self, size):
        """How far below the anchor the middle of a capital letter sits when text is centred on a point."""
        return ((self.asc - self.desc) / 2 - self.cap / 2) * size / (self.asc + self.desc)


CAPTION_LEAD = 0.0  # measured: word stamps are within 0.15 s of the voice, so no shift is needed
def install_fonts():
    """Copy the bundled caption fonts next to the data, so captions look the same on every machine and server."""
    src = Path(__file__).resolve().parent / "fonts"
    for f in src.glob("*.ttf"):
        if not (config.FONTS_DIR / f.name).exists():
            shutil.copyfile(f, config.FONTS_DIR / f.name)


LAYOUTS = {"blur": "Full frame + blurred fill", "zoom": "Zoomed in + blurred fill", "crop": "Fill the screen (centre crop)",
           "gameplay": "Gameplay fills the screen, facecam in the corner",
           "stack": "Split screen: Facecam on top, gameplay below"}


def _ts(t):
    t = max(0.0, t)
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def _clean(text):
    return re.sub(r"[\x00-\x1f\x7f]+", " ", re.sub(r"[{}\\]", "", str(text))).strip()


def _key(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def _groups(words, per):
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= per or sum(len(x["w"]) for x in cur) + len(w["w"]) > 16
                    or w["s"] - cur[-1]["e"] > 0.45 or cur[-1]["w"][-1:] in ".?!"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def stack_layout_metrics(out_w: int = None, out_h: int = None, split_ratio: float = DEFAULT_SPLIT_RATIO,
                         cam_swap: bool = False, divider: bool = DEFAULT_SPLIT_DIVIDER) -> dict:
    """Compute proportional split-screen section geometry dynamically from output resolution."""
    ow = max(2, int(out_w or config.OUT_W or W))
    oh = max(2, int(out_h or config.OUT_H or H))
    ow -= ow % 2
    oh -= oh % 2
    ratio = max(0.20, min(0.80, float(DEFAULT_SPLIT_RATIO if split_ratio is None else split_ratio)))

    cam_h = max(2, min(oh - 2, int(round(oh * ratio / 2.0)) * 2))
    game_h = oh - cam_h

    if cam_swap:
        game_top, game_bottom = 0, game_h
        cam_top, cam_bottom = game_h, oh
        seam_y = game_h
    else:
        cam_top, cam_bottom = 0, cam_h
        game_top, game_bottom = cam_h, oh
        seam_y = cam_h

    divider_h = max(2, int(round(oh * 0.003 / 2.0)) * 2) if divider else 0
    half_div = divider_h / 2.0
    pad = max(10, int(round(oh * 0.012)))

    # Place Big Text cleanly below the split seam (inside gameplay) so it never covers the YouTuber's facecam
    hook_offset = int(round(oh * 0.048))
    if cam_swap:
        hook_bounds = (float(game_top + pad), float(game_bottom - half_div - pad))
        hook_y = float(max(hook_bounds[0], game_bottom - half_div - hook_offset))
    else:
        hook_bounds = (float(game_top + half_div + pad), float(game_bottom - pad))
        hook_y = float(min(hook_bounds[1], game_top + half_div + hook_offset))

    # Keep subtitles centered inside the gameplay section with clean bottom spacing (~15% margin on default 35/65)
    cap_frac = 0.80 if cam_swap else 0.77
    cap_y = round(game_top + game_h * cap_frac, 1)
    cap_bounds = (
        float(game_top + max(12, int(round(game_h * 0.10)))),
        float(game_bottom - max(12, int(round(game_h * 0.09)))),
    )

    return {
        "out_w": ow,
        "out_h": oh,
        "ratio": ratio,
        "cam_h": cam_h,
        "game_h": game_h,
        "cam_top": cam_top,
        "cam_bottom": cam_bottom,
        "game_top": game_top,
        "game_bottom": game_bottom,
        "seam_y": seam_y,
        "divider_h": divider_h,
        "hook_y": round(hook_y, 1),
        "hook_bounds": hook_bounds,
        "cap_y": cap_y,
        "cap_bounds": cap_bounds,
    }


def geometry(src_w, src_h, layout, out_w=None, out_h=None, split_ratio=DEFAULT_SPLIT_RATIO,
             cam_swap=False, divider=DEFAULT_SPLIT_DIVIDER):
    """Where the video sits in the output frame, and where hook and captions go (proportional to resolution)."""
    ow = max(2, int(out_w or config.OUT_W or W))
    oh = max(2, int(out_h or config.OUT_H or H))
    if src_h >= src_w and layout not in ("crop", "gameplay", "stack"):
        layout = "crop"  # already vertical or square: nothing to gain from a blurred fill
    if layout == "blur":
        fg_h = min(oh, int(round(ow * src_h / max(1, src_w) / 2)) * 2)
    elif layout == "zoom":
        fg_h = int(round(oh * (1000.0 / 1920.0) / 2)) * 2
    elif layout == "stack":
        m = stack_layout_metrics(ow, oh, split_ratio=split_ratio, cam_swap=cam_swap, divider=divider)
        return layout, oh, m["hook_y"], m["cap_y"]
    else:
        fg_h = oh
    top = (oh - fg_h) // 2
    if layout in ("crop", "gameplay"):
        return layout, fg_h, round(oh * (300.0 / 1920.0), 1), round(oh * (1380.0 / 1920.0), 1)
    return (layout, fg_h,
            max(round(oh * (230.0 / 1920.0), 1), top - round(oh * (190.0 / 1920.0), 1)),
            min(round(oh * (1640.0 / 1920.0), 1), top + fg_h + round(oh * (180.0 / 1920.0), 1)))


def _wrap_hook_lines(words, measure, max_text_w):
    """Split heading words into 1, 2, or 3 balanced lines so large font sizes stay genuinely big."""
    one = " ".join(words)
    if len(words) <= 1 or measure(one) <= max_text_w:
        return [one]
    # Try best balanced 2-line split first
    best2 = None
    best2_w = float("inf")
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        w = max(measure(a), measure(b))
        if w < best2_w:
            best2_w = w
            best2 = [a, b]
    if best2_w <= max_text_w or len(words) < 3:
        return best2
    # If user picked a large font size and 2 lines still overflow, try a clean 3-line balanced split
    best3 = best2
    best3_w = best2_w
    for i in range(1, len(words) - 1):
        for j in range(i + 1, len(words)):
            a, b, c = " ".join(words[:i]), " ".join(words[i:j]), " ".join(words[j:])
            w = max(measure(a), measure(b), measure(c))
            if w < best3_w:
                best3_w = w
                best3 = [a, b, c]
    return best3


def _hook_events(hook, dur, y, font_size=None, style=None, enabled=True, out_w=None, out_h=None, bounds=None, max_w=None):
    """TikTok/Reels/Shorts-style video heading: bold ALL-CAPS geometric sans-serif with slightly wide
    tracking, centered horizontally and vertically inside a solid pure-white rounded pill banner.

    If `bounds=(min_y, max_y)` is provided, the entire badge is strictly contained inside `[min_y, max_y]`
    so it never overflows outside its assigned section."""
    if not enabled or not hook:
        return []
    ow = max(2, int(out_w or config.OUT_W or W))
    oh = max(2, int(out_h or config.OUT_H or H))
    scale = ow / 1080.0
    base_max_w = max(360.0, min(1020.0, float(max_w))) if max_w else float(HOOK["max_w"])
    max_box_w = base_max_w * scale

    f = _Font.get(HOOK["font"])
    size = int(round((int(font_size) if font_size else HOOK["size"]) * scale))
    min_size = max(18, int(round(36 * scale)))

    # Set heading in punchy ALL-CAPS for maximum mobile readability
    words = [w.upper() for w in hook.split() if w.strip()]
    if not words:
        return []

    track_ratio = float(HOOK.get("tracking", 0.032))

    def _metrics(sz):
        trk = round(max(1.0 * scale, sz * track_ratio), 2)
        px = max(24.0 * scale, sz * 0.44)
        py = max(14.0 * scale, sz * 0.25)
        rad = max(18.0 * scale, sz * 0.28)
        return trk, px, py, rad

    def _line_w(txt, sz, trk):
        # Slightly inflate glyph width by ~3% to account for ASS Bold=-1 synthetic thickening
        return f.width(txt, sz) * 1.03 + trk * max(0, len(txt) - 1)

    track, pad_x, pad_y, radius = _metrics(size)
    max_text_w = max(120.0 * scale, max_box_w - 2.0 * pad_x)
    lines = _wrap_hook_lines(words, lambda s: _line_w(s, size, track), max_text_w)
    widest = max(_line_w(l, size, track) for l in lines)

    # Only scale down if even the wrapped lines exceed the frame's safe horizontal span
    if widest > max_text_w:
        size = max(min_size, int(size * max_text_w / widest))
        track, pad_x, pad_y, radius = _metrics(size)
        widest = max(_line_w(l, size, track) for l in lines)

    line_h = f.em(size) * 1.15
    # Span cleanly across the horizontal frame when near full width, while staying snug for short labels
    bw = min(max_box_w, widest + 2.0 * pad_x)
    bh = line_h * len(lines) + 2.0 * pad_y

    # Enforce strict section containment if bounds are supplied, else keep safely inside the full frame
    if bounds and len(bounds) == 2:
        min_y, max_y = float(bounds[0]), float(bounds[1])
        avail_h = max(16.0, max_y - min_y)
        if bh > avail_h:
            shrink = avail_h / bh
            size = max(min_size, int(size * shrink))
            track, pad_x, pad_y, radius = _metrics(size)
            widest = max(_line_w(l, size, track) for l in lines)
            line_h = f.em(size) * 1.15
            bw = min(max_box_w, widest + 2.0 * pad_x)
            bh = line_h * len(lines) + 2.0 * pad_y
        y = max(min_y + bh / 2.0, min(max_y - bh / 2.0, float(y)))
    else:
        margin_y = 28.0 * scale
        y = max(margin_y + bh / 2.0, min(oh - margin_y - bh / 2.0, float(y)))

    x0, y0, r = (ow - bw) / 2.0, y - bh / 2.0, min(radius, bh / 2.0)
    x1, y1 = x0 + bw, y0 + bh
    n = lambda v: f"{v:.0f}"                                                         # noqa: E731
    box = (f"m {n(x0 + r)} {n(y0)} l {n(x1 - r)} {n(y0)} b {n(x1)} {n(y0)} {n(x1)} {n(y0)} {n(x1)} {n(y0 + r)} "
           f"l {n(x1)} {n(y1 - r)} b {n(x1)} {n(y1)} {n(x1)} {n(y1)} {n(x1 - r)} {n(y1)} "
           f"l {n(x0 + r)} {n(y1)} b {n(x0)} {n(y1)} {n(x0)} {n(y1)} {n(x0)} {n(y1 - r)} "
           f"l {n(x0)} {n(y0 + r)} b {n(x0)} {n(y0)} {n(x0)} {n(y0)} {n(x0 + r)} {n(y0)}")
    span = f"{_ts(0)},{_ts(dur)}"
    ev = []
    if style != "nobox":
        ev.append(f"Dialogue: 3,{span},Hook,,0,0,0,,{{\\an7\\pos(0,0)\\fad(160,0)\\c&H{HOOK['box']}&\\bord0\\shad0\\p1}}{box}{{\\p0}}")
    for i, line in enumerate(lines):
        cy = y - line_h * (len(lines) - 1) / 2.0 + i * line_h - f.cap_shift(size)
        if style == "nobox":
            bord = max(3, int(round(size * 0.085)))
            extra = f"\\b1\\fsp{track:.1f}\\c&HFFFFFF&\\3c&H000000&\\bord{bord}"
        else:
            extra = f"\\b1\\fsp{track:.1f}\\c&H{HOOK['ink']}&\\bord0"
        ev.append(f"Dialogue: 4,{span},Hook,,0,0,0,,{{\\an5\\pos({ow // 2},{cy:.0f})\\fad(160,0)\\fs{size}{extra}}}{line}")
    return ev


def build_ass(words, start, end, hook, style_name, highlight, hook_y, cap_y,
              cap_font_size=None, hook_font_size=None, hook_style=None,
              hook_enabled=True, cap_enabled=True, out_w=None, out_h=None, hook_bounds=None, hook_max_w=None) -> str:
    ow = max(2, int(out_w or config.OUT_W or W))
    oh = max(2, int(out_h or config.OUT_H or H))
    scale = ow / 1080.0
    cap_max_w = CAP_MAX_W * scale

    st = STYLES.get(style_name, STYLES["bounce"])
    cap_sz = int(round((int(cap_font_size) if cap_font_size else st['size']) * scale))
    hk_sz = int(round((int(hook_font_size) if hook_font_size else HOOK['size']) * scale))
    outline_px = max(2, int(round(st['outline'] * scale)))

    norm_words = []
    for w in words:
        txt = str(w.get("w") if "w" in w else w.get("text", ""))
        if roman.needs_roman(txt):
            txt = roman.to_roman(txt)
        s = float(w["s"] if "s" in w else w.get("start", 0.0))
        e = float(w["e"] if "e" in w else w.get("end", s + 0.3))
        norm_words.append({"w": txt, "s": s - CAPTION_LEAD, "e": e - CAPTION_LEAD})
    words = norm_words
    strong = {_key(h) for h in highlight if _key(h)}
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {ow}
PlayResY: {oh}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{cap_sz},&H00{st['base']},&H00{st['base']},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{outline_px},0,5,0,0,0,1
Style: Hook,{HOOK['font']},{hk_sz},&H00{HOOK['ink']},&H00{HOOK['ink']},&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dur = end - start
    ev = _hook_events(_clean(hook), dur, hook_y, font_size=hook_font_size or HOOK['size'],
                      style=hook_style, enabled=hook_enabled, out_w=ow, out_h=oh, bounds=hook_bounds, max_w=hook_max_w)
    if not cap_enabled:
        return head + "\n".join(ev)
    f = _Font.get(st["font"])
    inside = [dict(w, w=_clean(w["w"])) for w in words if w["s"] >= start - 0.05 and w["s"] < end]
    inside = sorted((w for w in inside if w["w"]), key=lambda w: w["s"])
    for i, w in enumerate(inside[:-1]):
        w["e"] = max(w["s"], min(w["e"], inside[i + 1]["s"]))
    groups = _groups(inside, st["per"])
    min_cap_sz = max(18, int(round(36 * scale)))
    for gi, g in enumerate(groups):
        g_start = max(0.0, g[0]["s"] - start)
        if g_start >= dur - 0.05:
            g_start = max(0.0, dur - 0.18)
        raw_end = min(g[-1]["e"] + 0.25, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else end) - start
        g_end = min(dur, max(g_start + 0.14, raw_end)) if gi + 1 == len(groups) else min(dur, raw_end)
        if g_end - g_start < 0.02:
            continue
        texts = []
        for x in g:
            t = re.sub(r"[.,;:]+$", "", x["w"])
            texts.append(t.upper() if st["upper"] else t)
        if not all(roman.drawable(t) for t in texts):
            # A script the caption font cannot draw (Japanese, Thai, Chinese...): its words cannot be measured, so
            # they are not placed one by one. The line is drawn whole and centred, in a font the system has for it.
            line = " ".join(texts)
            fs = max(int(round(44 * scale)), min(int(cap_sz * 0.7), int(1500 * scale / max(1, len(line)) * 1.6)))
            margin_lr = int(round(60 * scale))
            ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(g_end)},Cap,,{margin_lr},{margin_lr},0,,{{\\an5\\pos({ow // 2},{cap_y:.0f})\\q0\\fs{fs}}}{line}")
            continue
        size = cap_sz
        gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * outline_px + 6 * scale) * size / max(1, st["size"] * scale)   # a space, plus room for both outlines
        total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
        if total > cap_max_w:                                  # a long phrase: shrink it to fit, never run off the frame
            size = max(min_cap_sz, int(size * cap_max_w / total))
            gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * outline_px + 6 * scale) * size / max(1, st["size"] * scale)
            total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
        cy = cap_y - f.cap_shift(size)
        x = (ow - total) / 2
        p = st["pop"]
        for i, (w, t) in enumerate(zip(g, texts)):
            wd = f.width(t, size)
            p = max(104, min(st["pop"], 100 + int(2200 * scale / max(wd, 1))))   # long words grow less, so they never touch their neighbours
            pos = f"\\an5\\pos({x + wd / 2:.0f},{cy:.0f})\\fs{size}"
            x += wd + gap
            a = max(g_start, w["s"] - start)
            b = (g[i + 1]["s"] - start) if i + 1 < len(g) else g_end
            base = st["strong"] if _key(w["w"]) in strong else st["base"]
            if a - g_start >= 0.02:                                                   # waiting to be said
                ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(a)},Cap,,0,0,0,,{{{pos}\\c&H{base}&}}{t}")
            if b - a >= 0.02:                                                         # being said: colour and a quick pop, in front
                ev.append(f"Dialogue: 2,{_ts(a)},{_ts(b)},Cap,,0,0,0,,{{{pos}\\c&H{st['active']}&\\fscx94\\fscy94"
                          f"\\t(0,90,\\fscx{p}\\fscy{p})\\t(90,200,\\fscx{p - 5}\\fscy{p - 5})}}{t}")
            if g_end - b >= 0.02:                                                     # said
                ev.append(f"Dialogue: 1,{_ts(b)},{_ts(g_end)},Cap,,0,0,0,,{{{pos}\\c&H{base}&\\fscx{p - 5}\\fscy{p - 5}"
                          f"\\t(0,90,\\fscx100\\fscy100)}}{t}")
    return head + "\n".join(ev) + "\n"


PIP_W = 380            # how wide the facecam is in the 1080-wide frame
PIP_MARGIN = 26


def cover_crop_filter(cam, panel_w, panel_h, face=None, headroom=0.42, zoom=1.0, inset=0.0, src_w=None, src_h=None):
    """
    cam  = (x, y, w, h) facecam box in source pixels
    face = (cx, cy) face center in source pixels, optional (falls back to box center)
    inset = fractional inward edge trim (e.g. 0.035) to strip rounded webcam bezels / corner bleed
    Returns an ffmpeg filter that fills panel_w x panel_h edge-to-edge with NO gaps and NO stretch.
    """
    x, y, w, h = (int(v) for v in cam)
    if src_w and src_h and src_w > 0 and src_h > 0:
        x = max(0, min(x, int(src_w) - 20))
        y = max(0, min(y, int(src_h) - 20))
        w = max(20, min(w, int(src_w) - x))
        h = max(20, min(h, int(src_h) - y))
    else:
        x = max(0, x)
        y = max(0, y)
        w = max(20, w)
        h = max(20, h)

    # Optional inward inset trims rounded webcam corners, dark bezels, and slight edge slop
    inset_val = max(0.0, min(0.15, float(inset or 0.0)))
    if inset_val > 0.0:
        dx = int(round(w * inset_val))
        dy = int(round(h * inset_val))
        if w - 2 * dx >= 40 and h - 2 * dy >= 40:
            x += dx
            y += dy
            w -= 2 * dx
            h -= 2 * dy

    panel_w = max(2, int(panel_w) - (int(panel_w) % 2))
    panel_h = max(2, int(panel_h) - (int(panel_h) % 2))
    target = panel_w / panel_h

    # Biggest aspect-matched window strictly inside the cam box (zero stretch, zero black bars)
    if w / h > target:          # cam wider than panel -> trim sides
        cw, ch = int(round(h * target)), h
    else:                       # cam taller than panel -> trim top/bottom
        cw, ch = w, int(round(w / target))
    cw = max(2, min(w, cw - (cw % 2)))
    ch = max(2, min(h, ch - (ch % 2)))

    # Zoom in if requested (shrinks cw/ch before clamping inside cam bounds)
    zoom_val = max(1.0, min(3.0, float(zoom or 1.0)))
    if zoom_val > 1.0:
        cw = int(round(cw / zoom_val))
        ch = int(round(ch / zoom_val))
        cw -= cw % 2
        ch -= ch % 2
        cw = max(20, min(w, cw))
        ch = max(20, min(h, ch))

    fcx, fcy = face if face else (x + w / 2, y + h / 2)
    cx = int(round(fcx - cw / 2))
    cy = int(round(fcy - ch * headroom))   # face sits ~42% down the panel, leaving room below for the label
    cx = max(x, min(cx, x + w - cw))
    cy = max(y, min(cy, y + h - ch))

    return f"crop={cw}:{ch}:{cx}:{cy},scale={panel_w}:{panel_h}:flags=lanczos,setsar=1"


def build_stack_filter(cam, ratio=DEFAULT_SPLIT_RATIO, swap=False, face=None, W=None, H=None,
                       cam_delay=0.0, zoom=1.0, divider=DEFAULT_SPLIT_DIVIDER,
                       cam_inset=STACK_CAM_INSET, src_w=None, src_h=None):
    """Build a clean, gap-free, edge-to-edge 9:16 split-screen FFmpeg filter graph.

    Default: top 35% facecam, bottom 65% gameplay, optional subtle divider at the seam."""
    m = stack_layout_metrics(out_w=W, out_h=H, split_ratio=ratio, cam_swap=swap, divider=divider)
    out_w, out_h = m["out_w"], m["out_h"]
    cam_h, game_h = m["cam_h"], m["game_h"]

    cam_f = cover_crop_filter(cam, out_w, cam_h, face=face, zoom=zoom, inset=cam_inset, src_w=src_w, src_h=src_h)
    game_f = f"scale={out_w}:{game_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={out_w}:{game_h},setsar=1"

    delay_f = ""
    if cam_delay and abs(cam_delay) >= 0.02:
        if cam_delay > 0:
            delay_f = f"tpad=start_duration={cam_delay:.3f}:start_mode=clone,"
        else:
            delay_f = f"trim=start={-cam_delay:.3f},"

    div_f = ""
    if divider and m["divider_h"] > 0:
        div_y = max(0, min(out_h - m["divider_h"], m["seam_y"] - m["divider_h"] // 2))
        div_f = f",drawbox=x=0:y={div_y}:w={out_w}:h={m['divider_h']}:color=0x0B0B0D@0.95:t=fill"

    if swap:   # gameplay on top, facecam below
        return (f"[0:v]setpts=PTS-STARTPTS,fps={config.OUT_FPS},split=2[g0][c0];"
                f"[g0]{game_f},setpts=PTS-STARTPTS[g];"
                f"[c0]{cam_f},{delay_f}setpts=PTS-STARTPTS[c];"
                f"[g][c]vstack=inputs=2:shortest=1,setsar=1{div_f},setpts=PTS-STARTPTS[v0];")
    return (f"[0:v]setpts=PTS-STARTPTS,fps={config.OUT_FPS},split=2[c0][g0];"
            f"[c0]{cam_f},{delay_f}setpts=PTS-STARTPTS[c];"
            f"[g0]{game_f},setpts=PTS-STARTPTS[g];"
            f"[c][g]vstack=inputs=2:shortest=1,setsar=1{div_f},setpts=PTS-STARTPTS[v0];")


def _filter(layout, fg_h, facecam=None, cam_delay=0.0, split_ratio=DEFAULT_SPLIT_RATIO,
            cam_swap=False, focus=None, zoom=1.0, divider=DEFAULT_SPLIT_DIVIDER,
            out_w=None, out_h=None, src_w=None, src_h=None):
    ow = max(2, int(out_w or config.OUT_W or W))
    oh = max(2, int(out_h or config.OUT_H or H))
    ow -= ow % 2
    oh -= oh % 2
    pip_w = max(2, int(round(ow * (PIP_W / 1080.0) / 2.0)) * 2)
    pip_margin = max(2, int(round(ow * (PIP_MARGIN / 1080.0))))

    delay_f = ""
    if cam_delay and abs(cam_delay) >= 0.02:
        if cam_delay > 0:
            delay_f = f"tpad=start_duration={cam_delay:.3f}:start_mode=clone,"
        else:
            delay_f = f"trim=start={-cam_delay:.3f},"

    if layout == "gameplay" and facecam:
        # The game fills the screen, and the webcam sits in the top corner of it, so a viewer sees
        # the play and the face at the same time. Both branches use setpts=PTS-STARTPTS to guarantee
        # perfect timestamp synchronization and zero frame delay in overlay.
        fx, fy, fw, fh = (int(v) for v in facecam)
        if fw > 8 and fh > 8:
            pip_h = max(2, int(round(pip_w * fh / fw / 2)) * 2)
            return (f"[0:v]setpts=PTS-STARTPTS,fps={config.OUT_FPS},split=2[base][fc];"
                    f"[base]scale={ow}:{oh}:force_original_aspect_ratio=increase,crop={ow}:{oh},setsar=1,setpts=PTS-STARTPTS[b];"
                    f"[fc]crop={fw}:{fh}:{fx}:{fy},scale={pip_w}:{pip_h},setsar=1,{delay_f}setpts=PTS-STARTPTS[p];"
                    f"[b][p]overlay={ow - pip_w - pip_margin}:{pip_margin}:eof_action=repeat:shortest=1,setpts=PTS-STARTPTS[v0];")

    if layout == "stack":
        sw = int(src_w) if (src_w and src_w > 0) else 1920
        sh = int(src_h) if (src_h and src_h > 0) else 1080
        fx, fy, fw, fh = (int(v) for v in facecam) if (facecam and len(facecam) == 4 and facecam[2] > 8 and facecam[3] > 8) else (0, 0, sw, int(round(sh * 0.50)))
        ratio = max(0.20, min(0.80, float(DEFAULT_SPLIT_RATIO if split_ratio is None else split_ratio)))
        return build_stack_filter([fx, fy, fw, fh], ratio, swap=cam_swap, face=focus, W=ow, H=oh,
                                  cam_delay=cam_delay, zoom=zoom, divider=divider, src_w=sw, src_h=sh)

    if layout == "crop" or (layout == "gameplay" and not facecam):
        return f"[0:v]setpts=PTS-STARTPTS,fps={config.OUT_FPS},scale={ow}:{oh}:force_original_aspect_ratio=increase,crop={ow}:{oh},setsar=1,setpts=PTS-STARTPTS[v0];"
    bg_w = max(2, int(round(ow / 4.0 / 2.0)) * 2)
    bg_h = max(2, int(round(oh / 4.0 / 2.0)) * 2)
    return (f"[0:v]setpts=PTS-STARTPTS,fps={config.OUT_FPS},split=2[a][b];"
            f"[a]scale={bg_w}:{bg_h}:force_original_aspect_ratio=increase,crop={bg_w}:{bg_h},boxblur=6:2,"
            f"scale={ow}:{oh},eq=brightness=-0.14:saturation=1.15,setpts=PTS-STARTPTS[bg];"
            f"[b]scale={ow}:{fg_h}:force_original_aspect_ratio=increase,crop={ow}:{fg_h},setpts=PTS-STARTPTS[fg];"
            f"[bg][fg]overlay=0:(H-h)/2:eof_action=repeat:shortest=1,setsar=1,setpts=PTS-STARTPTS[v0];")


def _hook_y(layout, hook_y, facecam=None, split_ratio=DEFAULT_SPLIT_RATIO, cam_swap=False,
            out_w=None, out_h=None, divider=DEFAULT_SPLIT_DIVIDER):
    """Where the heading goes. In the gameplay layout the webcam sits in the top corner, so a heading
    across the top would run under it: it goes just below the picture instead. In stack layout,
    it stays cleanly inside the facecam section near the divider without overflowing."""
    ow = max(2, int(out_w or config.OUT_W or W))
    oh = max(2, int(out_h or config.OUT_H or H))
    if layout == "gameplay" and facecam and len(facecam) == 4 and facecam[2] > 8:
        pip_w = ow * (PIP_W / 1080.0)
        pip_margin = ow * (PIP_MARGIN / 1080.0)
        return round(pip_margin + int(round(pip_w * facecam[3] / facecam[2])) + round(oh * (32.0 / 1920.0)), 1)
    if layout == "stack":
        return stack_layout_metrics(ow, oh, split_ratio=split_ratio, cam_swap=cam_swap, divider=divider)["hook_y"]
    return hook_y


_nvenc_ok = None
_qsv_ok = None
_amf_ok = None
RENDER_SECONDS = 300      # one clip's video encode; far beyond normal, so a frozen encoder is stopped, not waited on


def _probe_encoder(codec: str) -> bool:
    """Quickly test a hardware H.264 encoder on a 1-frame null source so a broken driver never stalls a real render."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.04",
             "-c:v", codec, "-f", "null", "-"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=5, creationflags=NO_WINDOW,
        )
        return r.returncode == 0
    except Exception:
        return False


def check_nvenc() -> bool:
    """Checks whether NVIDIA NVENC hardware encoder is available on this system."""
    global _nvenc_ok
    if _nvenc_ok is None:
        _nvenc_ok = _probe_encoder("h264_nvenc")
    return bool(_nvenc_ok)


def _available_encoders(gpu_enabled: bool, draft: bool = False) -> list:
    """Return ordered list of working (label, ffmpeg_args) encoders using cached 5s probes."""
    global _nvenc_ok, _qsv_ok, _amf_ok
    if gpu_enabled:
        if _nvenc_ok is None:
            _nvenc_ok = _probe_encoder("h264_nvenc")
        if not _nvenc_ok and _qsv_ok is None:
            _qsv_ok = _probe_encoder("h264_qsv")
        if not _nvenc_ok and not _qsv_ok and _amf_ok is None:
            _amf_ok = _probe_encoder("h264_amf")
    encoders = []
    if draft:
        if gpu_enabled and _nvenc_ok:
            encoders.append(("nvenc", ["yuv420p", "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "28"]))
        if gpu_enabled and _qsv_ok:
            encoders.append(("qsv", ["nv12", "-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "28"]))
        if gpu_enabled and _amf_ok:
            encoders.append(("amf", ["yuv420p", "-c:v", "h264_amf", "-rc", "cqp", "-qp_p", "28", "-qp_i", "28"]))
        encoders.append(("x264", ["yuv420p", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"]))
    else:
        if gpu_enabled and _nvenc_ok:
            encoders.append(("nvenc", ["yuv420p", "-c:v", "h264_nvenc", "-cq", "21", "-preset", "p4"]))
        if gpu_enabled and _qsv_ok:
            encoders.append(("qsv", ["nv12", "-c:v", "h264_qsv", "-global_quality", "23", "-preset", "fast"]))
        if gpu_enabled and _amf_ok:
            encoders.append(("amf", ["yuv420p", "-c:v", "h264_amf", "-rc", "cqp", "-qp_p", "21", "-qp_i", "21"]))
        encoders.append(("x264", ["yuv420p", "-c:v", "libx264", "-crf", "21", "-preset", "veryfast"]))
    return encoders


def _safe_move(src_path: Path, dst_path: Path, retries: int = 8):
    """Move rendered output into place, retrying briefly on Windows if the browser still holds a read lock."""
    last_err = None
    for attempt in range(retries):
        try:
            if dst_path.exists():
                try:
                    os.replace(str(src_path), str(dst_path))
                    return
                except OSError:
                    pass
            shutil.move(str(src_path), str(dst_path))
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.25 * (attempt + 1))
    # Final fallback: copy bytes over if destination can be truncated
    try:
        shutil.copyfile(str(src_path), str(dst_path))
        src_path.unlink(missing_ok=True)
        return
    except Exception:
        if last_err:
            raise last_err
        raise


def _normalize_clip_words(words_to_use: list, start: float, end: float, off: float = 0.0) -> list:
    """Align a clip's word timestamps uniformly to the [start, end] range of the source file.
    Using a whole-list basis check prevents per-word double-offset subtraction (when off is small)
    and prevents pre-start words (e.g. start - 0.5s) from being teleported forward into the clip."""
    parsed = []
    for w in (words_to_use or []):
        if not isinstance(w, dict):
            continue
        t = str(w.get("w") if "w" in w else w.get("text", "")).strip()
        if not t:
            continue
        if roman.needs_roman(t):
            t = roman.to_roman(t)
        try:
            s_val = float(w.get("s") if "s" in w else w.get("start", 0.0))
            e_val = float(w.get("e") if "e" in w else w.get("end", s_val + 0.3))
        except (ValueError, TypeError):
            continue
        if e_val < s_val:
            e_val = s_val + 0.1
        parsed.append((t, s_val, e_val))
    if not parsed:
        return []

    dur = max(0.5, float(end) - float(start))
    in_clip = sum(1 for _, s, _ in parsed if (start - 2.0) <= s <= (end + 2.0))
    in_abs = sum(1 for _, s, _ in parsed if (start + off - 2.0) <= s <= (end + off + 2.0)) if off > 1.0 else -1
    in_rel = sum(1 for _, s, _ in parsed if -1.5 <= s <= (dur + 2.0)) if start > 2.0 else -1

    delta = 0.0
    if in_clip > 0 and in_clip >= in_abs and in_clip >= in_rel:
        delta = 0.0
    elif off > 1.0 and in_abs > in_clip and in_abs >= in_rel:
        delta = -off
    elif start > 2.0 and in_rel > in_clip:
        delta = start

    return [{"w": t, "s": round(s + delta, 2), "e": round(e + delta, 2)} for t, s, e in parsed]


def render_clip(src: Path, out_dir: Path, name: str, clip: dict, words: list, src_w: int, src_h: int,
                quality: str = "final", out_w: int = None, out_h: int = None) -> dict:
    """Writes <name>.mp4 (or <name>_draft.mp4), <name>.jpg, <name>.ass, etc. into out_dir. Returns file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ow = max(2, int(out_w or clip.get("out_w") or config.OUT_W or W))
    oh = max(2, int(out_h or clip.get("out_h") or config.OUT_H or H))
    ow -= ow % 2
    oh -= oh % 2

    edit_file = out_dir / f"{name}_edit.json"
    ed = edit.load_edit(edit_file) if edit_file.is_file() else None
    off = float(clip.get("src_offset", 0.0) or 0.0)

    if ed:
        start = float(ed["start"])
        end = float(ed["end"])
        if off > 0 and start >= max(0.0, off - 30.0):
            start = max(0.0, round(start - off, 2))
            end = max(start + 1.0, round(end - off, 2))
        layout = ed.get("layout", "blur")
        split_ratio = float(ed.get("split_ratio") if ed.get("split_ratio") is not None else DEFAULT_SPLIT_RATIO)
        split_divider = bool(ed.get("split_divider", clip.get("split_divider", DEFAULT_SPLIT_DIVIDER)))
        fc_obj = ed.get("facecam")
        face = [fc_obj["x"], fc_obj["y"], fc_obj["w"], fc_obj["h"]] if (fc_obj and all(k in fc_obj for k in ("x", "y", "w", "h"))) else None
        face_focus = fc_obj.get("focus") if (fc_obj and isinstance(fc_obj, dict)) else None
        face_zoom = float(fc_obj.get("zoom", 1.0)) if (fc_obj and isinstance(fc_obj, dict)) else 1.0
        cam_swap = bool(ed.get("cam_swap", clip.get("cam_swap", False)))
        cap_conf = ed.get("caption") or {}
        if "enabled" in cap_conf:
            cap_enabled = bool(cap_conf["enabled"])
        else:
            cap_enabled = bool(clip.get("captions", True))
        cap_style = str(cap_conf.get("style", "bounce"))
        cap_font_size = int(cap_conf.get("font_size", 132))
        cap_y = cap_conf.get("y_pos")
        words_to_use = cap_conf.get("words") if cap_conf.get("words") else words
        hook_conf = ed.get("hook") or {}
        if "enabled" in hook_conf:
            hook_enabled = bool(hook_conf["enabled"])
        else:
            hook_enabled = bool(clip.get("hook_enabled", bool(clip.get("hook"))))
        hook_text = str(hook_conf.get("text", "")).strip() if hook_enabled else ""
        hook_font_size = int(hook_conf.get("font_size", 96))
        hook_y = hook_conf.get("y_pos")
        hook_max_w = hook_conf.get("max_w") or clip.get("hook_max_w") or HOOK["max_w"]
        hook_style = str(hook_conf.get("style", "default"))
        audio_conf = ed.get("audio") or {}
        audio_opts = {
            "meme": bool(audio_conf.get("meme", False)),
            "music": bool(audio_conf.get("music", False)),
            "meme_mood": str(audio_conf.get("meme_mood", "none")),
            "music_bed": str(audio_conf.get("music_bed", "none")),
            "meme_vol": float(audio_conf.get("meme_vol", 0.0)),
            "music_db": float(audio_conf.get("music_db", -18.0)),
        }
    else:
        start, end = float(clip["start"]), float(clip["end"])
        if off > 0 and start >= off:
            start = round(start - off, 2)
            end = round(end - off, 2)
        layout = clip.get("layout", "blur")
        split_ratio = float(clip.get("split_ratio") if clip.get("split_ratio") is not None else DEFAULT_SPLIT_RATIO)
        split_divider = bool(clip.get("split_divider", DEFAULT_SPLIT_DIVIDER))
        face = clip.get("facecam")
        face_focus = clip.get("facecam_focus")
        face_zoom = float(clip.get("facecam_zoom", clip.get("face_zoom", 1.0)))
        cam_swap = bool(clip.get("cam_swap", False))
        cap_enabled = bool(clip.get("captions", True))
        cap_style = clip.get("style", "bounce")
        cap_font_size = int(clip.get("cap_font_size") or 132)
        cap_y = clip.get("cap_y")
        words_to_use = words
        hook_enabled = bool(clip.get("hook_enabled", bool(clip.get("hook"))))
        hook_text = str(clip.get("hook", "")).strip() if hook_enabled else ""
        hook_font_size = int(clip.get("hook_font_size") or 96)
        hook_y = clip.get("hook_y")
        hook_max_w = clip.get("hook_max_w") or HOOK["max_w"]
        hook_style = str(clip.get("hook_style") or "default")
        audio_opts = {
            "meme": bool(clip.get("meme") and clip.get("meme") != "none"),
            "music": bool(clip.get("music") and clip.get("music") != "none"),
            "meme_mood": str(clip.get("meme") or "none"),
            "music_bed": str(clip.get("music") or "none"),
            "meme_vol": float(clip.get("meme_vol") or 0.0),
            "music_db": float(clip.get("music_db") or -18.0),
        }

    if hook_text and roman.needs_roman(hook_text):
        hook_text = roman.to_roman(hook_text)

    # Whole-list word timestamp normalization: avoids per-word double-offset subtraction and pre-start word teleporting
    words_to_use = _normalize_clip_words(words_to_use, start, end, off)

    layout, fg_h, def_hook_y, def_cap_y = geometry(src_w, src_h, layout, out_w=ow, out_h=oh,
                                                   split_ratio=split_ratio, cam_swap=cam_swap, divider=split_divider)
    if face and len(face) == 4 and src_w > 0 and src_h > 0:
        if all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in face) and (float(face[2]) > 0 or float(face[3]) > 0):
            fx = int(round(float(face[0]) * src_w))
            fy = int(round(float(face[1]) * src_h))
            fw = int(round(float(face[2]) * src_w))
            fh = int(round(float(face[3]) * src_h))
        else:
            fx, fy, fw, fh = (int(v) for v in face)
        fx = max(0, min(fx, src_w - 10))
        fy = max(0, min(fy, src_h - 10))
        fw = max(10, min(fw, src_w - fx))
        fh = max(10, min(fh, src_h - fy))
        face = [fx, fy, fw, fh]
    if layout == "gameplay" and not (face and len(face) == 4):
        fb = clip.get("cam_fallback") or "blur"
        layout = fb if fb in ("blur", "zoom") else "blur"
        layout, fg_h, def_hook_y, def_cap_y = geometry(src_w, src_h, layout, out_w=ow, out_h=oh)
    elif layout == "stack" and not (face and len(face) == 4):
        if src and Path(src).is_file():
            try:
                from . import facecam as _fc
                det = _fc.detect(src, window=(start, end))
                if not det:
                    det = _fc.detect(src)
                if det and det.get("w", 0) > 10 and det.get("h", 0) > 10:
                    face = [int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])]
            except Exception:
                pass
        if not (face and len(face) == 4):
            sw = src_w if src_w > 0 else 1920
            sh = src_h if src_h > 0 else 1080
            face = [0, 0, sw, int(round(sh * 0.50))]

    stack_m = stack_layout_metrics(ow, oh, split_ratio=split_ratio, cam_swap=cam_swap, divider=split_divider) if layout == "stack" else None
    y_scale = oh / 1920.0

    cap_anchor = clip.get("cap_anchor")
    if cap_y is not None:
        try:
            cap_y = float(cap_y) * y_scale if oh != 1920 else float(cap_y)
        except (ValueError, TypeError):
            cap_y = float(def_cap_y)
    elif layout == "stack" and stack_m:
        if cap_anchor == "seam":
            cap_y = float(stack_m["seam_y"] + round(oh * 0.03) * (1 if not cam_swap else -1))
        elif cap_anchor == "top":
            cap_y = float(stack_m["game_top"] + round(stack_m["game_h"] * 0.16))
        else:
            cap_y = float(stack_m["cap_y"])
    elif cap_anchor == "safe_bottom":
        cap_y = round(oh * (1630.0 / 1920.0), 1)
    elif cap_anchor == "top":
        cap_y = round(oh * (420.0 / 1920.0), 1)
    else:
        cap_y = float(def_cap_y)

    # In stack layout, keep subtitles inside the gameplay section when using default/safe_bottom anchor
    if layout == "stack" and stack_m and cap_anchor != "seam":
        g_min, g_max = stack_m["cap_bounds"]
        if cap_y < stack_m["game_top"] or cap_y > stack_m["game_bottom"]:
            cap_y = float(stack_m["cap_y"])
        else:
            cap_y = max(g_min, min(g_max, cap_y))

    hook_bounds = None
    if hook_y is not None:
        try:
            hook_y = float(hook_y) * y_scale if oh != 1920 else float(hook_y)
        except (ValueError, TypeError):
            hook_y = _hook_y(layout, def_hook_y, face, split_ratio=split_ratio, cam_swap=cam_swap, out_w=ow, out_h=oh, divider=split_divider)
    else:
        hook_y = _hook_y(layout, def_hook_y, face, split_ratio=split_ratio, cam_swap=cam_swap, out_w=ow, out_h=oh, divider=split_divider)

    ass = build_ass(words_to_use if cap_enabled else [], start, end, hook_text, cap_style,
                    clip.get("highlight_words", []), hook_y, cap_y,
                    cap_font_size=cap_font_size, hook_font_size=hook_font_size,
                    hook_style=hook_style, hook_enabled=hook_enabled, cap_enabled=cap_enabled,
                    out_w=ow, out_h=oh, hook_bounds=hook_bounds, hook_max_w=hook_max_w)
    # The clip is made in a short work folder and moved into place afterwards.
    work = Path(tempfile.mkdtemp(prefix="pk"))
    try:
        return _render_in(work, src, out_dir, name, start, end, layout, fg_h, ass,
                          face=face, clip=clip, words=words_to_use, hook_y=hook_y, cap_y=cap_y,
                          quality=quality, audio_opts=audio_opts, split_ratio=split_ratio,
                          focus=face_focus, zoom=face_zoom, cam_swap=cam_swap,
                          divider=split_divider, out_w=ow, out_h=oh, src_w=src_w, src_h=src_h)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _extract_cover(video_file: Path, out_jpg: Path, duration: float, fallback_src: Path = None, fallback_start: float = 0.0):
    """Safely extracts a cover image for a clip. Tries rendered video first, then fallback source, and finally Pillow if needed."""
    if out_jpg.is_file() and out_jpg.stat().st_size > 0:
        return
    work_dir = out_jpg.parent
    cover_t = max(0.0, min(1.2, duration / 2.0))

    # 1. Try rendered video at cover_t
    if video_file.is_file():
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{cover_t:.2f}", "-i", str(video_file),
             "-frames:v", "1", "-strict", "-1", "-q:v", "3", str(out_jpg)],
            cwd=work_dir, stdin=subprocess.DEVNULL, capture_output=True, text=True, creationflags=NO_WINDOW, timeout=15
        )
        if out_jpg.is_file() and out_jpg.stat().st_size > 0:
            return

        # 2. Try first frame of rendered video
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(video_file),
             "-frames:v", "1", "-strict", "-1", "-q:v", "3", str(out_jpg)],
            cwd=work_dir, stdin=subprocess.DEVNULL, capture_output=True, text=True, creationflags=NO_WINDOW, timeout=15
        )
        if out_jpg.is_file() and out_jpg.stat().st_size > 0:
            return

    # 3. Try fallback src at fallback_start + cover_t
    if fallback_src and fallback_src.is_file():
        src_t = max(0.0, fallback_start + cover_t)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{src_t:.2f}", "-i", str(fallback_src),
             "-frames:v", "1", "-strict", "-1", "-q:v", "3", str(out_jpg)],
            cwd=work_dir, stdin=subprocess.DEVNULL, capture_output=True, text=True, creationflags=NO_WINDOW, timeout=15
        )
        if out_jpg.is_file() and out_jpg.stat().st_size > 0:
            return

    # 4. Emergency fallback: generate a solid dark frame with Pillow
    try:
        from PIL import Image
        img = Image.new("RGB", (W, H), color=(20, 20, 20))
        img.save(str(out_jpg), "JPEG", quality=85)
    except Exception:
        pass


def _render_in(work: Path, src: Path, out_dir: Path, name: str, start, end, layout, fg_h, ass: str,
               face=None, clip=None, words=None, hook_y=300, cap_y=1380, quality="final",
               audio_opts=None, split_ratio=DEFAULT_SPLIT_RATIO, focus=None, zoom=1.0, cam_swap=False,
               divider=DEFAULT_SPLIT_DIVIDER, out_w=None, out_h=None, src_w=None, src_h=None) -> dict:
    global _nvenc_ok, _qsv_ok, _amf_ok
    ow = max(2, int(out_w or config.OUT_W or W))
    oh = max(2, int(out_h or config.OUT_H or H))
    ow -= ow % 2
    oh -= oh % 2
    draft_w = max(2, (ow // 2) - ((ow // 2) % 2))
    draft_h = max(2, (oh // 2) - ((oh // 2) % 2))

    (work / f"{name}.ass").write_text(ass, encoding="utf-8")
    (work / "f").mkdir()
    for f in config.FONTS_DIR.glob("*.ttf"):
        shutil.copyfile(f, work / "f" / f.name)

    gpu_enabled = getattr(config, "USE_GPU", True)
    try:
        from . import accounts
        if not accounts.settings().get("gpu_enabled", True):
            gpu_enabled = False
    except Exception:
        pass
    encoders = _available_encoders(gpu_enabled, draft=False)

    cam_delay = float((clip or {}).get("cam_delay", 0.0) or 0.0)
    cam_swap = bool(cam_swap or (clip or {}).get("cam_swap", False))

    clip_dur = max(0.5, float(end) - float(start))

    if quality == "draft":
        fc = _filter(layout, fg_h, face, cam_delay=cam_delay, split_ratio=split_ratio, cam_swap=cam_swap,
                     focus=focus, zoom=zoom, divider=divider, out_w=ow, out_h=oh, src_w=src_w, src_h=src_h)
        fc_clean = fc.rstrip(";")
        fc_draft = f"{fc_clean};[v0]scale={draft_w}:{draft_h},ass={name}.ass:fontsdir=f[vdraft]"
        # Put -t AFTER -i (output duration, never input -t before -i) so demuxer interleaving never cuts audio early!
        base_draft = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-accurate_seek", "-ss", f"{start:.2f}", "-i", str(src),
                      "-t", f"{clip_dur:.2f}",
                      "-filter_complex", fc_draft, "-map", "[vdraft]", "-map", "0:a:0?",
                      "-af", "aresample=48000:async=1:first_pts=0",
                      "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-movflags", "+faststart", "-pix_fmt"]
        draft_encoders = _available_encoders(gpu_enabled, draft=True)

        err = None
        for label, enc in draft_encoders:
            try:
                p = subprocess.run(base_draft + enc + [f"{name}_draft.mp4"], cwd=work, stdin=subprocess.DEVNULL,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   creationflags=NO_WINDOW, timeout=RENDER_SECONDS)
            except subprocess.TimeoutExpired:
                err = f"{label} timed out"
                if label == "nvenc": _nvenc_ok = False
                elif label == "qsv": _qsv_ok = False
                elif label == "amf": _amf_ok = False
                continue
            if p.returncode == 0:
                break
            err = p.stderr[-1200:]
            if label == "nvenc": _nvenc_ok = False
            elif label == "qsv": _qsv_ok = False
            elif label == "amf": _amf_ok = False
        else:
            raise RuntimeError(f"Draft render failed: {err}")

        _extract_cover(work / f"{name}_draft.mp4", work / f"{name}.jpg", end - start, fallback_src=src, fallback_start=start)
        for ext in ("mp4", "jpg", "ass"):
            fpath = work / f"{name}_draft.mp4" if ext == "mp4" else work / f"{name}.{ext}"
            if fpath.is_file():
                _safe_move(fpath, out_dir / fpath.name)
        return {"file": f"{name}_draft.mp4", "cover": f"{name}.jpg", "layout": layout, "draft": True}

    # Final Render — single-pass when layout/time/audio changed (2x faster, no double encode!)
    existing_raw = out_dir / f"{name}_raw.mp4"
    fast_caption = bool(clip and clip.get("_fast_caption_only") and existing_raw.is_file())

    if fast_caption:
        # Caption/hook-only remake: re-burn ASS onto existing raw video in 1 quick pass (2-4s)
        raw_file = work / f"{name}_raw.mp4"
        shutil.copyfile(existing_raw, raw_file)
        if ass:
            burn_base = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(raw_file),
                         "-vf", f"ass={name}.ass:fontsdir=f",
                         "-c:a", "copy", "-movflags", "+faststart", "-pix_fmt"]
            err = None
            for label, enc in encoders:
                try:
                    p = subprocess.run(burn_base + enc + [f"{name}.mp4"], cwd=work, stdin=subprocess.DEVNULL,
                                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       creationflags=NO_WINDOW, timeout=RENDER_SECONDS)
                except subprocess.TimeoutExpired:
                    err = f"{label} timed out"
                    continue
                if p.returncode == 0:
                    break
                err = p.stderr[-1200:]
                if label == "nvenc": _nvenc_ok = False
                elif label == "qsv": _qsv_ok = False
                elif label == "amf": _amf_ok = False
            else:
                shutil.copyfile(raw_file, work / f"{name}.mp4")
        else:
            shutil.copyfile(raw_file, work / f"{name}.mp4")
    else:
        # Full layout/clip render: burn layout + captions + normalized audio in ONE single FFmpeg pass!
        fc = _filter(layout, fg_h, face, cam_delay=cam_delay, split_ratio=split_ratio, cam_swap=cam_swap,
                     focus=focus, zoom=zoom, divider=divider, out_w=ow, out_h=oh, src_w=src_w, src_h=src_h)
        fc_clean = fc.rstrip(";")
        if ass:
            fc_final = f"{fc_clean};[v0]ass={name}.ass:fontsdir=f[vout]"
            v_map = "[vout]"
        else:
            fc_final = fc_clean
            v_map = "[v0]"
        # Put -t AFTER -i (output duration) and use zero-tail-loss async resample + dynaudnorm + alimiter
        # instead of input -t + single-pass loudnorm (which frequently muted/truncated the last 1-3s of audio).
        base_final = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-accurate_seek", "-ss", f"{start:.2f}", "-i", str(src),
                      "-t", f"{clip_dur:.2f}",
                      "-filter_complex", fc_final, "-map", v_map, "-map", "0:a:0?",
                      "-af", "aresample=48000:async=1:first_pts=0,dynaudnorm=f=150:g=15:p=0.92:m=6,alimiter=limit=0.95",
                      "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", "-pix_fmt"]
        err = None
        for label, enc in encoders:
            try:
                p = subprocess.run(base_final + enc + [f"{name}.mp4"], cwd=work, stdin=subprocess.DEVNULL,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   creationflags=NO_WINDOW, timeout=RENDER_SECONDS)
            except subprocess.TimeoutExpired:
                err = f"{label} timed out after {RENDER_SECONDS}s"
                if label == "nvenc": _nvenc_ok = False
                elif label == "qsv": _qsv_ok = False
                elif label == "amf": _amf_ok = False
                continue
            if p.returncode == 0:
                break
            err = p.stderr[-1200:]
            if label == "nvenc": _nvenc_ok = False
            elif label == "qsv": _qsv_ok = False
            elif label == "amf": _amf_ok = False
        else:
            raise RuntimeError(f"Render failed: {err}")
        # Remove any stale raw file from a previous layout so Fast remake never uses the old layout
        if existing_raw.is_file():
            try:
                existing_raw.unlink(missing_ok=True)
            except OSError:
                pass

    # Save captions.json with word timestamps, style, and offsets
    cap_data = {
        "words": words or [],
        "hook": clip.get("hook", "") if clip else "",
        "style": clip.get("style", "bounce") if clip else "bounce",
        "hook_y": hook_y,
        "cap_y": cap_y,
        "cap_anchor": clip.get("cap_anchor", "bottom") if clip else "bottom",
        "split_ratio": split_ratio,
        "split_divider": divider,
        "cam_swap": cam_swap,
    }
    try:
        (work / f"{name}_captions.json").write_text(json.dumps(cap_data, indent=1), encoding="utf-8")
    except Exception:
        pass

    mp4_file = work / f"{name}.mp4"
    aopts = audio_opts or {}
    meme_mood = aopts.get("meme_mood") if aopts.get("meme") else None
    if meme_mood and meme_mood != "none":
        try:
            from . import meme
            vol = aopts.get("meme_vol")
            meme.apply_meme_effect(mp4_file, meme_mood, out=mp4_file, vol=vol)
        except Exception:
            pass
    music_mood = aopts.get("music_bed") if aopts.get("music") else None
    if music_mood and music_mood != "none":
        try:
            from . import meme
            db = aopts.get("music_db")
            if db is not None:
                meme.apply_music_bed(mp4_file, music_mood, out=mp4_file, db=float(db))
            else:
                meme.apply_music_bed(mp4_file, music_mood, out=mp4_file)
        except Exception:
            pass

    _extract_cover(work / f"{name}.mp4", work / f"{name}.jpg", end - start, fallback_src=src, fallback_start=start)
    for ext in ("mp4", "jpg", "ass"):
        fpath = work / f"{name}.{ext}"
        if fpath.is_file():
            _safe_move(fpath, out_dir / fpath.name)
    if (work / f"{name}_captions.json").is_file():
        _safe_move(work / f"{name}_captions.json", out_dir / f"{name}_captions.json")

    # Cleanup draft file if present
    draft_file = out_dir / f"{name}_draft.mp4"
    if draft_file.is_file():
        draft_file.unlink(missing_ok=True)

    qc_res = None
    try:
        from . import qc
        qc_res = qc.check_clip(out_dir / f"{name}.mp4")
    except Exception:
        pass

    out_res = {"file": f"{name}.mp4", "cover": f"{name}.jpg", "layout": layout, "raw": f"{name}_raw.mp4", "captions_file": f"{name}_captions.json"}
    if qc_res:
        out_res["qc"] = qc_res
    return out_res

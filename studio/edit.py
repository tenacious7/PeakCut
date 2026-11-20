"""Edit-state file ({clip_id}_edit.json) loader, saver, validator and deep-merge logic.
Single source of truth for clip draft and final renders.
"""
import copy
import json
import math
from pathlib import Path

from . import config, roman

DEFAULT_SPLIT_RATIO = 0.35
DEFAULT_SPLIT_DIVIDER = True

GLOBAL_FIELDS = {
    "layout",
    "split_ratio",
    "split_divider",
    "cam_swap",
    "facecam",  # job-level override
    "facecam.zoom",
    "caption.enabled",
    "caption.style",
    "caption.font_size",
    "caption.y_pos",
    "hook.enabled",
    "hook.font_size",
    "hook.y_pos",
    "hook.style",
    "audio.meme",
    "audio.music",
    "audio.meme_mood",
    "audio.music_bed",
    "audio.meme_vol",
    "audio.music_db",
}

PER_CLIP_FIELDS = {
    "start",
    "end",
    "hook.text",
    "caption.words",
    "facecam_override",
    "facecam.focus",
}

ALLOWED_LAYOUTS = {"blur", "zoom", "crop", "gameplay", "stack"}


def _stack_defaults(split_ratio: float = DEFAULT_SPLIT_RATIO, cam_swap: bool = False,
                    split_divider: bool = DEFAULT_SPLIT_DIVIDER) -> tuple[float, float]:
    """Proportional (hook_y, cap_y) for stack layout based on configured output height."""
    oh = max(2, int(getattr(config, "OUT_H", 1920) or 1920))
    ratio = max(0.20, min(0.80, float(DEFAULT_SPLIT_RATIO if split_ratio is None else split_ratio)))
    cam_h = max(2, min(oh - 2, int(round(oh * ratio / 2.0)) * 2))
    game_h = oh - cam_h
    div_h = max(2, int(round(oh * 0.003 / 2.0)) * 2) if split_divider else 0
    half_div = div_h / 2.0
    pad = max(6, int(round(oh * 0.008)))
    hook_off = int(round(oh * 0.042))
    if cam_swap:
        cam_top, cam_bot = game_h, oh
        game_top = 0
        hook_y = round(min(cam_bot - pad, cam_top + half_div + hook_off), 1)
        cap_y = round(game_top + game_h * 0.80, 1)
    else:
        cam_top, cam_bot = 0, cam_h
        game_top = cam_h
        hook_y = round(max(cam_top + pad, cam_bot - half_div - hook_off), 1)
        cap_y = round(game_top + game_h * 0.77, 1)
    return hook_y, cap_y


def default_edit(start=0.0, end=30.0, layout="blur") -> dict:
    oh = max(2, int(getattr(config, "OUT_H", 1920) or 1920))
    if layout == "stack":
        def_hook_y, def_cap_y = _stack_defaults(DEFAULT_SPLIT_RATIO, False, DEFAULT_SPLIT_DIVIDER)
    else:
        def_hook_y = round(oh * (300.0 / 1920.0), 1)
        def_cap_y = round(oh * (1630.0 / 1920.0), 1)
    return {
        "version": 1,
        "start": round(float(start), 2),
        "end": round(float(end), 2),
        "layout": layout if layout in ALLOWED_LAYOUTS else "blur",
        "split_ratio": DEFAULT_SPLIT_RATIO,
        "split_divider": DEFAULT_SPLIT_DIVIDER,
        "cam_swap": False,
        "facecam": None,
        "facecam_override": False,
        "caption": {
            "enabled": True,
            "style": "bounce",
            "font_size": 132,
            "y_pos": def_cap_y,
            "words": [],
            "words_edited": False
        },
        "hook": {
            "enabled": True,
            "text": "",
            "font_size": 96,
            "y_pos": def_hook_y,
            "style": "default"
        },
        "audio": {
            "meme": False,
            "music": False,
            "meme_mood": "none",
            "music_bed": "none",
            "meme_vol": 0.0,
            "music_db": -18.0
        }
    }


def init_edit(job: dict, clip: dict, words: list = None, facecam: list = None) -> dict:
    """Generate initial edit dict from scouted clip, job options, facecam, and words."""
    opts = job.get("options") or {}
    start = round(float(clip.get("start", 0.0)), 2)
    end = round(float(clip.get("end", start + 30.0)), 2)
    layout = clip.get("layout") or opts.get("layout") or "blur"
    if layout not in ALLOWED_LAYOUTS:
        layout = "blur"

    raw_sr = clip.get("split_ratio") if clip.get("split_ratio") is not None else opts.get("split_ratio")
    split_ratio = float(raw_sr if raw_sr is not None else DEFAULT_SPLIT_RATIO)
    split_ratio = round(max(0.20, min(0.80, split_ratio)), 2)
    split_divider = bool(clip.get("split_divider", opts.get("split_divider", DEFAULT_SPLIT_DIVIDER)))
    cam_swap = bool(clip.get("cam_swap") or opts.get("cam_swap", False))

    fc_raw = clip.get("facecam") or facecam or opts.get("facecam") or job.get("facecam")
    fc_dict = None
    if isinstance(fc_raw, dict) and all(k in fc_raw for k in ("x", "y", "w", "h")):
        fc_dict = {k: int(fc_raw[k]) for k in ("x", "y", "w", "h")}
    elif isinstance(fc_raw, (list, tuple)) and len(fc_raw) == 4:
        # Check if coordinates are normalized 0..1 or pixel integers
        sw = job.get("width") or 1920
        sh = job.get("height") or 1080
        if all(0.0 <= float(v) <= 1.0 for v in fc_raw):
            fc_dict = {
                "x": int(float(fc_raw[0]) * sw),
                "y": int(float(fc_raw[1]) * sh),
                "w": max(20, int(float(fc_raw[2]) * sw)),
                "h": max(20, int(float(fc_raw[3]) * sh)),
            }
        else:
            fc_dict = {
                "x": int(fc_raw[0]),
                "y": int(fc_raw[1]),
                "w": max(20, int(fc_raw[2])),
                "h": max(20, int(fc_raw[3])),
            }
    if fc_dict:
        focus = fc_raw.get("focus") if isinstance(fc_raw, dict) else clip.get("facecam_focus")
        if focus and isinstance(focus, (list, tuple)) and len(focus) == 2:
            try:
                fc_dict["focus"] = [int(focus[0]), int(focus[1])]
            except (ValueError, TypeError):
                fc_dict["focus"] = None
        else:
            fc_dict["focus"] = None
        try:
            fc_dict["zoom"] = round(max(1.0, min(3.0, float(fc_raw.get("zoom", 1.0) if isinstance(fc_raw, dict) else (clip.get("facecam_zoom") or 1.0)))), 2)
        except (ValueError, TypeError):
            fc_dict["zoom"] = 1.0

    # Normalize words
    cap_words = []
    w_src = clip.get("words") or words or []
    c_start = float(clip.get("start", 0.0) or 0.0)
    c_end = float(clip.get("end", c_start + 30.0) or (c_start + 30.0))
    for w in w_src:
        text = str(w.get("text") if "text" in w else w.get("w", ""))
        if roman.needs_roman(text):
            text = roman.to_roman(text)
        s = round(float(w.get("start") if "start" in w else w.get("s", 0.0)), 2)
        e = round(float(w.get("end") if "end" in w else w.get("e", s + 0.3)), 2)
        if text.strip() and (c_start - 2.0 <= s <= c_end + 2.0 or 0 <= s <= (c_end - c_start) + 2.0):
            cap_words.append({"text": text, "start": s, "end": e})

    hook_text = str(clip.get("hook") or "").strip()
    if roman.needs_roman(hook_text):
        hook_text = roman.to_roman(hook_text)
    hook_enabled = bool(clip.get("hook_enabled", bool(hook_text)))
    oh = max(2, int(getattr(config, "OUT_H", 1920) or 1920))
    if layout == "stack":
        def_hook_y, def_cap_y = _stack_defaults(split_ratio, cam_swap, split_divider)
        hook_y = float(clip["hook_y"]) if clip.get("hook_y") is not None else def_hook_y
    else:
        def_cap_y = round(oh * (1630.0 / 1920.0), 1)
        hook_y = float(clip["hook_y"]) if clip.get("hook_y") is not None else round(oh * (300.0 / 1920.0), 1)
    hook_style = str(clip.get("hook_style") or "default")
    hook_font_size = int(clip.get("hook_font_size") or 96)

    cap_style = clip.get("style") or opts.get("style") or "bounce"
    cap_enabled = clip.get("captions", opts.get("captions", True))
    cap_y = float(clip["cap_y"]) if clip.get("cap_y") is not None else def_cap_y

    meme_val = clip.get("meme") or opts.get("meme") or "none"
    music_val = clip.get("music") or opts.get("music") or "none"

    return {
        "version": int(clip.get("version", 0)) + 1,
        "start": start,
        "end": end,
        "layout": layout,
        "split_ratio": split_ratio,
        "split_divider": split_divider,
        "cam_swap": cam_swap,
        "facecam": fc_dict,
        "facecam_override": bool(clip.get("facecam_override", False)),
        "caption": {
            "enabled": bool(cap_enabled),
            "style": cap_style,
            "font_size": 132,
            "y_pos": cap_y,
            "words": cap_words,
            "words_edited": bool(clip.get("words_edited", False))
        },
        "hook": {
            "enabled": hook_enabled,
            "text": hook_text,
            "font_size": hook_font_size,
            "y_pos": hook_y,
            "style": hook_style
        },
        "audio": {
            "meme": meme_val != "none",
            "music": music_val != "none",
            "meme_mood": meme_val if meme_val != "none" else "none",
            "music_bed": music_val if music_val != "none" else "none",
            "meme_vol": float(clip.get("meme_vol") or opts.get("meme_vol") or 0.0),
            "music_db": float(clip.get("music_db") or opts.get("music_db") or -18.0)
        }
    }


def validate_edit(data: dict, src_w=1920, src_h=1080) -> dict:
    """Validate edit dictionary, clamping ranges and filling defaults safely."""
    out = default_edit()
    if not isinstance(data, dict):
        return out

    out["version"] = int(data.get("version", 1))

    # Times
    try:
        s = round(float(data.get("start", 0.0)), 2)
        e = round(float(data.get("end", s + 15.0)), 2)
        if s < 0:
            s = 0.0
        if e <= s:
            e = s + 0.5
        out["start"], out["end"] = s, e
    except (ValueError, TypeError):
        pass

    # Layout
    lay = data.get("layout")
    if lay in ALLOWED_LAYOUTS:
        out["layout"] = lay

    # Split ratio & divider
    try:
        sr = float(data.get("split_ratio", DEFAULT_SPLIT_RATIO))
        out["split_ratio"] = round(max(0.20, min(0.80, sr)), 2)
    except (ValueError, TypeError):
        pass

    out["split_divider"] = bool(data.get("split_divider", DEFAULT_SPLIT_DIVIDER))
    out["cam_swap"] = bool(data.get("cam_swap", False))

    # Facecam
    fc = data.get("facecam")
    if fc is None:
        out["facecam"] = None
    elif isinstance(fc, dict) and all(k in fc for k in ("x", "y", "w", "h")):
        try:
            x = max(0, min(src_w - 20, int(fc["x"])))
            y = max(0, min(src_h - 20, int(fc["y"])))
            w = max(20, min(src_w - x, int(fc["w"])))
            h = max(20, min(src_h - y, int(fc["h"])))
            focus = None
            if fc.get("focus") and isinstance(fc["focus"], (list, tuple)) and len(fc["focus"]) == 2:
                try:
                    fx = max(0, min(src_w, int(fc["focus"][0])))
                    fy = max(0, min(src_h, int(fc["focus"][1])))
                    focus = [fx, fy]
                except (ValueError, TypeError):
                    focus = None
            try:
                zoom = max(1.0, min(3.0, float(fc.get("zoom", 1.0))))
            except (ValueError, TypeError):
                zoom = 1.0
            out["facecam"] = {"x": x, "y": y, "w": w, "h": h, "focus": focus, "zoom": round(zoom, 2)}
        except (ValueError, TypeError):
            out["facecam"] = None
    out["facecam_override"] = bool(data.get("facecam_override", False))

    # Caption
    cap = data.get("caption") if isinstance(data.get("caption"), dict) else {}
    out["caption"]["enabled"] = bool(cap.get("enabled", True))
    out["caption"]["style"] = str(cap.get("style", "bounce"))
    try:
        fs = int(cap.get("font_size", 132))
        out["caption"]["font_size"] = max(40, min(240, fs))
    except (ValueError, TypeError):
        pass
    oh = max(2, int(getattr(config, "OUT_H", 1920) or 1920))
    cap_min_y = round(oh * (100.0 / 1920.0), 1)
    cap_max_y = round(oh * (1820.0 / 1920.0), 1)
    try:
        yp = float(cap.get("y_pos", round(oh * (1630.0 / 1920.0), 1)))
        out["caption"]["y_pos"] = round(max(cap_min_y, min(cap_max_y, yp)), 1)
    except (ValueError, TypeError):
        pass

    raw_words = cap.get("words")
    if isinstance(raw_words, list):
        clean_words = []
        for w in raw_words:
            if isinstance(w, dict):
                txt = str(w.get("text") if "text" in w else w.get("w", "")).strip()
                if roman.needs_roman(txt):
                    txt = roman.to_roman(txt)
                try:
                    ws = round(float(w.get("start") if "start" in w else w.get("s", 0.0)), 2)
                    we = round(float(w.get("end") if "end" in w else w.get("e", ws + 0.3)), 2)
                except (ValueError, TypeError):
                    continue
                if txt:
                    clean_words.append({"text": txt, "start": ws, "end": we})
        out["caption"]["words"] = clean_words
    out["caption"]["words_edited"] = bool(cap.get("words_edited", False))

    # Hook
    hk = data.get("hook") if isinstance(data.get("hook"), dict) else {}
    out["hook"]["enabled"] = bool(hk.get("enabled", True))
    htxt = str(hk.get("text", "")).strip()
    if roman.needs_roman(htxt):
        htxt = roman.to_roman(htxt)
    out["hook"]["text"] = htxt
    try:
        hfs = int(hk.get("font_size", 96))
        out["hook"]["font_size"] = max(36, min(160, hfs))
    except (ValueError, TypeError):
        pass
    hook_min_y = round(oh * (50.0 / 1920.0), 1)
    hook_max_y = round(oh * (1800.0 / 1920.0), 1)
    try:
        hyp = float(hk.get("y_pos", round(oh * (300.0 / 1920.0), 1)))
        out["hook"]["y_pos"] = round(max(hook_min_y, min(hook_max_y, hyp)), 1)
    except (ValueError, TypeError):
        pass
    out["hook"]["style"] = str(hk.get("style", "default"))

    # Audio
    aud = data.get("audio") if isinstance(data.get("audio"), dict) else {}
    out["audio"]["meme"] = bool(aud.get("meme", False))
    out["audio"]["music"] = bool(aud.get("music", False))
    out["audio"]["meme_mood"] = str(aud.get("meme_mood", "none"))
    out["audio"]["music_bed"] = str(aud.get("music_bed", "none"))
    try:
        out["audio"]["meme_vol"] = round(float(aud.get("meme_vol", 0.0)), 1)
    except (ValueError, TypeError):
        pass
    try:
        out["audio"]["music_db"] = round(float(aud.get("music_db", -18.0)), 1)
    except (ValueError, TypeError):
        pass

    return out


def deep_merge_global(target: dict, patch: dict) -> dict:
    """Deep-merge GLOBAL fields from patch into target.
    PER-CLIP fields (start, end, hook.text, caption.words, per-clip facecam override) are strictly preserved.
    """
    res = copy.deepcopy(target)
    if not isinstance(patch, dict):
        return res

    # Layout & Split ratio & Divider & Panel swap
    if "layout" in patch and patch["layout"] in ALLOWED_LAYOUTS:
        res["layout"] = patch["layout"]
    if "split_ratio" in patch:
        try:
            sr = float(patch["split_ratio"])
            res["split_ratio"] = round(max(0.20, min(0.80, sr)), 2)
        except (ValueError, TypeError):
            pass
    if "split_divider" in patch:
        res["split_divider"] = bool(patch["split_divider"])
    if "cam_swap" in patch:
        res["cam_swap"] = bool(patch["cam_swap"])

    # Global facecam override and style (zoom is global style; x/y/w/h apply if no per-clip override)
    if "facecam" in patch:
        fc = patch["facecam"]
        if fc is None and not res.get("facecam_override", False):
            res["facecam"] = None
        elif isinstance(fc, dict):
            if res.get("facecam") is None and not res.get("facecam_override", False):
                res["facecam"] = copy.deepcopy(fc)
            elif isinstance(res.get("facecam"), dict):
                if "zoom" in fc:
                    try:
                        res["facecam"]["zoom"] = round(max(1.0, min(3.0, float(fc["zoom"]))), 2)
                    except (ValueError, TypeError):
                        pass
                if not res.get("facecam_override", False):
                    for k in ("x", "y", "w", "h"):
                        if k in fc:
                            try:
                                res["facecam"][k] = int(fc[k])
                            except (ValueError, TypeError):
                                pass
                    if "focus" in fc and fc["focus"] is not None:
                        res["facecam"]["focus"] = fc["focus"]

    # Caption global fields
    if "caption" in patch and isinstance(patch["caption"], dict):
        p_cap = patch["caption"]
        if "enabled" in p_cap:
            res["caption"]["enabled"] = bool(p_cap["enabled"])
        if "style" in p_cap and p_cap["style"]:
            res["caption"]["style"] = str(p_cap["style"])
        if "font_size" in p_cap:
            try:
                res["caption"]["font_size"] = max(40, min(240, int(p_cap["font_size"])))
            except (ValueError, TypeError):
                pass
        if "y_pos" in p_cap:
            oh = max(2, int(getattr(config, "OUT_H", 1920) or 1920))
            try:
                res["caption"]["y_pos"] = round(max(round(oh * (100.0 / 1920.0), 1), min(round(oh * (1820.0 / 1920.0), 1), float(p_cap["y_pos"]))), 1)
            except (ValueError, TypeError):
                pass
        # Note: p_cap["words"] is PER-CLIP and intentionally IGNORED here!

    # Hook global fields
    if "hook" in patch and isinstance(patch["hook"], dict):
        p_hk = patch["hook"]
        if "enabled" in p_hk:
            res["hook"]["enabled"] = bool(p_hk["enabled"])
        if "font_size" in p_hk:
            try:
                res["hook"]["font_size"] = max(36, min(160, int(p_hk["font_size"])))
            except (ValueError, TypeError):
                pass
        if "y_pos" in p_hk:
            oh = max(2, int(getattr(config, "OUT_H", 1920) or 1920))
            try:
                res["hook"]["y_pos"] = round(max(round(oh * (50.0 / 1920.0), 1), min(round(oh * (1800.0 / 1920.0), 1), float(p_hk["y_pos"]))), 1)
            except (ValueError, TypeError):
                pass
        if "style" in p_hk and p_hk["style"]:
            res["hook"]["style"] = str(p_hk["style"])
        # Note: p_hk["text"] is PER-CLIP and intentionally IGNORED here!

    # Audio global fields
    if "audio" in patch and isinstance(patch["audio"], dict):
        p_aud = patch["audio"]
        if "meme" in p_aud:
            res["audio"]["meme"] = bool(p_aud["meme"])
        if "music" in p_aud:
            res["audio"]["music"] = bool(p_aud["music"])
        if "meme_mood" in p_aud:
            res["audio"]["meme_mood"] = str(p_aud["meme_mood"])
        if "music_bed" in p_aud:
            res["audio"]["music_bed"] = str(p_aud["music_bed"])
        if "meme_vol" in p_aud:
            try:
                res["audio"]["meme_vol"] = round(float(p_aud["meme_vol"]), 1)
            except (ValueError, TypeError):
                pass
        if "music_db" in p_aud:
            try:
                res["audio"]["music_db"] = round(float(p_aud["music_db"]), 1)
            except (ValueError, TypeError):
                pass

    res["version"] = int(res.get("version", 1)) + 1
    return res


def load_edit(path: Path | str) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return validate_edit(raw)
    except Exception:
        return None


def save_edit(path: Path | str, data: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    valid = validate_edit(data)
    valid["version"] = int(data.get("version", 1))
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(valid, indent=2), encoding="utf-8")
    tmp.replace(p)
    return valid

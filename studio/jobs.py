"""Job pipeline: get video -> audio -> transcript -> AI picks -> rendered clips. One heavy task at a time."""
import collections
import json
import os
import math
import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from . import accounts, ai, brain, config, creators, ctx, edit, facecam, games, load, media, meme, quality, render, roman, rules, safe, transcribe, usage

class _FairQueue:
    """Whose video is next. Each person has their own line and the lines take turns, so someone who adds ten
    videos does not make everyone else wait behind all ten. Small tasks (re-making one clip) go first."""

    def __init__(self):
        self.cv = threading.Condition()
        self.quick = collections.deque()
        self.lines = {}                       # user -> their waiting videos, oldest first
        self.turn = collections.deque()       # users with something waiting, in the order they will be served
        self.epoch = 0                        # goes up whenever the line moves
        self._pos = (0.0, {})
        self.running = 0                      # videos being worked on right now
        self.last_start = 0.0
        self.held = False                     # videos are waiting only because the machine is near full

    def _room(self) -> bool:
        """May one more video start? One always may. More only while the machine has room, and a few seconds
        after the last one started, so its weight is already showing in the measurements."""
        if getattr(load, "_force", None) is None and (getattr(config, "BYPASS_QUEUE_HOLD", True) or not config.ACCOUNTS):
            return self.running < max(1, config.JOB_WORKERS)
        if self.running == 0:
            return True
        return self.running < config.JOB_WORKERS and time.time() - self.last_start > 6 and not load.busy()

    def done(self):
        with self.cv:
            self.running = max(0, self.running - 1)
            self.cv.notify_all()

    def put(self, item, user=None):
        with self.cv:
            if user is None:
                self.quick.append(item)
            else:
                if user not in self.lines:
                    self.lines[user] = collections.deque()
                    self.turn.append(user)
                self.lines[user].append(item)
            self.epoch += 1
            self.cv.notify_all()

    def get(self, quick_only=False):
        with self.cv:
            while True:
                if self.quick:
                    self.epoch += 1
                    return self.quick.popleft()
                if not quick_only and self.turn:
                    if self._room():
                        break
                    if not self.held:
                        self.held = True
                        self.epoch += 1
                elif self.held:
                    self.held = False
                    self.epoch += 1
                self.cv.wait(1.0)             # looked at again every second: the line moves the moment there is room
            self.epoch += 1
            self.held = False
            self.running += 1
            self.last_start = time.time()
            user = self.turn.popleft()
            item = self.lines[user].popleft()
            if self.lines[user]:
                self.turn.append(user)
            else:
                del self.lines[user]
            return item

    def ahead(self) -> dict:
        """job id -> how many videos will start before it. Worked out at most once a second."""
        with self.cv:
            if time.time() - self._pos[0] < 1:
                return self._pos[1]
            out, lines, order, n = {}, {u: list(q) for u, q in self.lines.items()}, list(self.turn), 0
            depth = 0
            while order:
                nxt = []
                for u in order:
                    if depth < len(lines[u]):
                        out[lines[u][depth][1]] = n
                        n += 1
                        if depth + 1 < len(lines[u]):
                            nxt.append(u)
                order, depth = nxt, depth + 1
            self._pos = (time.time(), out)
            return out


_q = _FairQueue()
_live_q: "queue.Queue" = queue.Queue()   # live clips never wait behind a long video
_jobs: dict = {}
_by_user: dict = {}                       # user -> {job id: job}: listing one person's videos never scans everyone's
_lock = threading.RLock()
on_clip_ready: list = []   # callbacks (job, clip), run once when a clip first finishes

DEFAULTS = {"max_clips": 6, "min_len": 15, "max_len": 60, "style": "bounce", "layout": "blur", "quality": 720,
            "range_start": None, "range_end": None, "custom_ranges": None, "timestamp_mode": "direct",
            "captions": True, "meme": "none", "music": "none", "format": "moment", "cam_delay": 0.0,
            "meme_vol": None, "music_db": None, "cam_pos": "auto",
            "split_ratio": 0.35, "split_divider": True, "cam_swap": False, "cap_anchor": "safe_bottom", "cam_fallback": "blur", "facecam": None, "cap_y": None,
            "review_mode": "auto", "speech_lang": "auto"}


LIMITS = {"max_clips": (1, 20), "min_len": (5, 120), "max_len": (10, 180), "quality": (360, 1080)}


def _parse_time_sec(val):
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(val) and val >= 0 else None
    s = str(val).strip().lower()
    if s in ("start", "end", "none", "full", ""):
        return None

    import re
    h = re.search(r"(\d+(?:\.\d+)?)\s*h", s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*m", s)
    sec = re.search(r"(\d+(?:\.\d+)?)\s*s", s)
    if h or m or sec:
        tot = 0.0
        if h: tot += float(h.group(1)) * 3600
        if m: tot += float(m.group(1)) * 60
        if sec: tot += float(sec.group(1))
        return tot

    if ":" in s:
        try:
            parts = [float(x) for x in s.split(":") if x.strip()]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            if len(parts) == 1:
                return parts[0]
        except ValueError:
            return None

    if "." in s:
        parts = [x.strip() for x in s.split(".") if x.strip()]
        if len(parts) >= 3:
            try:
                nums = [float(p) for p in parts]
                if len(nums) == 3:
                    return nums[0] * 3600 + nums[1] * 60 + nums[2]
                elif len(nums) == 4:
                    return nums[0] * 86400 + nums[1] * 3600 + nums[2] * 60 + nums[3]
            except ValueError:
                return None
        elif len(parts) == 2:
            try:
                n0 = int(parts[0])
                n1 = int(parts[1])
                # M.S e.g. 23.3 -> 23m 3s = 1383s, 23.30 -> 23m 30s = 1410s
                if len(parts[1]) <= 2 and n0 < 60 and n1 < 60:
                    return float(n0 * 60 + n1)
                else:
                    return float(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    return None

    try:
        f = float(s)
        return f if math.isfinite(f) and f >= 0 else None
    except ValueError:
        return None


def clean_options(options) -> dict:
    """Only known settings, each inside its allowed range. Raises ValueError on nonsense."""
    out = {}
    for k, v in (options if isinstance(options, dict) else {}).items():
        if k == "captions":
            if not isinstance(v, bool):
                raise ValueError("Captions must be on or off.")
            out[k] = v
            continue
        if k not in DEFAULTS or v in ("", None):
            continue
        if k in LIMITS:
            lo, hi = LIMITS[k]
            v = max(lo, min(hi, int(float(v))))
        elif k == "style":
            v = v if v in render.STYLES else DEFAULTS["style"]
        elif k == "review_mode":
            v = v if v in ("auto", "review", "review_first") else "auto"
        elif k == "speech_lang":
            sv = str(v).strip().lower()
            v = sv if sv in ("auto", "en", "hi", "hinglish") else "auto"
        elif k == "layout":
            v = v if v in render.LAYOUTS else DEFAULTS["layout"]
        elif k == "cam_delay":
            try:
                v = round(max(-3.0, min(3.0, float(v))), 2)
            except (ValueError, TypeError):
                v = 0.0
        elif k == "meme":
            v = v if v in meme.MEME_MOODS else "none"
        elif k == "music":
            v = v if v in meme.MUSIC_BEDS else "none"
        elif k == "meme_vol":
            try:
                out[k] = round(max(-30.0, min(10.0, float(v))), 1)
            except (ValueError, TypeError):
                pass
            continue
        elif k == "music_db":
            try:
                out[k] = round(max(-40.0, min(0.0, float(v))), 1)
            except (ValueError, TypeError):
                pass
            continue
        elif k == "cam_pos":
            out[k] = v if v in ("auto", "top-right", "top-left", "bottom-right", "bottom-left") else "auto"
            continue
        elif k == "split_ratio":
            try:
                out[k] = round(max(0.2, min(0.8, float(v))), 2)
            except (ValueError, TypeError):
                pass
            continue
        elif k == "split_divider":
            out[k] = bool(v)
            continue
        elif k == "cam_swap":
            out[k] = bool(v)
            continue
        elif k == "cap_anchor":
            out[k] = v if v in ("safe_bottom", "seam", "top") else "safe_bottom"
            continue
        elif k == "cam_fallback":
            out[k] = v if v in ("blur", "zoom") else "blur"
            continue
        elif k == "cap_y":
            try:
                out[k] = int(float(v)) if v is not None else None
            except (ValueError, TypeError):
                pass
            continue
        elif k == "facecam":
            if isinstance(v, (list, tuple)) and len(v) == 4:
                try:
                    out[k] = [round(max(0.0, min(1.0, float(x))), 4) for x in v]
                except (ValueError, TypeError):
                    pass
            elif isinstance(v, dict) and all(axis in v for axis in ("x", "y", "w", "h")):
                try:
                    out[k] = [
                        round(max(0.0, min(1.0, float(v["x"]))), 4),
                        round(max(0.0, min(1.0, float(v["y"]))), 4),
                        round(max(0.01, min(1.0, float(v["w"]))), 4),
                        round(max(0.01, min(1.0, float(v["h"]))), 4),
                    ]
                except (ValueError, TypeError):
                    pass
            continue
        elif k == "format":
            v = v if v in ("moment", "callback", "waitforit", "reveal") else "moment"
        elif k == "timestamp_mode":
            out[k] = v if v in ("direct", "analyze") else "direct"
        elif k == "custom_ranges":
            if isinstance(v, list):
                ranges = []
                for item in v:
                    if isinstance(item, dict):
                        st = _parse_time_sec(item.get("start"))
                        en = _parse_time_sec(item.get("end"))
                        m = item.get("mode", options.get("timestamp_mode", "direct"))
                        if m not in ("direct", "analyze"):
                            m = "direct"
                        if st is not None and en is not None:
                            if en <= st:
                                raise ValueError("Each custom range end time must be after its start time.")
                            ranges.append({"start": round(st, 2), "end": round(en, 2), "mode": m})
                if ranges:
                    out[k] = ranges
            continue
        elif k in ("range_start", "range_end"):
            sec = _parse_time_sec(v)
            if sec is None or not math.isfinite(sec) or sec < 0:
                raise ValueError("That part of the video isn't valid.")
            v = round(sec, 2)
        out[k] = v
    if out.get("custom_ranges"):
        out["range_start"] = min(r["start"] for r in out["custom_ranges"])
        out["range_end"] = max(r["end"] for r in out["custom_ranges"])
        out["max_clips"] = max(out.get("max_clips", 6), len(out["custom_ranges"]))
    elif out.get("range_start") is not None and out.get("range_end") is not None:
        if out["range_end"] <= out["range_start"]:
            raise ValueError("The end of the chosen part must be after its start.")
        out["custom_ranges"] = [{
            "start": out["range_start"],
            "end": out["range_end"],
            "mode": out.get("timestamp_mode", "direct")
        }]
    elif out.get("range_end") is not None and out["range_end"] <= out.get("range_start", 0):
        raise ValueError("The end of the chosen part must be after its start.")
    return out


def job_dir(jid) -> Path:
    return Path(_jobs[jid]["dir"])


HIDDEN = ("dir", "user", "source_path", "recover")


def public(job) -> dict:
    """What the browser may see: no server paths, no owner id."""
    out = {k: v for k, v in job.items() if k not in HIDDEN}
    out["source"] = {k: v for k, v in job["source"].items() if k != "path"}
    out["clips"] = [{k: v for k, v in c.items() if k not in ("src", "fixed_words", "fixed_for")} for c in job["clips"]]
    ai = job.get("ai") or {}
    out["ai"] = {"tokens": ai.get("tokens", 0), "note": ai.get("note", ""), "used": bool(ai.get("model"))}   # never the model, never raw errors
    if job.get("status") == "queued":
        n = _q.ahead().get(job["id"])
        if n:
            out["stage"] = (f"High demand right now: {n} video{'s' if n != 1 else ''} ahead of yours" if _q.held else
                            f"Waiting in line: {n} video{'s' if n != 1 else ''} ahead of yours")
        elif n == 0:
            out["stage"] = "High demand right now: yours is next" if _q.held else "Starting download..."
    return out


def save(job):
    ctx.bump(job.get("user"))
    with _lock:
        d = Path(job["dir"])
        for _ in range(5):
            try:
                text = json.dumps(job, indent=1)
                break
            except RuntimeError:  # changed mid-dump by another thread
                time.sleep(0.05)
        else:
            return
        tmp = d / "job.json.tmp"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(d / "job.json")


def _set(job, **kw):
    with _lock:
        job.update(kw)
    save(job)


def _remap(path, folder: Path):
    """Paths saved before data moved into per-user folders (or on another machine): find the file by name."""
    if path and not Path(path).exists():
        cand = folder / Path(str(path).replace("\\", "/")).name
        if cand.exists():
            return str(cand)
    return path


def _load_file(f: Path, uid: str):
    try:
        job = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    job.update(dir=str(f.parent), user=uid)
    job.setdefault("kind", "video")
    job.setdefault("creator", None)
    job.setdefault("options", {}).setdefault("captions", True)
    if job.get("source_path"):
        job["source_path"] = _remap(job["source_path"], f.parent)
    if job.get("source", {}).get("path"):
        job["source"]["path"] = _remap(job["source"]["path"], f.parent)
    if job.get("status") in ("queued", "running") and job.get("kind") != "live":
        # Picked up again as soon as the server is back (jobs.start), from the last finished part of the transcript.
        job.update(status="queued", stage="Picking up where it stopped", error=None, resume=True)
    elif job.get("status") == "paused":
        job["stage"] = "Paused by you"
    if job.get("status") == "live":
        job.update(status="done", stage="Recording paused by a server update", was_live=True)
    buffered = job.get("kind") == "live" and any((f.parent / "buffer").glob("*.ts"))
    for c in job.get("clips", []):
        c.setdefault("captions", job["options"]["captions"])
        if c.get("src"):
            c["src"] = _remap(c["src"], f.parent / "clips")
        if c.get("status") in ("rendering", "rendering_final", "drafting", "queued", "approved"):
            if buffered and not c.get("src"):
                c["status"] = "queued"  # the stream is still on disk: live.recover() finishes these
                job["recover"] = True
            elif job.get("resume"):
                c["status"] = "queued"  # the video is being picked up again: this clip is made then
            elif job.get("kind") != "live" and (f.parent / "words.json").is_file():
                c["status"] = "queued"  # a single clip being re-made: jobs.start() makes it again
                job.setdefault("remake", []).append(c["id"])
            else:
                c.update(status="error", error="Interrupted by a restart. Open it and press Save and re-make.")
    _jobs[job["id"]] = job
    _by_user.setdefault(uid, {})[job["id"]] = job


def load_user(uid: str):
    d = config.USERS_DIR / uid
    for f in list((d / "jobs").glob("*/job.json")) + list((d / "creators").glob("*/*/*/job.json")):
        _load_file(f, uid)


def load_all():
    for d in config.USERS_DIR.iterdir():
        if d.is_dir():
            load_user(d.name)


def all_jobs() -> list:
    """Every user's jobs. Only for background work; never send this to a browser."""
    with _lock:
        return list(_jobs.values())


def list_jobs():
    u = ctx.uid()
    with _lock:
        return sorted((dict(j) for j in _by_user.get(u, {}).values()), key=lambda j: -j.get("created", 0))


def get(jid):
    """The job, only if it belongs to whoever this code is running for."""
    job = _jobs.get(jid)
    return job if job and job["user"] == ctx.uid() else None


def busy_count() -> int:
    u = ctx.uid()
    return sum(1 for j in list(_by_user.get(u, {}).values()) if j["kind"] == "video" and j["status"] in ("queued", "running"))


def waiting_total() -> int:
    return sum(1 for j in list(_jobs.values()) if j["kind"] == "video" and j["status"] == "queued")


def queue_epoch() -> int:
    return _q.epoch


def running_now() -> int:
    return _q.running


def create(source: dict, options: dict, enqueue=True, creator=None, kind="video") -> dict:
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:10]
    c = creators.get(creator) if creator else None
    if c:
        d = creators.root(c["id"]) / ("live" if kind == "live" else "videos") / jid
        base = {**DEFAULTS, **{k: c[k] for k in ("max_clips", "min_len", "max_len", "style", "layout", "captions")}}
    else:
        d, base, creator = config.user_dir(ctx.uid()) / "jobs" / jid, dict(DEFAULTS), None
    opts = {**base, **clean_options(options)}
    if opts["min_len"] >= opts["max_len"]:
        opts["min_len"] = max(5, opts["max_len"] - 5)
    job = {"id": jid, "user": ctx.uid(), "kind": kind, "creator": creator, "dir": str(d), "created": time.time(), "source": source,
           "options": opts, "status": "queued", "stage": "Waiting", "progress": 0.0,
           "title": source.get("name", ""), "uploader": "", "duration": 0, "clips": [], "error": None, "ai": {}}
    d.mkdir(parents=True)
    with _lock:
        _jobs[jid] = job
        _by_user.setdefault(job["user"], {})[jid] = job
    save(job)
    if enqueue:
        _q.put(("job", jid, None), user=job["user"])
    return job


def enqueue(job):
    save(job)
    _q.put(("job", job["id"], None), user=job["user"])


def submit(fn):
    """Run any heavy task (a render, a cleanup) on the single worker, in order."""
    _q.put(("fn", ctx.bind(fn), None))


def submit_live(fn):
    """Live-stream renders and cleanups: their own worker, so a long video can never hold them up."""
    _live_q.put(ctx.bind(fn))


def _live_worker():
    while True:
        fn = _live_q.get()
        try:
            fn()
        except Exception:  # noqa: BLE001
            traceback.print_exc()


_paused = set()


def is_paused(jid: str) -> bool:
    return jid in _paused


def pause(jid: str) -> dict:
    job = get(jid)
    if not job:
        raise ValueError("No such video")
    if job.get("status") == "paused":
        return job
    if job.get("status") in ("done", "error"):
        raise ValueError("This video is already finished or has stopped.")
    _paused.add(jid)
    _set(job, status="paused", stage="Paused by you")
    return job


def resume(jid: str) -> dict:
    job = get(jid)
    if not job:
        raise ValueError("No such video")
    if job.get("status") != "paused":
        raise ValueError("This video is not paused.")
    _paused.discard(jid)
    _set(job, status="queued", stage="Picking up where it stopped", resume=True, error=None)
    _q.put(("job", job["id"], None), user=job["user"])
    return job


def delete(jid):
    _paused.discard(jid)
    with _lock:
        job = _jobs.pop(jid, None)
        if job:
            _by_user.get(job["user"], {}).pop(jid, None)
    if job:
        ctx.bump(job["user"])
        shutil.rmtree(job["dir"], ignore_errors=True)
    return bool(job)


def request_render(jid, cid, changes: dict):
    job = _jobs[jid]
    clip = next(c for c in job["clips"] if c["id"] == cid)
    new = dict(clip)
    if "captions" in changes:
        new.update(clean_options({"captions": changes["captions"]}))
    for k in ("start", "end"):
        if k in changes:
            new[k] = round(float(changes[k]), 2)
    if "src_offset" in clip:  # live clip: can only move inside the saved piece of stream
        new["start"] = max(new["start"], clip["src_offset"])
        new["end"] = min(new["end"], clip["src_end"])
    if not (math.isfinite(new["start"]) and math.isfinite(new["end"])) or new["start"] < 0:
        raise ValueError("Those times aren't valid.")
    if not 3 <= new["end"] - new["start"] <= 180:
        raise ValueError("A clip can be 3 to 180 seconds long.")
    for k in ("hook", "title", "description", "style", "layout", "meme", "music", "format", "cap_anchor", "cam_fallback"):
        if k in changes:
            new[k] = str(changes[k])[:{"hook": 60, "title": 100, "description": 2000}.get(k, 30)]
    if "split_ratio" in changes:
        try:
            new["split_ratio"] = round(max(0.2, min(0.8, float(changes["split_ratio"]))), 2)
        except (ValueError, TypeError):
            pass
    if "split_divider" in changes:
        new["split_divider"] = bool(changes["split_divider"])
    if "cam_swap" in changes:
        new["cam_swap"] = bool(changes["cam_swap"])
    if "cap_y" in changes:
        try:
            new["cap_y"] = int(float(changes["cap_y"])) if changes["cap_y"] is not None else None
        except (ValueError, TypeError):
            pass
    if "hook_y" in changes:
        try:
            new["hook_y"] = int(float(changes["hook_y"])) if changes["hook_y"] is not None else None
        except (ValueError, TypeError):
            pass
    if "hook_enabled" in changes:
        new["hook_enabled"] = bool(changes["hook_enabled"])
    if "hook_font_size" in changes:
        try:
            new["hook_font_size"] = int(float(changes["hook_font_size"]))
        except (ValueError, TypeError):
            pass
    if "hook_max_w" in changes:
        try:
            new["hook_max_w"] = max(360, min(1020, int(float(changes["hook_max_w"]))))
        except (ValueError, TypeError):
            pass
    if "hook_style" in changes:
        new["hook_style"] = str(changes["hook_style"])
    if "meme_vol" in changes:
        try:
            new["meme_vol"] = round(max(-30.0, min(10.0, float(changes["meme_vol"]))), 1)
        except (ValueError, TypeError):
            pass
    if "music_db" in changes:
        try:
            new["music_db"] = round(max(-40.0, min(0.0, float(changes["music_db"]))), 1)
        except (ValueError, TypeError):
            pass
    if "facecam" in changes:
        fc = changes["facecam"]
        if isinstance(fc, (list, tuple)) and len(fc) == 4:
            new["facecam"] = [round(max(0.0, min(1.0, float(x))), 4) for x in fc]
        elif isinstance(fc, dict) and all(k in fc for k in ("x", "y", "w", "h")):
            new["facecam"] = [round(max(0.0, min(1.0, float(fc["x"]))), 4),
                              round(max(0.0, min(1.0, float(fc["y"]))), 4),
                              round(max(0.01, min(1.0, float(fc["w"]))), 4),
                              round(max(0.01, min(1.0, float(fc["h"]))), 4)]
        elif fc is None:
            new["facecam"] = None
    force_full = bool(changes.get("force_full"))
    words_dirty = bool(changes.get("words_dirty"))
    hook_dirty = bool(changes.get("hook_dirty"))

    if "words" in changes and isinstance(changes["words"], list) and (words_dirty or "words_dirty" not in changes):
        valid_words = []
        for w in changes["words"]:
            if isinstance(w, dict) and "w" in w:
                valid_words.append({
                    "w": str(w["w"]),
                    "s": round(float(w.get("s", 0.0)), 2),
                    "e": round(float(w.get("e", 0.0)), 2)
                })
        new["words"] = valid_words
        if words_dirty:
            new["words_edited"] = True
    elif force_full and not clip.get("words_edited"):
        # Re-apply latest caption normalization & Hinglish/Romanization pipeline on full remake
        new.pop("fixed_words", None)
        new.pop("fixed_for", None)

    if new.get("style") not in render.STYLES:
        new["style"] = DEFAULTS["style"]
    if new.get("layout") not in render.LAYOUTS:
        new["layout"] = DEFAULTS["layout"]
    if new.get("meme") not in meme.MEME_MOODS:
        new["meme"] = "none"
    if new.get("music") not in meme.MUSIC_BEDS:
        new["music"] = "none"
    if new.get("layout") in ("gameplay", "stack") and not new.get("facecam"):
        if job.get("facecam"):
            new["facecam"] = job["facecam"]

    raw_file = Path(job["dir"]) / "clips" / f"{cid}_raw.mp4"
    time_changed = (new.get("start") != clip.get("start") or new.get("end") != clip.get("end"))
    layout_changed = (
        new.get("layout") != clip.get("layout") or
        new.get("split_ratio") != clip.get("split_ratio") or
        new.get("split_divider") != clip.get("split_divider") or
        new.get("cam_swap") != clip.get("cam_swap") or
        new.get("facecam") != clip.get("facecam") or
        new.get("cam_fallback") != clip.get("cam_fallback")
    )
    audio_changed = (
        new.get("meme") != clip.get("meme") or
        new.get("music") != clip.get("music") or
        new.get("meme_vol") != clip.get("meme_vol") or
        new.get("music_db") != clip.get("music_db")
    )
    if force_full:
        new["_fast_caption_only"] = False
    elif raw_file.is_file():
        new["_fast_caption_only"] = not (time_changed or layout_changed or audio_changed)
    else:
        new["_fast_caption_only"] = False

    if force_full or layout_changed or time_changed or audio_changed:
        try:
            raw_file.unlink(missing_ok=True)
        except OSError:
            pass

    # On full remake without manual hook drag, let render.py apply the latest default below-split placement
    if force_full and not hook_dirty and new.get("layout") == "stack":
        new["hook_y"] = None

    # If switching FROM split-screen (stack) to a non-stack layout, reset stack-specific hook/caption Y if untouched
    old_layout = clip.get("layout")
    new_layout = new.get("layout")
    if old_layout == "stack" and new_layout != "stack":
        old_hy = clip.get("hook_y")
        if new.get("hook_y") == old_hy:
            new["hook_y"] = 300
        old_cy = clip.get("cap_y")
        if new.get("cap_y") == old_cy:
            new["cap_y"] = 1380 if new_layout in ("crop", "gameplay") else 1630

    edit_file = Path(job["dir"]) / "clips" / f"{cid}_edit.json"
    if edit_file.is_file():
        ed = edit.load_edit(edit_file)
        if ed:
            ed["start"] = new["start"]
            ed["end"] = new["end"]
            ed["layout"] = new["layout"]
            if "split_ratio" in new:
                ed["split_ratio"] = new["split_ratio"]
            if "split_divider" in new:
                ed["split_divider"] = bool(new["split_divider"])
            if "cam_swap" in new:
                ed["cam_swap"] = bool(new["cam_swap"])
            cap_ed = ed.setdefault("caption", {})
            if "captions" in new:
                cap_ed["enabled"] = bool(new["captions"])
            if "style" in new:
                cap_ed["style"] = new["style"]
            if "cap_font_size" in changes:
                try:
                    cap_ed["font_size"] = int(float(changes["cap_font_size"]))
                except (ValueError, TypeError):
                    pass
            if "cap_y" in new and new["cap_y"] is not None:
                cap_ed["y_pos"] = float(new["cap_y"])
            if words_dirty and "words" in new and isinstance(new["words"], list):
                cap_ed["words"] = [
                    {"text": str(w.get("w", "")), "start": float(w.get("s", 0.0)), "end": float(w.get("e", 0.0))}
                    for w in new["words"]
                ]
                cap_ed["words_edited"] = True
            hk_ed = ed.setdefault("hook", {})
            if "hook_enabled" in new:
                hk_ed["enabled"] = bool(new["hook_enabled"])
            if "hook" in new:
                hk_ed["text"] = str(new["hook"])
            if "hook_font_size" in new and new["hook_font_size"] is not None:
                hk_ed["font_size"] = int(new["hook_font_size"])
            if "hook_max_w" in new and new["hook_max_w"] is not None:
                hk_ed["max_w"] = max(360, min(1020, int(new["hook_max_w"])))
            if "hook_style" in new and new["hook_style"]:
                hk_ed["style"] = str(new["hook_style"])
            if force_full and not hook_dirty and new.get("layout") == "stack":
                hk_ed.pop("y_pos", None)
            elif "hook_y" in new and new["hook_y"] is not None:
                hk_ed["y_pos"] = float(new["hook_y"])
            aud_ed = ed.setdefault("audio", {})
            if "meme" in new:
                aud_ed["meme"] = bool(new["meme"] and new["meme"] != "none")
                aud_ed["meme_mood"] = str(new["meme"] or "none")
            if "music" in new:
                aud_ed["music"] = bool(new["music"] and new["music"] != "none")
                aud_ed["music_bed"] = str(new["music"] or "none")
            ed["version"] = int(ed.get("version", 1)) + 1
            edit.save_edit(edit_file, ed)

    dur = max(3.0, float(new["end"]) - float(new["start"]))
    est_sec = max(3, int(round(dur * (0.16 if new.get("_fast_caption_only") else (0.28 if render._nvenc_ok else 0.55)))))
    new["version"] = int(clip.get("version", 0)) + 1
    if force_full and not clip.get("words_edited"):
        clip.pop("fixed_words", None)
        clip.pop("fixed_for", None)
    clip.update(new, status="queued", error=None, error_detail=None, eta_sec=est_sec, render_started=time.time())
    save(job)
    _q.put(("clip", jid, cid))


def remake_job(jid: str, mode: str = "rerender", retranscribe: bool = False) -> dict:
    """Remake an existing video job using the already-downloaded video/audio files on disk,
    applying the latest code, layout rules, caption normalization, and AI model."""
    job = _jobs.get(jid)
    if not job or job.get("user") != ctx.uid():
        raise ValueError("No such video")
    if job.get("status") in ("queued", "running", "live"):
        raise ValueError("This video is still being processed. Wait until it finishes or pause it first.")
    d = Path(job["dir"])
    clips_dir = d / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    mode = (mode or "rerender").strip().lower()

    if mode == "rerender" and job.get("clips"):
        # Re-render every clip from the saved source video with the latest code & layout defaults
        for c in job["clips"]:
            cid = c["id"]
            # Delete stale _raw.mp4 and old edit.json geometry so latest renderer & below-split defaults apply
            for stale in (clips_dir / f"{cid}_raw.mp4", clips_dir / f"{cid}_edit.json"):
                try:
                    stale.unlink(missing_ok=True)
                except OSError:
                    pass
            c["_fast_caption_only"] = False
            c.pop("fixed_words", None)
            c.pop("fixed_for", None)
            c.pop("error_detail", None)
            if c.get("layout") == "stack":
                c["hook_y"] = None
            c["version"] = int(c.get("version", 0)) + 1
            dur = max(3.0, float(c.get("end", 30.0)) - float(c.get("start", 0.0)))
            est_sec = max(4, int(round(dur * (0.28 if render._nvenc_ok else 0.55))))
            c.update(status="queued", error=None, eta_sec=est_sec, render_started=time.time())
        job["error"] = None
        job.pop("error_detail", None)
        job["stage"] = f"Remaking {len(job['clips'])} clip{'s' if len(job['clips']) != 1 else ''} with latest code"
        save(job)
        for c in job["clips"]:
            _q.put(("clip", jid, c["id"]))
        return job

    # Full re-process using the saved local video on disk (no re-download!)
    if retranscribe:
        try:
            (d / "words.json").unlink(missing_ok=True)
        except OSError:
            pass
    # Clean old rendered clips so fresh clips are created cleanly
    for f in clips_dir.glob("*"):
        if f.is_file() and not f.name.startswith("piece_"):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
    job["clips"] = []
    job["done_blocks"] = 0
    job["error"] = None
    job.pop("error_detail", None)
    job.pop("ai", None)
    job["status"] = "queued"
    job["progress"] = 0.0
    job["stage"] = "Queued to remake with latest code (using saved video)"
    job["remaking"] = True
    save(job)
    _q.put(("job", jid))
    return job


def render_one(job, clip, words, quality="final"):
    d = Path(job["dir"])
    dur = max(3.0, float(clip.get("end", 30.0)) - float(clip.get("start", 0.0)))
    is_fast = bool(clip.get("_fast_caption_only")) or quality == "draft"
    est_sec = max(3, int(round(dur * (0.16 if is_fast else (0.28 if render._nvenc_ok else 0.55)))))
    if quality == "draft":
        clip["status"] = "drafting"
    else:
        clip["status"] = "rendering_final"
    clip["eta_sec"] = est_sec
    clip["render_started"] = time.time()
    save(job)
    try:
        # Resolve an existing source video on disk; re-fetch piece if missing
        src = None
        if clip.get("src") and Path(clip["src"]).is_file():
            src = Path(clip["src"])
        elif job.get("source_path") and Path(job["source_path"]).is_file():
            src = Path(job["source_path"])
            clip.pop("src", None)
            clip.pop("src_offset", None)
            clip.pop("src_end", None)
        if src is None:
            clip.pop("src", None)
            _piece_again(job, clip)
            src = Path(clip["src"])
        off = clip.get("src_offset", 0.0)
        if not clip.get("src") and job.get("duration"):
            clip["end"] = min(clip["end"], job["duration"])
        if clip.get("layout") in ("gameplay", "stack") and not clip.get("facecam"):
            clip_win = (max(0.0, float(clip["start"]) - off), max(0.0, float(clip["end"]) - off))
            face = job.get("facecam") or _facecam(job, src=src, window=clip_win)
            if face:
                clip["facecam"] = face
        edit_file = d / "clips" / f"{clip['id']}_edit.json"
        existing_ed = edit.load_edit(edit_file) if edit_file.is_file() else None

        # Respect explicit hook_enabled=False from either edit.json or clip dict
        if existing_ed and "enabled" in (existing_ed.get("hook") or {}):
            eff_hook_enabled = bool(existing_ed["hook"]["enabled"])
        else:
            eff_hook_enabled = bool(clip.get("hook_enabled", bool(clip.get("hook"))))

        local = dict(clip, start=clip["start"] - off, end=clip["end"] - off,
                     hook=roman.to_roman(clip.get("hook") or "") if eff_hook_enabled else "",
                     hook_enabled=eff_hook_enabled,
                     cam_delay=float(clip.get("cam_delay", 0.0) or 0.0))

        # Authoritative caption words: user-edited words in edit.json ALWAYS win over raw/cached transcript
        ed_cap = (existing_ed or {}).get("caption") or {}
        if existing_ed and ed_cap.get("words_edited") and isinstance(ed_cap.get("words"), list):
            near = roman.words([
                {
                    "w": str(w.get("text") if "text" in w else w.get("w", "")).strip(),
                    "s": round(float(w.get("start") if "start" in w else w.get("s", 0.0)), 2),
                    "e": round(float(w.get("end") if "end" in w else w.get("e", 0.0)), 2),
                }
                for w in ed_cap["words"]
                if str(w.get("text") if "text" in w else w.get("w", "")).strip()
            ])
        elif clip.get("captions", True):
            if clip.get("words"):
                near = [dict(w, s=float(w["s"]), e=float(w["e"])) for w in roman.words(clip["words"])]
            elif clip.get("src_words"):
                raw = [dict(w) for w in clip["src_words"] if local["start"] - 0.5 <= w["s"] <= local["end"] + 0.5]
                try:
                    raw = brain.fix_caption_words(raw, job.get("title") or "", job.get("lang") or "")
                except Exception:
                    pass
                near = [dict(w, s=w["s"], e=w["e"]) for w in roman.words(raw)]
            else:
                near = _caption_words(job, clip, words, off)
        else:
            near = []

        # Ensure {cid}_edit.json exists and stays synced with cleaned captions unless user manually edited words
        if not existing_ed:
            if clip.get("facecam") and not clip.get("facecam_focus") and src.is_file():
                try:
                    foc = facecam.find_clip_face_focus(src, clip["facecam"], clip["start"], clip["end"])
                    if foc:
                        clip["facecam_focus"] = foc
                except Exception:
                    pass
            ed = edit.init_edit(job, clip, near, clip.get("facecam"))
            edit.save_edit(edit_file, ed)
        else:
            ed = existing_ed
            if "captions" in clip:
                ed["caption"]["enabled"] = bool(clip["captions"])
            if "hook_enabled" in clip:
                ed.setdefault("hook", {})["enabled"] = bool(clip["hook_enabled"])
            if near and not ed["caption"].get("words_edited"):
                ed["caption"]["words"] = [
                    {
                        "text": str(w.get("w") if "w" in w else w.get("text", "")),
                        "start": round(float(w.get("s") if "s" in w else w.get("start", 0.0)), 2),
                        "end": round(float(w.get("e") if "e" in w else w.get("end", 0.0)), 2),
                    }
                    for w in near
                    if str(w.get("w") if "w" in w else w.get("text", "")).strip()
                ]
            edit.save_edit(edit_file, ed)

        out = render.render_clip(src, d / "clips", clip["id"], local, near, job.get("width") or 1920, job.get("height") or 1080, quality=quality)
        clip.pop("eta_sec", None)
        clip.pop("render_started", None)
        clip.pop("error_detail", None)
        if quality == "draft":
            clip.update(out, status="awaiting_review", error=None)
            clip["version"] = clip.get("version", 0) + 1
            clip["words"] = near
        else:
            final_status = "done"
            if out.get("qc") and out["qc"].get("status") == "FAIL":
                final_status = "qc_flagged"
                issues = out["qc"].get("issues", [])
                clip["error"] = ", ".join(issues) if issues else "Quality control check failed"
            clip.update(out, status=final_status, version=clip.get("version", 0) + 1, error=None if final_status == "done" else clip.get("error"))
            clip["words"] = near
            post = {k: clip[k] for k in ("title", "description", "hashtags", "hook", "start", "end", "score") if k in clip}
            (d / "clips" / f"{clip['id']}.json").write_text(json.dumps(post, indent=1), encoding="utf-8")
            if clip["version"] == 1 and final_status == "done":
                clip.setdefault("made_at", time.time())
                creators.log(job["creator"], f"Clip ready (score {clip['score']}): {clip.get('title') or clip.get('hook')}")
                save(job)
                for fn in on_clip_ready:
                    try:
                        fn(job, clip)
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()
    except Exception as e:  # noqa: BLE001
        clip.pop("eta_sec", None)
        clip.pop("render_started", None)
        where = f"Rendering clip {clip.get('id', '')} ({'draft preview' if quality == 'draft' else 'final 1080p'})"
        detail = _classify_error(e, where=where)
        clip.update(status="failed" if quality == "draft" else "error", error=detail["what"], error_detail=detail)
        creators.log(job["creator"], f"Clip failed ({where}): {detail['what'][:200]}")
    _refresh_review_stage(job)
    save(job)


def _classify_error(e: Exception, where: str = "Processing") -> dict:
    """Turn any backend exception into a clear, actionable diagnostic object for the UI error popup."""
    raw = safe.friendly(e, 450)
    low = raw.lower()
    tip = "Try again, or open the error details below if this keeps happening."
    action = ""
    if any(k in low for k in ("ollama", "11434", "connection refused")) or ("connect" in low and ai.provider() == "ollama"):
        tip = "Make sure Ollama is running (`ollama serve`) and the model name matches `ollama list` in Settings → AI."
        action = "ai"
    elif any(k in low for k in ("did not answer with json", "json", "401", "403", "404", "429", "api key", "quota", "rate limit", "not found", "ai did not answer")):
        tip = "Check your AI service, model name, or API key under Settings → AI, then press Save and test."
        action = "ai"
    elif any(k in low for k in ("cuda", "cublas", "cudnn", "out of memory", "vram", "nvenc")):
        tip = "Your GPU ran low on VRAM or hit a driver issue. Check Settings → Hardware & Speech (or switch to a smaller Whisper model / CPU)."
        action = "hardware"
    elif any(k in low for k in ("sign in to confirm", "bot", "cookies", "youtube", "403", "format", "yt-dlp", "could not fetch")):
        tip = "YouTube blocked or interrupted the download. Refresh your YouTube cookies in Settings → YouTube & Cookies, or retry."
        action = "youtube"
    elif any(k in low for k in ("permission", "being used by another process", "winerror 32", "access is denied")):
        tip = "Windows had the video file locked (often by the browser player or antivirus). Close the preview and click Save draft / Remake again."
    elif "ffmpeg" in low:
        tip = "FFmpeg failed while encoding this clip. Check that the source video isn't corrupted and disk space isn't full."
    return {
        "where": where or "Processing",
        "what": raw or "Something went wrong.",
        "tip": tip,
        "action": action,
        "type": type(e).__name__,
    }


def live_runtime() -> dict:
    """Live snapshot of active AI, Speech-to-Text, and GPU/Video rendering for the Home & Clip glowing indicators."""
    ai_rt = ai.runtime_state()
    asr_rt = transcribe.live_state()
    gpu_hw = transcribe.gpu_status() if hasattr(transcribe, "gpu_status") else {}
    has_nvenc = render.check_nvenc() if hasattr(render, "check_nvenc") else False
    try:
        gpu_on = accounts.settings().get("gpu_enabled", True) and getattr(config, "USE_GPU", True)
    except Exception:
        gpu_on = getattr(config, "USE_GPU", True)
    has_gpu = bool(has_nvenc or gpu_hw.get("available"))
    gpu_enabled = bool(gpu_on and has_gpu)

    # Check if any job or clip for this user is actively rendering/encoding video
    rendering_clips = 0
    running_jobs = 0
    uid = ctx.uid_or_none() if hasattr(ctx, "uid_or_none") else None
    for j in list(_jobs.values()):
        if uid and j.get("user") != uid:
            continue
        if j.get("status") in ("running", "queued"):
            running_jobs += 1
        for c in j.get("clips") or []:
            if c.get("status") in ("rendering", "rendering_final", "drafting"):
                rendering_clips += 1

    gpu_active = bool(gpu_enabled and (
        rendering_clips > 0
        or (asr_rt["active"] and asr_rt.get("device") == "cuda")
        or (ai_rt["active"] and ai_rt.get("provider") == "ollama")
    ))
    gpu_short = gpu_hw.get("name") or ("NVIDIA NVENC" if has_nvenc else "CPU")
    gpu_short = gpu_short.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").strip()
    epoch = int(ai_rt.get("epoch", 0)) + int(asr_rt.get("epoch", 0)) + rendering_clips * 17 + running_jobs * 7
    short_label = ai_rt["label"].replace(" (on this machine)", "").replace(" (OpenAI-compatible)", "")
    return {
        "epoch": epoch,
        "busy": bool(ai_rt["active"] or asr_rt["active"] or rendering_clips > 0 or running_jobs > 0),
        "ai": {
            "active": ai_rt["active"],
            "ready": ai_rt["ready"],
            "provider_id": ai_rt["provider"],
            "provider": short_label,
            "label": ai_rt["label"],
            "model": ai_rt["model"],
            "active_model": ai_rt["model"] if ai_rt["active"] else "",
            "purpose": ai_rt["purpose"],
            "active_purpose": ai_rt["purpose"],
        },
        "asr": {
            "active": asr_rt["active"],
            "engine": asr_rt["engine"],
            "label": "Faster-Whisper" if asr_rt["engine"] == "faster-whisper" else "Parakeet v2",
            "model": asr_rt["model"],
            "device": asr_rt["device"],
            "compute_type": asr_rt["compute_type"],
        },
        "gpu": {
            "active": gpu_active,
            "available": has_gpu,
            "enabled": gpu_enabled,
            "name": gpu_short,
            "full_name": gpu_hw.get("name") or ("NVIDIA GPU" if has_nvenc else "CPU"),
            "vram_mb": gpu_hw.get("total_mb") or 0,
            "nvenc": bool(has_nvenc),
            "rendering_clips": rendering_clips,
        },
    }


def regen_clip_titles(jid: str, cid: str) -> dict:
    """Regenerate 3 fresh scroll-stopping heading + post title options for a single clip."""
    job = _jobs.get(jid)
    if not job or job.get("user") != ctx.uid():
        raise ValueError("No such video")
    clip = next((c for c in job.get("clips") or [] if c["id"] == cid), None)
    if not clip:
        raise ValueError("No such clip")
    d = Path(job["dir"])
    ef = d / "clips" / f"{cid}_edit.json"
    ed = edit.load_edit(ef) if ef.is_file() else None

    words_src = []
    if ed and (ed.get("caption") or {}).get("words"):
        words_src = ed["caption"]["words"]
    elif clip.get("words"):
        words_src = clip["words"]
    else:
        wf = d / "words.json"
        if wf.is_file():
            try:
                all_w = json.loads(wf.read_text(encoding="utf-8"))
                s0, e0 = float(clip.get("start", 0.0)), float(clip.get("end", 30.0))
                words_src = [w for w in all_w if s0 - 1.0 <= float(w.get("s", 0.0)) <= e0 + 1.0]
            except Exception:
                words_src = []
    text = " ".join(str(w.get("text") or w.get("w") or "").strip() for w in words_src).strip()
    if not text:
        text = f"{clip.get('hook') or ''} {clip.get('title') or ''} {clip.get('reason') or ''}".strip()

    lang = (job.get("options") or {}).get("speech_lang") or job.get("lang") or ""
    res = brain.regenerate_titles(
        text=text,
        title=job.get("title") or "",
        uploader=job.get("uploader") or "",
        game=job.get("game"),
        lang=lang,
        current_hook=(ed.get("hook", {}).get("text") if ed else None) or clip.get("hook") or "",
        current_title=clip.get("title") or "",
    )
    clip["hook_options"] = res["hook_options"]
    clip["title_options"] = res["title_options"]
    save(job)
    return res


def _refresh_review_stage(job: dict) -> None:
    """Keep job['stage'] accurate as clips move from awaiting_review -> approved/rendering_final -> done."""
    if job.get("status") != "done":
        return
    clips = job.get("clips") or []
    if not clips:
        return
    awaiting = sum(1 for c in clips if c.get("status") in ("awaiting_review", "drafting"))
    rendering = sum(1 for c in clips if c.get("status") in ("approved", "rendering_final", "rendering", "queued"))
    if rendering:
        job["stage"] = f"Rendering {rendering} approved clip{'s' if rendering != 1 else ''}"
    elif awaiting:
        job["stage"] = f"{awaiting} clip{'s' if awaiting != 1 else ''} ready for review"
    else:
        job["stage"] = "Done"


def _mark_clip_approved(clip: dict) -> None:
    dur = max(3.0, float(clip.get("end", 30.0)) - float(clip.get("start", 0.0)))
    est = max(4, int(round(dur * (0.28 if render._nvenc_ok else 0.55))))
    clip["status"] = "approved"
    clip["error"] = None
    clip["eta_sec"] = est
    clip["render_started"] = time.time()


def approve_clip(jid: str, cid: str) -> dict:
    job = _jobs.get(jid)
    if not job or job.get("user") != ctx.uid():
        raise ValueError("No such video")
    clip = next((c for c in job["clips"] if c["id"] == cid), None)
    if not clip:
        raise ValueError("No such clip")
    if clip.get("status") == "rendering_final":
        return clip

    _mark_clip_approved(clip)
    _refresh_review_stage(job)
    save(job)
    _q.put(("clip_final", jid, cid))

    if (job.get("options") or {}).get("review_mode") == "review_first":
        edit_file = Path(job["dir"]) / "clips" / f"{cid}_edit.json"
        if edit_file.is_file():
            ed = edit.load_edit(edit_file)
            if ed:
                patch = {
                    "layout": ed.get("layout"),
                    "split_ratio": ed.get("split_ratio"),
                    "split_divider": ed.get("split_divider", True),
                    "cam_swap": ed.get("cam_swap", False),
                    "caption": {
                        "style": ed.get("caption", {}).get("style"),
                        "font_size": ed.get("caption", {}).get("font_size"),
                        "y_pos": ed.get("caption", {}).get("y_pos"),
                        "enabled": ed.get("caption", {}).get("enabled", True),
                    },
                    "hook": {
                        "enabled": ed.get("hook", {}).get("enabled", True),
                        "font_size": ed.get("hook", {}).get("font_size"),
                        "y_pos": ed.get("hook", {}).get("y_pos"),
                        "max_w": ed.get("hook", {}).get("max_w", 980),
                        "style": ed.get("hook", {}).get("style"),
                    },
                    "audio": ed.get("audio", {}),
                    "facecam": ed.get("facecam")
                }
                for other in job["clips"]:
                    if other["id"] != cid and other.get("status") not in ("approved", "rendering_final", "done"):
                        other_ef = Path(job["dir"]) / "clips" / f"{other['id']}_edit.json"
                        other_ed = edit.load_edit(other_ef) if other_ef.is_file() else edit.init_edit(job, other, None, other.get("facecam"))
                        merged = edit.deep_merge_global(other_ed, patch)
                        edit.save_edit(other_ef, merged)
                        _mark_clip_approved(other)
                        _q.put(("clip_final", jid, other["id"]))
                _refresh_review_stage(job)
                save(job)
    return clip


def approve_all(jid: str) -> list:
    job = _jobs.get(jid)
    if not job or job.get("user") != ctx.uid():
        raise ValueError("No such video")
    approved = []
    for c in job["clips"]:
        if c.get("status") in ("awaiting_review", "qc_flagged", "failed", "error", "drafting"):
            _mark_clip_approved(c)
            _q.put(("clip_final", jid, c["id"]))
            approved.append(c)
    if approved:
        _refresh_review_stage(job)
        save(job)
    return approved


def apply_to_all(jid: str, patch: dict, skip_draft: bool = False) -> list:
    job = _jobs.get(jid)
    if not job or job.get("user") != ctx.uid():
        raise ValueError("No such video")
    d = Path(job["dir"])
    updated = []
    for c in job["clips"]:
        if c.get("status") in ("approved", "rendering_final", "done"):
            continue
        cid = c["id"]
        ef = d / "clips" / f"{cid}_edit.json"
        ed = edit.load_edit(ef) if ef.is_file() else edit.init_edit(job, c, None, c.get("facecam"))
        merged = edit.deep_merge_global(ed, patch)
        edit.save_edit(ef, merged)
        if "layout" in merged:
            c["layout"] = merged["layout"]
        if "split_ratio" in merged:
            c["split_ratio"] = merged["split_ratio"]
        if "split_divider" in merged:
            c["split_divider"] = bool(merged["split_divider"])
        if "cam_swap" in merged:
            c["cam_swap"] = bool(merged["cam_swap"])
        if "hook" in merged and isinstance(merged["hook"], dict):
            hk = merged["hook"]
            c["hook_enabled"] = bool(hk.get("enabled", True))
            if hk.get("font_size") is not None:
                c["hook_font_size"] = int(hk["font_size"])
            if hk.get("y_pos") is not None:
                c["hook_y"] = float(hk["y_pos"])
            if hk.get("max_w") is not None:
                c["hook_max_w"] = int(hk["max_w"])
            if hk.get("style"):
                c["hook_style"] = str(hk["style"])
        if "caption" in merged and isinstance(merged["caption"], dict):
            cp = merged["caption"]
            c["captions"] = bool(cp.get("enabled", True))
            if cp.get("style"):
                c["style"] = str(cp["style"])
            if cp.get("y_pos") is not None:
                c["cap_y"] = float(cp["y_pos"])
        if "facecam" in merged and not merged.get("facecam_override"):
            fc = merged["facecam"]
            c["facecam"] = [fc["x"], fc["y"], fc["w"], fc["h"]] if fc else None
        if not skip_draft:
            c["status"] = "drafting"
            _q.put(("clip_draft", jid, cid))
        updated.append(c)
    if updated:
        save(job)
    return updated


def set_job_facecam(jid: str, facecam: dict | list) -> dict:
    job = _jobs.get(jid)
    if not job or job.get("user") != ctx.uid():
        raise ValueError("No such video")
    sw = job.get("width") or 1920
    sh = job.get("height") or 1080
    fc_dict = None
    if isinstance(facecam, dict) and all(k in facecam for k in ("x", "y", "w", "h")):
        fc_dict = {
            "x": max(0, min(sw - 20, int(facecam["x"]))),
            "y": max(0, min(sh - 20, int(facecam["y"]))),
            "w": max(20, min(sw, int(facecam["w"]))),
            "h": max(20, min(sh, int(facecam["h"]))),
        }
    elif isinstance(facecam, (list, tuple)) and len(facecam) == 4:
        if all(0.0 <= float(v) <= 1.0 for v in facecam):
            fc_dict = {
                "x": max(0, min(sw - 20, int(float(facecam[0]) * sw))),
                "y": max(0, min(sh - 20, int(float(facecam[1]) * sh))),
                "w": max(20, min(sw, int(float(facecam[2]) * sw))),
                "h": max(20, min(sh, int(float(facecam[3]) * sh))),
            }
        else:
            fc_dict = {
                "x": max(0, min(sw - 20, int(facecam[0]))),
                "y": max(0, min(sh - 20, int(facecam[1]))),
                "w": max(20, min(sw, int(facecam[2]))),
                "h": max(20, min(sh, int(facecam[3]))),
            }
    job["facecam"] = [fc_dict["x"], fc_dict["y"], fc_dict["w"], fc_dict["h"]] if fc_dict else None
    job["facecam_override"] = job["facecam"]
    d = Path(job["dir"])
    for c in job["clips"]:
        if c.get("status") in ("approved", "rendering_final", "done"):
            continue
        ef = d / "clips" / f"{c['id']}_edit.json"
        ed = edit.load_edit(ef) if ef.is_file() else edit.init_edit(job, c, None, job["facecam"])
        if not ed.get("facecam_override"):
            ed["facecam"] = fc_dict
            edit.save_edit(ef, ed)
            c["facecam"] = job["facecam"]
            c["status"] = "drafting"
            _q.put(("clip_draft", jid, c["id"]))
    save(job)
    return {"ok": True, "facecam": job["facecam"]}


def _caption_words(job, clip, words, off):
    span = [round(clip["start"], 2), round(clip["end"], 2), "en2"]      # "en2": captions made always-English
    if clip.get("fixed_for") == span and clip.get("fixed_words") and not any(
            roman.needs_roman(w.get("w", "")) for w in clip["fixed_words"]):
        raw = clip["fixed_words"]          # already tidied for this exact span: no second AI call
    else:
        raw = [w for w in words if clip["start"] - 1 <= w["s"] <= clip["end"] + 1]
        try:
            usage.set_context(job.get("creator"), job.get("title") or "")
            raw = brain.english_captions(raw, job.get("lang") or "", job.get("title") or "",
                                         youtube_english=lambda: _english_part(job, clip))
            clip.update(fixed_words=raw, fixed_for=span)
        except Exception:  # noqa: BLE001 - captions still render with the words as heard
            pass
    # the last word on it: whatever was said, nothing is drawn in Hindi or Urdu script
    return [dict(w, s=w["s"] - off, e=w["e"] - off) for w in roman.words(raw)]


CLIP_MINUTES = float(config.env("PEAKCUT_CLIP_MINUTES", "15"))     # the longest one clip may take to make


def _wait_for_clips(job, futures, began, workers):
    """Wait for a video's clips, updating live ETA and bounding hung clips."""
    left, given_up = list(futures), 0
    total = len(left)
    wait_t0 = time.time()
    last_eta_at = 0.0
    late = "This clip took too long and was stopped. Open it and press Save and re-make to try again."
    while left:
        if is_paused(job["id"]):
            raise media.JobPaused("Job paused by user")
        still = []
        for f, clip in left:
            if f.done():
                f.result()
                continue
            t = began.get(clip["id"])
            if (t and time.time() - t > CLIP_MINUTES * 60) or (not t and given_up >= workers):
                given_up += 1
                clip.update(status="error", error=late)
                clip.pop("eta_sec", None)
                creators.log(job["creator"], f"Clip stopped after {CLIP_MINUTES:g} minutes: {clip.get('title') or clip.get('hook') or clip['id']}")
                save(job)
                continue
            still.append((f, clip))
        left = still
        if left:
            now = time.time()
            if now - last_eta_at >= 2.0:
                last_eta_at = now
                done_n = max(0, total - len(left))
                if done_n > 0:
                    avg = (now - wait_t0) / done_n
                    eta = max(2, int(round(avg * len(left))))
                else:
                    eta = max(4, int(round(sum(max(5, int((c["end"] - c["start"]) * 0.4)) for _, c in left) / max(1, workers))))
                pct = 0.92 + 0.07 * (done_n / max(1, total))
                _set(job, progress=max(job.get("progress", 0.92), pct), eta_sec=eta)
            time.sleep(1)


def _english_part(job, clip):
    """YouTube's English captions for just this clip, for a video in another language. Fetched once per video."""
    if (job.get("source") or {}).get("type") != "url":
        return None
    f = Path(job["dir"]) / "words_en.json"
    if f.is_file():
        en = json.loads(f.read_text(encoding="utf-8"))
    else:
        try:
            en = media.youtube_english_words(job["source"]["url"]) or []
        except Exception:  # noqa: BLE001 - YouTube often refuses its translations: the AI translates instead
            en = []
        f.write_text(json.dumps(en), encoding="utf-8")       # asked once per video, refused or not
    return [w for w in en if clip["start"] - 0.5 <= w["s"] <= clip["end"] + 0.5]


LONG_VIDEO = getattr(config, "LONG_VIDEO", 12 * 60)   # above this, fetch only the pieces that become clips
PAD = 8                # seconds kept either side of a fetched piece, so clips can be re-trimmed


def _piece_again(job, clip):
    """Remaking a clip whose own piece of the video is missing and whose whole video is not here:
    fetch just the clip's range quickly (without slow force_keyframes_at_cuts full-video decoding)."""
    src = job.get("source") or {}
    if src.get("type") != "url" or not src.get("url"):
        raise RuntimeError("The original video is no longer here, so this clip can't be made again. Add the video again.")
    d = Path(job["dir"])
    want = max(0.0, clip["start"] - PAD)
    until = clip["end"] + PAD if not job.get("duration") else min(job["duration"], clip["end"] + PAD)
    try:
        piece = media.download_section(src["url"], d / "clips", f"{clip['id']}.src", want, until,
                                       int(job["options"]["quality"]), exact=False)
        meta = media.probe(piece)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not fetch this part of the video: {safe.friendly(e, 250)}") from e
    clip.update(src=str(piece), src_offset=round(want, 2), src_end=round(want + meta["duration"], 2))
    clip["end"] = min(clip["end"], clip["src_end"])
    job.setdefault("width", meta["width"])
    job.setdefault("height", meta["height"])


def _fetch_piece(job, clip, full_audio):
    """Long videos: download just this clip's stretch and pin down exactly where it starts."""
    d = Path(job["dir"])
    o = job["options"]
    try:
        want = max(0.0, clip["start"] - PAD)
        piece = media.download_section(job["source"]["url"], d / "clips", f"{clip['id']}.src", want,
                                       min(job["duration"], clip["end"] + PAD), int(o["quality"]))
        wav = d / "clips" / f"{clip['id']}.wav"
        media.extract_audio(piece, wav)
        part = media.load_audio(wav)
        wav.unlink(missing_ok=True)
        meta = media.probe(piece)
        off = media.find_offset(full_audio, part, want)
        clip.update(src=str(piece), src_offset=round(off, 2), src_end=round(off + meta["duration"], 2))
        clip["start"], clip["end"] = max(clip["start"], clip["src_offset"]), min(clip["end"], clip["src_end"])
        job.setdefault("width", meta["width"])
        job.setdefault("height", meta["height"])
    except Exception as e:  # noqa: BLE001
        detail = _classify_error(e, where=f"Downloading clip slice {clip.get('id', '')}")
        clip.update(status="error", error=f"Could not fetch this part of the video: {detail['what'][:250]}", error_detail=detail)


BLOCK = 10 * 60         # long videos are transcribed and clipped one block at a time
PROGRESSIVE_FROM = 20 * 60


def _covers(words: list, start: float, end: float) -> bool:
    """Is a borrowed transcript good enough to find clips with? It has to reach the end of the part being clipped
    and have a normal amount of talking in it; otherwise our own model writes the transcript."""
    inside = [w for w in words if start <= w["s"] <= end]
    minutes = max(0.5, (end - start) / 60)
    return len(inside) >= 25 * minutes and inside[-1]["s"] >= start + 0.85 * (end - start)


def _facecam(job, src=None, window=None) -> list | None:
    """Where the webcam is in the source: once for the whole video, for every clip of it.

    Uses multi-frame vision AI + full/quadrant border scan + OpenCV face clustering + corner stillness
    so real facecams are reliably auto-detected without falling back to gameplay."""
    if job.get("facecam"):
        return job["facecam"]
    path = src or job.get("source_path")
    cam_pos = (job.get("options") or {}).get("cam_pos", "auto")
    if cam_pos and cam_pos != "auto":
        # Snap tightly to the real webcam inside the requested corner so we don't capture surrounding gameplay
        if path and Path(path).is_file():
            try:
                detected = facecam.detect_in_corner(path, cam_pos, window=window)
                if not detected and window is not None:
                    detected = facecam.detect_in_corner(path, cam_pos, window=None)
                if detected:
                    job["facecam"] = [detected["x"], detected["y"], detected["w"], detected["h"]]
                    return job["facecam"]
            except Exception:  # noqa: BLE001
                pass
        src_w, src_h = 1920, 1080
        if path and Path(path).is_file():
            try:
                info = media.probe(path)
                src_w, src_h = info.get("w") or 1920, info.get("h") or 1080
            except Exception:
                pass
        bw, bh = int(src_w * 0.22), int(src_h * 0.24)
        margin = int(src_w * 0.015)
        if cam_pos == "top-left":
            box = [margin, margin, bw, bh]
        elif cam_pos == "bottom-right":
            box = [src_w - bw - margin, src_h - bh - margin, bw, bh]
        elif cam_pos == "bottom-left":
            box = [margin, src_h - bh - margin, bw, bh]
        else:
            box = [src_w - bw - margin, margin, bw, bh]
        job["facecam"] = box
        return job["facecam"]

    if not path or not Path(path).is_file():
        return None
    try:
        box = facecam.detect(path, ask=brain.look_for_facecam, window=window)
        if (not box or box.get("conf") == "none") and window is not None:
            box = facecam.detect(path, ask=brain.look_for_facecam, window=None)
    except Exception:  # noqa: BLE001 - a missing facecam is not a failure
        return None
    if not box or box.get("conf") == "none":
        return None
    job["facecam"] = [box["x"], box["y"], box["w"], box["h"]]
    return job["facecam"]


def _run_job(job):
    import math
    from concurrent.futures import ThreadPoolExecutor
    d = Path(job["dir"])
    o = job["options"]
    src = job["source"]
    # Which game this is: what the creator set, or what the title says. It decides the extra rules the AI
    # reads and the tags every post carries.
    game = job.get("game") or ""
    if not game and job.get("creator"):
        try:
            game = (creators.get(job["creator"]) or {}).get("game") or ""
        except Exception:  # noqa: BLE001
            game = ""
    if not game:
        game = games.detect(job["title"], src.get("name"))[0] or ""
    if game and job.get("game") != game:
        job["game"] = game
    t0 = time.time()
    notes = []
    face = False       # where the webcam is in the source, found once; False means not looked for yet
    usage.set_context(job["creator"], job["title"] or src.get("name"))
    resuming = bool(job.pop("resume", False)) and (d / "words.json").is_file()
    if resuming:   # after a restart: keep the transcript parts, clips and AI answers already made, and carry on
        _set(job, status="running", stage="Picking up where it stopped", error=None)
    else:
        part_audio = any(d.glob("audio_src.*.part")) or any(d.glob("audio_src.*.ytdl"))
        init_p = job.get("progress") if (part_audio and job.get("progress")) else 0.03
        _set(job, status="running", stage="Getting the audio", progress=init_p, error=None, clips=job.get("clips", []), ai=job.get("ai", {}), done_blocks=job.get("done_blocks", 0))
    if not transcribe.model_ready() and not transcribe.REMOTE:
        from . import fetch_model
        _set(job, stage="Getting the speech model (one time, about 480 MB)")
        fetch_model.ensure()
        _set(job, stage="Getting the audio")
    fetch = ThreadPoolExecutor(1)
    video = None
    existing_src = Path(job["source_path"]) if job.get("source_path") and Path(job["source_path"]).is_file() else None
    used_existing = False
    if existing_src is not None:
        # Reuse the already-downloaded video on disk (zero re-download on remake!)
        used_existing = True
        pieces, audio_file = False, existing_src
        info = media.probe(audio_file)
        _set(job, source_path=str(audio_file), width=info["width"], height=info["height"],
             duration=job.get("duration") or info["duration"], stage="Reusing saved video on disk", progress=0.08)
        if not job.get("thumb") and media.save_thumb(d, video=audio_file):
            _set(job, thumb=True)
    elif src["type"] == "url":
        meta = media.info(src["url"])
        if meta["kind"] != "video":
            raise RuntimeError("That link is a channel or playlist, not a single video. Paste the link of one video.")
        if ((o["range_end"] or meta["duration"]) - (o["range_start"] or 0)) > config.MAX_VIDEO_HOURS * 3600:
            raise RuntimeError(f"That is longer than {config.MAX_VIDEO_HOURS:g} hours. Choose a shorter part of it under Options.")
        if meta["is_live"]:
            raise RuntimeError("This link is a live stream, so it would never finish downloading. Open the creator, "
                               "put this link under Setup, and press Start recording instead.")
        _set(job, title=meta["title"], uploader=meta["uploader"], duration=meta["duration"])
        if not job.get("thumb") and media.save_thumb(d, url=src["url"]):
            _set(job, thumb=True)
        usage.set_context(job["creator"], meta["title"])
        span = (o["range_end"] or meta["duration"]) - (o["range_start"] or 0)
        pieces = span > LONG_VIDEO or meta["duration"] <= 0 or bool(o.get("range_start")) or bool(o.get("range_end"))
        if not pieces:  # shorter video: fetch all of it in the background while the audio is worked on
            video = fetch.submit(media.download, src["url"], d, int(o["quality"]))
        last_audio_tick = [0.0]

        def _audio_progress(p, d_info=None):
            if is_paused(job["id"]):
                raise media.JobPaused("Job paused by user")
            now = time.time()
            if now - last_audio_tick[0] < 0.5 and p < 0.99:
                return
            last_audio_tick[0] = now
            stage = "Getting the audio"
            kw = {}
            if d_info:
                speed = d_info.get("speed")
                downloaded = d_info.get("downloaded_bytes") or 0
                total = d_info.get("total_bytes") or d_info.get("total_bytes_estimate") or 0
                speed_str = (media.fmt_bytes(speed) + "/s") if speed else ""
                dl_str = f"{media.fmt_bytes(downloaded)} / {media.fmt_bytes(total)}" if total else media.fmt_bytes(downloaded)
                eta_str = f"ETA {media.fmt_eta(d_info.get('eta'))}" if d_info.get("eta") else ""

                parts = [x for x in (speed_str, dl_str, eta_str) if x]
                if o.get("range_start") or o.get("range_end"):
                    r_from = media.fmt_eta(o.get("range_start") or 0) or "0s"
                    r_to = media.fmt_eta(o["range_end"]) if o.get("range_end") else "end"
                    parts.insert(0, f"part {r_from} to {r_to}")
                if parts:
                    stage = f"Getting the audio ({' · '.join(parts)})"
                if speed_str:
                    kw["speed"] = speed_str
            _set(job, stage=stage, progress=0.03 + 0.1 * p, **kw)

        audio_file = media.download_audio(
            src["url"], d,
            on_progress=_audio_progress,
            is_paused=lambda: is_paused(job["id"]),
            start=o.get("range_start"),
            end=o.get("range_end"),
        )
    else:
        pieces, audio_file = False, Path(src["path"])
        info = media.probe(audio_file)
        if info["duration"] > config.MAX_VIDEO_HOURS * 3600:
            raise RuntimeError(f"That video is longer than {config.MAX_VIDEO_HOURS:g} hours, which is the limit.")
        _set(job, source_path=str(audio_file), width=info["width"], height=info["height"], duration=info["duration"])
        if not job.get("thumb") and media.save_thumb(d, video=audio_file):
            _set(job, thumb=True)
    wav = d / "audio.wav"
    media.extract_audio(audio_file, wav)
    audio = media.load_audio(wav)
    wav.unlink(missing_ok=True)
    if src["type"] == "url" and not used_existing:
        audio_file.unlink(missing_ok=True)
        if o.get("range_start"):
            silence = np.zeros(int(o["range_start"] * 16000), dtype=audio.dtype)
            audio = np.concatenate([silence, audio])
    if not job["duration"]:
        job["duration"] = len(audio) / 16000
    lo = int((o["range_start"] or 0) * 16000)
    hi = min(len(audio), int(o["range_end"] * 16000)) if o["range_end"] else len(audio)
    span = (hi - lo) / 16000
    db = media.loudness_per_second(audio[lo:hi])

    vlock, vstate = threading.Lock(), ({"path": existing_src} if used_existing else {})
    remaking = bool(job.pop("remaking", False))

    def whole_video():
        """Wait for the background download once; retry it, or give up and use pieces."""
        with vlock:
            if "path" not in vstate:
                path = None
                try:
                    path = video.result()
                except Exception as e:  # noqa: BLE001 - sites sometimes refuse a download
                    notes.append(f"Video download refused once ({str(e)[-80:]}), retrying")
                    try:
                        path = media.download(src["url"], d, int(o["quality"]))
                    except Exception as e2:  # noqa: BLE001
                        notes.append(f"Whole-video download failed ({str(e2)[-80:]}), fetching clip parts instead")
                if path:
                    info = media.probe(path)
                    job.update(source_path=str(path), width=info["width"], height=info["height"])
                vstate["path"] = path
            return vstate["path"]

    exact = {}      # clip id -> (from, to, words): our own model's words for just that clip

    began = {}      # clip id -> when work on it started

    def make(clip, words_now, quality="final"):
        began[clip["id"]] = time.time()
        if src["type"] == "url" and (pieces or not whole_video()):
            _fetch_piece(job, clip, audio)
        if clip.get("layout") in ("gameplay", "stack") and not clip.get("facecam"):
            src_file = Path(clip["src"]) if clip.get("src") else (Path(job["source_path"]) if job.get("source_path") else None)
            cam = job.get("facecam") or (src_file and _facecam(job, src=src_file))
            if cam:
                clip["facecam"] = cam
        speech_lang = (o.get("speech_lang") or "auto").strip().lower()
        whisper_active = (transcribe.active_engine() == "faster-whisper") or (
            speech_lang in ("hi", "hinglish") and transcribe.faster_whisper_installed()
        )
        should_retranscribe = bool(
            (yt and (
                yt[1].lower().startswith("en")
                or whisper_active
                or brain.needs_translation(yt[1])
            ))
            # When Faster-Whisper scanned the full video fast with word_timestamps=False,
            # run crisp word_timestamps=True ONLY on this short selected clip!
            or (not yt and whisper_active and not o.get("custom_ranges"))
        )
        if clip["status"] not in ("error", "failed") and clip.get("captions", True) and should_retranscribe:
            # Run crisp word-level timestamps on just this short clip (fast even on CPU, instant on GPU).
            a0, b0 = max(lo, int((clip["start"] - 1.5) * 16000)), min(hi, int((clip["end"] + 1.5) * 16000))
            eff_lang_hint = speech_lang if speech_lang in ("en", "hi", "hinglish") else (job.get("lang") or "")
            try:
                tw = [dict(w, s=round(w["s"] + a0 / 16000, 2), e=round(w["e"] + a0 / 16000, 2))
                      for w in transcribe.transcribe(audio[a0:b0], lang_hint=eff_lang_hint, word_timestamps=True)]
            except Exception:  # noqa: BLE001 - the clip is still made, with existing words
                tw = []
            if len(tw) >= 3:
                exact[clip["id"]] = (a0 / 16000, b0 / 16000, tw)
                words_now = tw
                clip.pop("fixed_for", None)
                clip.pop("fixed_words", None)
                clip.pop("words", None)
        if clip["status"] not in ("error", "failed"):
            render_one(job, clip, words_now, quality=quality)

    cpus = os.cpu_count() or 4
    renders = ThreadPoolExecutor(2 if cpus < 8 else min(6, cpus // 6))  # more clips made at once on a bigger machine
    futures = []
    step = BLOCK * 16000 if span > PROGRESSIVE_FROM else hi - lo
    blocks = [(x, min(hi, x + step)) for x in range(lo, hi, step)] or [(lo, hi)]
    words, tokens, model = [], 0, None
    done_blocks = 0
    yt = None
    speech_lang = (o.get("speech_lang") or "auto").strip().lower()
    if speech_lang in ("en", "hi", "hinglish"):
        job["lang"] = speech_lang
    if remaking and (d / "words.json").is_file():
        try:
            cached_w = json.loads((d / "words.json").read_text(encoding="utf-8"))
            if isinstance(cached_w, list) and _covers(cached_w, lo / 16000, hi / 16000):
                yt = (cached_w, job.get("lang") or "")
        except Exception:
            pass
    elif src["type"] == "url" and config.YT_TRANSCRIPT and media.is_youtube(src["url"]):
        try:
            yt = media.youtube_words(src["url"])
        except Exception:  # noqa: BLE001 - no transcript to borrow: ours is written as usual
            yt = None
        if yt and not _covers(yt[0], lo / 16000, hi / 16000):
            yt = None
        # If user confirmed Hindi or Hinglish, ignore mismatched YouTube English captions so Hindi speech isn't lost
        if yt and speech_lang in ("hi", "hinglish") and str(yt[1] or "").lower().startswith("en"):
            yt = None
        job["transcript_from"] = "youtube" if yt else "own"
        if speech_lang not in ("en", "hi", "hinglish"):
            # Never let YouTube's auto-tag override Auto speech detection unless we actually borrowed YouTube's captions
            job["lang"] = yt[1] if yt else ""
    if resuming:
        try:
            words = json.loads((d / "words.json").read_text(encoding="utf-8"))
        except ValueError:
            words = []
        done_blocks = min(int(job.get("done_blocks", 0)), len(blocks)) if words else 0
        tokens, model = int((job.get("ai") or {}).get("tokens", 0)), (job.get("ai") or {}).get("model")
        if done_blocks < len(blocks):     # words past the last finished part are transcribed again
            cut = blocks[done_blocks][0] / 16000
            words = [w for w in words if w["s"] < cut]
        for c in job["clips"]:            # clips found before the restart that were not finished yet
            if c.get("status") not in ("done", "error"):
                c["status"] = "queued"
                futures.append((renders.submit(ctx.bind(make), c, list(words)), c))
        save(job)
    # The transcript of the next part is written while the AI is still reading this one (the AI is a wait on the
    # network, the transcript is work on this machine), so neither waits for the other.
    ahead = ThreadPoolExecutor(2, thread_name_prefix="ahead")
    heard, now_on = {}, [done_blocks]
    asr_t0 = [0.0]
    asr_base_samples = blocks[done_blocks][0] if done_blocks < len(blocks) else hi

    def hear(i):
        if i < len(blocks) and i >= done_blocks and i not in heard:
            a_, b_ = blocks[i]
            last = [0.0, 0.0]  # [last_p, last_ts]
            whisper_active_block = (transcribe.active_engine() == "faster-whisper") or (
                speech_lang in ("hi", "hinglish") and transcribe.faster_whisper_installed()
            )
            use_word_ts = not whisper_active_block

            def tp(p, i=i, a_=a_, b_=b_):
                now = time.time()
                if asr_t0[0] <= 0.0:
                    asr_t0[0] = now
                if now_on[0] == i and (p - last[0] > 0.02 or now - last[1] >= 2.0):
                    last[0], last[1] = p, now
                    cur_samples = (a_ - lo) + p * (b_ - a_)
                    total_samples = max(1, hi - lo)
                    prog = 0.14 + 0.78 * cur_samples / total_samples
                    done_since = max(0.0, (a_ + p * (b_ - a_)) - asr_base_samples)
                    rem_samples = max(0.0, hi - (a_ + p * (b_ - a_)))
                    elapsed = now - asr_t0[0]
                    kw = {}
                    if elapsed >= 3.0 and done_since >= 16000 * 3.0:
                        rate = done_since / elapsed
                        kw["eta_sec"] = max(1, int(round(rem_samples / max(1.0, rate))))
                    _set(job, progress=max(job["progress"], prog), **kw)
            heard[i] = ahead.submit(ctx.bind(transcribe.transcribe), audio[a_:b_], tp, job.get("lang") or "", use_word_ts)

    # The ad breaks and static screens of the whole video, found once. It reads the video end to end,
    # so it is done here rather than for each clip, and only when asked for.
    statics = quality.static_ranges(job.get("source_path") or "") if config.SKIP_STATIC else []

    custom_ranges = o.get("custom_ranges")
    review_mode = o.get("review_mode", "auto")

    if custom_ranges:
        for i, crange in enumerate(custom_ranges):
            if is_paused(job["id"]):
                ahead.shutdown(wait=False)
                raise media.JobPaused("Job paused by user")
            r_start = float(crange["start"])
            r_end = float(crange["end"])
            r_mode = crange.get("mode", o.get("timestamp_mode", "direct"))
            cid = f"clip{len(job['clips']) + 1:02d}"

            _set(job, stage=f"Processing clip {i + 1} of {len(custom_ranges)} ({media.fmt_eta(r_start)} to {media.fmt_eta(r_end)})",
                 progress=0.15 + 0.75 * (i / len(custom_ranges)))

            a_s = int(r_start * 16000)
            b_s = min(len(audio), int(r_end * 16000))
            slice_audio = audio[a_s:b_s] if a_s < len(audio) else np.zeros(0, dtype=audio.dtype)

            clip_words = []
            if o.get("captions", True) and len(slice_audio) >= 1600:
                try:
                    if yt:
                        raw_w = [dict(w) for w in yt[0] if r_start <= w["s"] < r_end]
                    else:
                        raw_w = transcribe.transcribe(slice_audio, lang_hint=job.get("lang") or "")
                    clip_words = [dict(w, s=round(w["s"] + (0 if yt else r_start), 2),
                                       e=round(w["e"] + (0 if yt else r_start), 2)) for w in raw_w]
                except Exception as e:
                    notes.append(f"Notice: transcription for clip {cid} ({e})")
                    clip_words = []
            words.extend(clip_words)
            (d / "words.json").write_text(json.dumps(words), encoding="utf-8")

            if r_mode == "direct":
                # DIRECT CLIP: Never drop, never scout, exact start and end!
                transcript_text = " ".join(w.get("text") or w.get("w") or "" for w in clip_words).strip()
                hook_text = ""
                hook_opts = []
                title_text = ""
                title_opts = []
                highlight_words = []
                if transcript_text and len(transcript_text.split()) >= 3:
                    try:
                        desc = brain.describe(transcript_text, title=job.get("title") or "",
                                              uploader=job.get("uploader") or "", game=game)
                        if desc:
                            hook_text = desc.get("hook") or ""
                            hook_opts = desc.get("hook_options") or []
                            title_text = desc.get("title") or ""
                            title_opts = desc.get("title_options") or []
                            highlight_words = desc.get("highlight_words") or []
                    except Exception:
                        pass
                if not hook_text:
                    hook_text = f"CLIP {len(job['clips']) + 1}"
                if not title_text:
                    title_text = f"Clip {len(job['clips']) + 1}"

                c = {
                    "id": cid,
                    "start": round(r_start, 2),
                    "end": round(r_end, 2),
                    "hook": hook_text,
                    "hook_options": hook_opts,
                    "title": title_text,
                    "title_options": title_opts,
                    "score": 100,
                    "category": "custom",
                    "highlight_words": highlight_words,
                    "style": o["style"],
                    "layout": o["layout"],
                    "captions": o.get("captions", True),
                    "meme": o.get("meme", "none"),
                    "music": o.get("music", "none"),
                    "meme_vol": o.get("meme_vol"),
                    "music_db": o.get("music_db"),
                    "split_ratio": o.get("split_ratio", render.DEFAULT_SPLIT_RATIO),
                    "split_divider": o.get("split_divider", render.DEFAULT_SPLIT_DIVIDER),
                    "cam_swap": o.get("cam_swap", False),
                    "cap_anchor": o.get("cap_anchor", "safe_bottom"),
                    "cam_fallback": o.get("cam_fallback", "blur"),
                    "cap_y": o.get("cap_y"),
                    "version": 0,
                    "custom_mode": "direct"
                }

                if o.get("facecam"):
                    c["facecam"] = o["facecam"]
                elif o["layout"] in ("gameplay", "stack"):
                    if not face and job.get("source_path"):
                        face = _facecam(job, window=(r_start, r_end))
                    if face:
                        c["facecam"] = face

                if c.get("facecam") and job.get("source_path"):
                    try:
                        fc_focus = facecam.find_clip_face_focus(job["source_path"], c["facecam"], c["start"], c["end"])
                        if fc_focus:
                            c["facecam_focus"] = fc_focus
                    except Exception:
                        pass

                ed = edit.init_edit(job, c, clip_words, c.get("facecam"))
                (d / "clips").mkdir(parents=True, exist_ok=True)
                edit.save_edit(d / "clips" / f"{cid}_edit.json", ed)

                if review_mode in ("review", "review_first"):
                    c["status"] = "drafting"
                    q = "draft"
                else:
                    c["status"] = "queued"
                    q = "final"

                job["clips"].append(c)
                save(job)
                futures.append((renders.submit(ctx.bind(make), c, list(clip_words), q), c))

            else:
                # ANALYZE MODE: AI evaluates the range and may drop it
                bw = [dict(w, s=round(w["s"] - r_start, 2), e=round(w["e"] - r_start, 2)) for w in clip_words]
                s0 = int(r_start)
                dur = max(1.0, r_end - r_start)
                s_db = db[s0: s0 + int(dur) + 1] if s0 < len(db) else np.zeros(1, dtype=np.float32)
                found, ai = brain.find_clips(bw, s_db, dur, job["title"],
                                             job["uploader"], 1, int(o["min_len"]), int(o["max_len"]),
                                             notes.append, game=game)
                tokens += ai.get("tokens", 0)
                model = ai.get("model") or model
                if ai.get("note"):
                    notes.append(ai["note"])

                if found and statics:
                    keep = []
                    for fc in found:
                        a_s_chk, b_s_chk = fc["start"] + r_start, fc["end"] + r_start
                        inside = quality.static_overlap(a_s_chk, b_s_chk, statics)
                        if inside > 0.5 * max(1.0, b_s_chk - a_s_chk):
                            notes.append(f"Range {media.fmt_eta(r_start)} to {media.fmt_eta(r_end)} was dropped: sat in an ad break or static screen.")
                        else:
                            keep.append(fc)
                    found = keep

                if not found:
                    notes.append(f"Range {media.fmt_eta(r_start)} to {media.fmt_eta(r_end)} was dropped in Analyze mode: no clip-worthy moment found by AI.")
                    save(job)
                else:
                    for fc in found:
                        cid = f"clip{len(job['clips']) + 1:02d}"
                        fc.update(id=cid, start=round(fc["start"] + r_start, 2),
                                  end=round(fc["end"] + r_start, 2), style=o["style"], layout=o["layout"],
                                  captions=o.get("captions", True), meme=o.get("meme", "none"),
                                  music=o.get("music", "none"), meme_vol=o.get("meme_vol"),
                                  music_db=o.get("music_db"), split_ratio=o.get("split_ratio", render.DEFAULT_SPLIT_RATIO),
                                  split_divider=o.get("split_divider", render.DEFAULT_SPLIT_DIVIDER),
                                  cam_swap=o.get("cam_swap", False), cap_anchor=o.get("cap_anchor", "safe_bottom"),
                                  cam_fallback=o.get("cam_fallback", "blur"), cap_y=o.get("cap_y"),
                                  version=0, custom_mode="analyze")
                        if o.get("facecam"):
                            fc["facecam"] = o["facecam"]
                        elif o["layout"] in ("gameplay", "stack"):
                            if not face and job.get("source_path"):
                                face = _facecam(job, window=(fc["start"], fc["end"]))
                            if face:
                                fc["facecam"] = face
                        if fc.get("facecam") and job.get("source_path"):
                            try:
                                fc_focus = facecam.find_clip_face_focus(job["source_path"], fc["facecam"], fc["start"], fc["end"])
                                if fc_focus:
                                    fc["facecam_focus"] = fc_focus
                            except Exception:
                                pass
                        ed = edit.init_edit(job, fc, clip_words, fc.get("facecam"))
                        (d / "clips").mkdir(parents=True, exist_ok=True)
                        edit.save_edit(d / "clips" / f"{cid}_edit.json", ed)
                        if review_mode in ("review", "review_first"):
                            fc["status"] = "drafting"
                            q = "draft"
                        else:
                            fc["status"] = "queued"
                            q = "final"
                        job["clips"].append(fc)
                        save(job)
                        futures.append((renders.submit(ctx.bind(make), fc, list(clip_words), q), fc))
        job["ai"] = {"model": model, "tokens": tokens, "note": "", "log": notes[-5:]}
        save(job)
    else:
        for bi, (a, b) in enumerate(blocks):
            if is_paused(job["id"]):
                ahead.shutdown(wait=False)
                raise media.JobPaused("Job paused by user")
            if bi < done_blocks:
                continue                      # transcribed and checked by the AI before the restart
            shift = a / 16000
            where = f" (minute {round((a - lo) / 16000 / 60)} of {round(span / 60)})" if len(blocks) > 1 else ""
            if yt:
                _set(job, stage="Reading the transcript" + where)
                bw = [dict(w, s=round(w["s"] - shift, 2), e=round(w["e"] - shift, 2)) for w in yt[0] if a / 16000 <= w["s"] < b / 16000]
            else:
                if asr_t0[0] <= 0.0:
                    asr_t0[0] = time.time()
                _set(job, stage="Writing the transcript" + where)
                now_on[0] = bi
                hear(bi)
                # Parakeet can run multiple copies in parallel; Faster-Whisper shares one model lock
                if transcribe.active_engine() != "faster-whisper":
                    hear(bi + 1)
                bw = heard.pop(bi).result()
                if transcribe.active_engine() == "faster-whisper" and bi + 1 < len(blocks):
                    hear(bi + 1)  # starts next block while AI evaluates this block over the network
            _set(job, progress=max(job["progress"], 0.14 + 0.78 * (b - lo) / max(1, hi - lo)))
            words += [dict(w, s=round(w["s"] + shift, 2), e=round(w["e"] + shift, 2)) for w in bw]
            (d / "words.json").write_text(json.dumps(words), encoding="utf-8")
            _set(job, stage="Finding the best moments" + where, word_count=len(words))
            left = int(o["max_clips"]) - len(job["clips"])
            if left <= 0 or len(bw) < 8:
                job["done_blocks"] = bi + 1
                save(job)
                continue
            quota = max(1, math.ceil(left / (len(blocks) - bi)))
            s0 = int((a - lo) / 16000)
            found, ai = brain.find_clips(bw, db[s0: s0 + int((b - a) / 16000) + 1], (b - a) / 16000, job["title"],
                                         job["uploader"], quota, int(o["min_len"]), int(o["max_len"]), notes.append, game=game)
            tokens += ai.get("tokens", 0)
            model = ai.get("model") or model
            if ai.get("note"):
                notes.append(ai["note"])
            if found and statics:
                # A commercial or a "be right back" slate is not a clip. Any clip that sits mostly inside one
                # is dropped.
                keep = []
                for c in found:
                    a_s, b_s = c["start"] + shift, c["end"] + shift
                    inside = quality.static_overlap(a_s, b_s, statics)
                    if inside > 0.5 * max(1.0, b_s - a_s):
                        notes.append(f"A clip from {round(a_s)}s was left out: it sat in an ad break or a static screen.")
                    else:
                        keep.append(c)
                found = keep
            for c in found:
                cid = f"clip{len(job['clips']) + 1:02d}"
                c.update(id=cid, start=round(c["start"] + shift, 2),
                         end=round(c["end"] + shift, 2), style=o["style"], layout=o["layout"],
                         captions=o.get("captions", True), meme=o.get("meme", "none"),
                         music=o.get("music", "none"), meme_vol=o.get("meme_vol"),
                         music_db=o.get("music_db"), split_ratio=o.get("split_ratio", render.DEFAULT_SPLIT_RATIO),
                         split_divider=o.get("split_divider", render.DEFAULT_SPLIT_DIVIDER),
                         cam_swap=o.get("cam_swap", False), cap_anchor=o.get("cap_anchor", "safe_bottom"),
                         cam_fallback=o.get("cam_fallback", "blur"), cap_y=o.get("cap_y"),
                         version=0)
                if o.get("facecam"):
                    c["facecam"] = o["facecam"]
                elif o["layout"] in ("gameplay", "stack"):
                    if not face and job.get("source_path"):
                        face = _facecam(job, window=(c["start"], c["end"]))
                    if face:
                        c["facecam"] = face

                if c.get("facecam") and job.get("source_path"):
                    try:
                        fc_focus = facecam.find_clip_face_focus(job["source_path"], c["facecam"], c["start"], c["end"])
                        if fc_focus:
                            c["facecam_focus"] = fc_focus
                    except Exception:
                        pass

                ed = edit.init_edit(job, c, words, c.get("facecam"))
                (d / "clips").mkdir(parents=True, exist_ok=True)
                edit.save_edit(d / "clips" / f"{cid}_edit.json", ed)

                if review_mode in ("review", "review_first"):
                    c["status"] = "drafting"
                    q = "draft"
                else:
                    c["status"] = "queued"
                    q = "final"

                job["clips"].append(c)
                futures.append((renders.submit(ctx.bind(make), c, list(words), q), c))
            job["ai"] = {"model": model, "tokens": tokens, "note": "", "log": notes[-5:]}
            job["done_blocks"] = bi + 1       # a restart from here on carries on with the next part
            save(job)
    if futures:
        _set(job, stage="Finishing the clips", progress=max(job["progress"], 0.92))
    _wait_for_clips(job, futures, began, renders._max_workers)
    if exact:           # keep our own words for the clips in the saved transcript, so editing a clip later uses them
        for a_s, b_s, tw in exact.values():
            words = [w for w in words if not a_s <= w["s"] < b_s] + tw
        words.sort(key=lambda w: w["s"])
        (d / "words.json").write_text(json.dumps(words), encoding="utf-8")
    renders.shutdown(wait=False)
    ahead.shutdown(wait=False)
    fetch.shutdown(wait=False)
    del audio
    clips = job["clips"]
    job["ai"] = {"model": model, "tokens": tokens, "log": notes[-5:],
                 "note": "The AI was unavailable, so clips were picked by loudness only." if clips and not model else ""}
    awaiting = [c for c in clips if c.get("status") in ("awaiting_review", "drafting")]
    if awaiting:
        stage = f"{len(awaiting)} clip{'s' if len(awaiting) != 1 else ''} ready for review"
    elif clips:
        stage = "Done"
    else:
        stage = "Done, but nothing clip-worthy was found"
    _set(job, status="done", stage=stage, progress=1.0, eta_sec=None, took=round(time.time() - t0))
    try:
        from . import notices
        name = job.get("title") or job["source"].get("name") or "your video"
        notices.add(job.get("user"), "clips", f"{len(clips)} clip{'s' if len(clips) != 1 else ''} ready" if clips else "Nothing clip-worthy found",
                    name, f"/app/creators/{job['creator']}" if job.get("creator") else "/app/videos")
    except Exception:  # noqa: BLE001
        pass
    creators.log(job["creator"], f"Video finished with {len(clips)} clips in {round(time.time() - t0)}s: {job['title']}")


def _worker(quick_only=False):
    while True:
        kind, a, cid = _q.get(quick_only)
        if kind == "fn":
            try:
                a()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            continue
        job = _jobs.get(a)
        if not job or (kind == "job" and job.get("status") == "paused"):
            if kind == "job":
                _q.done()
            continue
        token = ctx._uid.set(job["user"])  # everything this job does runs as its owner
        try:
            if kind == "job":
                _run_job(job)
            elif kind in ("clip_final", "clip"):
                clip = next((c for c in job["clips"] if c["id"] == cid), None)
                if not clip or (kind == "clip_final" and clip.get("status") == "done"):
                    continue
                words = []
                if clip.get("captions", True):
                    wf = Path(job["dir"]) / "words.json"
                    if wf.is_file():
                        words = json.loads(wf.read_text(encoding="utf-8"))
                render_one(job, clip, words, quality="final")
            elif kind == "clip_draft":
                clip = next((c for c in job["clips"] if c["id"] == cid), None)
                # Never let an earlier queued draft overwrite a clip that was already approved or finished!
                if not clip or clip.get("status") in ("approved", "rendering_final", "done"):
                    continue
                words = []
                if clip.get("captions", True):
                    wf = Path(job["dir"]) / "words.json"
                    if wf.is_file():
                        words = json.loads(wf.read_text(encoding="utf-8"))
                render_one(job, clip, words, quality="draft")
        except media.JobPaused:
            _paused.discard(job["id"])
            _set(job, status="paused", stage="Paused by you", eta_sec=None)
        except Exception as e:  # noqa: BLE001
            (Path(job["dir"]) / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            failed_stage = job.get("stage") or "Processing video"
            if kind == "job":
                detail = _classify_error(e, where=failed_stage)
                for c in job["clips"]:
                    if c.get("status") in ("queued", "rendering", "rendering_final", "drafting", "approved"):
                        c.update(status="error", error="The video stopped before this clip was made.", error_detail=detail)
                        c.pop("eta_sec", None)
                _set(job, status="error", error=detail["what"], error_detail=detail, stage="Stopped", eta_sec=None)
                creators.log(job["creator"], f"Video failed ({failed_stage}): {detail['what'][:200]}")
            elif kind in ("clip", "clip_final", "clip_draft") and cid:
                clip = next((c for c in job.get("clips", []) if c["id"] == cid), None)
                if clip and clip.get("status") in ("queued", "rendering", "rendering_final", "drafting", "approved"):
                    detail = _classify_error(e, where=f"Clip {cid} ({kind})")
                    clip.pop("eta_sec", None)
                    clip.pop("render_started", None)
                    clip.update(status="failed" if kind == "clip_draft" else "error", error=detail["what"], error_detail=detail)
                    _refresh_review_stage(job)
                    save(job)
        finally:
            ctx._uid.reset(token)
            if kind == "job":
                _q.done()


def _pick_up_after_restart():
    """Videos and single clips that were being made when the server stopped: queued again, oldest first."""
    waiting = sorted((j for j in all_jobs() if j.get("status") == "queued" and j.get("resume")), key=lambda j: j["created"])
    for j in waiting:
        _q.put(("job", j["id"], None), user=j["user"])
    for j in all_jobs():
        for cid in j.pop("remake", []):
            _q.put(("clip", j["id"], cid))
        if j.get("status") == "done":
            save(j)


def start():
    load.start()
    load_all()
    _pick_up_after_restart()
    for _ in range(max(1, config.JOB_WORKERS)):
        threading.Thread(target=_worker, daemon=True).start()
    for _ in range(2):
        threading.Thread(target=_worker, args=(True,), daemon=True).start()   # re-making single clips never waits behind whole videos
    threading.Thread(target=_live_worker, daemon=True).start()

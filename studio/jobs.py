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

from . import brain, config, creators, ctx, edit, facecam, games, load, media, meme, quality, render, roman, rules, safe, transcribe, usage

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
            self.cv.notify()

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
            "range_start": None, "range_end": None, "captions": True, "meme": "none", "music": "none", "format": "moment", "cam_delay": 0.0,
            "meme_vol": None, "music_db": None, "cam_pos": "auto",
            "split_ratio": 0.4, "cam_swap": False, "cap_anchor": "safe_bottom", "cam_fallback": "blur", "facecam": None, "cap_y": None,
            "review_mode": "auto"}


LIMITS = {"max_clips": (1, 20), "min_len": (5, 120), "max_len": (10, 180), "quality": (360, 1080)}


def _parse_time_sec(val):
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    if s in ("start", "end", "none", "full"):
        return None
    if ":" in s:
        parts = [float(x) for x in s.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
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
    try:
        return float(s)
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
        elif k in ("range_start", "range_end"):
            sec = _parse_time_sec(v)
            if sec is None or not math.isfinite(sec) or sec < 0:
                raise ValueError("That part of the video isn't valid.")
            v = round(sec, 2)
        out[k] = v
    if out.get("range_end") is not None and out["range_end"] <= out.get("range_start", 0):
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
        if c.get("status") in ("rendering", "queued"):
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
    if "words" in changes and isinstance(changes["words"], list):
        valid_words = []
        for w in changes["words"]:
            if isinstance(w, dict) and "w" in w:
                valid_words.append({
                    "w": str(w["w"]),
                    "s": round(float(w.get("s", 0.0)), 2),
                    "e": round(float(w.get("e", 0.0)), 2)
                })
        new["words"] = valid_words
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
        else:
            face = _facecam(job, src=clip.get("src"))
            if face:
                new["facecam"] = face

    raw_file = Path(job["dir"]) / "clips" / f"{cid}_raw.mp4"
    if raw_file.is_file():
        time_changed = (new.get("start") != clip.get("start") or new.get("end") != clip.get("end"))
        layout_changed = (
            new.get("layout") != clip.get("layout") or
            new.get("split_ratio") != clip.get("split_ratio") or
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
        new["_fast_caption_only"] = not (time_changed or layout_changed or audio_changed)
    else:
        new["_fast_caption_only"] = False

    clip.update(new, status="queued")
    save(job)
    _q.put(("clip", jid, cid))


def render_one(job, clip, words, quality="final"):
    d = Path(job["dir"])
    if quality == "draft":
        clip["status"] = "drafting"
    else:
        clip["status"] = "rendering_final"
    save(job)
    try:
        if not clip.get("src") and not job.get("source_path"):
            _piece_again(job, clip)
        off = clip.get("src_offset", 0.0)
        src = Path(clip["src"]) if clip.get("src") else Path(job["source_path"])
        if not clip.get("src"):
            clip["end"] = min(clip["end"], job["duration"])
        if clip.get("layout") in ("gameplay", "stack") and not clip.get("facecam"):
            face = job.get("facecam") or _facecam(job, src=src)
            if face:
                clip["facecam"] = face
        local = dict(clip, start=clip["start"] - off, end=clip["end"] - off,
                     hook=roman.to_roman(clip.get("hook") or ""),
                     cam_delay=float(clip.get("cam_delay", 0.0) or 0.0))
        if clip.get("captions", True):
            if clip.get("words"):
                near = [dict(w, s=float(w["s"]), e=float(w["e"])) for w in clip["words"]]
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

        # Ensure {cid}_edit.json exists
        edit_file = d / "clips" / f"{clip['id']}_edit.json"
        if not edit_file.is_file():
            ed = edit.init_edit(job, clip, near, clip.get("facecam"))
            edit.save_edit(edit_file, ed)
        else:
            ed = edit.load_edit(edit_file)
            if ed and "captions" in clip:
                ed["caption"]["enabled"] = bool(clip["captions"])
                edit.save_edit(edit_file, ed)

        out = render.render_clip(src, d / "clips", clip["id"], local, near, job["width"], job["height"], quality=quality)
        if quality == "draft":
            clip.update(out, status="awaiting_review", error=None)
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
        clip.update(status="failed" if quality == "draft" else "error", error=safe.friendly(e, 400))
        creators.log(job["creator"], f"Clip failed: {safe.friendly(e, 200)}")
    save(job)


def approve_clip(jid: str, cid: str) -> dict:
    job = _jobs.get(jid)
    if not job or job.get("user") != ctx.uid():
        raise ValueError("No such video")
    clip = next((c for c in job["clips"] if c["id"] == cid), None)
    if not clip:
        raise ValueError("No such clip")
    if clip.get("status") == "rendering_final":
        raise ValueError("Clip is currently rendering")

    clip["status"] = "approved"
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
                        other["status"] = "approved"
                        _q.put(("clip_final", jid, other["id"]))
                save(job)
    return clip


def approve_all(jid: str) -> list:
    job = _jobs.get(jid)
    if not job or job.get("user") != ctx.uid():
        raise ValueError("No such video")
    approved = []
    for c in job["clips"]:
        if c.get("status") in ("awaiting_review", "qc_flagged", "failed", "error", "drafting"):
            c["status"] = "approved"
            _q.put(("clip_final", jid, c["id"]))
            approved.append(c)
    if approved:
        save(job)
    return approved


def apply_to_all(jid: str, patch: dict) -> list:
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
        if "facecam" in merged and not merged.get("facecam_override"):
            fc = merged["facecam"]
            c["facecam"] = [fc["x"], fc["y"], fc["w"], fc["h"]] if fc else None
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
    """Wait for a video's clips, but not for ever. A clip that hangs (a download that stalls, an encoder that
    freezes) used to keep the whole video "in progress", and no other clip could be re-made. Now it is marked as
    failed after CLIP_MINUTES and the video finishes with the clips that did work."""
    left, given_up = list(futures), 0
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
                creators.log(job["creator"], f"Clip stopped after {CLIP_MINUTES:g} minutes: {clip.get('title') or clip.get('hook') or clip['id']}")
                save(job)
                continue
            still.append((f, clip))
        left = still
        if left:
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
    """Remaking a clip whose own piece of the video was never fetched (its first try failed) and whose whole video
    is not here: fetch the piece now, cut exactly at the asked time, so no sound matching is needed."""
    src = job.get("source") or {}
    if src.get("type") != "url" or not src.get("url"):
        raise RuntimeError("The original video is no longer here, so this clip can't be made again. Add the video again.")
    d = Path(job["dir"])
    want = max(0.0, clip["start"] - PAD)
    until = clip["end"] + PAD if not job.get("duration") else min(job["duration"], clip["end"] + PAD)
    try:
        piece = media.download_section(src["url"], d / "clips", f"{clip['id']}.src", want, until,
                                       int(job["options"]["quality"]), exact=True)
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
        clip.update(status="error", error=f"Could not fetch this part of the video: {safe.friendly(e, 250)}")


BLOCK = 10 * 60         # long videos are transcribed and clipped one block at a time
PROGRESSIVE_FROM = 20 * 60


def _covers(words: list, start: float, end: float) -> bool:
    """Is a borrowed transcript good enough to find clips with? It has to reach the end of the part being clipped
    and have a normal amount of talking in it; otherwise our own model writes the transcript."""
    inside = [w for w in words if start <= w["s"] <= end]
    minutes = max(0.5, (end - start) / 60)
    return len(inside) >= 25 * minutes and inside[-1]["s"] >= start + 0.85 * (end - start)


def _facecam(job, src=None) -> list | None:
    """Where the webcam is in the source: once for the whole video, for every clip of it.

    Only a sure answer is used. A box that is a scoreboard drawn over the game would
    put a picture of a scoreboard in the corner of the clip. One frame is also shown to
    the AI, which tells a person from a game character better than any measurement can;
    what it says is measured again on the picture, so it cannot do any harm."""
    if job.get("facecam"):
        return job["facecam"]
    path = src or job.get("source_path")
    cam_pos = (job.get("options") or {}).get("cam_pos", "auto")
    if cam_pos and cam_pos != "auto":
        src_w, src_h = 1920, 1080
        if path and Path(path).is_file():
            try:
                info = media.probe(path)
                src_w, src_h = info.get("w") or 1920, info.get("h") or 1080
            except Exception:
                pass
        bw, bh = int(src_w * 0.25), int(src_h * 0.28)
        margin = int(src_w * 0.02)
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
        box = facecam.detect(path, ask=brain.look_for_facecam)
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
    if src["type"] == "url":
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
    if src["type"] == "url":
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

    vlock, vstate = threading.Lock(), {}

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
        if clip["status"] not in ("error", "failed") and clip.get("captions", True) and yt and yt[1].lower().startswith("en"):
            # The clip was found with YouTube's transcript. What is shown on screen comes from our own model,
            # run on just this clip: full punctuation, nothing blanked out, timings to the word.
            a0, b0 = max(lo, int((clip["start"] - 1.5) * 16000)), min(hi, int((clip["end"] + 1.5) * 16000))
            try:
                tw = [dict(w, s=round(w["s"] + a0 / 16000, 2), e=round(w["e"] + a0 / 16000, 2)) for w in transcribe.transcribe(audio[a0:b0])]
            except Exception:  # noqa: BLE001 - the clip is still made, with YouTube's words
                tw = []
            if len(tw) >= 3:
                exact[clip["id"]] = (a0 / 16000, b0 / 16000, tw)
                words_now = tw
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
    if src["type"] == "url" and config.YT_TRANSCRIPT and media.is_youtube(src["url"]):
        try:
            yt = media.youtube_words(src["url"])
        except Exception:  # noqa: BLE001 - no transcript to borrow: ours is written as usual
            yt = None
        if yt and not _covers(yt[0], lo / 16000, hi / 16000):
            yt = None
        job["transcript_from"] = "youtube" if yt else "own"
        job["lang"] = yt[1] if yt else ""          # the language spoken, when YouTube says (decides the captions)
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

    def hear(i):
        if i < len(blocks) and i >= done_blocks and i not in heard:
            a_, b_ = blocks[i]
            last = [0.0]

            def tp(p, i=i, a_=a_, b_=b_):
                if now_on[0] == i and p - last[0] > 0.05:      # only the part being shown moves the bar
                    last[0] = p
                    _set(job, progress=max(job["progress"], 0.14 + 0.78 * ((a_ - lo) + p * (b_ - a_)) / max(1, hi - lo)))
            heard[i] = ahead.submit(ctx.bind(transcribe.transcribe), audio[a_:b_], tp)

    # The ad breaks and static screens of the whole video, found once. It reads the video end to end,
    # so it is done here rather than for each clip, and only when asked for.
    statics = quality.static_ranges(job.get("source_path") or "") if config.SKIP_STATIC else []
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
            _set(job, stage="Writing the transcript" + where)
            now_on[0] = bi
            hear(bi)
            hear(bi + 1)
            bw = heard.pop(bi).result()
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
        review_mode = o.get("review_mode", "auto")
        for c in found:
            cid = f"clip{len(job['clips']) + 1:02d}"
            c.update(id=cid, start=round(c["start"] + shift, 2),
                     end=round(c["end"] + shift, 2), style=o["style"], layout=o["layout"],
                     captions=o.get("captions", True), meme=o.get("meme", "none"),
                     music=o.get("music", "none"), meme_vol=o.get("meme_vol"),
                     music_db=o.get("music_db"), split_ratio=o.get("split_ratio", 0.4),
                     cam_swap=o.get("cam_swap", False), cap_anchor=o.get("cap_anchor", "safe_bottom"),
                     cam_fallback=o.get("cam_fallback", "blur"), cap_y=o.get("cap_y"),
                     version=0)
            if o.get("facecam"):
                c["facecam"] = o["facecam"]
            elif o["layout"] in ("gameplay", "stack"):
                if face is False and job.get("source_path"):
                    face = _facecam(job)
                if face:
                    c["facecam"] = face

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
    _set(job, status="done", stage=stage, progress=1.0, took=round(time.time() - t0))
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
                clip = next(c for c in job["clips"] if c["id"] == cid)
                words = []
                if clip.get("captions", True):
                    wf = Path(job["dir"]) / "words.json"
                    if wf.is_file():
                        words = json.loads(wf.read_text(encoding="utf-8"))
                render_one(job, clip, words, quality="final")
            elif kind == "clip_draft":
                clip = next(c for c in job["clips"] if c["id"] == cid)
                words = []
                if clip.get("captions", True):
                    wf = Path(job["dir"]) / "words.json"
                    if wf.is_file():
                        words = json.loads(wf.read_text(encoding="utf-8"))
                render_one(job, clip, words, quality="draft")
        except media.JobPaused:
            _paused.discard(job["id"])
            _set(job, status="paused", stage="Paused by you")
        except Exception as e:  # noqa: BLE001
            (Path(job["dir"]) / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            if kind == "job":
                for c in job["clips"]:
                    if c.get("status") in ("queued", "rendering", "drafting"):
                        c.update(status="error", error="The video stopped before this clip was made.")
                _set(job, status="error", error=safe.friendly(e), stage="Stopped")
                creators.log(job["creator"], f"Video failed: {safe.friendly(e)[:200]}")
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
    threading.Thread(target=_worker, args=(True,), daemon=True).start()   # re-making a single clip never waits behind whole videos
    threading.Thread(target=_live_worker, daemon=True).start()

"""Getting video in: link download, probing, audio extraction, loudness."""
import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# ---------------------------------------------------------------- when YouTube refuses this server
# YouTube sometimes refuses a server's downloads altogether ("sign in to confirm you're not a bot"). It is a
# refusal of the server, not of one video, so once it happens YouTube links are turned away at once for a few
# minutes, with a clear message, instead of each one waiting to fail. Then the next link simply tries again.
YT_PAUSE = getattr(config, "YT_PAUSE", 0)
_yt_refused_until = 0.0
YT_BLOCKED = ("YouTube is asking for bot/sign-in verification for this video. "
              "To fix this, add your YouTube cookies in Settings > YouTube & Cookies, "
              "or upload the video file directly.")
_BOT_CHECK = re.compile(
    r"not a bot|confirm you.?re not|sign in to confirm(?! your age)|"
    r"use --cookies|Please sign in|HTTP Error 429|The page needs to be reloaded",
    re.I
)


class YouTubeBlocked(RuntimeError):
    pass


def is_youtube(url: str) -> bool:
    from urllib.parse import urlparse
    host = (urlparse((url or "").strip()).hostname or "").lower()
    return host in ("youtube.com", "youtu.be") or host.endswith((".youtube.com", ".youtu.be"))


def youtube_paused() -> bool:
    return time.time() < _yt_refused_until


def unpause_youtube():
    """Clear any temporary YouTube pause lock."""
    global _yt_refused_until
    _yt_refused_until = 0.0


def check_youtube(url: str):
    """Raise YouTubeBlocked straight away if YouTube is refusing this server right now."""
    if is_youtube(url) and youtube_paused():
        raise YouTubeBlocked(YT_BLOCKED)


def note_refusal(url: str, err) -> bool:
    """If this error is YouTube refusing the server, pause YouTube links for a while. True if it was."""
    global _yt_refused_until
    err_str = str(err).lower()
    if "live event has ended" in err_str or "not currently live" in err_str or "video unavailable" in err_str or "private video" in err_str:
        return False
    if is_youtube(url) and _BOT_CHECK.search(str(err)):
        if YT_PAUSE > 0:
            _yt_refused_until = time.time() + YT_PAUSE
        return True
    return False


@contextlib.contextmanager
def _youtube_guard(url):
    check_youtube(url)
    try:
        yield
    except YouTubeBlocked:
        raise
    except Exception as e:  # noqa: BLE001
        if note_refusal(url, e):
            raise YouTubeBlocked(YT_BLOCKED) from None
        raise


def get_cookie_file() -> Path | None:
    """Find an active Netscape cookies file from config or the data directory."""
    if getattr(config, "COOKIES_FILE", ""):
        p = Path(config.COOKIES_FILE)
        if p.is_file() and p.stat().st_size > 0:
            return p
    for name in ("cookies.txt", "youtube_cookies.txt", "youtube.cookies.txt"):
        p = config.DATA_DIR / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def get_browser_cookies() -> str:
    return (getattr(config, "COOKIES_BROWSER", "") or "").strip().lower()


def cookie_status() -> dict:
    cp = get_cookie_file()
    paused = youtube_paused()
    left = max(0, int(_yt_refused_until - time.time())) if paused else 0
    return {
        "has_cookies": bool(cp),
        "file_name": cp.name if cp else "",
        "size_bytes": cp.stat().st_size if cp else 0,
        "browser": get_browser_cookies(),
        "paused": paused,
        "pause_seconds_left": left,
        "message": YT_BLOCKED if paused else "",
    }


def save_cookies(text: str) -> dict:
    """Save Netscape cookies text to data/cookies.txt and unpause YouTube."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Cookie text cannot be empty.")
    target = config.DATA_DIR / "cookies.txt"
    target.write_text(text, encoding="utf-8")
    unpause_youtube()
    return cookie_status()


def remove_cookies() -> dict:
    """Remove any cookies.txt file from the data folder."""
    for name in ("cookies.txt", "youtube_cookies.txt", "youtube.cookies.txt"):
        p = config.DATA_DIR / name
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    return cookie_status()


def site_allowed(url: str) -> bool:
    """Links are fetched by the server, so only known video sites are accepted (never arbitrary addresses)."""
    from urllib.parse import urlparse
    try:
        u = urlparse(url.strip())
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    return u.scheme in ("http", "https") and any(host == s or host.endswith("." + s) for s in config.ALLOWED_SITES)


def run(cmd, cwd=None, timeout=3600):
    try:
        p = subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", creationflags=NO_WINDOW, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{Path(str(cmd[0])).name} took too long and was stopped.")
    if p.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed: {p.stderr[-1500:]}")
    return p.stdout


class _Quiet:
    def debug(self, m): pass
    def info(self, m): pass
    def warning(self, m): pass
    def error(self, m): pass


def get_aria2c_path() -> str | None:
    if not getattr(config, "USE_ARIA2", False):
        return None
    w = shutil.which("aria2c")
    if w:
        return w
    local_bin = config.ROOT / "data" / "bin" / ("aria2c.exe" if os.name == "nt" else "aria2c")
    if local_bin.is_file():
        return str(local_bin)
    data_bin = config.DATA_DIR / "bin" / ("aria2c.exe" if os.name == "nt" else "aria2c")
    if data_bin.is_file():
        return str(data_bin)
    if os.name == "nt":
        lad = os.environ.get("LOCALAPPDATA")
        if lad:
            wg = Path(lad) / "Microsoft" / "WinGet" / "Packages"
            if wg.is_dir():
                found = list(wg.glob("aria2.aria2_*/aria2-*/aria2c.exe"))
                if found and found[0].is_file():
                    return str(found[0])
    return None


def _get_aria2_args() -> list:
    args = [
        "-x", "8",
        "-s", "8",
        "-j", "8",
        "-k", "1M",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--summary-interval=0",
        "--timeout=30",
        "--max-tries=3",
        "--retry-wait=2",
    ]
    try:
        import certifi
        ca = certifi.where()
        if os.path.isfile(ca):
            args.extend(["--ca-certificate", ca])
        else:
            args.append("--check-certificate=false")
    except Exception:
        args.append("--check-certificate=false")
    return args


def _ydl(opts, url: str | None = None):
    import yt_dlp
    base = {
        "noplaylist": True,
        "ignoreconfig": True,
        "quiet": True,
        "no_warnings": True,
        "logger": _Quiet(),
        "extract_flat": "in_playlist",
        "playlistend": 1,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 8,
        "http_chunk_size": 10485760,
        "throttled_rate": "100K",
        "buffersize": 1048576,
    }
    cf = get_cookie_file()
    if cf and "cookiefile" not in opts:
        base["cookiefile"] = str(cf)
    elif get_browser_cookies() and "cookiesfrombrowser" not in opts:
        base["cookiesfrombrowser"] = (get_browser_cookies(),)

    url_is_yt = is_youtube(url) if url else False
    is_audio = bool(opts.get("is_audio")) or str(opts.get("format", "")).startswith("ba")

    allow_aria2 = (
        getattr(config, "USE_ARIA2", False)
        and opts.get("use_aria2", False) is True
        and "outtmpl" in opts
        and "download_ranges" not in opts
        and "external_downloader" not in opts
        and not url_is_yt
        and not is_audio
    )
    if allow_aria2:
        aria2 = get_aria2c_path()
        if aria2:
            base["external_downloader"] = {"default": aria2}
            base["external_downloader_args"] = {"default": _get_aria2_args()}

    merged = {**base, **opts}
    merged.pop("use_aria2", None)
    merged.pop("is_audio", None)
    if merged.get("external_downloader") is None:
        merged.pop("external_downloader", None)
        merged.pop("external_downloader_args", None)
    return yt_dlp.YoutubeDL(merged)


def _is_aria2_error(e: Exception) -> bool:
    s = str(e).lower()
    return "aria2" in s or "exited with code" in s or "external downloader" in s


def _execute_download(opts: dict, url: str):
    """Run yt-dlp download with complete fault tolerance:
    1. If an external downloader (e.g. aria2c) fails with exit code or error, retry immediately with native yt-dlp.
    2. If YouTube bot check is hit, retry with mweb/android player clients.
    3. Handles temporary network blips with retry.
    """
    cur_opts = dict(opts)
    if is_youtube(url) or cur_opts.get("is_audio") or str(cur_opts.get("format", "")).startswith("ba"):
        cur_opts["use_aria2"] = False
        cur_opts["external_downloader"] = None
        cur_opts.pop("external_downloader_args", None)

    try:
        with _ydl(cur_opts, url=url) as ydl:
            ydl.download([url])
            return
    except Exception as e:
        if isinstance(e, JobPaused) or "Job paused" in str(e):
            raise JobPaused("Job paused by user") from None

        # Fallback 1: If aria2c / external downloader failed, retry immediately using native yt-dlp downloader
        if _is_aria2_error(e) or cur_opts.get("external_downloader"):
            fallback_opts = dict(cur_opts)
            fallback_opts["use_aria2"] = False
            fallback_opts["external_downloader"] = None
            fallback_opts.pop("external_downloader_args", None)
            try:
                with _ydl(fallback_opts, url=url) as ydl:
                    ydl.download([url])
                    return
            except Exception as e2:
                if isinstance(e2, JobPaused) or "Job paused" in str(e2):
                    raise JobPaused("Job paused by user") from None
                e = e2

        # Fallback 2: If YouTube bot-check error, try alternate player clients
        if is_youtube(url) and _BOT_CHECK.search(str(e)):
            fb_opts = {
                **cur_opts,
                "use_aria2": False,
                "external_downloader": None,
                "extractor_args": {"youtube": {"player_client": ["mweb", "android"]}},
            }
            fb_opts.pop("external_downloader_args", None)
            try:
                with _ydl(fb_opts, url=url) as ydl:
                    ydl.download([url])
                    return
            except Exception as e3:
                if isinstance(e3, JobPaused) or "Job paused" in str(e3):
                    raise JobPaused("Job paused by user") from None
                raise e3 from None

        raise e


def _video_format(height):
    return f"bv*[height<={height}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={height}]+ba/b[height<={height}]/b"


def _biggest(job_dir: Path, stem: str) -> Path:
    files = sorted((p for p in job_dir.glob(stem + ".*") if p.suffix not in (".part", ".ytdl")),
                   key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("Download finished but no file was produced.")
    return files[0]


def _extract_info(url: str, download=False) -> dict:
    with _youtube_guard(url):
        try:
            with _ydl({}, url=url) as ydl:
                return ydl.extract_info(url, download=download)
        except Exception as e:
            if is_youtube(url) and _BOT_CHECK.search(str(e)):
                try:
                    fallback_opts = {"extractor_args": {"youtube": {"player_client": ["mweb", "android"]}}}
                    with _ydl(fallback_opts, url=url) as ydl:
                        return ydl.extract_info(url, download=download)
                except Exception:
                    raise e
            raise


_INFO_CACHE: dict = {}


def info(url: str) -> dict:
    """Title, channel and length, without downloading anything."""
    url = (url or "").strip()
    now = time.time()
    cached = _INFO_CACHE.get(url)
    if cached and now - cached[0] < 900:  # 15 minutes cache
        return dict(cached[1])
    try:
        i = _extract_info(url, download=False)
    except Exception as e:
        err_str = str(e).lower()
        if "live event has ended" in err_str or "not currently live" in err_str:
            raise ValueError("This YouTube live stream has ended. YouTube takes time to process the broadcast into a regular video before it can be clipped.") from e
        if "private video" in err_str or "sign in if you've been granted access" in err_str:
            raise ValueError("This video is private on YouTube. Make sure your account has access and cookies are saved.") from e
        if "members-only" in err_str or "join this channel" in err_str:
            raise ValueError("This video is for channel members only on YouTube.") from e
        raise
    out = {"title": i.get("title") or "", "uploader": i.get("uploader") or i.get("channel") or "",
           "duration": float(i.get("duration") or 0), "is_live": bool(i.get("is_live")),
           "thumbnail": i.get("thumbnail") or "", "kind": i.get("_type") or "video"}
    if len(_INFO_CACHE) > 300:
        _INFO_CACHE.pop(next(iter(_INFO_CACHE)))
    _INFO_CACHE[url] = (now, dict(out))
    return out


def _is_mismatched_auto_lang(lang: str, title: str, words: list) -> bool:
    """Detect when YouTube auto-captioning mis-tags a noisy English/Hinglish video as another script (e.g. Arabic)."""
    base = (lang or "").lower().split("-")[0]
    if base in ("", "en", "hi"):
        return False
    # Only suspect auto-caption mismatches for non-Latin scripts like Arabic/Persian/Hebrew/Cyrillic/CJK
    if base not in ("ar", "fa", "ur", "he", "iw", "ru", "uk", "ko", "ja", "zh", "th"):
        return False
    title_letters = [c for c in (title or "") if c.isalpha()]
    if not title_letters:
        return False
    ascii_ratio = sum(1 for c in title_letters if ord(c) < 128) / len(title_letters)
    # If the title is strongly Latin/ASCII (e.g. English/Hinglish gaming video) and has NO characters
    # from Arabic/Urdu/Cyrillic/CJK, reject a noisy auto-guessed Arabic/foreign caption track (except Urdu
    # when the title explicitly mentions Urdu/Pakistan).
    if ascii_ratio >= 0.75:
        if base == "ur" and any(k in (title or "").lower() for k in ("urdu", "pakistan", "pakistani")):
            return False
        sample = " ".join(w.get("w", "") for w in words[:80])
        non_ascii = sum(1 for c in sample if c.isalpha() and ord(c) >= 128)
        total = max(1, sum(1 for c in sample if c.isalpha()))
        if non_ascii / total >= 0.5:
            return True
    return False


def youtube_words(url: str, prefer_lang: str = ""):
    """YouTube's own transcript of a video, word by word: ([{'w', 's', 'e'}, ...], language), or None if the video
    has none. Only the transcript in the language that was spoken is used, never a translation."""
    if not is_youtube(url):
        return None
    import json
    i = _extract_info(url, download=False)
    auto = i.get("automatic_captions") or {}
    subs = i.get("subtitles") or {}
    pref = (prefer_lang or "").strip().lower()
    want_hi = pref in ("hi", "hinglish", "hi-en", "en-hi")
    want_en = pref == "en"

    def _pick_json3(group: dict, keys: tuple):
        for k in keys:
            if k in group:
                tr = next((f for f in group.get(k) or [] if f.get("ext") == "json3" and f.get("url")), None)
                if tr:
                    return k, tr
        return None, None

    key = None
    track = None
    is_auto = True

    if want_hi:
        # 1. Human-written Hindi / Hinglish subtitles first
        key, track = _pick_json3(subs, ("hi", "hi-IN", "hi-Latn"))
        if track:
            is_auto = False
        # 2. Original or auto-generated Hindi track (YouTube often tags Hinglish videos as hi-orig or hi)
        if not track:
            key, track = _pick_json3(auto, ("hi-orig", "hi", "hi-IN"))
            is_auto = True
    elif want_en:
        key, track = _pick_json3(subs, ("en", "en-US", "en-GB", "en-IN"))
        if track:
            is_auto = False
        if not track:
            key, track = _pick_json3(auto, ("en-orig", "en", "en-US", "en-GB"))
            is_auto = True

    if not track:
        key = next((k for k in auto if k.endswith("-orig")), None)
        if not key and i.get("language") in auto:
            key = i["language"]
        # Prefer human-written subtitles in the video's own language if available
        if i.get("language") and i["language"] in subs:
            track = next((f for f in subs.get(i["language"]) or [] if f.get("ext") == "json3" and f.get("url")), None)
            if track:
                key = i["language"]
                is_auto = False
        if not track:
            track = next((f for f in auto.get(key) or [] if f.get("ext") == "json3" and f.get("url")), None) if key else None
    if not track:
        return None
    with _ydl({}) as ydl:
        raw = ydl.urlopen(track["url"]).read(40_000_000)
    words = _caption_words(json.loads(raw))
    lang = (key or "").replace("-orig", "")
    if is_auto and not (want_hi and lang.lower().startswith("hi")) and _is_mismatched_auto_lang(lang, i.get("title") or "", words):
        return None
    return words, lang


def youtube_english_words(url: str):
    """YouTube's English captions of a video spoken in another language: ones a person wrote if there are any,
    otherwise YouTube's own translation of its transcript. Word list like youtube_words, or None."""
    if not is_youtube(url):
        return None
    import json
    i = _extract_info(url, download=False)
    tracks = []
    for group in (i.get("subtitles") or {}, i.get("automatic_captions") or {}):
        for k in ("en", "en-US", "en-GB"):
            tracks.append(next((f for f in group.get(k) or [] if f.get("ext") == "json3" and f.get("url")), None))
    track = next((t for t in tracks if t), None)
    if not track:
        return None
    with _ydl({}) as ydl:
        raw = ydl.urlopen(track["url"]).read(40_000_000)
    words = _caption_words(json.loads(raw))
    return words or None


def _caption_words(data: dict) -> list:
    out = []
    for ev in data.get("events") or []:
        segs = [x for x in ev.get("segs") or [] if (x.get("utf8") or "").strip()]
        if not segs:
            continue
        t0, dur = float(ev.get("tStartMs") or 0), float(ev.get("dDurationMs") or 0)
        timed = len(segs) > 1 or "tOffsetMs" in segs[0]
        for x in segs:
            parts = x["utf8"].split()
            for n, w in enumerate(parts):
                if re.fullmatch(r"[\[(].*[\])]", w) or w in ("[", "]", "__"):
                    continue                      # [Music], [Applause], and words YouTube blanked out
                # a line with no word timings: its words are spread evenly over the time the line was on screen
                at = t0 + float(x.get("tOffsetMs") or 0) if timed else t0 + dur * n / max(1, len(parts))
                out.append({"w": w, "s": round(at / 1000, 2)})
    out.sort(key=lambda w: w["s"])
    for n, w in enumerate(out):
        nxt = out[n + 1]["s"] if n + 1 < len(out) else w["s"] + 0.4
        w["e"] = round(max(w["s"] + 0.08, min(nxt, w["s"] + 0.6)), 2)
    return out


_PREVIEWS: dict = {}


def preview(url: str) -> dict:
    """What the user will get from a link, checked before any work starts. The thumbnail comes back inline
    (the page may only load images from this site). Cached, so retyping a link costs nothing."""
    url = url.strip()
    if url in _PREVIEWS:
        return _PREVIEWS[url]
    i = info(url)
    if i["kind"] != "video" or not i["title"]:
        raise ValueError("That link is a channel or playlist, not a single video.")
    thumb = ""
    t = i.get("thumbnail") or ""
    if t.startswith("https://"):
        import base64
        import urllib.request
        try:
            req = urllib.request.Request(t, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as r:
                ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip()
                data = r.read(600_000 + 1)
            if ctype in ("image/jpeg", "image/png", "image/webp") and len(data) <= 600_000:
                thumb = f"data:{ctype};base64," + base64.b64encode(data).decode()
        except Exception:
            thumb = ""
    out = {"title": i["title"][:200], "uploader": i["uploader"][:120], "duration": i["duration"],
           "is_live": i["is_live"], "thumb": thumb}
    if len(_PREVIEWS) > 300:
        _PREVIEWS.pop(next(iter(_PREVIEWS)))
    _PREVIEWS[url] = out
    return out


def save_thumb(job_dir: Path, url: str = "", video: Path = None) -> bool:
    """A small picture for the video list: the link's own thumbnail, or a frame from an uploaded file."""
    import base64
    out = Path(job_dir) / "clips" / "thumb.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if url:
            p = _PREVIEWS.get(url.strip()) or preview(url)
            if not p.get("thumb"):
                return False
            out.write_bytes(base64.b64decode(p["thumb"].split(",", 1)[1]))
            return True
        if video:
            d = probe(video).get("duration") or 0
            run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{min(3.0, d / 3):.2f}", "-i", str(video),
                 "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(out)])
            return out.is_file()
    except Exception:
        return False
    return False


class JobPaused(Exception):
    """Raised when a job is paused by the user."""
    pass


def fmt_bytes(n) -> str:
    if not n or n <= 0:
        return ""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.1f} GB"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{int(n)} B"


def fmt_eta(seconds) -> str:
    s = int(seconds or 0)
    if s <= 0:
        return ""
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _move_with_retry(src: Path, dst: Path, tries: int = 5) -> Path:
    """Move a downloaded file into place, retrying briefly if Windows antivirus or a handle briefly locks it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for i in range(tries):
        try:
            if dst.exists():
                try:
                    dst.unlink()
                except OSError:
                    pass
            try:
                src.replace(dst)
            except OSError:
                shutil.move(str(src), str(dst))
            return dst
        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(0.35 * (i + 1))
    raise last_err or RuntimeError(f"Could not save downloaded file to {dst.name}")


def _download_staged(url: str, out_dir: Path, stem: str, opts: dict) -> Path:
    """Download inside a short, space-free temp folder (nopart + windowsfilenames) and move the finished file
    into out_dir. Prevents Windows path-length and '.part' rename failures ('Giving up after 3 retries')."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="pkdl_"))
    try:
        staged_opts = dict(opts)
        staged_opts["outtmpl"] = str(tmp_dir / "f.%(ext)s")
        staged_opts["nopart"] = True
        staged_opts["windowsfilenames"] = True
        with _youtube_guard(url):
            _execute_download(staged_opts, url)
        got = _biggest(tmp_dir, "f")
        target = out_dir / f"{stem}{got.suffix}"
        return _move_with_retry(got, target)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_audio(url: str, job_dir: Path, on_progress=None, is_paused=None, start=None, end=None) -> Path:
    """Audio only: a few seconds even for long videos, and all the AI needs to pick clips."""
    def hook(d):
        if is_paused and is_paused():
            raise JobPaused("Job paused by user")
        if not on_progress:
            return
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            pct = (d.get("downloaded_bytes", 0) / total) if total else 0.0
            try:
                on_progress(pct, d)
            except TypeError:
                on_progress(pct)

    opts = {
        "format": "ba[ext=m4a]/ba/worst",
        "progress_hooks": [hook],
        "is_audio": True,
        "external_downloader": None,
    }
    start_sec = max(0.0, float(start or 0.0))
    end_sec = float(end) if end is not None else None
    if start_sec > 0 or end_sec is not None:
        from yt_dlp.utils import download_range_func
        opts["download_ranges"] = download_range_func(None, [(start_sec, end_sec or float("inf"))])
        opts["force_keyframes_at_cuts"] = True

    return _download_staged(url, job_dir, "audio_src", opts)


def download(url: str, job_dir: Path, height=720) -> Path:
    """The whole video. Used for shorter videos, where this is quicker than fetching pieces."""
    opts = {"format": _video_format(height), "merge_output_format": "mp4"}
    if is_youtube(url):
        opts["external_downloader"] = None
    return _download_staged(url, job_dir, "source", opts)


def download_section(url: str, out_dir: Path, stem: str, start: float, end: float, height=720, exact=False,
                     on_progress=None, is_paused=None) -> Path:
    """Just one stretch of the video. The cut lands on a keyframe, so find_offset() pins the true start; with
    exact=True the cut is made at exactly `start` instead (slower), for when there is nothing to match against."""
    from yt_dlp.utils import download_range_func
    def hook(d):
        if is_paused and is_paused():
            raise JobPaused("Job paused by user")
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            pct = (d.get("downloaded_bytes", 0) / total) if total else 0.0
            try:
                on_progress(pct, d)
            except TypeError:
                on_progress(pct)

    opts = {"format": _video_format(height), "merge_output_format": "mp4",
            "download_ranges": download_range_func(None, [(max(0.0, start), end)]),
            "progress_hooks": [hook],
            "external_downloader": None}
    if exact:
        opts["force_keyframes_at_cuts"] = True
    return _download_staged(url, out_dir, stem, opts)


def find_offset(full: np.ndarray, part: np.ndarray, expected: float, search=20.0) -> float:
    """Where in the full audio (seconds) a downloaded piece really begins, by matching the sound itself."""
    n = min(len(part), 30 * 16000)
    a0 = max(0, int((expected - search) * 16000))
    hay = full[a0: int((expected + search) * 16000) + n].astype(np.float32)
    needle = part[:n].astype(np.float32)
    if n < 16000 or len(hay) < n or float(np.abs(needle).mean()) < 20:
        return expected  # too short or silent to match: trust the requested time
    size = 1 << (len(hay) + n).bit_length()
    corr = np.fft.irfft(np.fft.rfft(hay, size) * np.conj(np.fft.rfft(needle, size)), size)[: len(hay) - n + 1]
    return (a0 + int(np.argmax(corr))) / 16000


def probe(path: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    data = json.loads(out)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not v:
        raise RuntimeError("This file has no video track.")
    if not any(s["codec_type"] == "audio" for s in data["streams"]):
        raise RuntimeError("This file has no audio track, so there is nothing to transcribe.")
    w, h = int(v["width"]), int(v["height"])
    if w * h > 3840 * 2160 or w < 16 or h < 16:
        raise RuntimeError("This video's resolution isn't supported (4K is the largest).")
    rot = str((v.get("side_data_list") or [{}])[0].get("rotation", "0"))
    if rot.lstrip("-") in ("90", "270"):
        w, h = h, w
    return {"width": w, "height": h, "duration": float(data["format"]["duration"])}


def extract_audio(src: Path, wav: Path):
    # first_pts=0 pads any gap before the audio starts, so second 0 of the wav is second 0 of the file
    # -t: never more audio than the longest video allowed, whatever the file claims about itself
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-t", str(int(config.MAX_VIDEO_HOURS * 3600) + 60),
         "-af", "aresample=async=1:first_pts=0", "-c:a", "pcm_s16le", str(wav)])


def load_audio(wav: Path) -> np.ndarray:
    data, sr = sf.read(str(wav), dtype="int16")
    assert sr == 16000
    return data


def loudness_per_second(audio: np.ndarray) -> np.ndarray:
    """RMS level in dB for each second of audio."""
    n = len(audio) // 16000
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    x = audio[: n * 16000].reshape(n, 16000).astype(np.float32) / 32768.0
    rms = np.sqrt((x ** 2).mean(axis=1) + 1e-10)
    return (20 * np.log10(rms)).astype(np.float32)

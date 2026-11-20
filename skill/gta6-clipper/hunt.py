import os
"""Live hunt: who of the user's streamers is live now, and start/stop live.py on them. Public endpoints only, no keys, read-only."""
import json, re, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (clipper hunt)"}
CACHE = {}; PROCS = {}; LOCK = threading.Lock()   # CACHE[(platform, handle)] = (time, info); PROCS[key] = Popen
def _get(url, t=10): return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=t).read().decode("utf-8", "replace")
def look_twitch(h):
    import live
    s = live.gql('{user(login:%s){stream{title viewersCount createdAt}}}' % json.dumps(h))["user"]; s = s and s["stream"]
    return dict(live=bool(s), title=s["title"] if s else "", viewers=s["viewersCount"] if s else 0, started=s["createdAt"] if s else "")
def look_kick(h):
    d = json.loads(_get("https://kick.com/api/v2/channels/" + h)); l = d.get("livestream")
    return dict(live=bool(l), title=(l or {}).get("session_title", ""), viewers=(l or {}).get("viewer_count", 0), started=(l or {}).get("created_at", ""))
def look_youtube(h):
    t = _get(f"https://www.youtube.com/{h if h.startswith('@') else '@' + h}/live")
    on = '"isLive":true' in t; m = re.search(r'<meta property="og:title" content="([^"]*)"', t)
    return dict(live=on, title=(m[1] if m and on else ""), viewers=0, started="")
LOOK = dict(twitch=look_twitch, kick=look_kick, youtube=look_youtube)
def lookup(p, h, now=None):
    now = now or time.time(); c = CACHE.get((p, h))
    if c and now - c[0] < 60: return c[1]
    try: r = LOOK[p](h)
    except Exception as e: r = dict(live=False, title="", viewers=0, started="", error=str(e)[:80])
    CACHE[(p, h)] = (now, r); return r
def key(p, h): return f"{p}:{h.lower()}"
def outdir(ws, p, h): return Path(ws) / "hunt" / f"{p}_{h.lstrip('@').lower()}"
def clips_found(d):
    try: return len(json.loads((d / "picks.json").read_text(encoding="utf-8")))
    except Exception: return 0
WS = None  # last workspace seen, so a restarted app finds hunts it did not start (pid files)
def _alive(pid):
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True)
        return "python" in r.stdout.lower()
    try: os.kill(pid, 0); return True
    except OSError: return False
def _pid(p, h):
    try: return int((outdir(WS, p, h) / "hunt.pid").read_text().strip()) if WS else None
    except Exception: return None
def watching(p, h):
    pr = PROCS.get(key(p, h))
    if pr: return pr.poll() is None
    pid = _pid(p, h); return bool(pid and _alive(pid))
def hunt(ws, streamers, lookup_fn=lookup, limit=25):
    global WS; WS = ws
    ss = streamers[:limit]
    with ThreadPoolExecutor(8) as ex: infos = list(ex.map(lambda s: lookup_fn(s["platform"], s["handle"]), ss))
    out = []
    for s, i in zip(ss, infos):
        w = watching(s["platform"], s["handle"]); d = outdir(ws, s["platform"], s["handle"])
        out.append(dict(handle=s["handle"], name=s.get("name", s["handle"]), platform=s["platform"], live=bool(i["live"]), title=i.get("title", ""), viewers=i.get("viewers", 0),
                        chat_per_min=None, watching=w, clips_found=clips_found(d), started=i.get("started", "")))   # chat_per_min: only live.py measures it (per stream, not in the 60 s lookup)
    return sorted(out, key=lambda r: (not r["live"], -r["viewers"]))
def src(p, h): return {"twitch": f"https://www.twitch.tv/{h}", "kick": f"kick:{h}", "youtube": f"https://www.youtube.com/{h if h.startswith('@') else '@' + h}/live"}[p]
def start(ws, p, h, py=sys.executable):
    if p not in LOOK or not re.fullmatch(r"@?[\w.\-]{1,60}", h or ""): raise ValueError("bad platform or handle")
    global WS; WS = ws
    if watching(p, h): return dict(ok=True, already=True)
    d = outdir(ws, p, h); d.mkdir(parents=True, exist_ok=True); log = open(d / "live.log", "ab")
    with LOCK: PROCS[key(p, h)] = subprocess.Popen([py, str(HERE / "live.py"), src(p, h), "--out", str(d)], cwd=str(d), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    (d / "hunt.pid").write_text(str(PROCS[key(p, h)].pid))
    return dict(ok=True, out=str(d), log=str(d / "live.log"))
def stop(p, h):
    pr = PROCS.pop(key(p, h), None); pid = None if pr else _pid(p, h)
    if pr and pr.poll() is None: pr.terminate()
    elif pid and _alive(pid):  # started by an earlier app run
        if os.name == "nt": subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else: os.kill(pid, 15)
    if WS: (outdir(WS, p, h) / "hunt.pid").unlink(missing_ok=True)
    return dict(ok=True, was_running=bool(pr or pid))
PCACHE = {}   # (platform, handle) -> (time, rows); 10 min
def past_twitch(h):
    import live
    e = live.gql('{user(login:%s){videos(type:ARCHIVE,first:5){edges{node{id title createdAt lengthSeconds viewCount}}}}}' % json.dumps(h))["user"]
    return [dict(title=n["title"] or "", url="https://www.twitch.tv/videos/" + n["id"], date=(n["createdAt"] or "")[:10], duration_s=n["lengthSeconds"] or 0, views=n["viewCount"] or 0)
            for n in (x["node"] for x in ((e or {}).get("videos") or {}).get("edges", []))]
def past_kick(h):
    d = json.loads(_get(f"https://kick.com/api/v2/channels/{h}/videos"))   # often Cloudflare-blocked: the caller turns that into a row
    return [dict(title=(v.get("session_title") or ""), url=f"https://kick.com/{h}/videos/{v.get('video', {}).get('uuid') or v.get('uuid', '')}", date=(v.get("created_at") or v.get("start_time") or "")[:10],
                 duration_s=int((v.get("duration") or 0) / 1000), views=v.get("views") or 0) for v in (d if isinstance(d, list) else d.get("data", []))[:5]]
def past_youtube(h):
    o = subprocess.run([sys.executable, "-m", "yt_dlp", "--flat-playlist", "-j", "--playlist-end", "5", "--no-warnings", f"https://www.youtube.com/{h if h.startswith('@') else '@' + h}/streams"],
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    rows = [json.loads(l) for l in o.stdout.splitlines() if l.startswith("{")]
    if not rows: raise RuntimeError((o.stderr.strip().splitlines() or ["no streams found"])[-1][:100])
    return [dict(title=r.get("title") or "", url=r.get("webpage_url") or r.get("url") or "", date=(r.get("upload_date") or "")[:4] + "-" + (r.get("upload_date") or "")[4:6] + "-" + (r.get("upload_date") or "")[6:8] if r.get("upload_date") else "",
                 duration_s=int(r.get("duration") or 0), views=r.get("view_count") or 0) for r in rows[:5]]
PAST = dict(twitch=past_twitch, kick=past_kick, youtube=past_youtube)
def past(p, h, now=None, fns=None):
    if p not in PAST or not re.fullmatch(r"@?[\w.\-]{1,60}", h or ""): raise ValueError("bad platform or handle")
    now = now or time.time(); c = PCACHE.get((p, h))
    if c and now - c[0] < 600: return c[1]
    try: r = (fns or PAST)[p](h)
    except Exception as e:
        msg = str(e)[:100]; r = [dict(title="", url="", date="", duration_s=0, views=0, error=("Kick blocks this lookup: " if p == "kick" else "Could not load: ") + msg)]
    PCACHE[(p, h)] = (now, r); return r

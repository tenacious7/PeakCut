# Live clipping: record a live stream into rolling segments, watch Twitch chat (anonymous IRC), on a chat spike cut a clip + write a batch.py picks entry.
# Usage: PY live.py https://www.twitch.tv/<channel> [--out clips/live] [--ratio 3] [--min 15]   |   PY live.py selftest
# Each spike: clips/live/spikeNN_HHMMSS.mp4 + .txt (transcript, [LOUD] tags) + a row in picks.json and spikes.log (chat rate + top lines). Stop with Ctrl+C.
# live.py top:"Grand Theft Auto VI"  = pick the live English channel in that category with the fastest chat (60 s measured on the top 5), then start on it.
# Sources: twitch link/name, kick:<slug> or kick.com/<slug>, youtube link / @handle / youtube:<id>. Chat for all three comes from chat.live_chat (no keys, read-only).
import collections, datetime as dt, json, os, re, socket, statistics, subprocess, sys, threading, time
from pathlib import Path
RENDER_LOCK = threading.Lock()  # one render at a time per stream (weak PCs)
HERE = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")
BIN, KEEP, PRE, POST, WIN, COOL = 10, 600, 25, 10, 10, 60  # bin s, seconds of video kept, clip seconds before/after spike, spike window s, cooldown s

class Detector:
    """Spike = different chatters in the last WIN s >= ratio x median of the previous 5 min's WIN-s counts, and >= min_n. Needs 2 min of history.
    People, not messages: one spammer or a bot can't fake a spike."""
    def __init__(s, ratio=3.0, min_n=15): s.ratio, s.min_n, s.ts, s.last = ratio, min_n, collections.deque(), -1e9
    def add(s, t, nick="?"): s.ts.append((t, nick))
    def check(s, now):
        while s.ts and s.ts[0][0] < now - 330: s.ts.popleft()
        cnt = lambda a, b: len({n for t, n in s.ts if a < t <= b})
        if not s.ts: return None
        cur = cnt(now - WIN, now); hist = [cnt(now - WIN * (i + 2), now - WIN * (i + 1)) for i in range(30) if now - WIN * (i + 2) >= s.ts[0][0]]  # only windows after the first message
        if len(hist) < 12: return None  # under 2 min of history
        base = statistics.median(hist)
        if cur >= s.min_n and cur >= s.ratio * max(base, 1) and now - s.last > COOL: s.last = now; return cur, base
        return None

BOTS = {"nightbot", "streamelements", "streamlabs", "moobot", "fossabot", "wizebot", "botrixoficial", "sery_bot", "kofistreambot", "pokemoncommunitygame"}
def is_bot(nick, text): return nick.lower() in BOTS or "http" in text or ".com/" in text or text.startswith("!")  # bots, ad links, chat commands

# ---- Twitch discovery (public GQL, no login): used by watch.py twitch: and `live.py top:"<category>"`
import urllib.request
CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
FALLBACK = {"grand theft auto vi": ("Grand Theft Auto V", r"GTA ?(6|VI)\b")}  # category not populated yet: look in the old one by title
def gql(q):
    r = urllib.request.Request("https://gql.twitch.tv/gql", json.dumps({"query": q}).encode(), {"Client-ID": CLIENT_ID, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=30))["data"]
def rank(streams, lang="en", rx=None):
    """streams = [dict(login, viewers, lang, title)] -> wanted language (any = all) and title regex only, most viewers first."""
    return sorted((s for s in streams if (lang == "any" or s["lang"].lower() == lang.lower()) and (not rx or re.search(rx, s["title"], re.I))), key=lambda s: -s["viewers"])
def category_streams(name, lang="en", n=40):
    """-> (category shown, ranked live streams). Exact category first; if it is empty or unknown, the FALLBACK category filtered by title regex."""
    cat, rx = name, None
    for _ in range(2):
        d = gql('{game(name:%s){name streams(first:100,options:{sort:VIEWER_COUNT}){edges{node{viewersCount title language broadcaster{login}}}}}}' % json.dumps(cat))["game"]
        ss = rank([dict(login=e["node"]["broadcaster"]["login"], viewers=e["node"]["viewersCount"], lang=e["node"]["language"] or "", title=e["node"]["title"]) for e in (d["streams"]["edges"] if d else [])], lang, rx)
        if ss or name.lower() not in FALLBACK or rx: return (d or {}).get("name", cat) + (f" (title ~ {rx})" if rx else ""), ss[:n]
        cat, rx = FALLBACK[name.lower()]
    return cat, []
def vod_window(length, win=None):
    """win None = last 2 h, '90m'/'3h' = last that long, 'A-B' = as given (H:MM:SS) -> 'start-end' text for yt-dlp and chat."""
    f = lambda x: f"{int(x) // 3600}:{int(x) // 60 % 60:02d}:{int(x) % 60:02d}"
    if win and "-" in win: return win
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([hm])", win or "2h"); last = float(m[1]) * (3600 if m[2] == "h" else 60)
    return f"{f(max(0, length - last))}-{f(length)}"
def recent_vods(logins, now=None, hours=24, min_len=1200):
    """-> newest finished-or-running broadcast of each channel from the last `hours` h, at least min_len s: [dict(login, id, title, length)]."""
    now, out = now or time.time(), []
    for l in logins:
        try: es = gql('{user(login:%s){videos(first:3,type:ARCHIVE,sort:TIME){edges{node{id title lengthSeconds publishedAt}}}}}' % json.dumps(l))["user"]["videos"]["edges"]
        except Exception as e: print("  vods failed", l, str(e)[:60]); continue
        for e in es:
            n = e["node"]; t = dt.datetime.fromisoformat(n["publishedAt"].replace("Z", "+00:00")).timestamp()
            if now - t <= hours * 3600 and n["lengthSeconds"] >= min_len: out.append(dict(login=l, id=n["id"], title=n["title"], length=n["lengthSeconds"], age_h=(now - t) / 3600)); break
    return out
def measure_chat(chans, secs=60):
    """Anonymous chat read, all channels at once for `secs` s -> {channel: human messages per minute}."""
    cnt = {c: 0 for c in chans}
    def one(c):
        try:
            s = socket.create_connection(("irc.chat.twitch.tv", 6667), timeout=5); s.sendall(f"NICK justinfan12345\r\nJOIN #{c}\r\n".encode()); end, buf = time.time() + secs, ""
            while time.time() < end:
                s.settimeout(max(0.5, end - time.time()))
                try: data = s.recv(8192)
                except socket.timeout: break
                if not data: break
                buf += data.decode("utf-8", "replace"); *ls, buf = buf.split("\r\n")
                for l in ls:
                    if l.startswith("PING"): s.sendall(b"PONG :tmi.twitch.tv\r\n")
                    elif m := re.match(r":(\w+)!\S+ PRIVMSG #\S+ :(.*)", l):
                        if not is_bot(m[1], m[2]): cnt[c] += 1
            s.close()
        except OSError as e: print("  chat read failed", c, e)
    ts = [threading.Thread(target=one, args=(c,)) for c in chans]; [t.start() for t in ts]; [t.join() for t in ts]
    return {c: n * 60 / secs for c, n in cnt.items()}
def top_channel(category, lang=None, secs=60):
    lang = lang or os.environ.get("CLIPPER_LANG", "en")
    cat, ss = category_streams(category, lang, 5)
    if not ss: sys.exit(f"nobody live in {cat} ({lang})")
    rates = measure_chat([s["login"] for s in ss], secs)
    for s in ss: print(f"  {s['login']:<20}{s['viewers']:>8,} viewers  chat {rates.get(s['login'], 0):>5.0f} msg/min  {s['title'][:50]}")
    best = max(ss, key=lambda s: rates.get(s["login"], 0)); print(f"-> {best['login']} (fastest chat of {len(ss)} in {cat})", flush=True)
    return "https://www.twitch.tv/" + best["login"]

# ---- AI check per spike (pick modes): free = off, smart = the clip's words + tags, deep = + last 5 min of chat. Provider = app.py's AI table (text command, prompt on stdin).
def provider(prompt, timeout=60):
    sys.path.insert(0, str(HERE)); import app
    return app.text_ai(app.ai_cmd(app.cur_ai()), prompt, timeout)
def live_cfg():
    try: sys.path.insert(0, str(HERE)); import app; c = app.pick_cfg()
    except Exception: c = {}
    return os.environ.get("LIVE_MODE") or c.get("live_mode", "free"), int(os.environ.get("LIVE_CAP") or c.get("live_cap", 20))
class AiGate:
    def __init__(s, mode, cap, ask=None): s.mode, s.cap, s.ask, s.ts = mode, cap, ask or provider, collections.deque()
    def check(s, words, tags, chat5=None, now=None, span=PRE + POST):
        """-> dict(keep, start, end, hook) from the AI, or None = no decision (free mode, cap reached, error): the caller keeps the clip as in free mode."""
        now = now or time.time()
        while s.ts and s.ts[0] < now - 3600: s.ts.popleft()
        if s.mode == "free" or len(s.ts) >= s.cap: return None
        s.ts.append(now)
        p = (f"A live GTA 6 stream chat just spiked. Clip of {span:.0f} s. Tags: {tags}. Words: {words[:1500]}\n" + (f"Chat in the last 5 minutes: {' | '.join(chat5 or [])[:2500]}\n" if s.mode == "deep" else "")
             + 'Reply with JSON only: {"keep": true/false, "start": seconds, "end": seconds, "hook": "max 60 chars, starts with a verb, no emoji"}. Drop it if there is no story or payoff.')
        try:
            o = json.loads(re.search(r"\{.*\}", s.ask(p), re.S)[0]); keep = bool(o.get("keep", True))
            a, b = float(o.get("start", 0)), float(o.get("end", span)); a = min(max(a, 0), span - 8); b = min(max(b, a + 8), span)
            return dict(keep=keep, start=a, end=b, hook=str(o.get("hook") or "")[:70])
        except Exception as e: print("AI check failed, keeping the clip:", str(e)[:80], flush=True); return None

def selftest():
    d, t, fired = Detector(3, 15), 0.0, []
    for i in range(300):  # 5 min at ~1 msg/s, then 40 msgs in 10 s at t=300..310
        t = float(i); d.add(t, f"u{i % 50}"); r = d.check(t)
        assert r is None, f"false spike at {t}: {r}"
    for i in range(40):
        t = 300 + i / 4; d.add(t, f"v{i}"); r = d.check(t)
        if r: fired.append((t, r))
    assert len(fired) == 1 and fired[0][0] < 310, fired  # fires once, early, then cooldown holds
    s = Detector(3, 15)  # one spammer flooding 200 messages is 1 chatter: no spike
    for i in range(300): s.add(float(i), f"u{i % 50}")
    assert all(s.check(300 + i / 20) is None for i in range(200) if not s.add(300 + i / 20, "spammer"))
    assert Detector().check(5) is None and is_bot("Nightbot", "hi") and is_bot("x", "bonus at https://bet.example") and not is_bot("x", "LETS GO")
    S = lambda l, v, g, t="x": dict(login=l, viewers=v, lang=g, title=t)
    r = rank([S("a", 5, "EN"), S("b", 50, "EN"), S("c", 99, "RU"), S("d", 7, "en")]); assert [x["login"] for x in r] == ["b", "d", "a"]
    assert [x["login"] for x in rank([S("a", 5, "EN", "GTA 6 live"), S("b", 50, "EN", "GTA V rp")], "en", FALLBACK["grand theft auto vi"][1])] == ["a"] and len(rank([S("c", 1, "RU")], "any")) == 1
    assert vod_window(30000) == "6:20:00-8:20:00" and vod_window(1800, "3h") == "0:00:00-0:30:00" and vod_window(9000, "90m") == "1:00:00-2:30:00" and vod_window(9000, "0:10:00-0:20:00") == "0:10:00-0:20:00"
    g = AiGate("smart", 2, lambda p: 'ok {"keep": false, "start": 3, "end": 50, "hook": "Watch it"}'); r = g.check("w", "LOUD", now=1000); assert r == dict(keep=False, start=3.0, end=35.0, hook="Watch it"), r
    assert g.check("w", "", now=1001) and g.check("w", "", now=1002) is None and g.check("w", "", now=5000), "cap 2 per hour"
    assert AiGate("free", 5, lambda p: 1 / 0).check("w", "") is None and AiGate("deep", 5, lambda p: 1 / 0).check("w", "") is None, "free = no call, error = fallback"
    seen = []; AiGate("deep", 5, lambda p: seen.append(p) or "{}").check("w", "", ["lets go"]); AiGate("smart", 5, lambda p: seen.append(p) or "{}").check("w", "", ["lets go"]); assert "lets go" in seen[0] and "lets go" not in seen[1]
    print("selftest ok", fired)

def chat(chan, on_msg):
    while True:
        try:
            s = socket.create_connection(("irc.chat.twitch.tv", 6667), timeout=330)
            s.sendall(f"NICK justinfan12345\r\nJOIN #{chan}\r\n".encode()); buf = ""
            while True:
                data = s.recv(8192)
                if not data: break
                buf += data.decode("utf-8", "replace"); *lines, buf = buf.split("\r\n")
                for l in lines:
                    if l.startswith("PING"): s.sendall(b"PONG :tmi.twitch.tv\r\n")
                    elif m := re.match(r":(\w+)!\S+ PRIVMSG #\S+ :(.*)", l): on_msg(time.time(), m[1], m[2])
        except OSError as e: print("chat reconnect:", e, flush=True); time.sleep(3)

def record(url, seg):  # shortcut: no stall watchdog, DST-ambiguous local names, children can outlive a crash; fine for an evening session, harden for 24/7
    while True:  # restart if the stream hiccups; segment names carry the wall-clock start, so restarts are harmless
        sl = subprocess.Popen([sys.executable, "-m", "streamlink", "--stdout", "--twitch-disable-ads", url, "720p60,720p,best"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "-", "-c", "copy", "-f", "segment", "-segment_time", "4", "-strftime", "1", "-reset_timestamps", "1", str(seg / "s_%Y%m%d_%H%M%S.ts")], stdin=sl.stdout)
        sl.kill(); time.sleep(5)

def segs(seg): return sorted((dt.datetime.strptime(p.stem[2:], "%Y%m%d_%H%M%S").timestamp(), p) for p in seg.glob("s_*.ts"))

def cut(seg, out, T, name):
    a, b = T - PRE, T + POST
    done = segs(seg)[:-1]  # newest segment is still being written. shortcut: a closed newest segment during a restart is skipped too
    first = max([i for i, (t, _) in enumerate(done) if t <= a], default=0)  # the segment holding the start, however long it ran
    sel = [(t, p) for t, p in done[first:] if t <= b]
    if not sel: return None
    lst = seg / f"{name}.txt"; lst.write_text("".join(f"file '{p.name}'\n" for _, p in sel), encoding="utf-8")  # names relative to the list: no quoting or code-page issues
    mp4 = out / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-ss", f"{max(a - sel[0][0], 0):.2f}", "-t", str(PRE + POST), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(mp4)], check=True)
    return mp4

def buffer(out):
    """The last ~10 minutes as one file, for quiet chats: then `watch.py <file>` and pick from the words and [LOUD] like any video."""
    done = segs(out / "seg")[:-1]
    if not done: sys.exit("no recorded segments in " + str(out / "seg"))
    name = f"buffer_{dt.datetime.now():%H%M%S}"; lst = out / "seg" / f"{name}.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for _, p in done), encoding="utf-8")
    mp4 = out / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-bsf:a", "aac_adtstoasc", str(mp4)], check=True)
    return mp4.resolve()

def main(url, out, ratio, min_n):
    import chat as chatlib
    kind, chan = chatlib.live_source(url); src = url
    if kind == "twitch": url = "https://www.twitch.tv/" + chan
    elif kind == "kick": url = "https://kick.com/" + chan
    else: vid = chatlib.yt_video_id(chan); chan = vid; url = "https://www.youtube.com/watch?v=" + vid
    out = Path(out); seg = out / "seg"; seg.mkdir(parents=True, exist_ok=True)
    pj = out / "picks.json"; picks = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else []  # a restart keeps earlier picks
    det, recent, n = Detector(ratio, min_n), collections.deque(maxlen=4000), len(picks)
    gate = AiGate(*live_cfg()); print(f"live pick mode: {gate.mode} (cap {gate.cap}/h)", flush=True)
    busy = threading.Semaphore(2)  # at most 2 clips cut/transcribed at once; spikes meanwhile are logged, not cut
    lock = threading.Lock()
    def on_msg(t, nick, text):
        if is_bot(nick, text): return  # bot ads and !commands are not a reaction
        with lock: det.add(t, nick); recent.append((t, text))
    threading.Thread(target=chatlib.live_chat, args=(src, on_msg, threading.Event()), daemon=True).start()
    threading.Thread(target=record, args=(url, seg), daemon=True).start()
    log = out / "spikes.log"; print(f"live on {chan}: ratio>={ratio} min>={min_n}", flush=True); last_clean, started, warned, last_rate = 0, time.time(), 0, 0
    while True:
        time.sleep(2); now = time.time()
        if now - last_clean > 30:
            last_clean = now
            if now - started > 90 and not segs(seg) and not warned: warned = 1; print("no video yet: is the channel live? (streamlink gets nothing)", flush=True)
            if now - started > 120 and now - last_rate >= 300:  # chat speed decides if spikes can work at all
                last_rate = now
                with lock: per_min = sum(t > now - 60 for t, _ in recent)
                print(f"chat: {per_min} messages in the last minute" + (" (too quiet for spikes: use `live.py buffer`)" if per_min < 30 else ""), flush=True)
            for t, p in segs(seg):
                if t < now - KEEP:
                    try: p.unlink(missing_ok=True)
                    except OSError: pass  # still open (Windows lock): next pass
        with lock: r = det.check(now); snap = list(recent)
        if not r: continue
        cur, base = r; n += 1; T = now; lines = collections.Counter(x.lower() for t, x in snap if now - WIN - 5 < t <= now).most_common(5)
        rec = {"n": n, "time": dt.datetime.fromtimestamp(T).isoformat(timespec="seconds"), "rate_per_10s": cur, "baseline": base, "top": lines}
        print("SPIKE", json.dumps(rec, ensure_ascii=False), flush=True); open(log, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not busy.acquire(blocking=False): print("busy: spike logged, not cut", flush=True); continue
        def finish(T=T, n=n, lines=lines):
            time.sleep(POST + 8)  # wait until the post-spike video has been written
            try:
                t0 = time.time(); name = f"spike{n:02d}_{dt.datetime.fromtimestamp(T):%H%M%S}"; mp4 = cut(seg, out, T, name)
                if not mp4: return print("no segments for", name, flush=True)
                tj = mp4.resolve().with_suffix(".json").as_posix()  # word timings for the captions + a .txt Claude reads; forward slashes stay valid when copied into keep.json by hand
                subprocess.run([sys.executable, str(HERE / "transcribe.py"), str(mp4), tj], check=True, stdout=subprocess.DEVNULL)
                gate_r = gate.check(Path(tj).with_suffix(".txt").read_text(encoding="utf-8", errors="replace") if Path(tj).with_suffix(".txt").exists() else "", "LOUD,CHAT", [x for t, x in snap if T - 300 < t <= T]) if gate.mode != "free" else None
                if gate_r and not gate_r["keep"]: return print("AI dropped", name, flush=True)
                entry = {"url": mp4.resolve().as_posix(), "transcript": tj, "start": 0, "end": PRE + POST, "hook": f"{chan.upper()} CHAT EXPLODED", "name": name, "title": f"{chan} chat explodes ({lines[0][0] if lines else ''})", "hashtags": "#twitch #shorts", "credit": chan, "keywords": [], "chat": [x for x, _ in lines]}
                if gate_r: entry.update(start=gate_r["start"], end=gate_r["end"], **({"hook": gate_r["hook"]} if gate_r["hook"] else {}))
                with lock:  # one writer at a time, and readers only ever see a complete file
                    picks.append(entry); tmp = pj.with_suffix(".tmp"); tmp.write_text(json.dumps(picks, indent=1, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, pj)
                print(f"clip ready {mp4} ({time.time() - t0:.0f}s to build)", flush=True)
                if os.environ.get("CLIPPER_LIVE_RENDER", "1") != "0":  # render the new pick into <stream>/out so the app Library shows it (batch skips moments already made)
                    with RENDER_LOCK: r = subprocess.run([sys.executable, str(HERE / "batch.py"), str(pj), str(pj.parent / "out")], capture_output=True, text=True)
                    print("short rendered" if r.returncode == 0 else "render failed: " + (r.stderr or r.stdout)[-300:], flush=True)
            except Exception as e: print("clip failed:", e, flush=True)
            finally: busy.release()
        threading.Thread(target=finish, daemon=True).start()

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]: selftest()
    elif a[:1] == ["buffer"]: print(buffer(Path(a[1] if len(a) > 1 else "clips/live")))
    elif a[0].startswith("top:"): a[0] = top_channel(a[0][4:].strip().strip("\"'")); main(a[0], *(lambda o: (o.get("--out", "clips/live"), float(o.get("--ratio", 3)), int(o.get("--min", 15))))({k: a[a.index(k) + 1] for k in ("--out", "--ratio", "--min") if k in a}))
    else:
        o = {k: a[a.index(k) + 1] for k in ("--out", "--ratio", "--min") if k in a}
        main(a[0], o.get("--out", "clips/live"), float(o.get("--ratio", 3)), int(o.get("--min", 15)))

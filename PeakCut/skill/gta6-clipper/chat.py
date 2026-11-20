#!/usr/bin/env python
"""Free chat-spike detector (replaces Eklipse/StreamLadder/Powder "hype moment" scoring).
  chat.py <url> [START-END]      START/END = seconds or H:MM:SS, e.g. 2:00:00-2:20:00
  chat.py selftest
  from chat import chat_tags     tags = chat_tags(url, "2:00:00-2:20:00")  -> {bin_start_sec: " [CHAT x4 KEKW,W]"}
                                 tag_for(tags, line_start, line_end) -> tag text for a transcript line
Twitch VOD: public GQL (VideoCommentsByOffsetOrCursor) paged by cursor, window only.
YouTube live replay: yt-dlp live_chat subtitle (yt-dlp has no window option, so the whole chat is fetched, then cut)."""
import json, math, re, statistics, subprocess, sys, tempfile, time, urllib.request
from collections import Counter
from pathlib import Path

BIN = 10
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"  # public web-player id, no login
HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"  # shortcut: persisted-query hash can rotate; re-read it from the twitch.tv network tab


def hms(s):
    n = 0
    for p in str(s).split(":"):
        n = n * 60 + float(p)
    return int(n)


def parse_win(w):
    if not w:
        return 0, None
    a, b = w.split("-")
    return hms(a), hms(b)


def fetch_twitch(vid, start, end, delay=0.15):
    """-> [(sec, text)] for start<=sec<end. One request at a time.
    The cursor path needs an integrity token, so page by offset instead: each reply is ~30 s of chat, next offset = last message second, dedupe by id."""
    out, seen, off = [], set(), start
    while off < end:
        q = [{"operationName": "VideoCommentsByOffsetOrCursor", "variables": {"videoID": vid, "contentOffsetSeconds": off},
              "extensions": {"persistedQuery": {"version": 1, "sha256Hash": HASH}}}]
        req = urllib.request.Request("https://gql.twitch.tv/gql", json.dumps(q).encode(),
                                     {"Client-ID": CLIENT_ID, "Content-Type": "application/json"})
        for t in range(4):  # retry with backoff
            try:
                c = json.load(urllib.request.urlopen(req, timeout=30))[0]["data"]["video"]["comments"]
                break
            except Exception:
                if t == 3:
                    raise
                time.sleep(2 ** t)
        new, last = 0, off
        for e in c["edges"]:
            n = e["node"]
            s = n["contentOffsetSeconds"]
            last = max(last, s)
            if n["id"] in seen or not start <= s < end:
                continue
            seen.add(n["id"])
            new += 1
            out.append((s, "".join(f["text"] or "" for f in n["message"]["fragments"]).strip()))
        off = last if new else off + 1  # shortcut: a page that is all one second repeats; skip ahead 1 s (loses a few msgs)
        time.sleep(delay)
    return out


def fetch_youtube(url, start, end):
    d = tempfile.mkdtemp(prefix="chat_")
    try:
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--skip-download", "--write-subs", "--sub-langs", "live_chat",
                            "--no-warnings", "-o", d + "/c.%(ext)s", url], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    except subprocess.TimeoutExpired:
        raise RuntimeError("chat download stalled for 300 s, stopped (yt-dlp live_chat); Short continues without chat")
    f = next(Path(d).glob("*live_chat.json"), None)
    if not f:
        raise RuntimeError("no live_chat for this video (not a live replay, or chat off): " + r.stderr[-300:])
    out = []
    for line in f.open(encoding="utf-8"):
        try:
            act = json.loads(line)["replayChatItemAction"]
            ms = int(act["videoOffsetTimeMsec"])
            for a in act["actions"]:
                m = a.get("addChatItemAction", {}).get("item", {}).get("liveChatTextMessageRenderer")
                if not m:
                    continue
                txt = "".join(x.get("text") or (x.get("emoji", {}).get("shortcuts") or [x.get("emoji", {}).get("emojiId", "")])[0]
                              for x in m["message"].get("runs", []))
                if ms / 1000 >= start and (end is None or ms / 1000 < end):
                    out.append((ms // 1000, txt.strip()))
        except (KeyError, ValueError):
            continue
    return out


def fetch(url, start=0, end=None):
    m = re.search(r"twitch\.tv/videos/(\d+)", url)
    if m:
        if end is None:
            raise SystemExit("Twitch VODs can be 35 h: give a START-END window")
        return fetch_twitch(m.group(1), start, end)
    return fetch_youtube(url, start, end)


HYPE = set("kekw lul lulw lol lmao lmfao omegalul icant omg wtf pog poggers pogchamp pogu w ww www l ll d: monkas widepeepohappy "
           "clipit letsgo nooo noooo no? holy bruh sheesh insane cinema cooked".split())  # words that mean "something just happened"


def top_phrases(msgs, n=3, min_count=3):
    """Most repeated hype words (KEKW, LMAO, W, L, OMG, CLIP IT ...) and short repeated messages. Each message counts a word once."""
    c, shown = Counter(), {}
    for t in msgs:
        w, hit = t.split(), set()
        for x in w:
            k = x.lower().strip(".,!?")
            if k in HYPE: hit.add(k)
        if "clip it" in t.lower() or "clip that" in t.lower(): hit.add("clip it")
        if w and len(t) <= 25 and len(w) <= 4:  # old rule: a short message repeated as a whole
            hit.add(w[0].lower() if len(set(x.lower() for x in w)) == 1 else t.lower())
        for k in hit:
            c[k] += 1
            shown.setdefault(k, next((x for x in w if x.lower().strip(".,!?") == k), t.upper() if k == "clip it" else k))
    return [(shown[k].upper() if shown[k].lower() in HYPE else shown[k], v) for k, v in c.most_common(n) if v >= min_count]


def find_spikes(msgs, start, end, pct=97, mult=1.6, min_msgs=5, local=True):
    """msgs [(sec,text)] -> ([(bin_start, count, ratio, top_phrases)] by time, global median).
    Spike = bin >= max(mult*median, pct-percentile, min_msgs). Median over ALL bins incl. empty ones (ratio = to that median).
    Huge chats (global median >= 20 msgs per bin, ~2/s) are busy all the time: a bin that is the top 2 % of ALL bins when measured against the
    median of the 5 minutes around it (>= 1.25x and beyond Poisson noise) also counts (ratio = to that LOCAL median). Slow chats stay strict."""
    nb = max(1, -(-(end - start) // BIN))
    counts, texts = [0] * nb, [[] for _ in range(nb)]
    for s, t in msgs:
        i = int((s - start) // BIN)
        if 0 <= i < nb:
            counts[i] += 1
            texts[i].append(t)
    med = statistics.median(counts)
    p = sorted(counts)[min(nb - 1, int(nb * pct / 100))]
    thr = max(mult * max(med, 1), p, min_msgs)
    hit = {i: c / max(med, 1) for i, c in enumerate(counts) if c >= thr}
    if local and med >= 20:
        loc = {}
        for i, c in enumerate(counts):
            lm = statistics.median(counts[max(0, i - 30):i] + counts[i + 1:i + 31] or [c])
            if i not in hit and c >= 1.25 * lm and c >= lm + 3 * lm ** .5: loc[i] = c / max(lm, 1)
        for i in sorted(loc, key=loc.get, reverse=True)[:max(3, nb // 50)]: hit[i] = loc[i]
    return [(start + i * BIN, counts[i], r, top_phrases(texts[i])) for i, r in sorted(hit.items())], med


def chat_tags(url, window=None, msgs=None):
    """-> {bin_start_second: ' [CHAT x4: "KEKW" x40, "W" x12]'}  (xN = times the normal message rate)"""
    start, end = parse_win(window)
    msgs = msgs if msgs is not None else fetch(url, start, end)
    end = end if end is not None else (max((s for s, _ in msgs), default=start) + 1)
    sp, _ = find_spikes(msgs, start, end)
    return {b: f" [CHAT x{r:.1f}" + (": " + ", ".join(f'"{p}" x{n}' for p, n in tp) if tp else "") + "]" for b, c, r, tp in sp}


def tag_for(tags, s, e):
    """Tag of the strongest spike bin overlapping transcript line [s,e], else ''."""
    hit = [tags[b] for b in range(int(s // BIN) * BIN, int(e) + 1, BIN) if b in tags]
    return max(hit, key=lambda t: float(re.search(r"x([\d.]+)", t).group(1))) if hit else ""


def fetch_clips(vid):
    """Viewer-made Twitch clips of VOD `vid` -> [dict(off, dur, views, title)]. GQL cannot list clips by VOD, so read the channel's top-100
    clips per period (week, month, all time) and keep those whose video is this VOD. Top 100 only: a huge channel loses the small clips."""
    def gql(q):
        r = urllib.request.Request("https://gql.twitch.tv/gql", json.dumps({"query": q}).encode(), {"Client-ID": CLIENT_ID, "Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(r, timeout=30))
    login = gql('{video(id:"%s"){owner{login}}}' % vid)["data"]["video"]["owner"]["login"]
    out = {}
    for per in ("LAST_DAY", "LAST_WEEK", "LAST_MONTH", "ALL_TIME"):
        try: es = gql('{user(login:"%s"){clips(first:100,criteria:{period:%s,sort:VIEWS_DESC}){edges{node{slug title viewCount videoOffsetSeconds durationSeconds video{id}}}}}}' % (login, per))["data"]["user"]["clips"]["edges"]
        except Exception: continue  # a period can answer "server error"; the others still work
        for e in es:
            n = e["node"]
            if n["video"] and n["video"]["id"] == vid and n["videoOffsetSeconds"] is not None:
                out[n["slug"]] = dict(off=n["videoOffsetSeconds"], dur=n["durationSeconds"], views=n["viewCount"], title=n["title"])
    return sorted(out.values(), key=lambda c: c["off"])


def clip_tag(clips, s, e):
    """' [CLIPPED x3 5.6k views]' when viewer clips cover transcript line [s,e]; x = how many clips, views summed. Clips are made right
    AFTER the moment, so a line counts when it lies inside the clip or up to 5 s before it starts."""
    h = [c for c in clips or [] if c["off"] - 5 < e and c["off"] + c["dur"] > s]
    if not h: return ""
    v = sum(c["views"] for c in h)
    return f" [CLIPPED x{len(h)} {v / 1000:.1f}k views]" if v >= 1000 else f" [CLIPPED x{len(h)} {v} views]"


def heat(line):
    """0-100 'how likely a moment' from the tags already on a transcript line. Noisy-or: every signal adds, none alone reaches 100.
    LOUD +dB (15 dB = 0.6), CHAT rate ratio (x4 = 0.7) + hype words, CLIPPED views (log: 100 views .35, 5k .75) and clip count, REPLAYED ratio."""
    P = []
    if m := re.search(r"\[LOUD \+(\d+)dB", line): P.append(min(1, int(m[1]) / 15) * .6)
    if m := re.search(r"\[CHAT x([\d.]+)", line): P.append(min(1, float(m[1]) / 6) * .8 + (.1 if '"' in line else 0))
    if m := re.search(r"\[CLIPPED x(\d+) ([\d.]+)(k?) views", line):
        v = float(m[2]) * (1000 if m[3] else 1); P.append(min(1, .35 + .15 * math.log10(max(v, 1) / 100) + .05 * (int(m[1]) - 1)) * .9)
    if m := re.search(r"\[REPLAYED x(\d+)", line): P.append(min(1, int(m[1]) / 8) * .8)
    if re.search(r"\[GAME (WASTED|BUSTED|MISSION FAILED|MISSION PASSED|HEIST PASSED|HEIST FAILED|VICTORY|ELIMINATED|GAME OVER|YOU DIED|DEFEAT|CHAMPION)", line): P.append(.6)  # a big game banner on screen (vision.py)
    elif "[GAME " in line: P.append(.2)
    q = 1
    for x in P: q *= 1 - max(0, min(x, .95))
    return round(100 * (1 - q))


def replay_tag(hm, s, e):
    """' [REPLAYED xN]' when YouTube's most-replayed graph (yt-dlp 'heatmap') peaks inside line [s,e]; N = times the median."""
    if not hm: return ""
    v = sorted(x["value"] for x in hm); med = max(v[len(v) // 2], 0.05)
    hit = [x["value"] for x in hm if x["start_time"] < e and x["end_time"] > s and x["value"] >= max(0.5, 3 * med)
           and x["start_time"] > 0.03 * hm[-1]["end_time"]]  # the first seconds are always replayed: not a moment
    return f" [REPLAYED x{max(hit) / med:.0f}]" if hit else ""


# ---------- LIVE chat (read-only, public endpoints, no keys): Twitch IRC, Kick Pusher websocket, YouTube innertube polling ----------
import socket, threading
UA = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9", "Cookie": "CONSENT=YES+1"}


def _get(u, data=None, hdr=None):
    return urllib.request.urlopen(urllib.request.Request(u, data, {**UA, **(hdr or {})}), timeout=20).read().decode("utf-8", "replace")


def live_source(s):
    """'kick:slug' / kick.com/slug / youtube link / @handle / 'youtube:...' / twitch link or name -> (kind, id)."""
    s = s.strip()
    if m := re.match(r"kick:(\w[\w-]*)$|.*kick\.com/(\w[\w-]*)", s): return "kick", (m[1] or m[2]).lower()
    if re.search(r"youtu|^(youtube|yt):|^@", s): return "youtube", re.sub(r"^(youtube|yt):", "", s)
    if m := re.search(r"twitch\.tv/(\w+)", s): return "twitch", m[1].lower()
    return "twitch", s.removeprefix("twitch:").lower()


def yt_video_id(x):
    """link (watch / live / youtu.be / @handle/live) or @handle or bare id -> 11-char video id of the live stream."""
    if "@" not in x and (m := re.search(r"(?:v=|youtu\.be/|/live/|^)([\w-]{11})(?:[&?/]|$)", x)): return m[1]
    h = x if x.startswith("http") else "https://www.youtube.com/" + x.lstrip("/")
    if "@" in h and not re.search(r"/(live|streams)$", h): h = h.rstrip("/") + "/live"
    page = _get(h)
    m = re.search(r'<link rel="canonical" href="https://www\.youtube\.com/watch\?v=([\w-]{11})', page) or re.search(r'"videoId":"([\w-]{11})"', page)
    if not m: raise RuntimeError("no live video found for " + x)
    return m[1]


def kick_chatroom(slug):
    d = json.loads(_get(f"https://kick.com/api/v2/channels/{slug}", hdr={"Accept": "application/json"}))
    return d["chatroom"]["id"], d["id"], bool(d.get("livestream"))


def kick_pusher_url(channel_id):
    """Kick hands out the Pusher app key + cluster per request (the key rotates), no login: web.kick.com realtime negotiation."""
    body = json.dumps({"client": {"id": "clipper", "type": "web"}, "capabilities": {"accepted_providers": [{"provider": "pusher"}]}}).encode()
    c = json.loads(_get(f"https://web.kick.com/api/v1/realtime/channels/{channel_id}/chat/connection", body, {"Content-Type": "application/json"}))["data"]["connections"][0]["credentials"]
    return f"wss://ws-{c['cluster']}.pusher.com/app/{c['app_key']}?protocol=7&client=js&version=8.4.0&flash=false"


def kick_msg(frame):
    """Pusher frame (dict) -> (nick, text) for a ChatMessageEvent, else None."""
    if "ChatMessageEvent" not in frame.get("event", ""): return None
    d = json.loads(frame["data"]) if isinstance(frame.get("data"), str) else frame.get("data", {})
    return d["sender"]["username"], d["content"]


def kick_chat(slug, on_msg, stop):
    from websockets.sync.client import connect
    rid, cid, _ = kick_chatroom(slug)
    while not stop.is_set():
        try:
            with connect(kick_pusher_url(cid), open_timeout=15, origin="https://kick.com") as ws:
                ws.send(json.dumps({"event": "pusher:subscribe", "data": {"auth": "", "channel": f"chatrooms.{rid}.v2"}}))
                while not stop.is_set():
                    try: raw = ws.recv(timeout=5)
                    except TimeoutError: continue
                    f = json.loads(raw)
                    if f.get("event") == "pusher:ping": ws.send(json.dumps({"event": "pusher:pong", "data": {}}))
                    elif (m := kick_msg(f)): on_msg(time.time(), *m)
        except Exception as e:
            if stop.is_set(): return
            print("kick chat reconnect:", str(e)[:80], flush=True); stop.wait(3)


def twitch_chat(chan, on_msg, stop):
    while not stop.is_set():
        try:
            s = socket.create_connection(("irc.chat.twitch.tv", 6667), timeout=5); s.sendall(f"NICK justinfan12345\r\nJOIN #{chan}\r\n".encode()); buf = ""
            while not stop.is_set():
                try: data = s.recv(8192)
                except socket.timeout: continue
                if not data: break
                buf += data.decode("utf-8", "replace"); *ls, buf = buf.split("\r\n")
                for l in ls:
                    if l.startswith("PING"): s.sendall(b"PONG :tmi.twitch.tv\r\n")
                    elif m := re.match(r":(\w+)!\S+ PRIVMSG #\S+ :(.*)", l): on_msg(time.time(), m[1], m[2])
            s.close()
        except OSError as e: print("chat reconnect:", e, flush=True); stop.wait(3)


def yt_actions(resp):
    """live_chat reply -> ([(nick, text)], next_continuation, wait_seconds)."""
    c = resp["continuationContents"]["liveChatContinuation"]; out = []
    for a in c.get("actions", []):
        m = a.get("addChatItemAction", {}).get("item", {}).get("liveChatTextMessageRenderer")
        if m: out.append((m["authorName"]["simpleText"], "".join(x.get("text") or (x.get("emoji", {}).get("shortcuts") or [""])[0] for x in m["message"].get("runs", [])).strip()))
    k = next((v for v in c.get("continuations", [{}])[0].values() if isinstance(v, dict) and "continuation" in v), None)
    return out, (k or {}).get("continuation"), (k or {}).get("timeoutMs", 3000) / 1000


def yt_init(vid):
    h = _get(f"https://www.youtube.com/watch?v={vid}")  # the live_chat popout page is an empty stub without a browser; the watch page carries the first continuation
    key, ver = re.search(r'"INNERTUBE_API_KEY":"([\w-]+)"', h), re.search(r'"INNERTUBE_CLIENT_VERSION":"([\d.]+)"', h)
    cont = re.search(r'"liveChatRenderer".*?"continuation":"([^"]+)"', h)
    if not (key and cont): raise RuntimeError("no live chat for " + vid + " (not live, or chat off)")
    return key[1], ver[1] if ver else "2.20240101.00.00", cont[1]


def youtube_chat(x, on_msg, stop):
    vid = yt_video_id(x); key, ver, cont = yt_init(vid); first = True
    while not stop.is_set() and cont:  # no dedupe needed: each continuation returns only new actions; the very first reply is old backlog, dropped
        wait = 5
        try:
            body = json.dumps({"context": {"client": {"clientName": "WEB", "clientVersion": ver}}, "continuation": cont}).encode()
            out, cont, wait = yt_actions(json.loads(_get(f"https://www.youtube.com/youtubei/v1/live_chat/get_live_chat?key={key}", body, {"Content-Type": "application/json"})))
            for n, t in out * (not first): on_msg(time.time(), n, t)
            first = False
        except Exception as e: print("youtube chat retry:", str(e)[:80], flush=True)
        stop.wait(min(max(wait, 1), 6))


def live_chat(src, on_msg, stop):
    """Blocking; src as live_source() accepts it. Calls on_msg(time, nick, text) until stop (threading.Event) is set."""
    k, i = live_source(src)
    {"kick": kick_chat, "twitch": twitch_chat, "youtube": youtube_chat}[k](i, on_msg, stop)


def measure(src, secs=60):
    """-> (messages, distinct chatters) read from a live chat for secs seconds."""
    got, stop, err = [], threading.Event(), []
    def run():
        try: live_chat(src, lambda *a: got.append(a), stop)
        except Exception as e: err.append(e)
    t = threading.Thread(target=run, daemon=True); t.start(); t.join(secs); stop.set()
    if err and not got: raise err[0]
    return len(got), len({n for _, n, _ in got})


def _live_ok(v):
    try: yt_init(v); return True
    except Exception: return False


def check(name, secs=10):
    """name = twitch | kick | youtube -> (ok, detail). Reads a big live channel's chat for ~secs s, read-only, no login."""
    try:
        if name == "twitch":
            import live
            src = "twitch:" + live.category_streams("Just Chatting", "en", 3)[1][0]["login"]
        elif name == "kick":
            src = "kick:" + json.loads(_get("https://kick.com/stream/livestreams/en?page=1&limit=5&sort=desc", hdr={"Accept": "application/json"}))["data"][0]["channel"]["slug"]
        elif name == "youtube":
            for v in [v for v in dict.fromkeys(re.findall(r'"videoId":"([\w-]{11})"', _get("https://www.youtube.com/results?search_query=live+chat&sp=EgJAAQ%253D%253D"))) if _live_ok(v)][:3]:
                n, u = measure("youtube:" + v, secs)  # a quiet stream can give 0 in 10 s: try up to 3
                if n: return True, f"youtube:{v}: {n} messages from {u} chatters in {secs} s"
            return False, "no live YouTube chat produced messages"
        else: return False, "unknown platform " + name
        n, u = measure(src, secs)
        return n > 0, f"{src}: {n} messages from {u} chatters in {secs} s"
    except Exception as e: return False, f"{type(e).__name__}: {str(e)[:120]}"


def selftest():
    hm = [dict(start_time=i * 10, end_time=i * 10 + 10, value=v) for i, v in enumerate([1.0] + [0.1] * 50 + [0.8] + [0.1] * 48)]
    assert replay_tag(hm, 512, 518) == " [REPLAYED x8]" and replay_tag(hm, 2, 8) == "" and replay_tag(hm, 100, 200) == "", replay_tag(hm, 512, 518)
    import random
    random.seed(1)
    msgs = [(s, "hello there") for s in range(0, 600) for _ in range(random.choice([0, 1, 1, 2]))]  # ~1 msg/s baseline
    msgs += [(300 + i % 10, "KEKW") for i in range(60)] + [(301, "KEKW KEKW KEKW")] + [(305, "W")] * 8  # spike in bin 300
    sp, med = find_spikes(msgs, 0, 600)
    assert [b for b, *_ in sp] == [300], sp
    assert sp[0][3][0][0] == "KEKW" and sp[0][3][0][1] == 61 and sp[0][2] >= 3, sp
    assert top_phrases(["a long sentence that should be ignored by the filter"] * 9) == []
    assert tag_for({300: " [CHAT x5 KEKW]"}, 296, 304) == " [CHAT x5 KEKW]" and tag_for({300: "x"}, 0, 10) == ""
    # words: hype words counted once per message, inside longer messages too
    assert top_phrases(["KEKW that was wild", "kekw", "KEKW KEKW KEKW", "clip it", "CLIP IT bro", "clip it now", "nothing special here at all"]) == [("KEKW", 3), ("CLIP IT", 3)], top_phrases(["KEKW that was wild", "kekw", "KEKW KEKW KEKW", "clip it", "CLIP IT bro", "clip it now"])
    # huge chat: 8 msgs/s all hour, one bin 1.5x busier than its 5 min neighbours: global rule (x1.6) misses it, local fallback tags it; a slow chat gets no fallback
    random.seed(2)
    big = [(s, "x") for s in range(0, 1200) for _ in range(random.choice([6, 8, 8, 10]))] + [(600 + i % 10, "LOL") for i in range(45)]
    assert find_spikes(big, 0, 1200, local=False)[0] == [] and [b for b, *_ in find_spikes(big, 0, 1200)[0]] == [600]
    slow = [(s, "x") for s in range(0, 1200)] + [(600 + i % 10, "x") for i in range(9)]
    assert find_spikes(slow, 0, 1200)[0] == find_spikes(slow, 0, 1200, local=False)[0]
    # clips + heat
    cl = [dict(off=100, dur=30, views=5632, title="a"), dict(off=110, dur=30, views=400, title="b")]
    assert clip_tag(cl, 120, 125) == " [CLIPPED x2 6.0k views]" and clip_tag(cl, 95, 98) == " [CLIPPED x1 5.6k views]" and clip_tag(cl, 200, 210) == "" and clip_tag([], 1, 2) == ""
    h0, h1, h2 = heat("hi"), heat("hi [LOUD +12dB]"), heat('hi [LOUD +12dB] [CHAT x4.0: "KEKW" x9] [CLIPPED x2 6.0k views]')
    assert h0 == 0 and 0 < h1 < 100 and h2 > h1 and h2 <= 100 and heat("x [CLIPPED x1 42 views]") < heat("x [CLIPPED x1 5.6k views]"), (h0, h1, h2)
    assert parse_win("2:00:00-2:20:00") == (7200, 8400)
    # live chat parsers, saved sample payloads
    fr = {"event": "App\Events\ChatMessageEvent", "channel": "chatrooms.668.v2", "data": json.dumps({"content": "LETS GO", "sender": {"username": "bob"}})}
    assert kick_msg(fr) == ("bob", "LETS GO") and kick_msg({"event": "pusher:pong", "data": "{}"}) is None
    yr = {"continuationContents": {"liveChatContinuation": {"continuations": [{"timedContinuationData": {"continuation": "NEXT", "timeoutMs": 4000}}], "actions": [
        {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {"authorName": {"simpleText": "amy"}, "message": {"runs": [{"text": "wow "}, {"emoji": {"shortcuts": [":fire:"]}}]}}}}}, {"other": 1}]}}}
    assert yt_actions(yr) == ([("amy", "wow :fire:")], "NEXT", 4.0) and yt_actions({"continuationContents": {"liveChatContinuation": {}}}) == ([], None, 3.0)
    assert [live_source(x) for x in ("kick:XQC", "https://kick.com/xqc", "https://youtu.be/abcdefghijk", "@GTA/live", "twitch:foo", "https://www.twitch.tv/Foo", "foo")] == [("kick", "xqc")] * 2 + [("youtube", "https://youtu.be/abcdefghijk"), ("youtube", "@GTA/live"), ("twitch", "foo"), ("twitch", "foo"), ("twitch", "foo")]
    assert yt_video_id("https://www.youtube.com/watch?v=abcdefghijk&t=3") == "abcdefghijk" and yt_video_id("https://youtu.be/abcdefghijk") == "abcdefghijk"
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
        sys.exit()
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    t0 = time.time()
    url, win = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    start, end = parse_win(win)
    msgs = fetch(url, start, end)
    end = end if end is not None else max((s for s, _ in msgs), default=start) + 1
    sp, med = find_spikes(msgs, start, end)
    print(f"{len(msgs)} msgs in {end - start}s ({len(msgs) / max(1, end - start):.1f}/s), median {med:.0f} per {BIN}s, fetched in {time.time() - t0:.0f}s, {len(sp)} spikes")
    for b, c, r, tp in sorted(sp, key=lambda x: -x[1])[:15]:
        print(f"  {b // 3600}:{b % 3600 // 60:02d}:{b % 60:02d}  {c:4d} msgs  x{r:.0f}  " + "  ".join(f"{p} x{n}" for p, n in tp))

"""How is the Short doing on YouTube, and why not better. Own keys only; read-only calls; stdlib only.
Level 1: YouTube Data API key (+ your channel id/@handle).  Level 2 (optional): your OAuth client JSON -> Analytics (retention, avg view %).
Mount in app.py: see handle(). Selftest: python ytstats.py --selftest"""
import json, re, sys, time, threading, statistics, secrets, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DATA, ANA = "https://www.googleapis.com/youtube/v3", "https://youtubeanalytics.googleapis.com/v2/reports"
SCOPES = "https://www.googleapis.com/auth/yt-analytics.readonly https://www.googleapis.com/auth/youtube.readonly"
VERBS = set("watch see look stop wait why how make build get find learn try use spot hear meet beat break catch clip drop expose reveal show survive steal crash escape ruin save take unlock leak hit run shoot play ask say tell think guess".split())
_get = None   # test hook: replaces the HTTP GET (url, headers) -> dict


def http(url, headers=None, data=None):
    if _get and data is None: return _get(url, headers or {})
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode() if data else None, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")[:300]
        try: msg = json.loads(msg)["error"]["message"]
        except Exception: pass
        raise RuntimeError(f"YouTube said {e.code}: {msg}")


def vid_id(s):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", s or "") or re.fullmatch(r"([\w-]{11})", (s or "").strip())
    return m.group(1) if m else None


def dur(iso):
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    h, mi, s = (int(x or 0) for x in m.groups()) if m else (0, 0, 0)
    return h * 3600 + mi * 60 + s


def data_api(path, key, **p): return http(f"{DATA}/{path}?" + urllib.parse.urlencode(dict(p, key=key)))


def norm(it):
    s, st = it["snippet"], it.get("statistics", {})
    return dict(id=it["id"], title=s["title"], published=s["publishedAt"], seconds=dur(it.get("contentDetails", {}).get("duration")),
                views=int(st.get("viewCount", 0)), likes=int(st.get("likeCount", 0)), comments=int(st.get("commentCount", 0)))


def videos(key, ids):
    out = []
    for i in range(0, len(ids), 50):
        out += [norm(x) for x in data_api("videos", key, part="snippet,statistics,contentDetails", id=",".join(ids[i:i + 50])).get("items", [])]
    return out


def channel_id(key, ch):
    if re.fullmatch(r"UC[\w-]{22}", ch or ""): return ch
    r = data_api("channels", key, part="id", forHandle=(ch or "").lstrip("@")).get("items")
    if not r: raise RuntimeError("channel not found: use the UC... id or the @handle")
    return r[0]["id"]


def latest_shorts(key, ch, n=20):
    up = "UU" + channel_id(key, ch)[2:]
    ids = [i["contentDetails"]["videoId"] for i in data_api("playlistItems", key, part="contentDetails", playlistId=up, maxResults=50).get("items", [])]
    return [v for v in videos(key, ids) if 0 < v["seconds"] <= 180][:n]


# ---------- OAuth (the user's own client, loopback flow) ----------
def tok_path(ws): return Path(ws) / "yt" / "token.json"


def check_client(c):  # the uploaded client JSON must point at Google only: the code and client_secret are posted to token_uri
    c = c.get("installed") or c.get("web") or c
    for k, host in (("auth_uri", "accounts.google.com"), ("token_uri", "oauth2.googleapis.com")):
        u = urllib.parse.urlparse(str(c.get(k) or ""))
        if u.scheme != "https" or u.hostname != host or u.port not in (None, 443) or u.username: raise ValueError(f"OAuth client JSON rejected: {k} must be https://{host}/...")
    return c
def oauth_start(ws, client_json):
    c = check_client(client_json); box = {}

    class H(BaseHTTPRequestHandler):
        def do_GET(s):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(s.path).query)
            if q.get("state", [""])[0] == box["state"] and q.get("code"):
                try:
                    t = http(c["token_uri"], data=dict(code=q["code"][0], client_id=c["client_id"], client_secret=c["client_secret"], redirect_uri=box["redir"], grant_type="authorization_code"))
                    t["at"] = time.time(); tok_path(ws).write_text(json.dumps(t)); msg = "Signed in. You can close this tab."
                except Exception as e: msg = "Sign-in failed: " + str(e)
            else: msg = "Wrong or missing code."
            s.send_response(200); s.send_header("Content-Type", "text/plain"); s.end_headers(); s.wfile.write(msg.encode())
            threading.Thread(target=srv.shutdown, daemon=True).start()

        def log_message(s, *a): pass
    srv = HTTPServer(("127.0.0.1", 0), H); box["state"] = secrets.token_urlsafe(16); box["redir"] = f"http://127.0.0.1:{srv.server_port}"
    threading.Thread(target=srv.serve_forever, daemon=True).start(); t = threading.Timer(300, srv.shutdown); t.daemon = True; t.start()
    return c["auth_uri"] + "?" + urllib.parse.urlencode(dict(client_id=c["client_id"], redirect_uri=box["redir"], response_type="code", scope=SCOPES, access_type="offline", prompt="consent", state=box["state"]))


def access_token(ws):
    tp, cp = tok_path(ws), Path(ws) / "yt" / "client.json"
    if not tp.exists() or not cp.exists(): return None
    t = json.loads(tp.read_text()); c = check_client(json.loads(cp.read_text()))
    if time.time() - t.get("at", 0) > t.get("expires_in", 0) - 60:
        if not t.get("refresh_token"): return None
        t.update(http(c["token_uri"], data=dict(client_id=c["client_id"], client_secret=c["client_secret"], refresh_token=t["refresh_token"], grant_type="refresh_token")), at=time.time())
        tp.write_text(json.dumps(t))
    return t["access_token"]


def analytics(tok, start, end, metrics, dims=None, filters=None, sort=None, n=None):
    p = dict(ids="channel==MINE", startDate=start, endDate=end, metrics=metrics)
    for k, v in (("dimensions", dims), ("filters", filters), ("sort", sort), ("maxResults", n)):
        if v: p[k] = v
    r = http(ANA + "?" + urllib.parse.urlencode(p), {"Authorization": "Bearer " + tok})
    cols = [h["name"] for h in r.get("columnHeaders", [])]
    return [dict(zip(cols, row)) for row in r.get("rows", [])]


def deep(ws, v):
    """Analytics for one video + channel median of the top 20 videos (90 d); {} when not signed in or no data yet."""
    tok = access_token(ws)
    if not tok: return {}
    s, e = v["published"][:10], datetime.now(timezone.utc).strftime("%Y-%m-%d"); out = {}; M = "views,averageViewDuration,averageViewPercentage,subscribersGained"
    try:
        r = analytics(tok, s, e, M, filters="video==" + v["id"])
        if r: out.update(r[0])
        rt = analytics(tok, s, e, "audienceWatchRatio", dims="elapsedVideoTimeRatio", filters="video==" + v["id"])
        if rt and v["seconds"]: out["kept_at_3s"] = min(rt, key=lambda x: abs(x["elapsedVideoTimeRatio"] - 3 / v["seconds"]))["audienceWatchRatio"]
        st = datetime.fromordinal(datetime.now(timezone.utc).date().toordinal() - 90).strftime("%Y-%m-%d")
        rows = analytics(tok, st, e, M, dims="video", sort="-views", n=20)
        if rows: out["median"] = {k: statistics.median(x[k] for x in rows) for k in ("averageViewDuration", "averageViewPercentage", "subscribersGained")}
    except Exception as ex: out["error"] = str(ex)[:200]
    return out


# ---------- diagnosis (rules, no AI) ----------
def age_h(v): return max(1, (datetime.now(timezone.utc) - datetime.fromisoformat(v["published"].replace("Z", "+00:00"))).total_seconds() / 3600)


def diagnose(v, others=(), deep_=None, hook=""):
    deep_ = deep_ or {}; R = []; peers = [o for o in others if o["id"] != v["id"]]
    mv = statistics.median(o["views"] for o in peers) if peers else None; a = age_h(v)
    if mv is not None and a >= 24 and v["views"] < 0.5 * mv:
        R.append(dict(key="views", level="bad", text=f"{v['views']} views after {a:.0f} h; your last {len(peers)} Shorts have a median of {mv:.0f}. Low views this early usually means the first second did not earn the swipe-stay."))
    elif a < 24: R.append(dict(key="views", level="info", text=f"Only {a:.0f} h old ({v['views']} views): judge it at 24 h and 48 h."))
    if v["views"] >= 50 and v["likes"] / v["views"] < 0.02: R.append(dict(key="likes", level="warn", text=f"Like rate {100 * v['likes'] / v['views']:.1f}% (typical Shorts 3-6%): viewers do not feel enough to react; end on the strongest beat."))
    avp, med = deep_.get("averageViewPercentage"), (deep_.get("median") or {}).get("averageViewPercentage")
    if avp is not None and (avp < 60 or (med and avp < 0.8 * med)):
        R.append(dict(key="avp", level="bad", text=f"Average viewed {avp:.0f}%" + (f" vs your median {med:.0f}%" if med else "") + ": people leave before the end. Cut the slow middle or shorten the clip."))
    k3 = deep_.get("kept_at_3s")
    if k3 is not None and k3 < 0.7: R.append(dict(key="hook3", level="bad", text=f"Only {100 * k3:.0f}% are still watching at second 3: the hook is not working. Start on the peak line or action."))
    t = v["title"]; w = re.findall(r"[\w']+", t)
    if len(t) > 60: R.append(dict(key="title_len", level="warn", text=f"Title is {len(t)} characters (over about 60): the end is cut in the feed."))
    if w and w[0].lower() not in VERBS: R.append(dict(key="verb", level="warn", text="Title does not start with a verb: try an action opener (Watch, Spot, Stop...)."))
    if not any(x[0].isupper() or x.isdigit() for x in w[1:]): R.append(dict(key="name", level="warn", text="No name or number in the title (GTA 6, Rockstar, the streamer): searchers have nothing to match."))
    hw, tw = {x.lower() for x in re.findall(r"\w{4,}", hook)}, {x.lower() for x in re.findall(r"\w{4,}", t)}
    if hw and tw and len(hw & tw) / len(hw | tw) < 0.1: R.append(dict(key="mismatch", level="warn", text="Title and on-screen hook say different things: pick one promise and use it in both."))
    if len(peers) >= 6:
        hr = lambda o: datetime.fromisoformat(o["published"].replace("Z", "+00:00")).hour
        hs = {hr(o) for o in sorted(peers, key=lambda o: -o["views"])[:len(peers) // 3]}
        if hr(v) not in hs: R.append(dict(key="hour", level="info", text=f"Posted at {hr(v):02d}:00 UTC; your best Shorts went out around {', '.join(f'{h:02d}:00' for h in sorted(hs))} UTC."))
    return dict(reasons=R, median_views=mv, age_hours=round(a, 1), level="analytics" if deep_.get("averageViewPercentage") is not None else "key")


def ai_prompt(v, reasons, transcript="", hook=""):
    """Prompt for any AI provider; the answer is JSON {titles[5], hooks[3], recut}."""
    return ("A YouTube Short about GTA 6 is underperforming. Give fresh packaging.\n"
            f"Current title: {v['title']}\nOn-screen hook: {hook or '(none)'}\nStats: {v['views']} views, {v['likes']} likes, {v['comments']} comments, {v['seconds']} s, posted {v['published']}\n"
            "Problems found:\n" + "\n".join("- " + r["text"] for r in reasons) + f"\nTranscript of the Short:\n{transcript[:3000]}\n\n"
            'Return ONLY JSON: {"titles":[5 titles, each starts with a verb, under 60 chars, contains a name or number],"hooks":[3 on-screen hooks, max 7 words, same promise the clip delivers],"recut":"one idea to re-cut the clip (where to start, what to remove)"}')


# ---------- app glue ----------
def _links(ws):
    p = Path(ws) / "yt" / "links.json"
    return p, (json.loads(p.read_text()) if p.exists() else {})


def _title(ws, run, file):
    t = (Path(ws) / run / file).with_suffix(".txt")
    return (t.read_text(encoding="utf-8").strip().splitlines() or [""])[0].strip() if t.exists() else ""


def _find(ws, key, q, e):
    p, L = _links(ws); k = f"{q['run']}|{q['file']}"
    if k in L: return L[k]
    ttl = _title(ws, q["run"], q["file"]).lower()
    if ttl and e.get("YT_CHANNEL"):
        for o in latest_shorts(key, e["YT_CHANNEL"], 20):  # auto-match: identical title among the latest uploads
            if o["title"].strip().lower() == ttl: L[k] = o["id"]; p.write_text(json.dumps(L)); return o["id"]


def handle(method, path, query, body, ws, env, save_env):
    """-> (status, json). Mount in app.py (inside do_GET and do_POST, after the token check, where p = the path):
        import ytstats                                   # top of file
        if p.startswith("/api/yt/"): return self.send(*ytstats.handle(METHOD, p, q, d, WS, env(), save_env))   # GET: METHOD="GET", d=None; POST: "POST"
    """
    e = dict(env); Path(ws, "yt").mkdir(parents=True, exist_ok=True); key = e.get("YT_API_KEY"); P = path.rstrip("/"); q = query or {}; b = body or {}
    try:
        if P == "/api/yt/status" and method == "GET":
            return 200, dict(has_key=bool(key), channel=e.get("YT_CHANNEL", ""), has_oauth=(Path(ws) / "yt" / "client.json").exists(), signed_in=tok_path(ws).exists(), network="Calls go to Google APIs (googleapis.com) only when a key is saved or a linked Short is opened; nothing else leaves this PC.")
        if P == "/api/yt/key" and method == "POST":
            k = (b.get("api_key") or "").strip()
            if k:
                if not re.fullmatch(r"[\w-]{20,60}", k): return 400, dict(error="That does not look like an API key.")
                data_api("videos", k, part="id", id="dQw4w9WgXcQ"); e["YT_API_KEY"] = k  # cheap read-only check
            if b.get("channel") is not None: e["YT_CHANNEL"] = b["channel"].strip()
            save_env(e); return 200, dict(ok=True)
        if P == "/api/yt/oauth" and method == "POST":
            cj = b.get("client_json"); cj = json.loads(cj) if isinstance(cj, str) else cj
            if not (cj and (cj.get("installed") or cj.get("web"))): return 400, dict(error="Upload the OAuth client JSON (Desktop app) from Google Cloud.")
            try: check_client(cj)
            except ValueError as x: return 400, dict(error=str(x))
            Path(ws, "yt", "client.json").parent.mkdir(parents=True, exist_ok=True); Path(ws, "yt", "client.json").write_text(json.dumps(cj)); return 200, dict(ok=True, auth_url=oauth_start(ws, cj))
        if not key: return 400, dict(error="Add your YouTube API key in Settings first.")
        if P == "/api/yt/link" and method == "POST":
            i = vid_id(b.get("url"))
            if not i: return 400, dict(error="Paste a YouTube link or the 11-character video id.")
            p, L = _links(ws); L[f"{b['run']}|{b['file']}"] = i; p.write_text(json.dumps(L)); return 200, dict(ok=True, id=i)
        if P == "/api/yt/channel":
            if not e.get("YT_CHANNEL"): return 400, dict(error="Add your channel (@handle or UC id) in Settings.")
            return 200, dict(shorts=latest_shorts(key, e["YT_CHANNEL"], 20))
        if P in ("/api/yt/stats", "/api/yt/diagnose") and method == "GET":
            vi = _find(ws, key, q, e)
            if not vi: return 200, dict(linked=False, hint="Paste the YouTube link of this Short.")
            v = (videos(key, [vi]) or [None])[0]
            if not v: return 404, dict(error="Video not found (private or deleted?).")
            others = latest_shorts(key, e["YT_CHANNEL"], 20) if e.get("YT_CHANNEL") else []
            d = deep(ws, v); med = dict(d.get("median") or {})
            if others: med.update(views=statistics.median(o["views"] for o in others), likes=statistics.median(o["likes"] for o in others))
            out = dict(linked=True, id=vi, video=v, median=med, analytics={k: x for k, x in d.items() if k != "median"}, title=_title(ws, q["run"], q["file"]))
            if P.endswith("diagnose"): out.update(diagnose(v, others, d, hook=q.get("hook", "")))
            return 200, out
    except RuntimeError as ex: return 502, dict(error=str(ex))
    except (KeyError, ValueError) as ex: return 400, dict(error=f"bad request: {ex}")
    return 404, dict(error="unknown route")


def selftest():
    global _get
    import tempfile
    now = time.time(); iso = lambda h: datetime.fromtimestamp(now - h * 3600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mk = lambda i, t, views, h: dict(id=i, snippet=dict(title=t, publishedAt=iso(h)), statistics=dict(viewCount=str(views), likeCount=str(views // 100), commentCount="1"), contentDetails=dict(duration="PT30S"))
    items = [mk("abcdefghij" + str(i), "Watch Rockstar 6 trailer", 1000 + i * 10, 100 + i * 24) for i in range(8)]
    items.append(mk("zzzzzzzzzzz", "gta 6 some very long title about nothing in particular here, honestly", 20, 30))

    def fake(url, h):
        if "playlistItems" in url: return dict(items=[dict(contentDetails=dict(videoId=x["id"])) for x in items])
        if "/channels" in url: return dict(items=[dict(id="UC" + "a" * 22)])
        if "youtubeanalytics" in url:
            if "elapsedVideoTimeRatio" in url: return dict(columnHeaders=[dict(name="elapsedVideoTimeRatio"), dict(name="audienceWatchRatio")], rows=[[0.1, 0.5], [0.5, 0.4]])
            if "dimensions=video" in url: return dict(columnHeaders=[dict(name=n) for n in "video views averageViewDuration averageViewPercentage subscribersGained".split()], rows=[["x", 100, 15, 70, 2]])
            return dict(columnHeaders=[dict(name=n) for n in "views averageViewDuration averageViewPercentage subscribersGained".split()], rows=[[100, 8, 40, 1]])
        ids = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("id", [""])[0].split(",")
        return dict(items=[x for x in items if x["id"] in ids])
    _get = fake; ws = tempfile.mkdtemp(); st = {"YT_CHANNEL": "@x"}

    def sv(d): st.clear(); st.update(d)
    assert vid_id("https://youtube.com/shorts/zzzzzzzzzzz?x=1") == "zzzzzzzzzzz" and dur("PT1M5S") == 65
    assert handle("POST", "/api/yt/key", {}, {"api_key": "A" * 39}, ws, st, sv)[0] == 200 and st["YT_API_KEY"] == "A" * 39
    assert handle("GET", "/api/yt/diagnose", dict(run="r", file="01_a.mp4"), None, ws, st, sv)[1]["linked"] is False
    assert handle("POST", "/api/yt/link", {}, {"run": "r", "file": "01_a.mp4", "url": "https://youtu.be/zzzzzzzzzzz"}, ws, st, sv)[0] == 200
    s, d = handle("GET", "/api/yt/diagnose", dict(run="r", file="01_a.mp4"), None, ws, st, sv)
    keys = {r["key"] for r in d["reasons"]}; assert s == 200 and {"views", "title_len", "verb", "hour"} <= keys and d["level"] == "key", d
    Path(ws, "yt", "client.json").write_text(json.dumps({"installed": dict(client_id="i", client_secret="s", auth_uri="https://accounts.google.com/o/oauth2/auth", token_uri="https://oauth2.googleapis.com/token")})); tok_path(ws).write_text(json.dumps(dict(access_token="t", at=time.time(), expires_in=3600)))
    d = handle("GET", "/api/yt/diagnose", dict(run="r", file="01_a.mp4"), None, ws, st, sv)[1]
    assert {"avp", "hook3"} <= {r["key"] for r in d["reasons"]} and d["level"] == "analytics" and d["median"]["averageViewPercentage"] == 70, d
    assert "5 titles" in ai_prompt(d["video"], d["reasons"]) and len(handle("GET", "/api/yt/channel", {}, None, ws, st, sv)[1]["shorts"]) == 9
    good = {"installed": dict(client_id="i", client_secret="s", auth_uri="https://accounts.google.com/o/oauth2/auth", token_uri="https://oauth2.googleapis.com/token")}
    assert check_client(good)["client_id"] == "i"
    for bad in ("https://evil.example/token", "http://oauth2.googleapis.com/token", "https://oauth2.googleapis.com.evil.io/t", "https://x@evil.io/t"):
        try: check_client({"installed": dict(good["installed"], token_uri=bad)}); raise SystemExit("accepted " + bad)
        except ValueError: pass
    r = handle("POST", "/api/yt/oauth", {}, {"client_json": {"installed": dict(good["installed"], auth_uri="https://evil.io/a")}}, ws, st, sv); assert r[0] == 400 and "rejected" in r[1]["error"], r
    assert "Google" in handle("GET", "/api/yt/status", {}, None, ws, st, sv)[1]["network"]
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else:
        import os
        k, v = os.environ.get("YT_API_KEY"), (vid_id(sys.argv[1]) if len(sys.argv) > 1 else None)
        print(json.dumps(videos(k, [v]), indent=1) if k and v else "usage: YT_API_KEY=... python ytstats.py <video url>")

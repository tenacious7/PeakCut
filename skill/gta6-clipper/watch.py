# Source watcher: checks each channel's free RSS feed (no API key), finds new uploads matching a keyword,
# downloads only their audio and transcribes it. Claude then reads clips/src/<id>.txt and picks the moments.
# Twitch launch day: python watch.py twitch:"Grand Theft Auto VI" [N] [--window 2h|90m|H:MM:SS-H:MM:SS]  = top live streams (prints the live.py command) + last-24h VODs of the top channels, transcribed
#        channels.txt line 'twitch: Grand Theft Auto VI' does the same inside the normal run (3 VODs per line).
# Usage: python watch.py [TITLE_REGEX='GTA ?(6|VI)'] [MAX_NEW=3]      (run it on a schedule, or let the skill run it)
#        python watch.py <link or file>   one video from any site yt-dlp knows (YouTube, Twitch VODs, Kick) or a local file
import json, os, re, subprocess, sys, threading, urllib.parse, urllib.request
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # titles carry emoji (Windows console is cp1252); each line shows at once, also in a log file
from pathlib import Path
from live import category_streams, recent_vods, vod_window, top_channel
from chat import chat_tags, tag_for, replay_tag, fetch_clips, clip_tag, heat
CK = ["--cookies-from-browser", os.environ["CLIPPER_BROWSER"]] if os.environ.get("CLIPPER_BROWSER") else []  # set CLIPPER_BROWSER=firefox (or edge, chrome) for videos that need a YouTube login
import datetime
T_META, T_AUDIO = int(os.environ.get("CLIPPER_META_TIMEOUT", 120)), int(os.environ.get("CLIPPER_DL_TIMEOUT", 1200))  # seconds: info lookups / audio download
def run_t(cmd, timeout, what="command", **k):
    """subprocess.run with a hard timeout: a stalled download prints one clear line and returns rc 124 (stderr = the reason) instead of freezing the run."""
    try: return subprocess.run(cmd, timeout=timeout, **k)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {what} stalled for {timeout} s, stopped", flush=True); return subprocess.CompletedProcess(cmd, 124, "", f"timeout after {timeout} s: {what}")
LANG = os.environ.get("CLIPPER_LANG", "en").lower()  # language of the videos you want; others are skipped from the title alone
_W = {"fr": "je tu il elle les des une est pour avec sur dans quand vous nous ce cette au aux du réagit", "es": "el los las una para con por que cuando mi su pero esto reacciono",
      "pt": "uma não você para com por mais isso meu minha quando reagindo"}
_W = {k: set(v.split()) for k, v in _W.items()}
def title_lang(t):
    """Cheap language guess from a title: 'en' unless accented/non-Latin letters or >=2 common FR/ES/PT words say otherwise."""
    w = re.findall(r"[^\W\d_]+", t.lower())
    hits = {k: sum(x in ws for x in w) for k, ws in _W.items()}; k = max(hits, key=hits.get)
    if hits[k] >= 2: return k
    return "other" if any(c.isalpha() and ord(c) > 0x24F for c in t) else ("fr" if any(c in "éèêàçùôâîûœ" for c in t.lower()) else "en")  # one accent = French guess; shortcut: ñ/ã titles read as French, pass CLIPPER_LANG=any
def skip_reason(v, today=None):
    """Why NOT to transcribe a found video (from metadata only), else ''. v = dict(title, date YYYY-MM-DD|YYYYMMDD|'', duration s|None, lang|None, search=True for search hits)."""
    t, today = v["title"], today or datetime.date.today()
    lg = (v.get("lang") or title_lang(t))[:2]
    if LANG != "any" and lg != LANG: return f"language {lg}, wanted {LANG}"
    if (v.get("duration") or 0) > 7200: return f"over 2 h ({v['duration'] / 3600:.1f} h)"
    if v.get("search"):
        d = re.sub(r"\D", "", v.get("date") or "")[:8]
        if len(d) == 8 and (today - datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))).days > 30: return "older than 30 days"
        if not d and (y := re.search(r"\b(20[0-2]\d)\b", t)) and int(y[1]) < today.year: return f"old year {y[1]} in title"
    return ""
if sys.argv[1:] == ["selftest"]:
    T = datetime.date(2026, 11, 17); S = lambda **k: skip_reason({"date": "", "duration": None} | k, T)
    assert title_lang("JE REACT AU GAMEPLAY DE GTA 6 😱") == "fr" and title_lang("Kai Cenat Reacts To GTA 6") == "en" and title_lang("Fat Giorno JDG complètement BLUFFÉ") == "fr"
    assert S(title="JE REACT AU GAMEPLAY DE GTA 6").startswith("language fr") and S(title="Kai plays GTA 6") == ""

    assert S(title="Kai plays GTA 6", duration=36000).startswith("over 2 h")
    assert S(title="GTA 6 live", search=True, date="2026-09-01").startswith("older") and S(title="GTA 6 live", date="2026-09-01") == "" and S(title="GTA 6 live", search=True, date="2026-11-10") == ""
    assert S(title="GTA 6 Funny Moments 2023 Compilation", search=True).startswith("old year")
    LANG = "fr"; assert S(title="JE REACT AU GAMEPLAY DE GTA 6") == "" and S(title="Kai plays GTA 6").startswith("language en")
    r = run_t([sys.executable, "-c", "import time; time.sleep(30)"], 1, "test", capture_output=True, text=True); assert r.returncode == 124 and "timeout" in r.stderr  # a stalled download is stopped
    sp = lambda a: a[7:].strip().strip(chr(34) + chr(39)); assert sp('twitch:"Grand Theft Auto VI"') == "Grand Theft Auto VI" and sp("twitch: Grand Theft Auto VI") == "Grand Theft Auto VI"
    print("selftest ok"); sys.exit()
def twitch_sources(spec, n, w=None, lang=None):
    """Live list (printed, with the live.py command for the top 3) + the n best VODs from the last 24 h: channels of the category, most viewers first, one VOD each, last `w` of it."""
    cat, ss = category_streams(spec, lang or LANG, 40)
    print(f"twitch {cat}: {len(ss)} live ({lang or LANG})"); [print(f"  live {x['viewers']:>7,}  {x['login']:<18}{x['lang']}  {x['title'][:60]}") for x in ss[:10]]
    for x in ss[:3]: print(f"  clip it live: python live.py https://www.twitch.tv/{x['login']}   (or fastest chat: python live.py top:{chr(34)}{spec}{chr(34)})")
    order = {x["login"]: i for i, x in enumerate(ss)}
    vods = sorted(recent_vods([x["login"] for x in ss[:12]]), key=lambda v: order[v["login"]])[:n]
    for v in vods: print(f"  VOD {v['login']} {v['length'] / 3600:.1f} h, {v['age_h']:.0f} h ago | {v['title'][:60]}")
    return [dict(id=f"tw{v['id']}_" + "".join(c for c in vod_window(v["length"], w) if c.isalnum()), title=v["title"], date="", channel=v["login"], views=ss[order[v["login"]]]["viewers"],
                 url="https://www.twitch.tv/videos/" + v["id"], win=vod_window(v["length"], w), duration=v["length"], twitch=True) for v in vods]
def _arg(k): return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
HERE = Path(__file__).resolve().parent; SRC = Path.cwd() / "clips" / "src"; SRC.mkdir(parents=True, exist_ok=True)  # results go where the user works, not into the skill
SEEN = SRC / "seen.json"; PY = sys.executable
tw_spec = sys.argv[1][7:].strip().strip(chr(34) + chr(39)) if sys.argv[1:2] and sys.argv[1].startswith("twitch:") else None
link = sys.argv[1] if len(sys.argv) > 1 and not tw_spec and (sys.argv[1].startswith("http") or Path(sys.argv[1]).is_file()) else None
key = re.compile(sys.argv[1] if len(sys.argv) > 1 and not link and not tw_spec else r"GTA ?(6|VI)\b", re.I)  # title filter (regex)
max_new = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() and not link else 3

def feed(cid):
    xml = urllib.request.urlopen(f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}", timeout=20).read().decode("utf-8")
    return [dict(id=i, title=t, date=d[:10]) for i, t, d in
            re.findall(r"<yt:videoId>(.+?)</yt:videoId>.*?<title>(.+?)</title>.*?<published>(.+?)</published>", xml, re.S)]

def search(q):
    """YouTube's own results page, newest first (sp=CAI%3D), read by yt-dlp: no API key. Long videos only, biggest first."""
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q) + "&sp=CAI%3D"
    out = run_t([PY, "-m", "yt_dlp", *CK, "--flat-playlist", "-I", "1:40", "-J", url], T_META, "YouTube search", capture_output=True, text=True, encoding="utf-8")
    es = [e for e in (json.loads(out.stdout or "null") or {}).get("entries", []) if (e.get("duration") or 0) > 180]
    return [dict(id=e["id"], title=e["title"], date=e.get("upload_date") or "", channel=e.get("channel", ""), views=e.get("view_count") or 0, duration=e.get("duration"), lang=e.get("language"), search=True)
            for e in sorted(es, key=lambda e: -(e.get("view_count") or 0))]

FAILS = SRC / "fails.json"; fails = json.loads(FAILS.read_text()) if FAILS.exists() else {}
seen = set(json.loads(SEEN.read_text())) if SEEN.exists() else set(); rss, found, tw = [], [], []
if tw_spec: tw += twitch_sources(tw_spec, max_new, _arg("--window"))
for line in ([] if link or tw_spec else (HERE / "channels.txt").read_text(encoding="utf-8").splitlines()):
    cid = line.split("#")[0].strip()
    if not cid: continue
    if cid.startswith("twitch:"):
        try: tw += twitch_sources(cid[7:].strip(), 3)
        except Exception as e: print("twitch failed", cid, e)
        continue
    if cid.startswith("search:"):
        try: found += search(cid[7:].strip())
        except Exception as e: print("search failed", cid, e)
        continue
    try: vids = feed(cid)
    except Exception as e: print("feed failed", cid, e); continue
    rss += [v | dict(channel=line.split("#")[-1].strip()) for v in vids]
rss.sort(key=lambda v: v["date"], reverse=True)
new, ids = [], set()
for v in rss + found:  # named channels first (newest), then search hits (most viewed)
    if key.search(v["title"]) and v["id"] not in seen | ids:
        if why := skip_reason(v): print("skip:", why, v["title"]); continue  # no audio download, no Whisper time
        new.append(v); ids.add(v["id"])
new += [v for v in tw if v["id"] not in seen]
if link:  # one pasted link or file: ask yt-dlp for its id and title (works for Twitch/Kick too)
    j = {} if Path(link).is_file() else json.loads(run_t([PY, "-m", "yt_dlp", *CK, "-J", "--no-warnings", link], T_META, "video info", capture_output=True, text=True, encoding="utf-8").stdout or "{}")
    new = [dict(id=re.sub(r"\W", "_", j.get("id") or Path(link).stem), title=j.get("title") or Path(link).stem, date=(j.get("upload_date") or "")[:8],
                channel=j.get("uploader") or j.get("channel") or "", views=j.get("view_count") or 0, url=link)]; max_new = 1
    win = sys.argv[2] if len(sys.argv) > 2 else ("0-3:00:00" if (j.get("duration") or 0) > 4 * 3600 else None)  # a 40-hour VOD would take an hour to fetch
    if win: new[0]["id"] += "_" + re.sub(r"\W", "", win); print("  window", win, "of", f"{(j.get('duration') or 0) / 3600:.1f} h")
else: win = None
sec = lambda t: sum(float(x) * 60 ** i for i, x in enumerate(reversed(t.split(":"))))
done_n = 0  # only real transcripts count toward the number asked for: Shorts and not-yet-aired streams are skipped for free
for v in new:
    if done_n >= max_new: break
    url = v.get("url") or f"https://www.youtube.com/watch?v={v['id']}"
    a = Path(url) if Path(url).is_file() else SRC / f"{v['id']}.m4a"  # a local file is transcribed as it is: nothing to download
    print("new:", v["date"] or f"{v['views']:,} {'viewers' if v.get('twitch') else 'views'}", v["channel"], "|", v["title"])
    # audio only: small and fast; the video is fetched later, only for the moments Claude picks (clip.py with the URL)
    r = subprocess.CompletedProcess([], 0) if a.exists() else run_t([PY, "-m", "yt_dlp", *CK, "-q", "-f", "ba[ext=m4a]/audio_only/ba", "--match-filter", "duration>180" if link else "duration>180 & duration<=7200", *(["--download-sections", "*" + (v.get("win") or win)] if v.get("win") or win else []), "-o", str(a), url], T_AUDIO, "audio download " + v["id"], capture_output=True, text=True, encoding="utf-8", errors="replace")  # resume: audio already fetched
    if r.returncode and "confirm your age" in (r.stderr or ""):
        print("  skipped: age-restricted (needs a login)"); seen.add(v["id"]); SEEN.write_text(json.dumps(sorted(seen))); continue
    if r.returncode and re.search(r"live event will begin|Premieres in|is_upcoming", r.stderr or ""):
        print("  not aired yet: picked up once it has started"); continue  # never a failure: launch streams must not be given up on before they start
    if r.returncode:
        fails[v["id"]] = fails.get(v["id"], 0) + 1; FAILS.write_text(json.dumps(fails))
        print("  download failed, try later:", (r.stderr or "").strip()[-160:])
        if fails[v["id"]] >= 3: print("  failed 3 times, giving up on it"); seen.add(v["id"]); SEEN.write_text(json.dumps(sorted(seen)))
        continue
    if a.exists():
        chat = {}  # stream replay: fetch chat while Whisper runs; spikes show where viewers lost it
        live = v.get("twitch") or (link and (j.get("was_live") or "twitch.tv" in url))
        def sig():  # chat spikes + viewer-made clips (Twitch only; a failure of one must not lose the other)
            try: chat.update(t=chat_tags(url, v.get("win") or win or f"0-{int(j.get('duration') or 0)}"))
            except Exception as e: print("  chat skipped:", str(e)[:100])
            m = re.search(r"twitch\.tv/videos/(\d+)", url)
            try: chat.update(c=fetch_clips(m[1]) if m else [])
            except Exception as e: print("  clips skipped:", str(e)[:100])
        if live: th = threading.Thread(target=sig, daemon=True); th.start()
        env = os.environ | {"TRANSCRIBE_OFFSET": str(sec((v.get("win") or win).split("-")[0])) if v.get("win") or win else "0"}  # transcript times = times in the full VOD
        if subprocess.run([PY, str(HERE / "transcribe.py"), str(a), str(SRC / f"{v['id']}.json")], env=env).returncode:
            print("  transcript failed, will retry next run"); continue  # not marked seen
        if live: th.join()
        hm = []  # YouTube's "most replayed" graph: where real viewers rewatched
        if "youtube.com" in url or "youtu.be" in url:
            try: hm = (j if link else json.loads(run_t([PY, "-m", "yt_dlp", *CK, "-J", "--no-warnings", url], T_META, "replay graph", capture_output=True, text=True, encoding="utf-8").stdout or "{}")).get("heatmap") or []
            except Exception as e: print("  replay graph skipped:", str(e)[:100])
        if chat.get("t") or chat.get("c") or hm:
            txt = SRC / f"{v['id']}.txt"
            try:
                def tagged(l):
                    s, e = map(float, re.match(r"\[([\d.]+)-([\d.]+)\]", l).groups())
                    l += tag_for(chat.get("t") or {}, s, e) + clip_tag(chat.get("c"), s, e) + replay_tag(hm, s, e)
                    return l + (f" [HEAT {h}]" if (h := heat(l)) else "")  # 0-100 from all tags: sort by it
                txt.write_text(chr(10).join(tagged(l) for l in txt.read_text(encoding="utf-8").splitlines()), encoding="utf-8")
                print("  chat:", len(chat.get("t") or {}), "spikes; clips:", len(chat.get("c") or []), "; replay graph:", "yes" if hm else "none")
            except Exception as e: print("  tags skipped:", str(e)[:100])
        (SRC / f"{v['id']}.meta.json").write_text(json.dumps(v | dict(url=url), ensure_ascii=False), encoding="utf-8"); done_n += 1
    else: print("  skipped: under 3 min (a Short) or over 2 h")
    seen.add(v["id"]); SEEN.write_text(json.dumps(sorted(seen)))  # only after the transcript exists, so a crash retries
print(len(new), "new matching uploads,", done_n, "transcribed -> clips/src/*.txt")

---
name: gta6-clipper
description: Free automated clipping machine. Finds new GTA 6 videos on YouTube (Rockstar, big streamers) or takes any YouTube, Twitch or Kick link, transcribes them on your own PC, picks the best moments and renders up to 10 finished 9:16 Shorts per run, with face-tracked framing, word-by-word captions, a hook title, a cover and ready-to-paste post text. No editing. Also clips a Twitch stream while it is live, on chat spikes, and has a local web app for the results. Use for "make 10 GTA 6 Shorts", "clip the newest GTA 6 videos", "make Shorts from <link>", "clip <streamer> live", "add my voice to this clip", "open the clipper app".
---
# GTA 6 clipper
Free, on the user's PC. `SKILL` = this folder, `PY` = `SKILL\.venv\Scripts\python.exe` (Mac/Linux `SKILL/.venv/bin/python`). Run from the user's folder; results go to `clips/`. Steps 1 and 3 take 10+ min: run in the background and check output.
Rare options (formats, caption styles, branding, memes, music, translation, vision, live, voice, app, callbacks, learn, Twitch details, setup) are in `SKILL-REFERENCE.md`: read a section only when the user asks or a rule below sends you there. Default "make 10 Shorts, just run it": steps 1, 2, peek, 3, 4, hand over; no questions, default look, no extras. Only posting needs the user. Missing `SKILL/.venv`: see Setup in the reference. Never read whole transcripts: use candidates. Pick mode (app setting `PICK_MODE`): smart = candidates (default), deep = read full transcripts, free = no AI (`nopick.py N` + `batch.py`); the run prompt says which. Live: `LIVE_MODE` free/smart/deep in live.py.

## 1. Find videos
```
PY SKILL\watch.py "GTA ?(6|VI)" 5
```
Checks `channels.txt` (YouTube RSS, `search:` lines, Twitch category), downloads audio only, writes `clips/src/<id>.txt` (sentences with timestamps and tags), `<id>.json` (word timings, needed to render), `<id>.meta.json` (title, channel, link). Done, other-language and over-2 h videos are skipped. One link: `PY SKILL\watch.py <link> [START-END]` (streams over 4 h: pass a window of about 1 h; times stay full-video times; a given link is never skipped).
Tags say where to look, not what is good: `[LOUD +12dB]`, `[REPLAYED x5]`, `[CHAT x4 KEKW]` (chat spike), `[CLIPPED x2 5.6k views]` (viewers clipped it), `[MOMENT chase 0.8]`, `[HEAT 0-100]` (all signals in one number). Two tags on one line is usually the best moment.
When a video file exists (not only audio), label it free, no AI: `PY SKILL\moments.py <video|audio> [START-END] --txt clips\src\<id>.txt [--cam x0,y0,x1,y1]` labels chase/crash/fight/stunt/reaction/funny (~85x realtime), writes `<src>.moments.json` for the app timeline and adds `[MOMENT type score]` tags. chase/funny/banner = clip-worthy beats; crash/fight/stunt = hints only (they misfire on scene cuts).
Clip rules: before clipping a named streamer, look them up in `SKILL\clip_rules.json` (`platform:handle`, lowercase). `no` = stop and tell the user they said no to clipping (go on only if the user says they have permission); `ask` = tell the user the note; `unknown` = remind them to check the channel's rules before posting.
Download fails: `PY -m pip install -U yt-dlp` and retry; bot or age check: wait 10 min or set `CLIPPER_BROWSER=firefox`. 10 Shorts need 5+ usable sources (2 per source max): else rerun with a higher number or more `search:` lines.

## 2. Pick (your job)
1. `PY SKILL\nopick.py clips --candidates 30` (a few minutes on big transcripts: run in the background). Read only `clips/candidates.txt`: the 30 best windows, one line each: `id start-end score tags | words (head ... tail)`. Do not open the full transcripts.
2. Judge, rank, pick the best N (10 by default). To widen, shift or check a window, read only that region of `clips/src/<id>.txt` (grep the time range). A window is a suggestion: move its start or end to a sentence boundary; a story that needs more room gets it.
3. Fewer than N pass the rules: rerun with `--candidates 60`, then fetch more videos once, then make fewer and say why. Skip ids in `clips/src/used.json` unless asked; never overlap ranges in `clips/src/moments.json`.
No AI at all: `PY SKILL\nopick.py clips [N]` writes `picks.json` directly (weaker, no premise check).

**A pick must have**
1. **A premise and a payoff.** A peak (`[LOUD]`, `[CHAT]`, `[CLIPPED]`, `[REPLAYED]`, `[MOMENT chase/funny]`, laughter, a big thing on screen) AND something a stranger can name (object, person, number) said or shown BEFORE the peak. Shouting with no premise = 0. A reaction to a named thing on screen beats talk about it. Interviews, backstory, opinions, scenery = 0. A straight-faced description is no reaction: a face must change or something big must happen.
2. **Picture first.** Second 1 shows the moment or a reacting face, not a menu, loading screen or someone settling in. Setup over about 8 s: start later.
3. **Only the game asked for** (GTA 6 unless another is named): the screen shows it or someone reacting to it.
4. **No spoilers or leaks:** no story endings, cutscene runs, leaked pre-release footage.
5. **Credit:** the streamer's or channel's real name (meta or words), else skip. User's language only.
6. **Stands alone.** Never open on a pronoun with no antecedent, an answer without its question, or "and / but / so"; never cut mid-sentence. No two Shorts with the same point.
7. **Reveal, don't tease.** The viewer must SEE or hear the payoff: it sits in the last 40 % and the Short ends right on it (never on a logo, end card, sponsor read, credits, or a request repeated 10+ s). Creator feedback 18 Nov: Shorts were too short and revealed nothing. A Short holds the whole story (setup, turn, payoff) so a cold viewer knows what happened and why it mattered.

**Length follows the story, not a fixed cap:** one complete beat, start one sentence before the setup, never cut the payoff to hit a number. The renderer snaps to whole sentences and trims dead air. `batch.py` rejects a window over `MAX_SPAN` (see top of batch.py); if a story needs more, say so in the pick list instead of cutting it.
**Choose:** (a) today's topic: names or nouns recurring 5+ times in `[CHAT]`/`[HEAT]` lines beat one-offs; (b) score each 0-100 for how likely a stranger keeps watching (best 80-95, weak under 50); (c) stranger test, a 1 a.m. scroller: second 1 stops them, the payoff is seen not only said, they would send it to a friend; under 2 yes = replace; (d) max 2 Shorts per source (batch.py skips a 3rd); one link and N asked = N from it.

**Text:** hook = 6 words max, caps, name first then the moment (`SPEED FOUND KFC IN GTA 6`), says only what the clip shows. Title up to 70 chars starting with the name; 3-5 hashtags; keywords = 1-3 words as spoken (stay green in captions).
Write `clips/picks.json`, best first (forward slashes):
```json
[{"url": "<link from meta>", "transcript": "clips/src/<id>.json", "start": 2396.8, "end": 2416.6, "hook": "KAI CENAT SAYS THIS IS CAP",
  "name": "kai_cap", "score": 82, "title": "Kai Cenat thinks the GTA 6 map is cap", "hashtags": "#gta6 #kaicenat #shorts", "credit": "Kai Cenat", "keywords": ["CAP", "MAP"]}]
```
Formats (callback, compilation, waitforit...), `style`, `layout`, `lang`, watermark, memes, music: reference, only on request or when a moment clearly earns one.

**Look first:** `PY SKILL\peek.py clips\picks.json clips\peek` saves 4 frames per moment as `clips/peek/<name>.jpg`. Open them. Drop a pick if the wrong game, editor, menu, ad, logo card or still score screen fills the picture more than a couple of seconds, or the face never changes; fix a hook that mismatches the picture; replace and re-peek. Show the user the list (score, range, hook, why) unless told to just run.

## 3. Render
```
PY SKILL\batch.py clips\picks.json clips\out
```
Downloads only each moment, picks the layout itself, crops overlays, adds zoom, captions and hook. Per Short: `NN_<name>.mp4`, `.txt` (title, hashtags, credit), `_cover.jpg`; numbering continues; `clips/out/overview.jpg` = 4 frames per Short. About an hour for 10 (3 at once; `CLIPPER_WORKERS=1` on a weak PC).

## 4. Check
batch.py runs `qc.py` (PASS/WARN/FAIL, details `clips/out/qc.json`). FAIL: re-cut once with clip.py a sentence later or earlier as the reason says, re-run qc; still fails = drop it and say so. WARN: look at that overview row. Each `overview.jpg` row: first frame = a face or the moment (not a menu or empty street); last frame = still the moment (not a logo, end card, credits); face in frame, captions off faces, hook readable, no black bars. Fix one alone: `PY SKILL\clip.py <url> <transcript> START END "HOOK" clips\out\<name>.mp4`, or drop it.

## 5. Hand over
Never upload. Give each Short with its `.txt` and the list with QC status. Posting help, the app, live clipping, voice lines: reference.

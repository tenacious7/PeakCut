# GTA 6 clipper: reference (read only the section you need; SKILL.md holds the default run)

SKILL, PY as in SKILL.md.

## Setup
If `SKILL/.venv` is missing, set it up and tell the user in one line what you installed:
```
ffmpeg -version && ffprobe -version   # missing? Windows: winget install Gyan.FFmpeg (then open a new terminal) · Mac: brew install ffmpeg · Linux: apt install ffmpeg
python -m venv SKILL\.venv            # Mac/Linux: python3 -m venv SKILL/.venv
PY -m pip install -r SKILL\requirements.txt
nvidia-smi                            # works? then also: PY -m pip install -r SKILL\requirements-gpu.txt
```
Tested on Windows with Python 3.11 and 3.14. The GPU pack is about 2 GB; install it only if `nvidia-smi` works (then 1 hour of video transcribes in about 4 minutes on the `small` model; the CPU uses the faster `base` model, slower). Macs use the CPU.


## Learn (channel with 8+ Shorts)
Before picking, if the user's channel has 8 or more Shorts and they gave its handle, run `PY SKILL\learn.py @handle` (public view counts, 1 request, cached a day, no key) and read `clips\LEARN.md`; when two moments score about the same, take the one its "Rules" lines favour. With fewer than 8 Shorts it gives no rules. Otherwise skip it.


## Post (free)
Never upload for the user. Hand over each Short with its `.txt` (title, hashtags, credit). Free schedulers: YouTube Studio (Create -> Upload -> Visibility -> Schedule), Meta Business Suite (Instagram and Facebook Reels), TikTok web upload (Schedule). Spread the 10 Shorts across the day.


## Moments (video labels, no AI)
`PY SKILL\moments.py <video|audio> [START-END] --txt clips\src\<id>.txt [--cam x0,y0,x1,y1]` labels chase/crash/fight/stunt/reaction/funny (~85x realtime), writes `<src>.moments.json` for the app timeline and adds `[MOMENT type score]` tags. chase/funny/banner = clip-worthy beats; crash/fight/stunt = hints only (they misfire on scene cuts). SKILL.md runs it by default when a video file exists.

## Twitch and more tags
From launch day (19 Nov) the best material is Twitch.** `channels.txt` has a `twitch: Grand Theft Auto VI` line, so the plain run above also takes the best English VODs of the last 24 h in that Twitch category (the last 2 hours of each, with chat spikes and viewer clips tagged). By hand: `PY SKILL\watch.py twitch:"Grand Theft Auto VI" 3 [--window 2h]`; it also prints a `live.py` command for each big live stream. To clip whoever is live right now with the busiest chat: `PY SKILL\live.py top:"Grand Theft Auto VI"` (see Live).


Game events (gameplay streams only, when asked or the user has time):** `PY SKILL\vision.py <link> START-END --tag clips\src\<id>.txt` reads big on-screen banners (WASTED, BUSTED, MISSION PASSED/FAILED, VICTORY, GAME OVER, big centred text in new games) from a 360p copy of that window and adds `[GAME WASTED]` etc. to those lines (heat goes up). A `[GAME ...]` line is a clip-worthy beat: take the whole reaction around it. 7-20 minutes per hour of stream; run it before step 2 because it edits the transcript. First time: `PY -m pip install -r SKILL\requirements-vision.txt`.


## Callbacks
Claim then proof (a callback; try it on streams over 20 min).** Find the payoff first, then search the transcript 1 minute to 3 hours earlier for the same noun said as a claim, bet, prediction, promise, insult or gag ("I told you", "I bet", "watch", "I'm a pro", "y'all thought", "I'll never", a word said 3+ times). `PY SKILL\callbacks.py clips\src\<id>.txt` lists candidate pairs (a finder: most are not real, you decide). If one is real, make it a `"format": "callback"` pick (see Formats); same person and game in both halves; peek both. None: a single moment.


## Options (leave out unless the rule says to use them)
**Formats.** Set `"format"` on a pick; default `moment` (one clip, usually best on Shorts). Use another format only when the stream really has it (never invent a streak or pairing), and in a batch of 10 at most 2-3 picks are non-`moment`. `callback` needs a real claim; `compilation` only when the SAME event happens 3+ times, each part 10 s at most; `waitforit` only when the payoff lands by 15 s once the renderer squeezes the silent wait (it does by itself; `"squeeze": 0` turns it off). Two picks about one event or one running point: keep the better one. A compilation only suits long-reaction channels. Parts use the normal pick fields (`start`, `end`, optional `label`). Any format also takes `"meme"`.
- `callback`: `"parts": [{"start": .., "end": ..}, {"start": .., "end": ..}]`: the claim as one whole sentence (3-6 s), a **hard cut**, then 8-15 s of payoff. Put the gap in the hook (`"AFTER HE SAID HE'S A PRO"`); batch.py adds a small "EARLIER" tag on the claim by itself when the gap is over 10 minutes. A full card ("40 MINUTES LATER") only if you set `"jump_card": true` (optional `"jump": "text"`); the default is no card.
- `waitforit`: a slow build into one payoff; a "WAIT FOR IT" tag and bar run until it (`"payoff"`: second in the Short, optional).
- `reveal`: the thing on screen shows first, then the reacting face (stack layout; `"reveal"`: seconds).
- `chat`: chat's lines are the joke: real Twitch chat pops up over it (or give `"chat"` lines yourself).
- `compilation`: 3-5 times the same thing happens, numbered #3 #2 #1, best last; `"parts"` each with a `"label"`; the hook is the title (`"3 TIMES KAI LOST IT"`).
- `escalation`: it really gets worse in order (`"word": "STRIKE"` -> STRIKE 1, 2, 3).
- `thenvsnow`: a claim or plan, then what happened (`"labels": ["THEN", "NOW"]`, two parts).

**Caption style and layout.** Only if the user asks (or `brand.json` sets them); otherwise leave both out (`classic`, own framing). Set per pick (`"style"`, `"layout"`) or per run (`set CLIPPER_STYLE=hormozi`, `set CLIPPER_LAYOUT=bubble`). "Hormozi style", "MrBeast captions", "round cam", "facecam bubble" mean these. One style per batch unless asked; "make them look viral" = `word`.
- Styles: `classic` (default), `word` (one big word at a time; the look most viral streamer clips use), `hormozi`, `beast`, `karaoke`, `minimal`, `mozi`, `pill`, `lemon`, `bubble`, `sweep`, `stamp`, `neon`, `ember`, `retro`, `crimson`, `glow`, `twotone`, `hollow`, `comic`, `whisper`, `pop`, `typewriter`, `slide`, `shake` (keywords wobble), `punch`, `drop`, `duo` (two-line karaoke), `emoji` (needs an emoji font), `lower`, `meme` (Impact-style top and bottom), `tiktok`. Names say the look (`mozi` green box, `pill` purple box, `lemon` dark on yellow, `bubble` blue box, `neon` cyan glow); `classic` is white caps with a yellow spoken word. Keywords stay green in almost all of them.
- Layouts. Streamer with a cam: `stack` (default: cam on top, game below) or `bubble` (game fills the frame, round cam at the top). Use `bubble` only for a streamer with a webcam box over gameplay (Speed, Kai, xQc); on a studio or couch video it puts the wrong face in the circle, so keep the normal framing. No cam: own framing. `full`: whole picture over a blurred background, nobody cropped; for video calls (Zoom/FaceTime tiles), group shots and grids, and as the fix when the overview shows someone important cropped out.

**Branding and extras.** Per pick, as `CLIPPER_<NAME>` env, or once for all Shorts in `brand.json` in the user's folder (example: `SKILL/brand.example.json`). Use only what the user asked for or their `brand.json` holds.
- `position` (`auto` / `top` / `middle` / `bottom`); `watermark` (PNG path or text like `@mychannel`) with `watermark_pos` (`tl` `tr` `bl` `br` `bounce`), `watermark_opacity`, `watermark_size`; `progress` (bar along the bottom); `endcard` (text for the last second, e.g. `FOLLOW FOR MORE`); `flash` (white flash on the peak zoom, off by default).
- `aspects` (`["4:5", "1:1"]`): extra files for Instagram and X.
- `lang` (`es` `fr` `pt` `de` `ar` `hi`): captions and hook translated offline. Only when asked. First time `PY -m pip install --no-deps -r SKILL\requirements-translate.txt`, then about 100 MB per language once.
- Automatic, no setting: dark webcams are brightened, standalone "um/uh" cut, light voice noise reduction, short black flashes covered. If the render log says `THIN`, widen the pick by a sentence or drop it.

**Meme sound.** `"meme": "<mood>"` plus optional `"meme_at"` (the second in the Short where it lands; else the loudest moment late in the Short). Use it only when one beat is clearly the payoff and the mood is obvious: one per Short, on 2-3 Shorts of a batch of 10 at most, unless the user wants more. It never touches the first 3 s and may add up to 3 s at the end. Moods: `embarrassing` (freeze, slow push-in, dark vignette, sad song), `fail` (sad trombone), `shock` (vine boom + flash), `hype` (airhorn + shake), `win` (bass drop), `awkward` (crickets + freeze), `cringe` (record scratch + freeze), `funny` (rimshot), `suspense` (heartbeat). Famous meme songs are not included: the user adds a copy once, e.g. `PY SKILL\meme.py add hello_darkness <link> 0:05-0:12` (also `curb_enthusiasm`, `sad_violin`); that mood uses it from then on (`SKILL/sounds/moods.json` maps moods to sounds).
**Music bed.** `"music": "hype|chill|tense|funny"` (or a file name in `sounds/mine/`; `brand.json` `music` = default for every Short): a low synthesised beat under the finished Short that ducks under speech, mix kept at -14 LUFS, picture untouched; only when the user asks. Also `meme.py bed SHORT.mp4 --mood chill`. Music goes on the 9:16 file only (not the `aspects` 4:5 / 1:1 extras).


## Live: a stream that is on right now (Twitch only)
When the user wants a live stream clipped ("clip <channel> live", a twitch.tv link, `kick:<slug>` / kick.com link, or a YouTube live link / @handle), start this in the background from the user's folder and leave it running:
```
PY SKILL\live.py https://www.twitch.tv/<channel>
(Kick: `PY SKILL\live.py kick:<slug>`; YouTube: `PY SKILL\live.py "https://www.youtube.com/watch?v=<id>"` or `@handle`. Chat for all three needs no key; test with `chat.check("kick")`.)
```
It records the stream (last 10 minutes kept) and reads chat with no login. When chat hits 3x its normal rate it cuts 25 s before to 10 s after (chat reacts late), transcribes the clip and adds it to `clips/live/picks.json`; `clips/live/spikes.log` has the chat rate and top chat lines. A spike is a vote, not a pick. Every few minutes read the new `clips/live/spike*.txt`, drop spam (bot ads, emote floods while nothing happens in the words), and write the keepers, with a real hook, title and keywords by the step 2 rules, into `clips/live/keep.json` (never edit `picks.json`: live.py rewrites it; keep its forward-slash paths). Chat calling it "scammed" is not proof for a hook. Then render:
```
PY SKILL\batch.py clips\live\keep.json clips\out
```
- After 2 minutes live.py prints `chat: N messages in the last minute`. Under about 30, spikes won't come (viewer count says nothing: a big stream can have a dead chat): tell the user, suggest a channel with a faster chat, and use the words instead.
- Words route (also for more moments on a busy chat, every ~10 minutes): `PY SKILL\live.py buffer` joins the last ~10 minutes into `clips/live/buffer_HHMMSS.mp4`; then `PY SKILL\watch.py <that file>` and pick from its `.txt` by the step 2 rules (`[LOUD]` lines first), without overlapping the spike clips.
- A very busy chat that spams false spikes: restart with `--ratio 2 --min 40` (defaults: `--ratio 3 --min 15`).
- Stop it when the user says so. For Kick or YouTube Live say spikes are Twitch only, and use the link mode after the stream.


## 5. Make it yours (only if the user gives a voice line)
The user records one short line about the moment (phone or mic, any format). Then:
```
PY SKILL\voice.py clips\out\<name>.mp4 line.wav AT clips\out\<name>_voice.mp4
```
`AT` is the second the line starts, usually right after the payoff. The game drops about 18 dB under the line. Several Shorts: run it once per Short with that Short's line.


## The app (optional: when the user wants to see, review or send the results in a browser)
Start it in the background from the user's folder (the workspace; it lists every `clips/out` folder under it as a tab):
```
PY SKILL\app.py <workspace folder> [--port 8765] [--no-open]
```
It opens `http://localhost:8765` in the browser (`--no-open` skips that; if the port is taken it uses another, so read the address it prints). Tell the user the address; it keeps running until closed. Nothing extra to install. Nothing is posted unless the user has set a Blotato key and clicks "Confirm send" on a Short. The app also has an autopilot panel that schedules a weekly run through the OS scheduler (`app.py --autopilot <workspace>`); set it up from the app only when the user asks, and never run `--autopilot` yourself.

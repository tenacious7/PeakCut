# One finished Short from a long video: cut -> auto 9:16 -> dead air out -> word-by-word captions + hook title -> loudness -14 LUFS.
# Usage: python clip.py (src.mp4 or link) transcript.json START END "HOOK TEXT" out.mp4
# START/END in seconds (or m:ss). Transcript from transcribe.py (word timings). START/END snap to whole sentences.
import json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np, cv2
sys.stdout.reconfigure(encoding="utf-8")  # hooks and paths can carry emoji; a cp1252 pipe would crash the print
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reframe
CK = ["--cookies-from-browser", os.environ["CLIPPER_BROWSER"]] if os.environ.get("CLIPPER_BROWSER") else []  # set CLIPPER_BROWSER=firefox (or edge, chrome) for videos that need a YouTube login

FONT = "Montserrat ExtraBold"
FONTS = Path(__file__).resolve().parent / "fonts"  # free Google fonts (OFL), shipped with the skill
FONT_FILE = FONTS / "Montserrat-ExtraBold.ttf"
# Caption looks. ASS size = em px x the font's (winAscent+winDescent)/unitsPerEm (Anton 1.733, Poppins 1.762).
# All but "classic", "hormozi" and "karaoke" (tuned to what users call them, research/round12/templates.md) from Cutawan's caption table (MIT, github.com/JeremySNR/cutawan, src/shared/captionStyles.ts). Keywords stay green in every style.
# name: font file, family, bold, size, CAPS, text, spoken word, outline colour, outline px, shadow, pill behind the spoken word, words per line, y
STYLES = {
    "classic": ("Montserrat-ExtraBold", FONT, -1, 84, 1, "FFFFFF", "FFE500", "000000", 7, 3, None, 3, 1250),
    "beast":   ("Anton-Regular", "Anton", 0, 173, 1, "FFFFFF", "FFD400", "000000", 3.2, 1, None, 3, 1382),
    "hormozi": ("Anton-Regular", "Anton", 0, 166, 1, "FFFFFF", "F7C204", "000000", 5, 2, None, 3, 1344),  # as users know it: white caps, yellow spoken word, heavy outline
    "mozi":    ("Anton-Regular", "Anton", 0, 166, 1, "FFFFFF", "FFFFFF", "000000", 2.8, 1, "16A34A", 3, 1344),  # Cutawan's "hormozi": green pill
    "karaoke": ("Poppins-Bold", "Poppins", -1, 155, 0, "A3A3A3", "F7C204", "000000", 2.6, 1, None, 4, 1421),  # unspoken grey, spoken yellow
    "pill":    ("Anton-Regular", "Anton", 0, 146, 1, "FFFFFF", "FFFFFF", "000000", 2, 0, "7C3AED", 3, 1421),
    "lemon":   ("Anton-Regular", "Anton", 0, 152, 1, "FFFFFF", "111111", "000000", 1.8, 0, "FACC15", 3, 1421),
    "neon":    ("Poppins-Bold", "Poppins", -1, 155, 0, "FFFFFF", "22D3EE", "164E63", 3, 2, None, 4, 1421),
    "ember":   ("Anton-Regular", "Anton", 0, 173, 1, "FFF7ED", "FB923C", "431407", 3.2, 1, None, 3, 1382),
    "bubble":  ("Poppins-Bold", "Poppins", -1, 143, 0, "FFFFFF", "FFFFFF", "000000", 2, 0, "2563EB", 4, 1440),
    "retro":   ("Anton-Regular", "Anton", 0, 166, 1, "FDE68A", "F472B6", "3B0764", 3, 2, None, 3, 1382),
    "crimson": ("Poppins-Bold", "Poppins", -1, 162, 1, "FFFFFF", "EF4444", "000000", 3, 1, None, 3, 1382),
    "minimal": ("Poppins-Medium", "Poppins Medium", 0, 129, 0, "FFFFFF", "FFFFFF", "000000", 1.6, 0, None, 5, 1440),
    "word":    ("Anton-Regular", "Anton", 0, 210, 1, "FFFFFF", "FFE500", "000000", 6, 3, None, 1, 1250),  # one big word at a time, centred: what 8 of 12 viral streamer clips use (research/round12/clipref)
    "pop":     ("LilitaOne-Regular", "Lilita One", 0, 106, 1, "FFFFFF", "FFD60A", "000000", 6, 2, None, 3, 1340),
    "typewriter": ("CourierPrime-Bold", "Courier Prime", -1, 108, 0, "F8F8F2", "4ADE80", "000000", 5, 2, None, 3, 1400),
    "slide":   ("ArchivoBlack-Regular", "Archivo Black", 0, 104, 1, "FFFFFF", "FF4D6D", "000000", 6, 2, None, 3, 1360),
    "shake":   ("Anton-Regular", "Anton", 0, 173, 1, "FFFFFF", "FF3B30", "000000", 5, 2, None, 3, 1382),
    "sweep":   ("BowlbyOne-Regular", "Bowlby One", 0, 116, 1, "FFFFFF", "FFFFFF", "000000", 4, 1, "FF2E93", 3, 1382),
    "twotone": ("ArchivoBlack-Regular", "Archivo Black", 0, 103, 1, "FFF200", "FF3D00", "000000", 6, 2, None, 3, 1382),
    "emoji":   ("LilitaOne-Regular", "Lilita One", 0, 113, 1, "FFFFFF", "FFD60A", "000000", 6, 2, None, 3, 1380),
    "lower":   ("Poppins-Medium", "Poppins Medium", 0, 100, 0, "FFFFFF", "FFFFFF", "000000", 14, 0, None, 6, 1480),
    "duo":     ("Poppins-Bold", "Poppins", -1, 123, 1, "9CA3AF", "FFFFFF", "000000", 4, 1, None, 6, 1380),
    "meme":    ("Anton-Regular", "Anton", 0, 200, 1, "FFFFFF", "FFFFFF", "000000", 9, 0, None, 6, 1380),
    "glow":    ("Righteous-Regular", "Righteous", 0, 124, 0, "FFFFFF", "FF4FD8", "FF00C8", 3, 0, None, 3, 1380),
    "comic":   ("Bangers-Regular", "Bangers", 0, 211, 1, "FFE600", "FFFFFF", "000000", 9, 5, None, 3, 1380),
    "whisper": ("Poppins-Medium", "Poppins Medium", 0, 106, 0, "FFFFFF", "FFFFFF", "000000", 1.4, 3, None, 5, 1420),
    "punch":   ("BowlbyOne-Regular", "Bowlby One", 0, 196, 1, "FFFFFF", "FFE500", "000000", 8, 4, None, 1, 1150),
    "tiktok":  ("Poppins-Bold", "Poppins", -1, 113, 0, "FFFFFF", "FFFFFF", "000000", 4.5, 2, None, 4, 1380),
    "drop":    ("Montserrat-ExtraBold", "Montserrat ExtraBold", -1, 107, 1, "FFFFFF", "A78BFA", "000000", 6, 2, None, 3, 1360),
    "hollow":  ("BebasNeue-Regular", "Bebas Neue", 0, 184, 1, "FFFFFF", "FFE500", "000000", 5, 3, None, 3, 1380),
    "stamp":   ("ArchivoBlack-Regular", "Archivo Black", 0, 103, 1, "FFFFFF", "FFFFFF", "000000", 2, 0, "E11D48", 3, 1400),
}
def bgr(h): return h[4:6] + h[2:4] + h[0:2]  # ASS colours are &HBBGGRR
QDB = 20  # dB under typical speech that counts as dead air for quiet_cut
MIN_LEN = 12.0  # cutting dead air never takes a Short under this (critics prefer a tight 12 s to a padded 20 s)
LONG = 2.0  # a word gap longer than this is dead even over game sound, unless something louder than the voice happens in it
GAP = 0.35  # pause (s) between words that gets cut when the audio is quiet there too
HOOK_S = 3.0          # hook title stays this long
CAP_Y = 1250          # caption baseline (px from top of 1920): clear of the bottom ~400 px UI
HOOK_Y = 330          # hook centre, clear of the top ~180 px UI
BAD = re.compile(r"^(fuck|shit|bitch|dick|pussy|cock|nigga|nigger|faggot|fag|cunt)(s|in|ing|er|ers|ed|y|ass)?$", re.I)  # masked on screen (audio untouched)

def secs(v): return sum(float(x) * 60 ** i for i, x in enumerate(reversed(str(v).split(":"))))
def ts(t): h, r = divmod(max(t, 0), 3600); m, s = divmod(r, 60); return f"{int(h)}:{int(m):02d}:{s:05.2f}"
def esc(s): return s.replace("{", "(").replace("}", ")").replace("\n", " ")
def clean(w):
    w = re.sub(r'[,.;:"]+|^-+', "", w)  # Whisper punctuation reads as noise on screen; ? and ! stay
    return w[0] + "*" * (len(w) - 1) if BAD.match(w) else w

def snap(words, start, end):
    """Start on the first spoken word, end on the next sentence end (at most 4 s later) plus a short breath: never stop mid-thought."""
    i = next((k for k, w in enumerate(words) if w["e"] > start + 0.05), None)
    if i is None: return start, end
    start = max(words[i]["s"] - 0.2, words[i - 1]["e"] if i else 0)
    j = max((k for k, w in enumerate(words) if w["s"] < end), default=i)
    full = lambda k: bool(re.search(r"[.?!]$", words[k]["w"]))
    stop = lambda k: full(k) or k + 1 == len(words) or words[k + 1]["s"] - words[k]["e"] >= 0.25  # sentence end or a breath (no gap cap: a long pause is a pause)
    ahead = [k for k in range(j, len(words)) if words[k]["e"] <= end + 4]
    j = next((k for k in ahead if full(k)),                                                        # 1 sentence end within 4 s
        next((k for k in ahead[:1] if stop(k)),                                                     # 2 the pick already ends on a breath: keep it
        next((k for k in range(j - 1, i - 1, -1) if stop(k) and words[k]["e"] - start >= MIN_LEN),  # 3 back off to the last stop that keeps MIN_LEN
        next((k for k in ahead if stop(k)), j))))                                                   # 4 else the next breath, else the pick as it is
    nxt = words[j + 1]["s"] - 0.05 if j + 1 < len(words) else words[j]["e"] + 0.25
    return start, max(words[j]["e"] + 0.1, min(words[j]["e"] + 0.25, nxt))  # 0.25 s breath (was 0.4): blind critics marked silent tails down

def later_starts(words, start, end, most=8, keep=12):
    """Sentence starts (or, in an unpunctuated transcript, starts after a 0.35 s pause) up to `most` s after start that still leave `keep` s:
    where to begin instead when the first second has no face (a logo card, an empty chair)."""
    return [w["s"] - 0.2 for a, w in zip(words, words[1:]) if (re.search(r"[.?!]$", a["w"]) or w["s"] - a["e"] >= 0.35) and start < w["s"] - 0.2 <= start + most and end - w["s"] >= keep]

def has_face(src, t0):
    """A face (>= 2.5 % of the width) in at least 2 of 3 frames of the hook second: an empty cam chair, a street or a menu fails."""
    cap = cv2.VideoCapture(str(src)); W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT); sz = (640, int(640 * H / W)) if W else (640, 360)
    yn, hits = cv2.FaceDetectorYN.create("onnx", reframe.YUNET, np.empty(0, np.uint8), sz, 0.6), 0
    for t in (0.1, 0.6, 1.1):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + t) * 1000); ok, f = cap.read()
        fs = yn.detect(cv2.resize(f, sz))[1] if ok else None
        hits += fs is not None and (fs[:, 2] / 640).max() >= 0.025
    cap.release(); return hits >= 2

def durations(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout
    return {l.split(",")[0]: float(l.split(",")[1]) for l in out.split() if l.count(",") == 1 and l.split(",")[1] not in ("", "N/A")}

def loudnorm_stats(stderr): return json.loads(re.findall(r"\{[^{}]*\}", stderr, re.S)[-1])

def speech_db(words, wav_db):
    """Typical speech loudness: median of each word's peak."""
    return np.median([wav_db[int(w["s"] * 20):int(w["e"] * 20) + 1].max(initial=-99) for w in words]) if words else 0

TAIL = 2.0  # a Short may run at most this long past the end of the sentence holding its loudest second (the payoff)
def tail_trim(spans, words, wav_db, tail=TAIL):
    """Cut the tail after the peak: end the spans on the first stop (sentence end or breath) after the loudest 1 s, if more than `tail` s of kept time follows it."""
    if not spans or not words or len(wav_db) < 20: return spans
    c = np.concatenate([[0], np.cumsum(wav_db)]); m = (c[20:] - c[:-20]) / 20  # mean level of each 1 s window, start index i = i / 20 s
    ok = [i for i in range(len(m)) if any(a <= i / 20 + .5 <= b for a, b in spans)]
    if not ok: return spans
    pi = max(ok, key=lambda i: m[i]); pk = pi / 20 + .5
    k = next((k for k, w in enumerate(words) if w["e"] >= pk and (w.get("end") or k + 1 == len(words) or words[k + 1]["s"] - w["e"] >= 0.25)), None)
    if k is None: return spans
    et = words[k]["e"] + 0.25
    if int(et * 20) < len(m) and m[int(et * 20):].max() >= m[pi] - 2: return spans  # a second nearly as loud later (a reaction repeat) = the payoff may be there, keep it
    if sum(max(0, b - max(a, et)) for a, b in spans) <= tail: return spans
    return [(a, min(b, et)) for a, b in spans if a < et]

SQ_LOUD = 8  # squeeze keeps any stretch holding a peak >= 8 dB over the speech level (a wordless bang/scream/laugh payoff), only steady game sound is cut
def keep_spans(words, wav_db, dur, gap=GAP, pad=0.15, long=None, loud=3, hold=0.3):  # gap 0.35 (was 0.45, 0.6 before): refs have 16 % silence, ours had 28 %
    """Spans to keep: pauses over `gap` s between words are cut, but only when the audio is quiet there too (a silent crash is action, not dead air)."""
    speech = speech_db(words, wav_db)
    # second 1 needs a voice: start on the first word, even over game sound (critics: 2-4 s of quiet opening = swiped)
    spans, t = [], (max(words[0]["s"] - pad, 0.0) if words and words[0]["s"] > 0.6 else 0.0)
    for a, b in zip(words, words[1:]):
        g0, g1 = a["e"] + pad, b["s"] - pad; seg = wav_db[int(g0 * 20):int(g1 * 20) + 1]
        if g1 - g0 > gap - 2 * pad and len(seg) and np.percentile(seg, 90) < speech - 8: spans.append((t, g0)); t = g1
        # nobody talks for > LONG s over game sound: keep 0.3 s each side unless a bang louder than the voice is in there (critics: '10 s of near-nothing')
        elif long and g1 - g0 > long and seg.max(initial=-99) < speech + loud: spans.append((t, g0 + hold)); t = g1 - hold
    return spans + [(t, dur)]

def quiet_cut(spans, wav_db, floor, qmin=0.35, pad=0.10):
    """Audio-driven dead air: Whisper stretches word ends over silence, so word gaps miss it. Inside each span, runs quieter than `floor` dB for >= qmin s
    are cut down to `pad` s each side; quiet at the very end of the last span / start of the first is cut to 0.12 s / 0.05 s."""
    out = []
    for a, b in spans:
        q = wav_db[int(a * 20):int(b * 20)] < floor; t, k = a, 0
        for i, v in enumerate(list(q) + [False]):
            if v: k += 1; continue
            r0, r1 = a + (i - k) / 20, a + i / 20  # the quiet run [r0, r1)
            last, first = r1 >= b - 0.15 and (a, b) == spans[-1], r0 <= a + 0.05 and (a, b) == spans[0]
            if k / 20 >= (0.12 if last or first else qmin):
                if last: out.append((t, r0 + 0.12)); t = None; break  # trailing silence: end the Short on the sound
                if first: t = r1 - 0.05; k = 0; continue  # leading silence: start on the sound
                out.append((t, r0 + pad)); t = r1 - pad
            k = 0
        if t is not None: out.append((t, b))
    return [(x, y) for x, y in out if y - x > 0.05]

def audio_db(p):
    """Loudness per 50 ms of the clip."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw[: len(raw) // 800 * 800], np.int16).astype(np.float32).reshape(-1, 400)
    return 10 * np.log10((x ** 2).mean(1) + 1)

ZOOM, ZIN, ZOUT = 0.10, 0.2, 0.3  # punch-in: +10 % over 0.2 s, ease back over 0.3 s
HIT_ZIN, HIT_AMP, FLASH = 0.07, 1.6, 0.1  # the payoff gets a snap: +16 % in 70 ms, plus a 0.1 s soft white flash
GRADE = "eq=saturation=1.3:contrast=1.07"  # game footage pops like the refs; applied before captions so the text colours stay exact

def peaks(db, speech, dur, ok=lambda t: True, over=0, lo=1.5, gap=6.0, n=2):  # n 2: viral streamer clips average 1.3 zooms (research/round12/clipref)
    """Reaction moments = loudest 150 ms spots above speech level (+`over` dB; streamer audio is compressed, its 99th percentile is only ~4 dB over the median); n at most, `gap` s apart, none in the hook (before lo)."""
    sm, out = np.convolve(db, np.ones(3) / 3, "same"), []
    for i in np.argsort(-sm):
        t = i / 20
        if sm[i] < speech + over or len(out) == n: break
        if lo + 0.15 <= t <= dur - 1.5 and all(abs(t - u) >= gap for u in out) and ok(t): out.append(t)
    return sorted(out)

def zoom_filter(ps, stack, hit=None, flip=False):
    """Smoothstep in/out zoom as one ffmpeg expression of t. scale(eval=frame)+crop on an whole-pixel crop on a 2 px size grid: no zoompan, so no whole-pixel shake.
    stack: only the top cam panel (rows 0-763, seam kept); else the whole frame about the face (upper-middle)."""
    u = lambda x: f"clip({x},0,1)"; ease = lambda x: f"({u(x)}*{u(x)}*(3-2*{u(x)}))"
    zi = lambda a: HIT_ZIN if a == hit else ZIN  # the hit (loudest peak) snaps in, the others drift
    z = "(" + "+".join(f"{HIT_AMP if a == hit else 1}*{ease(f'(t-{a:.3f})/{zi(a)}')}*(1-{ease(f'(t-{b:.3f})/{ZOUT}')})" for a, b in ps) + ")"  # 0..1 (1.6 on the hit)
    d = lambda n: f"{n}+4*round({n}*{ZOOM}*{z}/4)"  # scaled size on a 4 px grid: centred crop offset is a whole even pixel (2 px grid showed 1 px spikes)
    py = 1156 if flip else 0  # flip: the cam panel is the bottom 764 rows, its seam box sits above it
    if not stack: return f"scale=w='{d(1080)}':h='{d(1920)}':eval=frame:flags=bicubic,crop=1080:1920:'({d(1080)}-1080)/2':'round(0.3*({d(1920)}-1920))'"
    return (f"split[p][q];[p]crop=1080:764:0:{py},scale=w='{d(1080)}':h='{d(764)}':eval=frame:flags=bicubic,crop=1080:764:'({d(1080)}-1080)/2':'({d(764)}-764)/2'[z];"
            f"[q][z]overlay=0:{py},drawbox=x=0:y={1149 if flip else 764}:w=1080:h=7:color=black:t=fill:enable='{'+'.join(f'between(t,{a - .05:.3f},{b + ZOUT + .05:.3f})' for a, b in ps)}'")
    # the 764-row panel leaves a 1 px strip above the 6 px seam: paint it black, but only while a zoom runs (the Short may switch to full screen elsewhere)

def flash(hit):
    """Soft white flash on the first frames of the hit zoom ('' when there is none): the refs flash on their loudest cut."""
    return "" if hit is None else f",drawbox=x=0:y=0:w=iw:h=ih:color=white@0.35:t=fill:enable='between(t,{hit:.3f},{hit + FLASH:.3f})'"

def whoosh(ps):
    """A soft whoosh (band-passed pink noise, swelling into the zoom) per punch-in, made by ffmpeg: no sound files to ship. Ends the chain '[an]... ,' ready for the fade."""
    if not ps: return "[an]"
    sfx = "".join(f"anoisesrc=d=0.45:c=pink:r=48000:a=0.5,bandpass=f=1800:width_type=o:w=2.5,afade=t=in:d=0.3:curve=exp,afade=t=out:st=0.3:d=0.15,"
                  f"aformat=channel_layouts=stereo,volume=10dB,adelay={max(int((a - .2) * 1000), 0)}:all=1[s{i}];" for i, (a, _) in enumerate(ps))
    return sfx + "[an]" + "".join(f"[s{i}]" for i in range(len(ps))) + f"amix=inputs={len(ps) + 1}:normalize=0:duration=first,"

def groups(words, size=3, gap=0.6):
    """1-3 words per caption line; break on pauses and after a sentence ends so lines follow speech rhythm.
    A sentence's last word joins a full line (4 words) instead of standing alone: the last frame never reads 'BOTTOM OF THE'."""
    g = []
    for w in words:
        full = len(g[-1]) >= size and not (w.get("end") and len(g[-1]) == size) if g else False
        if g and (full or g[-1][-1].get("end") or w["s"] - g[-1][-1]["e"] > gap): g.append([w])
        elif g: g[-1].append(w)
        else: g.append([w])
    return g

KEY = "5CFF3D"  # keyword colour (brand.json "key" overrides)
# kind of motion per style (see ass()): none = the old look; bounce/shake/slide/drop/type/stamp/punch/comic animate with \t transforms; box/phrase = one event per line;
# duo = 2-line \kf karaoke; meme = top + bottom text; sweep/glow/twotone add a second layer; hollow/whisper change the style alpha
ANIM = {"pop": "bounce", "typewriter": "type", "slide": "slide", "shake": "shake", "sweep": "sweep", "twotone": "twotone", "emoji": None, "lower": "box", "duo": "duo",
        "meme": "meme", "glow": "glow", "comic": "comic", "whisper": "whisper", "punch": "punch", "tiktok": "phrase", "drop": "drop", "hollow": "hollow", "stamp": "stamp"}
NOW = {  # extra override tags on the spoken word, per kind
    "bounce": r"\fscx72\fscy72\t(0,100,\fscx117\fscy117)\t(100,200,\fscx100\fscy100)", "punch": r"\fscx170\fscy170\t(0,110,\fscx100\fscy100)",
    "comic": r"\fscx70\fscy70\t(0,90,\fscx115\fscy115)\t(90,180,\fscx100\fscy100)", "stamp": r"\fscx165\fscy165\t(0,100,\fscx100\fscy100)",
    "hollow": r"\1a&H00&", "whisper": r"\1a&H00&", "shake": r"\fscx108\fscy108", "twotone": "", "sweep": ""}
SHAKE = r"\frz-4\t(0,45,\frz4)\t(45,90,\frz-4)\t(90,135,\frz4)\t(135,180,\frz0)"  # keywords wobble in kind "shake"
PUNCH_COL = ["FFFFFF", "FFE500", "22D3EE", "F472B6"]  # kind "punch": each word its own colour
EMOJI = {"fire": "🔥", "hot": "🔥", "burn": "🔥", "money": "💰", "cash": "💰", "dollars": "💰", "rich": "💰", "paid": "💰", "crazy": "🤯", "insane": "🤯", "wild": "🤯",
         "mind": "🤯", "laugh": "😂", "funny": "😂", "lol": "😂", "haha": "😂", "love": "❤️", "heart": "❤️", "omg": "😱", "wtf": "😱", "scared": "😱", "dead": "💀", "die": "💀",
         "died": "💀", "kill": "💀", "win": "🏆", "won": "🏆", "winner": "🏆", "champion": "🏆", "gun": "🔫", "shoot": "🔫", "car": "🚗", "drive": "🚗", "fast": "⚡", "speed": "⚡",
         "cry": "😭", "sad": "😭", "look": "👀", "watch": "👀", "see": "👀", "boom": "💥", "bomb": "💥", "explode": "💥", "game": "🎮", "gaming": "🎮", "hundred": "💯", "ghost": "👻",
         "chicken": "🍗", "eat": "🍗", "king": "👑", "stop": "🛑", "time": "⏰", "secret": "🤫", "shh": "🤫", "kidding": "🤨", "joking": "🤨", "wait": "✋", "help": "🆘", "gold": "🥇", "boss": "😎", "gas": "⛽", "bro": "😤", "no": "🚫", "yes": "✅"}
EMOJI_GAP = 1.4  # s between two emoji pops
POS_Y = {"top": 450, "middle": 960, "bottom": 1440}  # caption position option; "auto" = the style's own height (stacked layout: the cam/game seam)
ASPECTS = {"1:1": (1080, 1080, 320), "4:5": (1080, 1350, 190)}  # width, height, where the window starts in the 9:16 (hook, captions and the seam stay inside)

def font_factor(path):
    """ASS size = em px x this (winAscent+winDescent over unitsPerEm)."""
    try:
        from fontTools.ttLib import TTFont
        t = TTFont(path); o = t["OS/2"]; return (o.usWinAscent + o.usWinDescent) / t["head"].unitsPerEm
    except Exception: return 1.5
def text_w(st, text):
    """Rendered width in px of `text` in the style's font and size (PIL), for the highlight sweep."""
    from PIL import ImageFont
    p = fpath(st); return ImageFont.truetype(str(p), max(int(st[3] / font_factor(p)), 8)).getlength(text)

def word(x, now, keys, st=STYLES["classic"], kind=None, idx=0):
    """Caption word: the spoken one in the style's colour and popped (pill styles: a box behind it), keywords (the thing the clip is about) green all the time."""
    w = esc(x["w"].upper() if st[4] else x["w"]); key = re.sub(r"\W", "", w.upper()) in keys
    if not (now or key): return w
    tag = r"\c&H" + (bgr(KEY) if key else bgr(PUNCH_COL[idx % 4] if kind == "punch" and now else st[6])) + "&"
    if kind == "shake" and key: tag += SHAKE
    if now and kind in NOW: tag += NOW[kind] + (rf"\bord12\3c&H{bgr(st[10])}&" if kind == "stamp" else "")
    elif now: tag += r"\fscx112\fscy112" if st[:2] == STYLES["classic"][:2] and st[3] == 84 else r"\fscx100\fscy100\t(0,70,\fscx109\fscy109)" + (rf"\bord10\3c&H{bgr(st[10])}&" if st[10] else "")
    return "{" + tag + "}" + w + r"{\r}"

def plain(g, st, keys, top=None):
    """The words of a line as one string with no highlight; keywords green (back to `top` colour after each when given)."""
    f = lambda x: esc(x["w"].upper() if st[4] else x["w"])
    return " ".join(("{\\c&H" + bgr(KEY) + "&}" + f(x) + ("{\\c&H" + bgr(top) + "&}" if top else r"{\r}")) if re.sub(r"\W", "", f(x).upper()) in keys else f(x) for x in g)

def style_line(name, st, kind, box=False):
    """One ASS style. Primary/secondary: karaoke sweeps secondary -> primary; hollow hides the fill, whisper dims it; twotone draws its base layer in the second colour."""
    c1, c2 = bgr(st[5]), bgr(st[5])
    if kind == "duo": c1 = bgr(st[6])
    if kind == "twotone": c1 = c2 = bgr(st[6])
    a1 = {"hollow": "FF", "whisper": "8C"}.get(kind, "00")
    ang = {"comic": 3, "stamp": 3}.get(kind, 0)
    ol = bgr(st[5]) if kind == "hollow" else bgr(st[7])
    return f"Style: {name},{st[1]},{st[3]},&H{a1}{c1},&H00{c2},&H00{ol},&H64000000,{st[2]},0,0,0,100,100,0,{ang},{3 if box else 1},{st[8]},{st[9]},5,60,60,0,1"

def cap_y(st, pos="auto", stacked=None):
    """Caption centre (px of 1920). Stacked layout: seam unless middle/bottom is asked (top would cover the cam face)."""
    stacked = STACKED if stacked is None else stacked
    if stacked: return POS_Y[pos] if pos in ("middle", "bottom") else CAP_Y
    return POS_Y.get(pos, st[12])

def ass(words, hook, dur, keys=(), style="classic", pos="auto", H=1920, y0=0, endcard="", hookc=None, emoji=None, st=None, cy=None):
    """Caption file. H/y0: a re-layout for a cropped export (window of H rows starting at y0 of the 9:16): every y is shifted and kept inside.
    emoji: list that collects (time, emoji) for the overlay pass."""
    st = st or STYLES.get(style, STYLES["classic"]); kind = ANIM.get(style); hookc = hookc or {}
    sh = bool(y0) or H != 1920  # a cropped export: every y shifted into the window and kept inside it
    my = (lambda v: int(min(max(v - y0, 150), H - 170))) if sh else (lambda v: int(v))
    y = my(cy or cap_y(st, pos)); fs = st[3]  # cy: caller-given caption y (no module globals needed)
    hf = bgr(hookc.get("text", "FFFFFF")); hb = bgr(hookc.get("box", "000000"))
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line("Cap", st, kind)}
{style_line("Box", st, kind, True)}
Style: Hook,{FONT},76,&H00{hf},&H00{hf},&H00{hb},&H00000000,-1,0,0,0,100,100,0,0,3,22,0,5,70,70,0,1
Style: Cta,{FONT},96,&H00FFFFFF,&H00FFFFFF,&H00{hb},&H00000000,-1,0,0,0,100,100,0,0,3,26,0,5,70,70,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = [f"Dialogue: 1,{ts(0)},{ts(min(HOOK_S, dur))},Hook,,0,0,0,,{{\\pos(540,{my(HOOK_Y)})\\3c&H{hb}&\\fad(0,250)}}{esc(hook.upper())}"]
    ey = my(y + 230 if y < 1000 else y - 270)  # the end card sits beside the captions, never on them
    if endcard and dur > 2.5: ev.append(f"Dialogue: 2,{ts(dur - 1.0)},{ts(dur)},Cta,,0,0,0,,{{\\pos(540,{ey})\\3c&H{hb}&\\fad(150,0)\\fscx70\\fscy70\\t(0,180,\\fscx100\\fscy100)}}{esc(endcard.upper())}")
    gs = groups(words, st[11])
    E = lambda a, b, lay, tags, txt, sty="Cap": ev.append(f"Dialogue: {lay},{ts(a)},{ts(b)},{sty},,0,0,0,,{{{tags}}}{txt}")
    last_emoji = -9.0
    for gi, g in enumerate(gs):
        nxt = gs[gi + 1][0]["s"] if gi + 1 < len(gs) else dur
        gend = min(g[-1]["e"] + 0.15, nxt)  # never overlap the next line
        half = (len(g) + 1) // 2; brk = lambda L: "\\N".join(x for x in (" ".join(L[:half]), " ".join(L[half:])) if x)
        if emoji is not None:
            for x in g:
                k = re.sub(r"\W", "", x["w"].lower())
                if k in EMOJI and x["s"] - last_emoji >= EMOJI_GAP and HOOK_S <= x["s"] < dur - 0.4: emoji.append((x["s"], EMOJI[k])); last_emoji = x["s"]
        if kind in ("box", "phrase"):  # one event per line (subtitle look / TikTok auto-captions): no per-word highlight
            per = 3 if kind == "box" else 2; txt = "\\N".join(plain(g[k:k + per], st, keys) for k in range(0, len(g), per))
            E(g[0]["s"], gend, 0, rf"\pos(540,{y})" + (r"\fad(70,0)" if kind == "phrase" else "") + ("\\3c&H000000&\\3a&H5A&" if kind == "box" else ""), txt, "Box" if kind == "box" else "Cap")
        elif kind == "duo":  # two lines, each word sweeps grey -> white over its own duration (\kf)
            cs = [max(int(round(((g[i + 1]["s"] if i + 1 < len(g) else min(w["e"] + .1, gend)) - w["s"]) * 100)), 1) for i, w in enumerate(g)]
            parts = [f"{{\\kf{c}}}" + esc(w["w"].upper() if st[4] else w["w"]) for w, c in zip(g, cs)]
            E(g[0]["s"], gend, 0, rf"\pos(540,{y})", " ".join(parts[:half]) + ("\\N" + " ".join(parts[half:]) if g[half:] else ""))
        elif kind == "meme" and not STACKED:  # white Anton caps, black outline: first half on top of the frame, second half at the bottom
            E(g[0]["s"], gend, 0, rf"\pos(540,{my(520)})", plain(g[:half], st, keys)) if g[half:] else None
            E((g[half]["s"] if g[half:] else g[0]["s"]), gend, 0, rf"\pos(540,{my(1380)})", plain(g[half:] or g, st, keys))
        elif kind == "meme": E(g[0]["s"], gend, 0, rf"\pos(540,{y})", brk([plain([x], st, keys) for x in g]))
        else:
            for i, w in enumerate(g):  # one event per spoken word: whole line shown, current word highlighted
                end = g[i + 1]["s"] if i + 1 < len(g) else gend
                pre = rf"\move(540,{y + 70},540,{y},0,140)\fad(90,0)" if kind == "slide" and i == 0 else rf"\pos(540,{y})"
                if kind == "type":  # typed char by char while spoken, the rest of the line is invisible (keeps the line steady)
                    n = len(w["w"]); step = max(min((end - w["s"]) * 1000 * .6 / max(n, 1), 60), 15)
                    cur = "".join(rf"{{\alpha&HFF&\t({int(k * step)},{int(k * step) + 1},\alpha&H00&)}}{esc(c)}" for k, c in enumerate(w["w"].upper() if st[4] else w["w"])) + r"{\r}"
                    line = " ".join([word(x, False, keys, st) for x in g[:i]] + [cur] + [r"{\alpha&HFF&}" + word(x, False, keys, st) + r"{\r}" for x in g[i + 1:]])
                elif kind == "drop":  # words fall in as they are spoken
                    cur = r"{\alpha&HFF&\fscy150\t(0,110,\alpha&H00&\fscy100)}" + word(w, False, keys, st) + r"{\r}"
                    line = " ".join([word(x, False, keys, st) for x in g[:i]] + [cur] + [r"{\alpha&HFF&}" + word(x, False, keys, st) + r"{\r}" for x in g[i + 1:]])
                else: line = " ".join(word(x, j == i, keys, st, kind, j) for j, x in enumerate(g))
                if kind == "sweep":  # a box slides across behind the spoken word (clip grows left -> right over 0.14 s)
                    sp = text_w(st, " "); ws = [text_w(st, (x["w"].upper() if st[4] else x["w"])) for x in g]; tot = sum(ws) + sp * (len(g) - 1)
                    a = 540 - tot / 2 + sum(ws[:i]) + sp * i; b = a + ws[i]; pad = 22
                    E(w["s"], end, 0, rf"\pos(540,{y})\1a&HFF&\bord22\3c&H{bgr(st[10])}&\shad0\clip({a - pad:.0f},0,{a - pad:.0f},{H})\t(0,140,\clip({a - pad:.0f},0,{b + pad:.0f},{H}))", plain(g, st, ()))
                if kind == "glow":
                    E(w["s"], end, 0, rf"\pos(540,{y})\1a&H40&\1c&H{bgr(st[7])}&\3c&H{bgr(st[7])}&\bord16\blur14\shad0", plain(g, st, ()))
                if kind == "twotone":  # top half in the first colour over a base layer in the second
                    E(w["s"], end, 2, rf"\pos(540,{y})\1c&H{bgr(st[5])}&\bord0\shad0\clip(0,{y - fs},1080,{y - int(fs * .02)})", plain(g, st, keys, st[5]))
                E(w["s"], end, 1 if kind in ("sweep", "glow") else 0 if kind != "twotone" else 1, pre, line)
    return head + "\n".join(ev) + "\n"

FILLER = {"um", "uh", "uhm", "umm", "er", "erm", "hmm", "uh-huh"}
def fillers(words, gap=0.12):
    """Indices of filler words to cut: um/uh/er standing alone with a pause (>= gap s) on both sides; 'like' only as the first word of a sentence with a pause after it."""
    out = []
    for i, w in enumerate(words):
        t = re.sub(r"\W", "", w["w"].lower()); prev, nxt = words[i - 1] if i else None, words[i + 1] if i + 1 < len(words) else None
        before = prev is None or w["s"] - prev["e"] >= gap or prev.get("end"); after = nxt is None or nxt["s"] - w["e"] >= gap
        if (t in FILLER and before and after) or (t == "like" and (prev is None or prev.get("end")) and nxt is not None and nxt["s"] - w["e"] >= gap): out.append(i)
    return out
def cut_out(spans, holes):
    """Spans minus the (a, b) holes."""
    for a, b in sorted(holes):
        spans = [p for s in spans for p in ([s] if b <= s[0] or a >= s[1] else [(s[0], max(a, s[0])), (min(b, s[1]), s[1])])]
    return [(a, b) for a, b in spans if b - a > 0.05]

def heard(words, spans):
    """Words whose middle falls inside a kept span (what the viewer hears)."""
    return [w for w in words if any(a - 0.02 <= (w["s"] + w["e"]) / 2 <= b + 0.02 for a, b in spans)]
def longest_gap(words, dur):
    """Longest stretch of the cut Short with nobody talking (the opening and the end count too)."""
    ts = [0.0] + [t for w in words for t in (w["s"], w["e"])] + [dur]
    return max([b - a for a, b in zip(ts[::2], ts[1::2])] + [0.0]) if words else dur
def runs_from_luma(lum, fps, thr=10.0, max_s=0.8):
    """(first, last) frame indices of runs of near-black frames shorter than max_s: a game-over flash; a black start (nothing to hold) or a long run (qc's job) is left alone."""
    out, i, n = [], 0, len(lum); w = int(fps)
    # black = under thr, or a dip to under 35 % of the picture around it (+-1 s) that is itself dark (< 30): a near-black frame in a bright scene
    dark = [x < thr or (x < 30 and x < 0.35 * float(np.median(lum[max(k - w, 0):k + w + 1]))) for k, x in enumerate(lum)]
    while i < n:
        if not dark[i]: i += 1; continue
        j = i
        while j + 1 < n and dark[j + 1]: j += 1
        if i > 0 and j + 1 < n and (j - i + 1) / fps < max_s: out.append((i, j))
        i = j + 1
    return out
def black_runs(v, step=1):
    """Black runs of a video file: mean luma of every frame (reduced to 64 px wide), see runs_from_luma."""
    cap = cv2.VideoCapture(str(v)); fps = cap.get(cv2.CAP_PROP_FPS) or 30; lum = []
    while True:
        ok, f = cap.read()
        if not ok: break
        lum.append(float(cv2.resize(f, (64, 114), interpolation=cv2.INTER_AREA).mean()))
    cap.release(); return runs_from_luma(lum, fps), fps
def freeze_filters(runs, fps, spans):
    """ffmpeg chain that holds the last good frame over each black run (video only: the audio is untouched). Run frames are source-video frames,
    spans the kept (a, b) seconds: runs inside a dropped part vanish with it, the rest move to cut-video frame numbers."""
    f, acc = "", 0.0
    for a, b in spans:
        for i, j in runs:
            if a <= i / fps and (j + 1) / fps <= b + 0.05:
                s = int(round((i / fps - a + acc) * fps)); e = s + (j - i)
                f += f"split[fz{s}a][fz{s}b];[fz{s}a][fz{s}b]freezeframes=first={s}:last={e}:replace={max(s - 1, 0)},"
        acc += b - a
    return f

LIFT_BELOW, LIFT_TARGET = 70, 100  # webcam panel mean luma (0-255) under this gets a gamma lift towards the target
def lift_gamma(mean):
    """eq gamma that brings a dark cam panel to a mean of ~100 (None when bright enough; capped at 2.2 so noise isn't blown up)."""
    if mean >= LIFT_BELOW: return None
    return round(min(max(np.log(max(mean, 8) / 255) / np.log(LIFT_TARGET / 255), 1.1), 2.2), 2)
GAME_BELOW, GAME_TARGET, GAME_MAX = 80, 92, 1.5  # game footage mean luma under this gets a gentle gamma lift (h17 Shorts 31-70 got "dark" from critics, refs sit at 100-135)
def game_gamma(mean):
    """gentle eq gamma for dark game footage towards a mean of ~92 (None when bright enough; capped at 1.5 so night scenes stay night and noise isn't blown up)."""
    if mean >= GAME_BELOW: return None
    return round(min(max(np.log(max(mean, 8) / 255) / np.log(GAME_TARGET / 255), 1.1), GAME_MAX), 2)
def game_luma(v, stacked, n=8, flip=False):
    """Mean luma of the game region over n frames: rows 768+ of the stack, else the whole frame."""
    cap = cv2.VideoCapture(str(v)); N = cap.get(cv2.CAP_PROP_FRAME_COUNT); m = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(N * (k + .5) / n)); ok, f = cap.read()
        if ok: m.append(cv2.cvtColor((f[:1140] if flip else f[780:]) if stacked else f, cv2.COLOR_BGR2GRAY).mean())
    cap.release(); return float(np.mean(m)) if m else 255.0
def cam_luma(v, n=5, flip=False):
    """Mean luma of the top 764 rows (the cam panel of the stacked layout) over n frames."""
    cap = cv2.VideoCapture(str(v)); N = cap.get(cv2.CAP_PROP_FRAME_COUNT); m = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(N * (k + .5) / n)); ok, f = cap.read()
        if ok: m.append(cv2.cvtColor(f[1160:] if flip else f[:760], cv2.COLOR_BGR2GRAY).mean())
    cap.release(); return float(np.mean(m)) if m else 255.0

def load_brand(out):
    """brand.json: CLIPPER_BRAND, else the one in the current folder, next to the Shorts, or one folder up."""
    for p in (os.environ.get("CLIPPER_BRAND"), Path.cwd() / "brand.json", out.parent / "brand.json", out.parent.parent / "brand.json"):
        if p and Path(p).is_file():
            try: return json.load(open(p, encoding="utf-8"))
            except ValueError as e: print(f"  brand.json unreadable ({e}), ignoring it")
    return {}
def opt(name, opts, brand, default=None):
    """Pick field > env CLIPPER_<NAME> > brand.json > default."""
    if opts.get(name) not in (None, ""): return opts[name]
    if os.environ.get("CLIPPER_" + name.upper()): return os.environ["CLIPPER_" + name.upper()]
    return brand.get(name, default)
def cap_scale(v):
    """cap_size pick: S/M/L or a number -> font multiplier clamped 0.7-1.5."""
    f = {"S": .85, "M": 1.0, "L": 1.2}.get(str(v).strip().upper())
    if f is None:
        try: f = float(v)
        except (TypeError, ValueError): f = 1.0
    return min(max(f, .7), 1.5)
THEMES_FILE = Path(__file__).resolve().parent / "themes.json"
THEME_KEYS = ("style", "colors", "hook_text", "hook_box", "progress_color", "watermark_color", "watermark_opacity", "watermark_size", "watermark_pos", "progress", "font")
def load_themes():
    try: return json.load(open(THEMES_FILE, encoding="utf-8"))["themes"]
    except (OSError, ValueError, KeyError) as e: raise SystemExit(f"themes.json unreadable ({e})")
def with_theme(brand, opts):
    """Look order: theme < brand.json < pick fields (env CLIPPER_<KEY> still sits over brand.json/theme in opt()). `theme` comes from the pick, CLIPPER_THEME or brand.json; colors merge key by key.
    Unknown theme = SystemExit listing the names."""
    name = opts.get("theme") or os.environ.get("CLIPPER_THEME") or brand.get("theme") or ""
    th = {}
    if name:
        t = {t["name"]: t["brand"] for t in load_themes()}
        if name not in t: raise SystemExit(f"unknown theme {name!r}; themes: {', '.join(t)}")
        th = t[name]
    pick = {k: opts[k] for k in THEME_KEYS if opts.get(k) not in (None, "")}
    return th | brand | pick | dict(colors=th.get("colors", {}) | brand.get("colors", {}) | (opts.get("colors") or {}))
def truthy(v): return str(v).lower() not in ("", "0", "false", "no", "off", "none")
def brand_style(st, brand, work=None):
    """The style tuple with the brand's colours / font put in (colors: text, spoken, outline, box)."""
    c = brand.get("colors", {}); st = list(st)
    for k, v in (("text", 5), ("spoken", 6), ("outline", 7), ("box", 10)):
        if c.get(k) and (v != 10 or st[10]): st[v] = c[k].lstrip("#").upper()
    f = brand.get("font")
    if f:
        p = Path(f) if Path(f).is_file() else FONTS / (f if f.endswith(".ttf") else f + ".ttf")
        if p.is_file():
            from PIL import ImageFont
            fam, sty = ImageFont.truetype(str(p), 20).getname(); old = font_factor(FONTS / f"{st[0]}.ttf")
            st[0], st[1], st[2], st[3] = str(p.resolve()), fam, -1 if "Bold" in sty or "Black" in sty else 0, int(st[3] / old * font_factor(p))
    return tuple(st)
def fpath(st): return Path(st[0]) if Path(st[0]).suffix else FONTS / f"{st[0]}.ttf"

EMOJI_FONTS = (r"C:\Windows\Fonts\seguiemj.ttf", "/System/Library/Fonts/Apple Color Emoji.ttc", "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", "/usr/share/fonts/noto/NotoColorEmoji.ttf")  # Apple/Noto bitmap fonts only render at size 109/160; fallback = no emoji PNG
def make_emoji(e, path, px=150):
    """Colour emoji as a PNG (Segoe UI Emoji through PIL; libass can't draw colour glyphs). False when the font is missing."""
    from PIL import Image, ImageDraw, ImageFont
    f = next((p for p in EMOJI_FONTS if Path(p).exists()), None)
    if not f: return False
    im = Image.new("RGBA", (220, 220), (0, 0, 0, 0)); ImageDraw.Draw(im).text((20, 10), e, font=ImageFont.truetype(f, 109), embedded_color=True)
    im = im.crop(im.getbbox()) if im.getbbox() else im; im.thumbnail((px, px)); im.save(path); return True
def make_wm(spec, path, color="FFFFFF"):
    """Watermark PNG: the user's logo file, or `spec` text ('@mychannel') in Poppins Bold with a dark outline."""
    from PIL import Image, ImageDraw, ImageFont
    if Path(spec).is_file(): Image.open(spec).convert("RGBA").save(path); return
    f = ImageFont.truetype(str(FONTS / "Poppins-Bold.ttf"), 64); w = int(f.getlength(spec)) + 24
    im = Image.new("RGBA", (w, 100), (0, 0, 0, 0)); ImageDraw.Draw(im).text((12, 12), spec, font=f, fill="#" + color, stroke_width=4, stroke_fill="#000000")
    im.crop(im.getbbox()).save(path)
def wm_xy(pos, W, H, y0=0):
    """overlay x/y expressions for the watermark: corners sit clear of the Shorts UI and the caption band; bounce = slow slide along the top band (a face is never up there)."""
    top = max(160 - y0, 40)
    return {"tl": (40, top), "tr": ("W-w-40", top), "bl": (40, f"H-h-{190 if H > 1500 else 60}"), "br": ("W-w-40", f"H-h-{190 if H > 1500 else 60}"),
            "bounce": ("'40+(W-w-80)*abs(mod(t/7,2)-1)'", top)}.get(pos, (40, top))

def progress_bar(dur, color="FFE500", h=12):
    """Source + overlay pieces for a bar along the bottom edge that fills left to right over the Short (track in dim black)."""
    return f"color=c=0x{color}:s=1080x{h}:r=30:d={dur:.2f}", f"overlay=x='-w+w*t/{dur:.3f}':y=H-{h}"

STACKED = False
def need_ffmpeg():  # one clear install line per OS instead of a FileNotFoundError deep in a render
    miss = [x for x in ("ffmpeg", "ffprobe") if not shutil.which(x)]
    if miss: raise SystemExit(f"{'/'.join(miss)} not found on PATH. Install ffmpeg: Windows `winget install Gyan.FFmpeg` | macOS `brew install ffmpeg` | Linux `sudo apt install ffmpeg` (then open a new terminal).")
def main(src, tjson, start, end, hook, out, keys="", style="", layout="", opts=""):
    need_ffmpeg()
    opts = opts if isinstance(opts, dict) else (json.loads(opts) if str(opts).strip().startswith("{") else {})  # pick fields: position, watermark, watermark_pos/opacity/size, progress, endcard, aspects, emoji, denoise
    brand = with_theme(load_brand(Path(out)), opts)
    style = style or os.environ.get("CLIPPER_STYLE") or brand.get("style") or "classic"
    layout = layout or os.environ.get("CLIPPER_LAYOUT", "stack")  # layout for streams with a cam: stack (cam top, game below) | bubble (game full, round cam)
    if style not in STYLES: print(f"unknown style {style!r}, using classic. Styles: {', '.join(STYLES)}"); style = "classic"
    pos = opt("position", opts, brand, "auto"); pos = pos if pos in ("auto", "top", "middle", "bottom") else "auto"
    allw = [w for s in json.load(open(tjson, encoding="utf-8"))["segments"] for w in s["words"] if w["w"]]
    _a, _b = secs(start), secs(end); talk = sum(min(w["e"], _b) - max(w["s"], _a) for w in allw if w["e"] > _a and w["s"] < _b) / max(_b - _a, 1)
    wordless = talk < 0.12  # a game-action / chat-spike moment with almost no talk (live clips): word snapping and dead-air cuts would leave ~1 s, so keep the whole window
    start, end = (_a, _b) if wordless else snap(allw, _a, _b); out = Path(out); work = out.parent / ".work" / out.stem  # temp files kept out of sight
    work.mkdir(parents=True, exist_ok=True)
    seg, v916, sub = work / "seg.mp4", work / "seg_916.mp4", work / "captions.ass"
    seek = start
    if src.startswith("http"):  # a link: fetch only this moment (full downloads 403; H.264 because OpenCV decodes AV1 slowly)
        part = work / "src.mp4"
        for attempt in range(3):  # section downloads sometimes come back with the audio cut short: check and fetch again
            for retry in range(2):  # one retry on a stalled download
                try:
                    subprocess.run([sys.executable, "-m", "yt_dlp", *CK, "-q", "--force-overwrites", "-S", "vcodec:h264,res:1080",
                                    "--download-sections", f"*{start:.2f}-{end:.2f}", "--force-keyframes-at-cuts", "--merge-output-format", "mp4", "-o", str(part), src], timeout=600); break
                except subprocess.TimeoutExpired: print(f"  DOWNLOAD TIMEOUT (600 s) for {src} {start:.0f}-{end:.0f}s" + (", retrying once" if retry == 0 else ""))
            else: raise SystemExit(f"download timed out twice (600 s each): {src}; try this Short again later")
            d = durations(part) if part.exists() else {}  # a crashed download (ffmpeg died mid-merge) is retried like a short one
            if d.get("video") and d.get("audio", 0) >= min(d["video"], end - start) - 0.3: break
            print(f"  download came back with {d.get('audio', 0):.1f}s of audio of {end - start:.1f}s, downloading again")
        else: raise SystemExit("audio keeps coming back short; try this Short again later")
        src, seek = str(part), 0
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{seek:.3f}", "-to", f"{seek + end - start:.3f}", "-i", src,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "aac", "-b:a", "192k", str(seg)], check=True, timeout=300 + 20 * (end - start))
    reframe.crop_bars(seg)  # black bars baked into the stream frame (trailer/cutscene letterbox) would show as strips or a small picture
    global CAP_Y, HOOK_Y, STACKED, KEY
    flip = truthy(opt("flip", opts, brand, "")); reframe.FLIP = flip
    KEY = str(brand.get('colors', {}).get("key", "5CFF3D")).lstrip('#')
    if layout != "full" and reframe.is_call(seg): layout = "full"; print("  video call (two tiles side by side): tile stack")
    cam = None if layout == "full" else reframe.find_facecam(seg)  # full: never look for a cam (a video-call tile or a grid is not a facecam)
    info = reframe.render_stack(seg, v916, cam, layout) if cam else None  # None = bubble asked but no real webcam box: normal framing below
    if info:  # streamer reaction: facecam on top, game below, captions on the seam
        info["shots"] = []
        if layout == "bubble": HOOK_Y = reframe.BUB_Y + reframe.BUB + 90  # hook under the bubble, captions at the style's height
        else:
            STACKED = True; CAP_Y, HOOK_Y = (reframe.stack_geom(True)["seam"] - 100, 330) if flip else (reframe.TOP_H + 120, reframe.TOP_H - 90)  # flip: captions just above the seam (the cam face is below), hook in the top part
    else:
        info, xs, mode = reframe.plan(seg)
        if layout == "full": mode = ["fill"] * len(mode); info["shots"] = [s | dict(mode="fill") for s in info["shots"]]  # whole frame over a blurred fill: nobody cut out
        reframe.render(seg, v916, info, xs, mode)
    lead = 0.0  # the first second decides: checked on the finished 9:16 (what the viewer sees), late start = trim, no re-render
    if not has_face(v916, 0):
        lead = next((t - start for t in later_starts(allw, start, end) if has_face(v916, t - start)), 0.0)
        if lead: print(f"  no face in the first second: starting {lead:.1f}s later, at the next sentence")
    lang = str(opt("lang", opts, brand, "") or "")  # caption language (es fr pt de ar hi): translate.py is only imported when one is set
    words = [dict(w=w["w"] if lang else clean(w["w"]), s=w["s"] - start, e=w["e"] - start, end=bool(re.search(r"[.?!]$", w["w"]))) for w in allw if start + lead <= w["s"] and w["e"] <= end + 0.05]
    if lang:
        import translate; words = [w | dict(w=clean(w["w"])) for w in translate.translate_words(words, lang)]; hook = translate.translate_text(hook, lang); keys = ""  # keywords are English
    words = [w for w in words if w["w"]]
    dur = durations(v916).get("video", end - start)
    wav_db = audio_db(v916); speech = speech_db(words, wav_db)
    tot = lambda s: sum(b - a for a, b in s)
    sq = float(opt("squeeze", opts, brand, 0) or 0)  # waitforit: talk-free stretches over `sq` s are jump-cut to 0.2 s each side even over loud game/call sound
    def cut(long, quiet, trim=True):  # long=None is the back-off cut: no squeeze either
        s = [(max(a, lead), b) for a, b in (keep_spans(words, wav_db, dur, long=sq, loud=SQ_LOUD, hold=0.2) if sq and long else keep_spans(words, wav_db, dur, long=long)) if b > lead]  # drop the faceless opening
        s = quiet_cut(s, wav_db, speech - QDB) if quiet else s
        return tail_trim(s, words, wav_db) if trim and not sq else s  # nothing runs on 2 s past the payoff; waitforit (sq) ends on its reveal by design, never trimmed
    # tightest cut that still leaves MIN_LEN s (or the whole moment if it is shorter): dead air goes first, a too-thin Short backs off
    cs = [cut(LONG, True), cut(None, True)]; full = cut(None, False, False)
    spans = next((s for s in cs + [cut(None, False)] if tot(s) >= min(MIN_LEN, tot(full) - 0.01)), full); backed = spans is not cs[0]
    if wordless: spans, backed = [(0.0, dur)], False; print(f"WORDLESS: {talk:.0%} talk, kept the whole {dur:.0f} s window", flush=True)  # backed = the tight cut was under MIN_LEN, long gaps stayed in
    if spans is full and tot(cut(None, False)) < tot(full) - 0.01: print(f"THIN: {tot(full) - tot(cut(None, False)):.1f} s of tail after the payoff stays (trimming would go under {MIN_LEN:.0f} s); pick it a sentence later or drop it")
    elif tot(spans) < tot(cut(LONG, True, False)) - 0.01: print(f"  TAIL: cut {tot(cut(LONG, True, False)) - tot(spans):.1f} s after the payoff")
    fl = fillers(words) if truthy(opt('fillers', opts, brand, '1')) else []
    if fl: spans = cut_out(spans, [(words[i]['s'] - .04, words[i]['e'] + .04) for i in fl]); print('  filler words cut:', [words[i]['w'] for i in fl]); words = [w for i, w in enumerate(words) if i not in fl]
    n0 = len(words); words = heard(words, spans)  # a word that sits in a cut hole is not heard: no caption for it (it landed on the previous line before)
    if len(words) < n0: print(f'  {n0 - len(words)} caption words dropped (inside cut dead air)')
    def to_cut(w):  # clamp the word into the kept span holding its midpoint, then map to cut time (a word straddling a cut edge must not get e < s)
        m = (w["s"] + w["e"]) / 2
        for k, (a, b) in enumerate(spans):
            if a - 0.02 <= m <= b + 0.02:
                o = sum(y - x for x, y in spans[:k]); s0 = min(max(w["s"], a), b); e0 = min(max(w["e"], a), b)
                s1, e1 = s0 - a + o, e0 - a + o; assert 0 <= s1 <= e1 + 1e-9, (w, a, b)
                return w | dict(s=s1, e=max(e1, s1 + 0.05))
        return None  # fully inside cut audio
    words = [x for x in map(to_cut, words) if x]
    gap_max = longest_gap(words, sum(b - a for a, b in spans))
    if backed and gap_max >= 1.0: print(f"THIN: kept a {gap_max:.1f} s talk-free gap to stay over {MIN_LEN:.0f} s; widen the pick by a sentence or drop it")
    new_dur = sum(b - a for a, b in spans)
    punch, hit = [], None
    if words:  # punch-ins at reaction peaks, only where the layout is the zoomable one (stack: seam rows black; crop: not a fill/split shot)
        orig = lambda t: next((a + t - sum(y - x for x, y in spans[:k]) for k, (a, b) in enumerate(spans) if t <= sum(y - x for x, y in spans[:k + 1])), t)  # cut time -> v916 time
        cap, fps = cv2.VideoCapture(str(v916)), info.get("fps") or 30
        def zoomable(t):
            for u in (t - .15, t + 1, t + 2.3):
                o = orig(min(max(u, 0), new_dur))
                if cam: cap.set(cv2.CAP_PROP_POS_MSEC, o * 1000); ok, f = cap.read(); good = ok and f[(1150 if flip else 766):(1152 if flip else 768)].max() < 12
                else: good = all(s["mode"] == "crop" for s in info["shots"] if s["start"] <= o * fps < s["end"])
                if not good: return False
            return True
        cut_db = np.concatenate([wav_db[int(a * 20):int(b * 20)] for a, b in spans])
        for t in peaks(cut_db, speech, new_dur, zoomable):
            a = t - .15; se = next((w["e"] for w in words if w["end"] and w["e"] >= t), a + 2)  # hold to the end of the sentence, 1-2.5 s
            b = min(max(se, a + 1), a + 2.5, new_dur - ZOUT - .05)
            if sum(a <= w["s"] <= b for w in words) >= 2: punch.append((a, b))  # a zoom on a lone "Yo" reads as a glitch
        cap.release()
        if punch: hit = max(punch, key=lambda p: cut_db[int((p[0] + .15) * 20):int((p[0] + .15) * 20) + 4].max(initial=-99))[0]  # the loudest of them = the payoff
        print("punch-ins (cut time, said):", [(round(float(a), 2), round(float(b), 2), " ".join(w["w"] for w in words if a <= w["s"] <= b)) for a, b in punch])
    kset = {re.sub(r"\W", "", k.upper()) for k in keys.split(",") if k.strip()}
    st = brand_style(STYLES[style], brand)
    LF = {"ar": "Noto Sans Arabic", "hi": "Noto Sans Devanagari"}.get(lang)  # scripts Montserrat lacks: Noto Bold for captions and hook
    if LF: st = (str(FONTS / (LF.replace(" ", "") + "-Bold.ttf")), LF, -1, int(st[3] * 1.3)) + st[4:]  # Noto sets smaller than the Latin faces: 30 % bigger
    cs = cap_scale(opt("cap_size", opts, brand, 1))
    if cs != 1: st = list(st); st[3] = int(st[3] * cs); st[8] = st[8] * cs; st[11] = max(1, int(st[11] / cs + .5)); st = tuple(st)  # font, outline, words per line
    hookc = {"text": str(brand.get("hook_text", "FFFFFF")).lstrip("#"), "box": str(brand.get("hook_box", "000000")).lstrip("#")}
    ec = opt("endcard", opts, brand, ""); emo = [] if (style == "emoji" or truthy(opt("emoji", opts, brand, ""))) else None
    asp = [a for a in (opt("aspects", opts, brand, []) if isinstance(opt("aspects", opts, brand, []), list) else str(opt("aspects", opts, brand, "")).replace(" ", "").split(",")) if a in ASPECTS]
    branches = [("", 1920, 0)] + [("_" + a.replace(":", "x"), ASPECTS[a][1], (1920 - ASPECTS[a][1] if flip else 0) if STACKED else ASPECTS[a][2]) for a in asp]  # stacked: window from the top (the cam face stays whole)
    for k, (suf, H, y0) in enumerate(branches):  # one caption file per export: the 9:16 and each cropped window (positions shifted into it)
        (work / f"cap{k}.ass").write_text(ass(words, hook, new_dur, kset, style, pos, H, y0, ec, hookc, emo if k == 0 else None, st).replace(FONT, LF or FONT).replace(f"Style: Hook,{LF},76", f"Style: Hook,{LF},104"), encoding="utf-8")
    # ass filter wants plain relative paths on Windows: font copied next to the captions, ffmpeg run from the work folder
    shutil.copy(FONT_FILE, work); shutil.copy(fpath(st), work); LF and shutil.copy(fpath(st), work)  # hook font + caption font
    ln = "loudnorm=I=-14:TP=-1.5:LRA=11"; dn = "afftdn=nr=8:nf=-45," if truthy(opt("denoise", opts, brand, "1")) else ""  # light voice noise reduction
    cuts = "".join(f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS[v{i}];[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS[a{i}];" for i, (a, b) in enumerate(spans))
    acuts = "".join(f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS[a{i}];" for i, (a, b) in enumerate(spans))
    mgraph = acuts + "".join(f"[a{i}]" for i in range(len(spans))) + f"concat=n={len(spans)}:v=0:a=1[ac];[ac]{dn}{ln}:print_format=json"
    try:  # linear two-pass loudnorm avoids dynamic-mode pumping on speech
        r = subprocess.run(["ffmpeg", "-v", "info", "-i", v916.name, "-filter_complex", mgraph, "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, cwd=work, timeout=600 + 20 * dur)
        m = loudnorm_stats(r.stderr)
        ln += f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true"
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError): pass
    frz = ""  # a game-over black flash under 0.8 s: hold the last good frame over it (video only, the audio keeps its sync)
    if truthy(opt("blackfix", opts, brand, "1")):
        runs, bfps = black_runs(v916); frz = freeze_filters(runs, bfps, spans)
        if runs: print(f"  black flash held: {[(round(i / bfps, 2), round((j + 1) / bfps, 2)) for i, j in runs]}")
    # cam panel too dark (caseOh): gamma lift on the top panel only, the game panel stays as is
    lift = ""
    if cam and layout != "bubble" and truthy(opt("lift", opts, brand, "1")):
        lm = cam_luma(v916, flip=flip); g = lift_gamma(lm)
        if g: cy = 1156 if flip else 0; lift = f"split[l0][l1];[l0]crop=1080:764:0:{cy},eq=gamma={g}:brightness=0.02[ll];[l1][ll]overlay=0:{cy},"; print(f"  dark cam panel (luma {lm:.0f}): gamma {g} on the cam panel")
    # dark game footage (critics: "the game footage is dark"): gentle gamma lift on the game region only (same switch: lift=0)
    if truthy(opt("lift", opts, brand, "1")):
        stk = bool(cam and layout != "bubble"); gm = game_luma(v916, stk, flip=flip); gg = game_gamma(gm)
        if gg:
            eqf = f"eq=gamma={gg}:contrast=1.04"
            gy = 0 if flip else 768; lift += (f"split[g0][g1];[g0]crop=1080:1152:0:{gy},{eqf}[gl];[g1][gl]overlay=0:{gy}," if stk else f"{eqf},")
            print(f"  dark game footage (luma {gm:.0f}): gamma {gg} on the game panel")
    # overlays per export: watermark, emoji pops, progress bar (inputs after the video: indexes count up)
    extra, wmspec = [], str(opt("watermark", opts, brand, "") or "")
    wmp = None
    if wmspec: wmp = work / "wm.png"; make_wm(wmspec, wmp, str(opt("watermark_color", opts, brand, "FFFFFF")).lstrip("#"))
    emos = {}
    for t, e in (emo or []):
        if e not in emos: emos[e] = work / f"emo{len(emos)}.png"; emos[e] = emos[e] if make_emoji(e, emos[e]) else None
    wmw = int(1080 * float(opt("watermark_size", opts, brand, 0.22))); wmo = float(opt("watermark_opacity", opts, brand, 0.7)); wpos = str(opt("watermark_pos", opts, brand, "tl"))
    bar = truthy(opt("progress", opts, brand, "")); pcol = str(opt("progress_color", opts, brand, "FFE500")).lstrip("#")
    def deco(k, H, y0, src_lab, dst):
        f, cur = "", src_lab
        def ov(png, chain, over):
            nonlocal f, cur
            extra.extend(["-i", Path(png).name])  # one frame: overlay repeats it (a looped input deadlocked ffmpeg with many trim/concat segments)
            n = f"x{k}_{len(extra)}"; f += f"[{len(extra) // 2}:v]{chain}[{n}n];[{cur}][{n}n]{over}[{n}];"; cur = n
        if wmp:
            x, y = wm_xy(wpos, 1080, H, y0); ov(wmp, f"format=rgba,scale={wmw}:-1,colorchannelmixer=aa={wmo}", f"overlay=x={x}:y={y}")
        for t, e in (emo or []):
            if emos.get(e):
                Y = max(my_y(H, y0) - 235, 200 - y0 if y0 else 200)
                ov(emos[e], "format=rgba", f"overlay=x=(W-w)/2:y='{Y}+50*max(0,1-(t-{t:.2f})*7)':enable='between(t,{t:.2f},{t + 1.0:.2f})'")
        if bar:
            s, o = progress_bar(new_dur, pcol); f += f"{s}[bar{k}];[{cur}]drawbox=x=0:y=ih-12:w=iw:h=12:color=black@0.35:t=fill[tr{k}];[tr{k}][bar{k}]{o}[{dst}];"; cur = dst
        return f + (f"[{cur}]null[{dst}];" if cur != dst else "")
    my_y = lambda H, y0: (lambda v: int(min(max(v - y0, 150), H - 170)) if (y0 or H != 1920) else int(v))(cap_y(st, pos))
    nb = len(branches); flx = flash(hit) if truthy(opt('flash', opts, brand, '')) else ''  # the white flash on the peak zoom read as a glitch to critics: off unless asked
    fg = (cuts + "".join(f"[v{i}][a{i}]" for i in range(len(spans))) + f"concat=n={len(spans)}:v=1:a=1[vc][ac];"
          + (f"[vc]{frz}{lift}null[vc2];" if (lift or frz) else "[vc]null[vc2];")
          + (f"[vc2]{zoom_filter(punch, bool(cam), hit, flip)}{flx}[vz0];" if punch else "[vc2]null[vz0];")  # zoom first: captions and hook never zoom
          + f"[vz0]{GRADE}[vz];"
          + (f"[vz]split={nb}" + "".join(f"[s{k}]" for k in range(nb)) + ";" if nb > 1 else "[vz]null[s0];"))
    for k, (suf, H, y0) in enumerate(branches):
        fg += f"[s{k}]" + (f"crop=1080:{H}:0:{y0}," if H != 1920 else "") + f"ass=cap{k}.ass:fontsdir=.[c{k}];" + deco(k, H, y0, f"c{k}", f"o{k}")
    fg += f"[ac]{dn}{ln}[an];" + whoosh(punch) + f"afade=t=out:st={max(new_dur - 0.08, 0):.2f}:d=0.08" + (f",asplit={nb}" + "".join(f"[a{k}]" for k in range(nb)) if nb > 1 else "[a0]") + ";"  # hard cut on the punchline (refs end on the peak, no fade); 80 ms only to avoid a click
    outs = []
    for k, (suf, H, y0) in enumerate(branches): outs += ["-map", f"[o{k}]", "-map", f"[a{k}]", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str((out.parent / (out.stem + suf + out.suffix)).resolve())]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", v916.name, *extra, "-filter_complex", fg.rstrip(";"), *outs], check=True, cwd=work, timeout=600 + 30 * new_dur)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", str(out.resolve()), "-frames:v", "1", "-q:v", "2", str(out.with_name(out.stem + "_cover.jpg").resolve())])  # fresh cover (frame at 1 s) for a Short re-cut with clip.py alone
    cut = dur - new_dur - lead
    json.dump(dict(length=round(new_dur, 2), source_start=round(start + lead, 2), seg_start=round(start, 3), source_end=round(end, 2), kept_spans=[[round(a, 2), round(b, 2)] for a, b in spans], dead_air_cut=round(cut, 2),
                   longest_talk_free_gap=round(gap_max, 2), thin=bool(backed and gap_max >= 1.0), punch_ins=[[round(float(a), 2), round(float(b), 2)] for a, b in punch], style=style, layout=layout, lang=lang or "", caption_y=int(cap_y(st, pos)), stacked=STACKED,
                   watermark=wmspec, position=pos, aspects=asp, filler_cut=len(fl), black_held=[[round(i / bfps, 2), round((j + 1) / bfps, 2)] for i, j in runs] if frz else []),
              open(out.with_name(out.stem + ".render.json"), "w", encoding="utf-8"), indent=1)  # qc.py reads this
    print("ok", out, f"{new_dur:.1f}s ({start:.1f}-{end:.1f}, {cut:.1f}s dead air cut),", len(words), "caption words,",
          sum(1 for s in info["shots"] if s["mode"] == "fill"), "fill shots")
    shutil.rmtree(work, ignore_errors=True)  # ~100 MB of temp video per Short

if __name__ == "__main__":
    assert [len(g) for g in groups([dict(w="a", s=0, e=.2), dict(w="b", s=.3, e=.5), dict(w="c", s=2, e=2.2)])] == [2, 1], "pause must split lines"
    T = [dict(w=x, s=i * .3, e=i * .3 + .2, end=x.endswith(".")) for i, x in enumerate("in the bottom of the ocean bro. next one".split())]
    assert [" ".join(x["w"] for x in g) for g in groups(T)] == ["in the bottom", "of the ocean bro.", "next one"], groups(T)
    W = [dict(w="So", s=1.0, e=1.2), dict(w="this", s=1.3, e=1.5), dict(w="is", s=1.6, e=1.8), dict(w="crazy.", s=1.9, e=2.4), dict(w="Next", s=3.0, e=3.2)]
    assert later_starts(W, 0.5, 20, keep=1) == [2.8] and later_starts(W, 0.5, 3.5) == [], later_starts(W, 0.5, 20, keep=1)
    assert later_starts([dict(w="yo", s=1.0, e=1.2), dict(w="look", s=1.3, e=1.5), dict(w="bro", s=2.0, e=2.2)], 0.5, 20, keep=1) == [1.8]  # no full stops: a 0.5 s pause still counts
    assert snap(W, 0.5, 1.7) == (0.8, 2.65), snap(W, 0.5, 1.7)  # starts at the first word, ends after "crazy." not mid-sentence
    U = [dict(w="x", s=i * .3 + (.5 if i > 45 else 0), e=i * .3 + .25 + (.5 if i > 45 else 0)) for i in range(70)]  # unpunctuated, one 0.5 s pause after word 45
    assert abs(snap(U, 0, 17.0)[1] - 14.0) < .01, snap(U, 0, 17.0)  # mid-sentence pick, no stop within 4 s: back off to the pause (was 17.05 mid-run)
    assert clean("shit,") == "s***" and clean("crazy.") == "crazy" and clean("Dickens") == "Dickens"
    _w = [dict(w="yo", s=2.0, e=2.3, end=False), dict(w="look", s=2.4, e=2.7, end=True)]
    assert keep_spans(_w, np.full(200, 60.0), 10.0)[0][0] == 1.85, keep_spans(_w, np.full(200, 60.0), 10.0)  # 2 s of game sound before the first word: dropped
    _g = [dict(w="go", s=1.0, e=1.3, end=True), dict(w="yes", s=6.0, e=6.3, end=True)]  # 4.7 s without words over steady game sound (60 dB, voice peaks 60)
    assert keep_spans(_g, np.full(200, 60.0), 8.0, long=LONG) == [(0.85, 1.75), (5.55, 8.0)] and keep_spans(_g, np.full(200, 60.0), 8.0) == [(0.85, 8.0)]
    _s = np.full(400, 60.0); _s[300] = 72; _sw = [dict(w="a", s=0.0, e=1.0, end=True), dict(w="b", s=18.0, e=19.0, end=True)]  # voice 0-1 s, bang at 15 s, voice 18 s: squeeze keeps the bang
    assert keep_spans(_sw, _s, 20.0, long=1.5, loud=SQ_LOUD, hold=0.2) == [(0.0, 20.0)] and len(keep_spans(_sw, np.full(400, 60.0), 20.0, long=1.5, loud=SQ_LOUD, hold=0.2)) == 2  # steady sound still squeezed
    _b = np.full(200, 60.0); _b[70] = 70  # a crash louder than the voice inside the gap: keep it all
    assert keep_spans(_g, _b, 8.0, long=LONG) == [(0.85, 8.0)]
    _tw = [dict(w="a", s=1.0, e=1.5, end=False), dict(w="hard", s=2.0, e=2.5, end=True), dict(w="sweet", s=6.0, e=6.5, end=True), dict(w="new", s=7.0, e=7.5, end=True)]
    _td = np.full(200, 50.0); _td[40:50] = 75  # loudest second at 2-2.5 s ("hard"), 5 s of talk after it
    assert tail_trim([(0.0, 10.0)], _tw, _td) == [(0.0, 2.75)], tail_trim([(0.0, 10.0)], _tw, _td)
    _te = _td.copy(); _te[150:170] = 74  # a second within 2 dB of the peak later on: nothing is trimmed
    assert tail_trim([(0.0, 10.0)], _tw, _te) == [(0.0, 10.0)], tail_trim([(0.0, 10.0)], _tw, _te)
    assert tail_trim([(0.0, 4.0)], _tw, _td) == [(0.0, 4.0)], "tail of 1.5 s is kept"
    assert word(dict(w="kfc"), False, {"KFC"}).startswith(r"{\c&H3DFF5C&}KFC") and word(dict(w="a"), False, {"KFC"}) == "A"
    db = np.full(400, -30.0); db[100:104] = 8; db[150:153] = 3; db[10:13] = 9; db[300:304] = 5  # loud spots at 5 s, 7.5 s (too close to 5 s), 0.5 s (the hook), 15 s
    assert [round(t) for t in peaks(db, -20, 20)] == [5, 15], peaks(db, -20, 20)  # the hook one and the one within 6 s of a louder one are out
    assert peaks(db, 10, 20) == [], "nothing above speech level = no punch-in"
    assert whoosh([]) == "[an]" and whoosh([(1.0, 2.0)]).endswith("[an][s0]amix=inputs=2:normalize=0:duration=first,") and "adelay=800:" in whoosh([(1.0, 2.0)])
    _d = np.concatenate([np.full(20, 70.0), np.full(30, 20.0), np.full(20, 70.0), np.full(10, 20.0), np.full(20, 70.0), np.full(40, 20.0)])  # sound 0-1, quiet 1-2.5, sound 2.5-3.5, short quiet 3.5-4, sound 4-5, quiet 5-7
    _q = quiet_cut([(0.0, 7.0)], _d, 50.0)
    assert [(round(a, 2), round(b, 2)) for a, b in _q] == [(0.0, 1.1), (2.4, 3.6), (3.9, 5.12)], _q  # 1.5 s gap -> 0.2 s, 0.5 s gap -> 0.2 s, tail 2 s -> 0.12 s
    _h = quiet_cut([(0.0, 2.0)], np.concatenate([np.full(4, 10.0), np.full(36, 70.0)]), 50.0); assert [round(x, 2) for x in _h[0]] == [0.15, 2.0], _h  # 0.2 s of dead air before the first sound
    assert quiet_cut([(0.0, 3.5)], _d[:70], 50.0) == [(0.0, 1.1), (2.4, 3.5)] and quiet_cut([(0.0, 2.0)], np.full(40, 70.0), 50.0) == [(0.0, 2.0)]
    sample = 'noise {"old": "block"}\n{\n "input_i" : "-23.50", "input_tp" : "-4.20", "input_lra" : "7.10", "input_thresh" : "-34.00", "target_offset" : "0.30"\n}\n'
    assert loudnorm_stats(sample) == {"input_i": "-23.50", "input_tp": "-4.20", "input_lra": "7.10", "input_thresh": "-34.00", "target_offset": "0.30"}
    assert all(len(s) == 13 and (FONTS / f"{s[0]}.ttf").exists() for s in STYLES.values()), "every style needs its 13 fields and its font file"
    assert "fscx109" in word(dict(w="go"), True, set(), STYLES["hormozi"]) and "3c&H4AA316" in word(dict(w="go"), True, set(), STYLES["mozi"]) and "fscx112" in word(dict(w="go"), True, set())
    # --- round 18: caption styles, position, brand, watermark, progress, aspects, lift, fillers ---
    _T = {t["name"] for t in load_themes()}; assert _T == {"clean", "nocodehack", "neon", "bold-yellow", "minimal-white"} and all(load_themes()[0]["brand"]["style"] in STYLES for _ in [0]) and all(t["brand"]["style"] in STYLES for t in load_themes()), _T  # themes: 5 names, every style real
    _b = {"hook_box": "123456", "colors": {"spoken": "ABCDEF"}}; _m = with_theme(_b, {"theme": "nocodehack", "progress_color": "00FF00"})  # theme < brand.json < pick
    assert _m["hook_box"] == "123456" and _m["progress_color"] == "00FF00" and _m["colors"]["spoken"] == "ABCDEF" and _m["colors"]["text"] == "FFF6E3" and _m["style"] == "hormozi", _m
    assert with_theme({}, {}) == {"colors": {}} and with_theme({}, {"theme": "clean"})["style"] == "classic"
    try: with_theme({}, {"theme": "nope"}); raise AssertionError("unknown theme must fail")
    except SystemExit as _e: assert "unknown theme" in str(_e) and "neon" in str(_e)
    assert reframe.stack_geom(True)["seam"] == 1152 and "1156" in zoom_filter([(1, 2)], True, None, True) and "1156" not in zoom_filter([(1, 2)], True)  # flip: cam panel below the game
    assert len(STYLES) >= 32 and set(ANIM) <= set(STYLES), "14 old + 18 new styles, every animated one is a style"
    _W = [dict(w=x, s=i * .35, e=i * .35 + .3, end=False) for i, x in enumerate("I just won a million dollars look at this crazy game".split())]
    for _n in STYLES:  # every style writes a caption file: header + hook + one Dialogue per spoken word (or per line)
        _e = []; _a = ass(_W, "hook", 6, {"MILLION"}, _n, "auto", 1920, 0, "FOLLOW", None, _e)
        assert _a.count("Dialogue:") >= 4 and "Style: Cap," in _a and "\\pos(" in _a, _n
    _a = ass(_W, "hook", 6, set(), "typewriter"); assert "\\alpha&HFF&\\t(" in _a                     # chars appear one by one
    _a = ass(_W, "hook", 6, {"MILLION"}, "shake"); assert "\\frz-4" in _a                             # keyword wobble
    _a = ass(_W, "hook", 6, set(), "slide"); assert "\\move(540," in _a and "\\fad(90,0)" in _a
    _a = ass(_W, "hook", 6, set(), "sweep"); assert "\\clip(" in _a and "\\1a&HFF&\\bord22" in _a      # box sweeps behind the spoken word
    _a = ass(_W, "hook", 6, set(), "twotone"); assert _a.count("Dialogue: 2,") >= 4                   # second layer, top half
    _a = ass(_W, "hook", 6, set(), "duo"); assert "\\kf" in _a and "\\N" in _a
    _a = ass(_W, "hook", 6, set(), "meme"); assert "\\pos(540,520)" in _a and "\\pos(540,1380)" in _a  # top and bottom text
    _a = ass(_W, "hook", 6, set(), "glow"); assert "\\blur14" in _a
    assert [cap_scale(x) for x in ("S", "l", 1.1, 9, "x", None)] == [.85, 1.2, 1.1, 1.5, 1.0, 1.0]
    _a = ass(_W, "hook", 6, set(), "hollow"); assert "Style: Cap,Bebas Neue,184,&HFF" in _a
    _e = []; ass(_W, "hook", 6, set(), "emoji", emoji=_e); assert _e and _e[0][1] == "🤯" and _e[0][0] >= HOOK_S, _e  # none under the hook title
    assert EMOJI["money"] == "💰" and all(t2 - t1 >= EMOJI_GAP for (t1, _), (t2, _) in zip(_e, _e[1:]))
    if any(Path(p).exists() for p in EMOJI_FONTS):
        import tempfile; _p = Path(tempfile.gettempdir()) / f"clip_selftest_emoji_{os.getpid()}.png"; assert make_emoji("🔥", _p) and _p.stat().st_size > 500; _p.unlink()
    assert "FOLLOW FOR MORE" in ass(_W, "hook", 6, set(), "classic", endcard="follow for more") and "Cta," not in ass(_W, "hook", 6, set(), "classic").split("[Events]")[1]
    # caption position: top / middle / bottom, auto = style height; stacked layout keeps the seam unless middle/bottom is asked
    assert [cap_y(STYLES["classic"], p, False) for p in ("auto", "top", "middle", "bottom")] == [1250, 450, 960, 1440] and cap_y(STYLES["classic"], "top", True) == CAP_Y and cap_y(STYLES["classic"], "middle", True) == 960
    assert "\\pos(540,960)" in ass(_W, "hook", 6, set(), "classic", "middle")
    # brand.json: pick > env > brand > default; colours and font put into the style
    assert opt("style", {"style": "pop"}, {"style": "word"}, "classic") == "pop" and opt("style", {}, {"style": "word"}, "classic") == "word" and opt("style", {}, {}, "classic") == "classic"
    _s = brand_style(STYLES["hormozi"], {"colors": {"spoken": "#00ff00", "outline": "112233"}, "font": "Bangers-Regular"})
    assert _s[6] == "00FF00" and _s[7] == "112233" and _s[1] == "Bangers" and fpath(_s).exists() and _s[3] != STYLES["hormozi"][3], _s
    assert brand_style(STYLES["pill"], {"colors": {"box": "FF0000"}})[10] == "FF0000" and brand_style(STYLES["hormozi"], {"colors": {"box": "FF0000"}})[10] is None
    assert truthy("1") and not truthy("0") and not truthy("") and not truthy("off")
    # watermark: corners clear of the UI, bounce slides along the top band
    assert wm_xy("tl", 1080, 1920) == (40, 160) and wm_xy("br", 1080, 1920)[0] == "W-w-40" and "mod(t/7,2)" in wm_xy("bounce", 1080, 1920)[0] and wm_xy("tl", 1080, 1350, 190)[1] == 40
    import tempfile; _p = Path(tempfile.gettempdir()) / f"clip_selftest_wm_{os.getpid()}.png"; make_wm("@mychannel", _p); assert _p.stat().st_size > 500; _p.unlink()
    assert progress_bar(10.0)[1] == "overlay=x='-w+w*t/10.000':y=H-12"
    assert ASPECTS["4:5"][:2] == (1080, 1350) and ASPECTS["1:1"][:2] == (1080, 1080)
    _a = ass(_W, "hook", 6, set(), "meme", "auto", 1080, 320); assert "\\pos(540,520)" not in _a  # cropped export: positions shifted into the window
    assert all(150 <= int(m) <= 1080 - 170 for m in re.findall(r"\\pos\(540,(\\d+)\)", ass(_W, "hook", 6, set(), "classic", "top", 1080, 320)))
    # dark cam panel lift, filler trim, voice denoise
    assert game_gamma(100) is None and game_gamma(79) is not None and 1.1 <= game_gamma(31) <= 1.5 and game_gamma(31) >= game_gamma(64)
    assert lift_gamma(120) is None and lift_gamma(69.9) is not None and lift_gamma(40) > lift_gamma(60) and lift_gamma(2) == 2.2 and 1.1 <= lift_gamma(69) <= 2.2
    _F = [dict(w="So", s=0, e=.2, end=False), dict(w="um", s=.6, e=.8, end=False), dict(w="we", s=1.2, e=1.4, end=False), dict(w="um", s=1.45, e=1.6, end=False), dict(w="go.", s=1.65, e=1.9, end=True),
          dict(w="Like", s=2.4, e=2.6, end=False), dict(w="this", s=3.0, e=3.2, end=True), dict(w="like", s=3.3, e=3.4, end=False), dict(w="it", s=3.41, e=3.6, end=False)]
    assert fillers(_F) == [1, 5], fillers(_F)  # the um glued to its neighbours stays, 'Like' after a full stop with a pause goes, mid-sentence 'like' stays
    assert cut_out([(0, 4)], [(.56, .84)]) == [(0, .56), (.84, 4)] and cut_out([(0, 1), (2, 3)], [(.5, 2.5)]) == [(0, .5), (2.5, 3)] and cut_out([(0, 1)], [(5, 6)]) == [(0, 1)]
    _l = [100] * 5 + [3] * 12 + [100] * 5 + [2] * 30 + [100] * 3 + [4] * 4  # 12 black frames at 30 fps = 0.4 s flash, 30 = 1 s (stays), the last 4 touch the end (stays)
    assert runs_from_luma(_l, 30) == [(5, 16)] and runs_from_luma([2] * 5 + [100] * 9, 30) == [] and runs_from_luma([100, 100, 100], 30) == []
    assert runs_from_luma([100] * 40 + [25] * 6 + [100] * 40, 30) == [(40, 45)] and runs_from_luma([30] * 90, 30) == []  # a near-black dip in a bright scene is held; a steadily dark scene (tunnel) is not
    assert "first=5:last=16:replace=4" in freeze_filters([(5, 16)], 30, [(0, 1.0)]) and freeze_filters([(75, 86)], 30, [(0, 1.0)]) == ""
    assert freeze_filters([(75, 86)], 30, [(0, 1.0), (2.0, 5.0)]) == "split[fz45a][fz45b];[fz45a][fz45b]freezeframes=first=45:last=56:replace=44,", freeze_filters([(75, 86)], 30, [(0, 1.0), (2.0, 5.0)])
    assert longest_gap([dict(s=1.0, e=1.4), dict(s=1.5, e=2.0)], 3.0) == 1.0 and longest_gap([dict(s=0.2, e=1.0), dict(s=1.1, e=1.5)], 6.5) == 5.0 and longest_gap([], 4.0) == 4.0
    assert [w["w"] for w in heard([dict(w="a", s=0, e=.4), dict(w="b", s=1.2, e=1.6), dict(w="c", s=2.1, e=2.5)], [(0, 1), (2, 3)])] == ["a", "c"]
    if len(sys.argv) < 7: print("clip.py selftest ok; usage: clip.py SRC TRANSCRIPT.json START END HOOK OUT [keys style layout opts]"); sys.exit()
    main(*sys.argv[1:11])

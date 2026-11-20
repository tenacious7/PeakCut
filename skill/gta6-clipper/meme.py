# Meme layer: a sound + visual gag at the payoff of a finished Short.
#   python meme.py SHORT.mp4 --mood embarrassing [--at SEC] [--out OUT.mp4] [--sheet SHEET.jpg]
#   python meme.py add NAME URL_OR_FILE [START-END]   # your own copy of a famous song -> sounds/mine/NAME.m4a
#   python meme.py sounds                              # (re)synthesise the shipped sounds (all ours, no samples)
#   python meme.py bed SHORT.mp4 --mood hype|chill|tense|funny [--file F] [--db -14] [--out OUT]   # low music bed, ducks under speech
#   python meme.py selftest
# sounds/moods.json: mood -> {sound: name or [names, first that exists wins], effect: [...], volume: dB, offset: s, hold: s, tail: s|"auto"}
# effects: zoom (slow push-in) punch (quick zoom) shake flash vignette desat | hold = freeze-frame seconds at the payoff
import json, os, subprocess, sys, wave, tempfile, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SND = HERE / "sounds"
SR = 48000
HOOK_SAFE = 3.0  # never put a gag in the first seconds (the hook)
MAX_EXT = 3.0    # a meme ending may extend the Short by at most this

def run(cmd, timeout=900, **k):
    """subprocess with a timeout (download or render stalled -> one clear line, rc 124, the Short is kept as is)."""
    try: return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, **k)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {Path(str(cmd[0])).name} stalled for {timeout} s, stopped", flush=True); return subprocess.CompletedProcess(cmd, 124, "", f"timeout after {timeout} s")
def probe(f):
    o = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration", "-of", "json", str(f)])
    j = json.loads(o.stdout); return j["streams"][0]["width"], j["streams"][0]["height"], float(j["format"]["duration"])

# ---------------------------------------------------------------- synthesis
def _t(d): return np.arange(int(d * SR)) / SR
def _env(n, a=0.005, r=0.05):
    e = np.ones(n); na, nr = max(int(a * SR), 1), max(int(r * SR), 1); e[:na] = np.linspace(0, 1, na); e[-nr:] = np.minimum(e[-nr:], np.linspace(1, 0, nr)); return e
def _lp_fast(x, fc, n=3):  # FFT low-pass
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); X *= 1 / (1 + (f / fc) ** (2 * n)); return np.fft.irfft(X, len(x))
def _bp(x, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); X *= ((f > lo) & (f < hi)); return np.fft.irfft(X, len(x))
def _reverb(x, tail=0.5, mix=0.25, seed=1):
    r = np.random.default_rng(seed); n = int(tail * SR); ir = r.standard_normal(n) * np.exp(-np.arange(n) / (SR * tail / 5)); ir = _lp_fast(ir, 3000)
    y = np.fft.irfft(np.fft.rfft(x, len(x) + n) * np.fft.rfft(ir, len(x) + n), len(x) + n); y = y / (np.abs(y).max() + 1e-9)
    z = np.zeros(len(y)); z[: len(x)] = x; return z * (1 - mix) + y * mix * np.abs(x).max()
def _norm(x, peak_db=-3.0): return x / (np.abs(x).max() + 1e-9) * 10 ** (peak_db / 20)

def vine_boom():
    t = _t(1.8); f = 38 + 120 * np.exp(-t * 14); ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.tanh(2.2 * np.sin(ph)) * np.exp(-t * 2.6) + 0.35 * np.sin(2 * ph) * np.exp(-t * 6)
    x += 0.5 * _bp(np.random.default_rng(3).standard_normal(len(t)), 80, 900) * np.exp(-t * 30)  # impact thump
    return _norm(_reverb(x * _env(len(t), 0.002, 0.2), 0.6, 0.18))
def sad_trombone():
    notes = [(233.1, .55), (220.0, .55), (207.7, .55), (196.0, 1.5)]; out = []
    for i, (f0, d) in enumerate(notes):
        t = _t(d); last = i == 3
        f = f0 * (1 + (0.004 * np.sin(2 * np.pi * 5.5 * t) if last else 0)) * (1 - 0.06 * np.clip((t - 0.5) / 1.0, 0, 1) if last else 1)
        ph = 2 * np.pi * np.cumsum(f) / SR; wah = 0.25 + 0.75 * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 0.7  # mute opens then closes
        x = sum(np.sin(k * ph) * k ** -1.0 * np.exp(-k / (1 + 9 * wah)) for k in range(1, 22))
        x = x + 0.04 * np.random.default_rng(i).standard_normal(len(t))  # breath
        out.append(x * _env(len(t), 0.04, 0.12 if not last else 0.35) * (0.6 + 0.4 * wah)); out.append(np.zeros(int(0.05 * SR)))
    x = np.concatenate(out); return _norm(_reverb(_bp(_lp_fast(x, 2200), 80, 9000), 0.4, 0.15))
def record_scratch():
    t = _t(0.75); rng = np.random.default_rng(5)
    f = 900 * np.exp(-t * 2.2) * (1 + 0.5 * np.sin(2 * np.pi * 9 * t)); ph = 2 * np.pi * np.cumsum(f) / SR
    saw = 2 * ((ph / (2 * np.pi)) % 1) - 1; noise = _bp(rng.standard_normal(len(t)), 1500, 9000)
    x = (0.6 * saw + 0.9 * noise) * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 14 * t))); x += 0.6 * (rng.random(len(t)) > 0.9985) * rng.standard_normal(len(t))  # vinyl crackle
    return _norm(x * _env(len(t), 0.003, 0.03) * np.exp(-t * 1.5))
def airhorn():
    out = []
    for d in (0.28, 0.28, 0.28, 1.1):
        t = _t(d); x = 0
        for f in (415.3, 523.3, 622.3):  # MLG-horn chord: Ab, C, Eb
            ph = 2 * np.pi * np.cumsum(f * (1 + 0.003 * np.sin(2 * np.pi * 7 * t))) / SR
            x = x + sum(np.sin(k * ph) * k ** -0.65 for k in range(1, 16))
        x = np.tanh(0.25 * x) * (1 + 0.15 * np.random.default_rng(1).standard_normal(len(t))) * _env(len(t), 0.012, 0.06 if d < 1 else 0.3)
        out += [x, np.zeros(int(0.07 * SR))]
    return _norm(_reverb(_lp_fast(np.concatenate(out), 5000), 0.5, 0.15))
def bass_drop():
    t = _t(2.2); f = np.where(t < 0.9, 220 * np.exp(-t * 2.4) + 38, 38 + 0 * t); ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.tanh(1.6 * np.sin(ph)) * np.where(t < 0.9, 0.7, 1.0) * np.exp(-np.maximum(t - 0.9, 0) * 1.4)
    x += 0.5 * _bp(np.random.default_rng(2).standard_normal(len(t)), 2000, 9000) * np.exp(-t * 4) * (t < 0.9)  # riser hiss
    return _norm(x * _env(len(t), 0.01, 0.3))
def crickets():
    t = _t(3.4); x = np.zeros(len(t)); rng = np.random.default_rng(7)
    for f, rate, ph0 in [(4300, 2.6, 0.0), (4750, 2.1, 0.3), (3900, 3.0, 0.6)]:
        gate = (((t * rate + ph0) % 1) < 0.35).astype(float); pulse = (np.sin(2 * np.pi * 38 * t) > 0.2).astype(float)  # chirp trains
        x += np.sin(2 * np.pi * f * t) * gate * pulse * 0.4
    x += 0.012 * _lp_fast(rng.standard_normal(len(t)), 1500)  # night air
    return _norm(x * _env(len(t), 0.2, 0.5), -8)
def heartbeat():
    d = 3.4; x = np.zeros(int(d * SR))
    for b in np.arange(0.0, d - 0.6, 0.85):
        for off, g, f0 in [(0, 1.0, 62), (0.2, 0.7, 52)]:
            n = int(0.22 * SR); t = np.arange(n) / SR; i = int((b + off) * SR)
            x[i:i + n] += np.sin(2 * np.pi * (f0 + 30 * np.exp(-t * 30)) * t) * np.exp(-t * 18) * g
    return _norm(_reverb(x, 0.3, 0.1), -4)
def whoosh():
    t = _t(0.7); n = np.random.default_rng(4).standard_normal(len(t)); y = np.zeros(len(t)); fc = 400 + 5500 * np.sin(np.pi * t / 0.7) ** 2  # band sweeps up and back
    for lo in range(0, len(t), 480):
        seg = n[lo:lo + 960]; k = len(seg); y[lo:lo + k] += _bp(seg, fc[lo] * .6, fc[lo] * 1.5) * np.hanning(k)
    return _norm(y * np.sin(np.pi * t / 0.7) ** 1.5)
def ding():
    t = _t(1.6); x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in [(1318.5, 1, 3), (2637, .35, 5), (3951, .15, 8)])
    return _norm(x * _env(len(t), 0.002, 0.05))
def rimshot():  # ba-dum-tss
    out = np.zeros(int(1.7 * SR)); rng = np.random.default_rng(6)
    def drum(at, f, d, noise):
        t = _t(d); i = int(at * SR); out[i:i + len(t)] += np.sin(2 * np.pi * (f + 90 * np.exp(-t * 40)) * t) * np.exp(-t * 14) + noise * _bp(rng.standard_normal(len(t)), 1500, 9000) * np.exp(-t * 25)
    drum(0.0, 190, 0.25, 0.9); drum(0.16, 110, 0.35, 0.1)
    t = _t(1.2); i = int(0.42 * SR); out[i:i + len(t)] += 0.7 * _bp(rng.standard_normal(len(t)), 5000, 14000) * np.exp(-t * 4)
    return _norm(out)
SOUNDS = dict(vine_boom=vine_boom, sad_trombone=sad_trombone, record_scratch=record_scratch, airhorn=airhorn, bass_drop=bass_drop,
              crickets=crickets, heartbeat=heartbeat, whoosh=whoosh, ding=ding, rimshot=rimshot)

def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
def make_sounds(d=None):
    d = Path(d or SND); d.mkdir(parents=True, exist_ok=True)
    for n, f in SOUNDS.items(): write_wav(d / f"{n}.wav", f())
    return sorted(SOUNDS)

# ---------------------------------------------------------------- moods
DEFAULT_MOODS = {
    "embarrassing": {"sound": ["mine/hello_darkness", "sad_trombone"], "effect": ["zoom", "vignette", "desat"], "volume": -3, "offset": 0.0, "hold": 1.2, "tail": "auto"},
    "fail":         {"sound": ["mine/sad_violin", "sad_trombone"], "effect": ["punch", "desat"], "volume": -2, "offset": 0.1, "hold": 0.0, "tail": "auto"},
    "shock":        {"sound": "vine_boom", "effect": ["punch", "flash"], "volume": 0, "offset": 0.0, "hold": 0.0, "tail": 0},
    "hype":         {"sound": ["airhorn", "bass_drop"], "effect": ["punch", "shake", "flash"], "volume": -2, "offset": 0.0, "hold": 0.0, "tail": 0},
    "win":          {"sound": "bass_drop", "effect": ["punch", "flash"], "volume": -2, "offset": -0.15, "hold": 0.0, "tail": 0},
    "awkward":      {"sound": ["mine/curb_enthusiasm", "crickets"], "effect": ["zoom", "desat"], "volume": -4, "offset": 0.0, "hold": 0.8, "tail": "auto"},
    "cringe":       {"sound": "record_scratch", "effect": ["flash", "desat"], "volume": -2, "offset": 0.0, "hold": 1.0, "tail": 0.4},
    "funny":        {"sound": ["rimshot", "ding"], "effect": ["punch"], "volume": -3, "offset": 0.1, "hold": 0.0, "tail": 0.6},
    "suspense":     {"sound": "heartbeat", "effect": ["zoom", "vignette"], "volume": -3, "offset": -0.3, "hold": 0.0, "tail": 0},
}
def load_moods():
    p = SND / "moods.json"
    if not p.exists(): SND.mkdir(exist_ok=True); p.write_text(json.dumps(DEFAULT_MOODS, indent=1), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))
def resolve_sound(spec):
    names = [spec] if isinstance(spec, str) else spec
    for _ in range(2):  # second pass after synthesising the shipped sounds
        for n in names:
            for ext in ("wav", "m4a", "mp3"):
                if (SND / f"{n}.{ext}").exists(): return SND / f"{n}.{ext}"
        if not any(SND.glob("*.wav")): make_sounds()
    raise SystemExit(f"no sound found for {spec} in {SND}")

# ---------------------------------------------------------------- payoff
def find_payoff(mp4, dur):
    """loudest 100 ms in the last 60 % of the Short, never in the hook zone."""
    o = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True, timeout=300).stdout
    x = np.frombuffer(o, "<i2").astype(float); w = 1600; n = len(x) // w
    if n < 5: return max(HOOK_SAFE, dur * 0.7)
    r = np.sqrt((x[: n * w].reshape(n, w) ** 2).mean(1)); lo = int(max(dur * 0.4, HOOK_SAFE) * 10); hi = max(int((dur - 0.6) * 10), lo + 1)
    return float(np.argmax(r[lo:hi]) + lo) / 10

# ---------------------------------------------------------------- render
def meme(src, mood, out=None, at=None, moods=None, sheet=None):
    m = (moods or load_moods())[mood]; W, H, dur = probe(src); out = Path(out or Path(src).with_name(Path(src).stem + f"_{mood}.mp4"))
    T = float(at) if at is not None else find_payoff(src, dur); T = min(max(T, HOOK_SAFE), dur - 0.4)
    snd = resolve_sound(m["sound"]); L = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(snd)]).stdout.strip())
    hold = float(m.get("hold", 0)); S = max(HOOK_SAFE, T + float(m.get("offset", 0)))
    tail = m.get("tail", 0)
    tail = max(0.0, S + L - (dur + hold)) + 0.3 if tail == "auto" else float(tail)
    tail = max(0.0, min(tail, MAX_EXT - hold)); total = dur + hold + tail; fx = set(m.get("effect", []))
    # transient effects ride the first segment after the payoff (the freeze if there is one, else the live video)
    zt = []; dx = dy = "0"
    if "zoom" in fx: zt.append(f"0.14*min(t/{hold if hold else 1.5:.3f},1)")
    if "punch" in fx: zt.append("0.12*max(0,1-t/0.35)")
    if "shake" in fx: zt.append("0.05"); dx = "20*sin(t*70)*max(0,1-t/0.5)"; dy = "14*cos(t*83)*max(0,1-t/0.5)"
    Z = "1+" + "+".join(zt) if zt else None
    trans = (f"scale=w='2*trunc(iw*({Z})/2)':h='2*trunc(ih*({Z})/2)':eval=frame,crop={W}:{H}:'(iw-{W})/2+{dx}':'(ih-{H})*0.4+{dy}'," if Z else "")
    flash = "fade=t=in:st=0:d=0.22:color=white," if "flash" in fx else ""
    stat = ("eq=saturation=0.25:contrast=1.05," if "desat" in fx else "") + ("vignette=angle=0.95," if "vignette" in fx else "")
    base = "fps=30,setsar=1,format=yuv420p,"
    v = [f"[0:v]split={3 if hold else 2}[a][c]" if not hold else "[0:v]split=3[a][b][c]", f"[a]trim=0:{T:.3f},setpts=PTS-STARTPTS,{base}null[pre]"]
    if hold: v.append(f"[b]trim=start={T:.3f}:end={T + 0.1:.3f},setpts=PTS-STARTPTS,fps=30,trim=end_frame=1,tpad=stop_mode=clone:stop_duration={hold:.3f},setpts=PTS-STARTPTS,{trans}{flash}{stat}{base}null[hold]")
    postx = ("" if hold else trans + flash) + stat
    v.append(f"[c]trim=start={T:.3f},setpts=PTS-STARTPTS,{postx}{base}" + (f"tpad=stop_mode=clone:stop_duration={tail:.3f}," if tail else "") + "null[post]")
    v.append(f"[pre]{'[hold]' if hold else ''}[post]concat=n={3 if hold else 2}:v=1:a=0[vo]")
    # audio: original (paused during the freeze) ducked under the sound, plus the sound; loudness -14 LUFS at the end
    aud = [f"[0:a]aresample={SR},aformat=channel_layouts=stereo,asplit=2[x][y]", f"[x]atrim=0:{T:.3f},asetpts=PTS-STARTPTS[ap]", f"[y]atrim=start={T:.3f},asetpts=PTS-STARTPTS[aq]"]
    if hold: aud += [f"anullsrc=r={SR}:cl=stereo,atrim=0:{hold:.3f},asetpts=PTS-STARTPTS[ah]", "[ap][ah][aq]concat=n=3:v=0:a=1[ao]"]
    else: aud.append("[ap][aq]concat=n=2:v=0:a=1[ao]")
    Lm = min(L, total - S)
    aud.append(f"[ao]apad=whole_dur={total:.3f},volume=0.3:enable='between(t,{S:.3f},{S + Lm:.3f})'[ad]")
    aud.append(f"[1:a]aresample={SR},aformat=channel_layouts=stereo,atrim=0:{Lm:.3f},afade=t=out:st={max(Lm - 0.25, 0):.3f}:d=0.25,volume={m.get('volume', 0)}dB,adelay={int(S * 1000)}|{int(S * 1000)}[as]")
    aud.append(f"[ad][as]amix=inputs=2:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=9,atrim=0:{total:.3f}[ao2]")
    gf = Path(tempfile.mkdtemp(prefix="meme_")) / "g.txt"; gf.write_text(";".join(v + aud), encoding="utf-8")
    r = run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-i", str(snd), "-filter_complex_script", str(gf), "-map", "[vo]", "-map", "[ao2]", "-t", f"{total:.3f}",
             "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out)])
    if r.returncode or not out.exists(): raise SystemExit("ffmpeg failed:\n" + r.stderr[-1500:])
    if sheet: contact(out, [T - 0.8, T + 0.02, T + 0.3, T + hold * 0.9 + 0.3, total - 0.1], sheet)
    return dict(out=str(out), at=round(T, 2), sound=snd.name, hold=hold, tail=round(tail, 2), duration=round(total, 2))

def contact(mp4, times, sheet):
    """5 frames side by side at the given seconds: the proof picture for one effect."""
    ins = []
    for t in times: ins += ["-ss", f"{max(t, 0):.2f}", "-i", str(mp4)]
    n = len(times); fc = "".join(f"[{i}:v]scale=216:-2[s{i}];" for i in range(n)) + "".join(f"[s{i}]" for i in range(n)) + f"hstack=inputs={n}"
    run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", fc, "-frames:v", "1", "-q:v", "3", str(sheet)])

def loudness(f):
    o = run(["ffmpeg", "-nostats", "-i", str(f), "-af", "ebur128=peak=true", "-f", "null", "-"]).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", o)[-1]), float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", o)[-1])

# ---------------------------------------------------------------- add your own song
def sec(s):
    p = [float(x) for x in str(s).split(":")]; return sum(v * 60 ** i for i, v in enumerate(reversed(p)))
def add(name, src, span=None):
    (SND / "mine").mkdir(parents=True, exist_ok=True); dst = SND / "mine" / f"{name}.m4a"; a = b = None
    if span: a, b = [sec(x) for x in span.split("-")]
    if Path(src).exists():
        cmd = ["ffmpeg", "-y", "-loglevel", "error"] + (["-ss", str(a), "-to", str(b)] if span else []) + ["-i", src, "-vn", "-c:a", "aac", "-b:a", "192k", str(dst)]
    else:
        cmd = [sys.executable, "-m", "yt_dlp", "-x", "--audio-format", "m4a", "--no-playlist", "-o", str(SND / "mine" / f"{name}.%(ext)s")] + (["--download-sections", f"*{a}-{b}", "--force-keyframes-at-cuts"] if span else []) + [src]
    r = run(cmd, timeout=600)
    if r.returncode or not dst.exists(): raise SystemExit("download failed: " + (r.stderr or r.stdout)[-600:])
    print("saved", dst, "(your own copy, not for the shared zip)")


# ---------------------------------------------------------------- music bed
def _kick(d=0.25):
    t = _t(d); return np.sin(2 * np.pi * (45 * t + 90 * (1 - np.exp(-t * 30)) / 30)) * np.exp(-t * 14)
def _hat(d=0.06, seed=3): return np.random.RandomState(seed).randn(int(d * SR)) * np.exp(-_t(d) * 60)
def _note(f, d, shape="saw"):
    t = _t(d); w = (2 * ((t * f) % 1) - 1) if shape == "saw" else np.sin(2 * np.pi * f * t); return w * _env(len(t), 0.01, 0.06)
def bed_loop(mood):
    """4-bar loop synthesised here (no samples): bpm, kick pattern, hat, bass/lead notes per mood."""
    bpm, kick, hat, notes, shape, f0 = {"hype": (128, [0, 4, 8, 12], 1, [0, 0, 3, 5], "saw", 55), "chill": (80, [0, 10], 0, [0, 3, 5, 3], "sin", 55), "tense": (90, [0, 8], 0, [0, 0, 1, 0], "saw", 41.2), "funny": (110, [0, 6, 8, 14], 1, [0, 4, 7, 4], "sin", 82.4)}[mood]
    st = 60 / bpm / 4; n = int(16 * 4 * st * SR); x = np.zeros(n + SR)
    def put(y, k, g=1.0): i = int(k * st * SR); x[i:i + len(y)] += y * g
    for bar in range(4):
        for k in kick: put(_kick(), bar * 16 + k, 1.0)
        if hat:
            for k in range(2, 16, 4): put(_hat(), bar * 16 + k, 0.35)
        for q in range(4): put(_note(f0 * 2 ** (notes[(bar + q) % 4] / 12), st * 3.5, shape), bar * 16 + q * 4, 0.45 if mood != "tense" else 0.7)
    return _norm(_lp_fast(x[:n], 2500 if mood != "hype" else 5000), -3)  # whole loop length is exact: it repeats without a gap

def resolve_bed(mood, file=None):
    if file:
        f = Path(file); f = f if f.exists() else SND / "mine" / file
        if not f.exists(): raise SystemExit(f"music file not found: {file}")
        return f
    for e in ("wav", "m4a", "mp3"):
        if (SND / "mine" / f"bed_{mood}.{e}").exists(): return SND / "mine" / f"bed_{mood}.{e}"
    f = SND / f"bed_{mood}.wav"; SND.mkdir(exist_ok=True)
    if not f.exists():  # atomic: parallel picks must never read a half-written wav
        t = f.with_name(f"{f.stem}.{os.getpid()}.tmp"); write_wav(t, bed_loop(mood)); os.replace(t, f)
    return f

def bed(src, mood="chill", out=None, file=None, db=-14, solo=None):
    """Music under the finished Short, ducked by its own audio (sidechaincompress); picture copied, mix back at -14 LUFS."""
    _, _, dur = probe(src); out = Path(out or Path(src).with_name(Path(src).stem + "_bed.mp4")); snd = resolve_bed(mood, file)
    pre = (f"[0:a]aresample={SR},aformat=channel_layouts=stereo,asplit=3[sp][k][m];"
           f"[1:a]aresample={SR},aformat=channel_layouts=stereo,atrim=0:{dur:.3f},volume={db}dB,afade=t=in:d=0.3,afade=t=out:st={max(dur - 0.5, 0):.3f}:d=0.5[b];"
           "[b][k]sidechaincompress=threshold=0.05:ratio=6:attack=15:release=350:makeup=1")
    mix = f"[m][bd]amix=inputs=2:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=9,atrim=0:{dur:.3f}[ao]"
    ins = ["-i", str(src), "-stream_loop", "-1", "-i", str(snd)]
    r = run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", f"{pre}[bd];[sp]anullsink;{mix}", "-map", "0:v", "-map", "[ao]", "-t", f"{dur:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out)])
    if r.returncode or not out.exists(): raise SystemExit("ffmpeg failed:\n" + r.stderr[-1500:])
    if solo: run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", f"{pre}[bs];[sp]anullsink;[m]anullsink", "-map", "[bs]", "-t", f"{dur:.3f}", str(solo)])  # the ducked bed alone, for measuring
    return dict(out=str(out), bed=snd.name, duration=round(dur, 2))

def rms_db(f, a, b):
    o = run(["ffmpeg", "-nostats", "-ss", str(a), "-t", str(b - a), "-i", str(f), "-af", "astats=measure_overall=RMS_level:measure_perchannel=0", "-f", "null", "-"]).stderr
    return float(re.findall(r"RMS level dB:\s+(-?[\d.inf]+)", o)[-1])

# ---------------------------------------------------------------- selftest
def selftest():
    global SND
    tmp = Path(tempfile.mkdtemp(prefix="memetest_")); old = SND; SND = tmp / "s"; make_sounds()
    try:
        for n, fn in SOUNDS.items():  # audible, not clipping, sane length
            f = fn(); assert 0.3 < len(f) / SR < 4 and 0.3 < np.abs(f).max() <= 1 and not np.isnan(f).any(), n
        f = vine_boom(); assert np.fft.rfftfreq(len(f), 1 / SR)[np.abs(np.fft.rfft(f)).argmax()] < 150, "boom must be sub-bass"
        f = crickets(); assert 3500 < np.fft.rfftfreq(len(f), 1 / SR)[np.abs(np.fft.rfft(f)).argmax()] < 5200, "crickets are high"
        src = tmp / "t.mp4"; r = run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=360x640:r=30:d=10", "-f", "lavfi", "-i", "sine=f=300:d=10", "-af", "volume=0.3", "-shortest", "-pix_fmt", "yuv420p", str(src)])
        assert r.returncode == 0, r.stderr
        assert 3.0 <= find_payoff(src, 10) <= 9.4
        for mood in DEFAULT_MOODS:
            res = meme(src, mood, tmp / f"{mood}.mp4", at=5.0, moods=DEFAULT_MOODS); w, h, d = probe(tmp / f"{mood}.mp4")
            assert (w, h) == (360, 640) and abs(d - res["duration"]) < 0.15 and 10 <= d <= 10 + MAX_EXT + 0.1, (mood, res, d)
            i, _ = loudness(tmp / f"{mood}.mp4"); assert -17 < i < -11, (mood, i)
        assert meme(src, "shock", tmp / "hook.mp4", at=0.5, moods=DEFAULT_MOODS)["at"] >= HOOK_SAFE  # hook zone is protected
        # bed: speech 0-5 s (sine), gap 5-10 s; duration + picture untouched, bed quieter in speech than in the gap
        sp = tmp / "sp.mp4"; run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=360x640:r=30:d=10", "-f", "lavfi", "-i", "sine=f=300:d=10", "-af", "volume='if(lt(t,5),2,0)':eval=frame", "-shortest", "-pix_fmt", "yuv420p", str(sp)])
        for mood in ("hype", "chill", "tense", "funny"):
            b = bed(sp, mood, tmp / f"b_{mood}.mp4", solo=tmp / f"s_{mood}.wav"); w, h, d = probe(tmp / f"b_{mood}.mp4")
            assert (w, h) == (360, 640) and abs(d - 10) < 0.1, (mood, d)
            i, _ = loudness(tmp / f"b_{mood}.mp4"); assert -17 < i < -11, (mood, i)
            assert rms_db(tmp / f"s_{mood}.wav", 2, 4) < rms_db(tmp / f"s_{mood}.wav", 6.5, 9) - 5, mood
        h = lambda f: run(["ffmpeg", "-loglevel", "error", "-i", str(f), "-map", "0:v", "-c", "copy", "-f", "md5", "-"]).stdout
        assert h(sp) == h(tmp / "b_hype.mp4"), "picture must be copied untouched"
    finally: SND = old
    r = run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1); assert r.returncode == 124 and "timeout" in r.stderr  # stalled tool is stopped, not waited for
    print("meme selftest ok:", len(DEFAULT_MOODS), "moods,", len(SOUNDS), "sounds")

if __name__ == "__main__":
    import argparse
    a = sys.argv[1:]
    if a[:1] == ["selftest"]: selftest()
    elif a[:1] == ["sounds"]: print("wrote", ", ".join(make_sounds()), "->", SND)
    elif a[:1] == ["add"]: add(*a[1:4])
    elif a[:1] == ["bed"]:
        ap = argparse.ArgumentParser(); ap.add_argument("_"); ap.add_argument("short"); ap.add_argument("--mood", default="chill", choices=["hype", "chill", "tense", "funny"]); ap.add_argument("--file"); ap.add_argument("--db", type=float, default=-14); ap.add_argument("--out"); ap.add_argument("--solo")
        x = ap.parse_args(a); print(json.dumps(bed(x.short, x.mood, x.out, x.file, x.db, x.solo)))
    else:
        ap = argparse.ArgumentParser(); ap.add_argument("short"); ap.add_argument("--mood", required=True); ap.add_argument("--at", type=float); ap.add_argument("--out"); ap.add_argument("--sheet")
        x = ap.parse_args(a); print(json.dumps(meme(x.short, x.mood, x.out, x.at, sheet=x.sheet)))

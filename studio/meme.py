"""Viral meme sound effects, visual punchlines, and background music beds for PeakCut clips."""
import json
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SND = HERE / "sounds"
SR = 48000
HOOK_SAFE = 3.0  # never put a meme gag in the first 3 seconds (the hook zone)
MAX_EXT = 3.0

MEME_MOODS = {
    "none": "None",
    "shock": "Shock (Vine boom)",
    "hype": "Hype (Airhorn / Bass drop)",
    "fail": "Fail (Sad trombone)",
    "funny": "Funny (Rimshot / Boing)",
    "awkward": "Awkward (Record scratch)",
    "embarrassing": "Embarrassing (Darkness zoom)",
    "cringe": "Cringe (Crickets)",
    "win": "Win (Ding)",
    "suspense": "Suspense (Heartbeat)",
}

MUSIC_BEDS = {
    "none": "None",
    "chill": "Chill Lofi",
    "hype": "Hype Phonk",
    "tense": "Tense Thriller",
    "funny": "Funny Cartoon",
}


def _t(d):
    return np.arange(int(d * SR)) / SR


def _env(n, a=0.005, r=0.05):
    e = np.ones(n)
    na, nr = max(int(a * SR), 1), max(int(r * SR), 1)
    e[:na] = np.linspace(0, 1, na)
    e[-nr:] = np.minimum(e[-nr:], np.linspace(1, 0, nr))
    return e


def _lp_fast(x, fc, n=3):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (f / fc) ** (2 * n))
    return np.fft.irfft(X, len(x))


def _bp(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= ((f > lo) & (f < hi))
    return np.fft.irfft(X, len(x))


def _reverb(x, tail=0.5, mix=0.25, seed=1):
    r = np.random.default_rng(seed)
    n = int(tail * SR)
    ir = r.standard_normal(n) * np.exp(-np.arange(n) / (SR * tail / 5))
    ir = _lp_fast(ir, 3000)
    y = np.fft.irfft(np.fft.rfft(x, len(x) + n) * np.fft.rfft(ir, len(x) + n), len(x) + n)
    y = y / (np.abs(y).max() + 1e-9)
    z = np.zeros(len(y))
    z[: len(x)] = x
    return z * (1 - mix) + y * mix * np.abs(x).max()


def _norm(x, peak_db=-3.0):
    return x / (np.abs(x).max() + 1e-9) * 10 ** (peak_db / 20)


def vine_boom():
    t = _t(1.8)
    f = 38 + 120 * np.exp(-t * 14)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.tanh(2.2 * np.sin(ph)) * np.exp(-t * 2.6) + 0.35 * np.sin(2 * ph) * np.exp(-t * 6)
    x += 0.5 * _bp(np.random.default_rng(3).standard_normal(len(t)), 80, 900) * np.exp(-t * 30)
    return _norm(_reverb(x * _env(len(t), 0.002, 0.2), 0.6, 0.18))


def sad_trombone():
    notes = [(233.1, .55), (220.0, .55), (207.7, .55), (196.0, 1.5)]
    out = []
    for i, (f0, d) in enumerate(notes):
        t = _t(d)
        last = i == 3
        f = f0 * (1 + (0.004 * np.sin(2 * np.pi * 5.5 * t) if last else 0)) * (1 - 0.06 * np.clip((t - 0.5) / 1.0, 0, 1) if last else 1)
        ph = 2 * np.pi * np.cumsum(f) / SR
        wah = 0.25 + 0.75 * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 0.7
        x = sum(np.sin(k * ph) * k ** -1.0 * np.exp(-k / (1 + 9 * wah)) for k in range(1, 22))
        x = x + 0.04 * np.random.default_rng(i).standard_normal(len(t))
        out.append(x * _env(len(t), 0.04, 0.12 if not last else 0.35) * (0.6 + 0.4 * wah))
        out.append(np.zeros(int(0.05 * SR)))
    x = np.concatenate(out)
    return _norm(_reverb(_bp(_lp_fast(x, 2200), 80, 9000), 0.4, 0.15))


def record_scratch():
    t = _t(0.75)
    rng = np.random.default_rng(5)
    f = 900 * np.exp(-t * 2.2) * (1 + 0.5 * np.sin(2 * np.pi * 9 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    saw = 2 * ((ph / (2 * np.pi)) % 1) - 1
    noise = _bp(rng.standard_normal(len(t)), 1500, 9000)
    x = (0.6 * saw + 0.9 * noise) * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 14 * t)))
    x += 0.6 * (rng.random(len(t)) > 0.9985) * rng.standard_normal(len(t))
    return _norm(x * _env(len(t), 0.003, 0.03) * np.exp(-t * 1.5))


def airhorn():
    out = []
    for d in (0.28, 0.28, 0.28, 1.1):
        t = _t(d)
        x = 0
        for f in (415.3, 523.3, 622.3):
            ph = 2 * np.pi * np.cumsum(f * (1 + 0.003 * np.sin(2 * np.pi * 7 * t))) / SR
            x = x + sum(np.sin(k * ph) * k ** -0.65 for k in range(1, 16))
        x = np.tanh(0.25 * x) * (1 + 0.15 * np.random.default_rng(1).standard_normal(len(t))) * _env(len(t), 0.012, 0.06 if d < 1 else 0.3)
        out += [x, np.zeros(int(0.07 * SR))]
    return _norm(_reverb(_lp_fast(np.concatenate(out), 5000), 0.5, 0.15))


def bass_drop():
    t = _t(2.2)
    f = np.where(t < 0.9, 220 * np.exp(-t * 2.4) + 38, 38 + 0 * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.tanh(1.6 * np.sin(ph)) * np.where(t < 0.9, 0.7, 1.0) * np.exp(-np.maximum(t - 0.9, 0) * 1.4)
    x += 0.5 * _bp(np.random.default_rng(2).standard_normal(len(t)), 2000, 9000) * np.exp(-t * 4) * (t < 0.9)
    return _norm(x * _env(len(t), 0.01, 0.3))


def crickets():
    t = _t(3.4)
    x = np.zeros(len(t))
    for f, rate, ph0 in [(4300, 2.6, 0.0), (4750, 2.1, 0.3), (3900, 3.0, 0.6)]:
        gate = (((t * rate + ph0) % 1) < 0.35).astype(float)
        pulse = (np.sin(2 * np.pi * 38 * t) > 0.2).astype(float)
        x += np.sin(2 * np.pi * f * t) * gate * pulse * 0.4
    x += 0.012 * _lp_fast(np.random.default_rng(7).standard_normal(len(t)), 1500)
    return _norm(_env(len(t), 0.2, 0.5) * x, -8)


def heartbeat():
    d = 3.4
    x = np.zeros(int(d * SR))
    for b in np.arange(0.0, d - 0.6, 0.85):
        for off, g, f0 in [(0, 1.0, 62), (0.2, 0.7, 52)]:
            n = int(0.22 * SR)
            t = np.arange(n) / SR
            i = int((b + off) * SR)
            x[i:i + n] += np.sin(2 * np.pi * (f0 + 30 * np.exp(-t * 30)) * t) * np.exp(-t * 18) * g
    return _norm(_reverb(x, 0.3, 0.1), -4)


def whoosh():
    t = _t(0.7)
    n = np.random.default_rng(4).standard_normal(len(t))
    y = np.zeros(len(t))
    fc = 400 + 5500 * np.sin(np.pi * t / 0.7) ** 2
    for lo in range(0, len(t), 480):
        seg = n[lo:lo + 960]
        k = len(seg)
        y[lo:lo + k] += _bp(seg, fc[lo] * .6, fc[lo] * 1.5) * np.hanning(k)
    return _norm(y * np.sin(np.pi * t / 0.7) ** 1.5)


def ding():
    t = _t(1.6)
    x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in [(1318.5, 1, 3), (2637, .35, 5), (3951, .15, 8)])
    return _norm(x * _env(len(t), 0.002, 0.05))


def rimshot():
    out = np.zeros(int(1.7 * SR))
    rng = np.random.default_rng(6)
    def drum(at, f, d, noise):
        t = _t(d)
        i = int(at * SR)
        out[i:i + len(t)] += np.sin(2 * np.pi * (f + 90 * np.exp(-t * 40)) * t) * np.exp(-t * 14) + noise * _bp(rng.standard_normal(len(t)), 1500, 9000) * np.exp(-t * 25)
    drum(0.0, 190, 0.25, 0.9)
    drum(0.16, 110, 0.35, 0.1)
    t = _t(1.2)
    i = int(0.42 * SR)
    out[i:i + len(t)] += 0.7 * _bp(rng.standard_normal(len(t)), 5000, 14000) * np.exp(-t * 4)
    return _norm(out)


SOUNDS = {
    "vine_boom": vine_boom, "sad_trombone": sad_trombone, "record_scratch": record_scratch,
    "airhorn": airhorn, "bass_drop": bass_drop, "crickets": crickets, "heartbeat": heartbeat,
    "whoosh": whoosh, "ding": ding, "rimshot": rimshot
}


def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def ensure_sound(name):
    p = SND / f"{name}.wav"
    if p.exists():
        return p
    if name in SOUNDS:
        SND.mkdir(parents=True, exist_ok=True)
        write_wav(p, SOUNDS[name]())
        return p
    return None


def probe_dur(f):
    try:
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                           capture_output=True, text=True, check=True)
        return float(o.stdout.strip())
    except Exception:
        return 0.0


def find_payoff(mp4, dur):
    """Loudest moment in the last 45% of the clip, after the hook zone."""
    try:
        o = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                           capture_output=True, timeout=60).stdout
        x = np.frombuffer(o, "<i2").astype(float)
        w = 1600
        n = len(x) // w
        if n < 5:
            return max(HOOK_SAFE, dur * 0.7)
        r = np.sqrt((x[: n * w].reshape(n, w) ** 2).mean(1))
        lo = int(max(dur * 0.5, HOOK_SAFE) * 10)
        hi = max(int((dur - 0.5) * 10), lo + 1)
        return float(np.argmax(r[lo:hi]) + lo) / 10
    except Exception:
        return max(HOOK_SAFE, dur * 0.75)


def apply_meme_effect(src: Path, mood: str, out: Path = None, at: float = None, vol: float = None) -> Path:
    """Applies a meme sound effect & visual punchline to a finished video clip."""
    if not mood or mood == "none":
        return src
    mood = mood.lower().strip()
    src = Path(src)
    dur = probe_dur(src)
    if dur < 4.0:
        return src

    moods_file = SND / "moods.json"
    m_info = {}
    if moods_file.exists():
        try:
            m_info = json.loads(moods_file.read_text(encoding="utf-8")).get(mood, {})
        except Exception:
            pass

    sound_name = m_info.get("sound") or mood
    if isinstance(sound_name, list):
        sound_name = sound_name[0]
    if sound_name.startswith("mine/"):
        sound_name = sound_name.split("/")[-1]

    snd_path = ensure_sound(sound_name) or SND / f"{sound_name}.wav"
    if not snd_path or not snd_path.exists():
        return src

    T = float(at) if at is not None else find_payoff(src, dur)
    T = min(max(T, HOOK_SAFE), dur - 0.4)
    out = Path(out or src.with_name(f"{src.stem}_meme.mp4"))

    # Build audio filter: blend sound effect at payoff moment T
    S = max(HOOK_SAFE, T + float(m_info.get("offset", 0.0)))
    if vol is not None:
        try:
            vol = float(vol)
        except (ValueError, TypeError):
            vol = m_info.get("volume", -2)
    else:
        vol = m_info.get("volume", -2)

    af = (f"[0:a]aresample={SR}:async=1:first_pts=0,volume=0.4:enable='between(t,{S:.3f},{S + 1.2:.3f})'[ad];"
          f"[1:a]aresample={SR},aformat=channel_layouts=stereo,volume={vol}dB,adelay={int(S * 1000)}|{int(S * 1000)}[as];"
          f"[ad][as]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95[ao]")

    # Build subtle visual punch effect at payoff T if specified
    effects = set(m_info.get("effect", []))
    vf = "null"
    if "flash" in effects:
        # brief 0.1s white flash at S
        vf = f"drawbox=t=fill:color=white@0.65:enable='between(t,{S:.3f},{S + 0.12:.3f})'"
    elif "punch" in effects:
        # brief punch zoom
        vf = f"scale=1080:1920"

    if vf == "null":
        vcodec = ["-c:v", "copy"]
    else:
        from . import render
        vcodec = ["-c:v", "h264_nvenc", "-cq", "18", "-preset", "p4"] if getattr(render, "_nvenc_ok", None) else ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]

    same_file = out.resolve() == src.resolve()
    tmp_out = out.with_name(f"{out.stem}_memetmp{out.suffix}") if same_file else out
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-i", str(snd_path),
           "-filter_complex", af]
    if vf != "null":
        cmd += ["-vf", vf]
    cmd += ["-map", "0:v", "-map", "[ao]", "-t", f"{dur:.3f}"] + vcodec + [
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-movflags", "+faststart", str(tmp_out)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0:
            if same_file:
                os.replace(str(tmp_out), str(out))
            return out
    except Exception:
        pass
    finally:
        if same_file and tmp_out.exists():
            try:
                tmp_out.unlink(missing_ok=True)
            except OSError:
                pass
    return src


def apply_music_bed(src: Path, mood: str = "chill", out: Path = None, db: float = -14) -> Path:
    """Adds a ducking background music bed under the finished clip."""
    if not mood or mood == "none":
        return src
    mood = mood.lower().strip()
    src = Path(src)
    dur = probe_dur(src)
    if dur < 4.0:
        return src

    bed_path = SND / f"bed_{mood}.wav"
    if not bed_path.exists():
        return src

    out = Path(out or src.with_name(f"{src.stem}_bed.mp4"))

    # Ducking filter: speech in [0:a] ducks background music in [1:a]
    pre = (f"[0:a]aresample={SR}:async=1:first_pts=0,aformat=channel_layouts=stereo,asplit=3[sp][k][m];"
           f"[1:a]aresample={SR},aformat=channel_layouts=stereo,atrim=0:{dur:.3f},volume={db}dB,afade=t=in:d=0.3,afade=t=out:st={max(dur - 0.5, 0):.3f}:d=0.5[b];"
           f"[b][k]sidechaincompress=threshold=0.05:ratio=6:attack=15:release=350:makeup=1")
    mix = f"[m][bd]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95,atrim=0:{dur:.3f}[ao]"

    same_file = out.resolve() == src.resolve()
    tmp_out = out.with_name(f"{out.stem}_bedtmp{out.suffix}") if same_file else out
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(src), "-stream_loop", "-1", "-i", str(bed_path),
           "-filter_complex", f"{pre}[bd];[sp]anullsink;{mix}",
           "-map", "0:v", "-map", "[ao]", "-t", f"{dur:.3f}",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-movflags", "+faststart", str(tmp_out)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0:
            if same_file:
                os.replace(str(tmp_out), str(out))
            return out
    except Exception:
        pass
    finally:
        if same_file and tmp_out.exists():
            try:
                tmp_out.unlink(missing_ok=True)
            except OSError:
                pass
    return src

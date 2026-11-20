# Free local transcript with word timings (faster-whisper on GPU if present, else CPU), plus loud moments.
# Usage: python transcribe.py in.m4a out.json [model]     (model default: small on GPU, base on CPU)
import glob, json, os, subprocess, sys
import numpy as np
for d in glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")) if os.name == "nt" else []:  # pip CUDA libs (cublas, cudnn) -> GPU
    os.add_dll_directory(d); os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
from faster_whisper import WhisperModel
size = sys.argv[3] if len(sys.argv) > 3 else None
# names Whisper should know; it also makes Whisper punctuate, which the sentence snapping and caption lines need (Speed test: no full stops -> full sentences)
VOCAB = os.environ.get("TRANSCRIBE_PROMPT", "GTA 6, Grand Theft Auto VI, GTA 5, San Andreas, Vice City, Leonida, Lucia, Jason, Rockstar, PS5, Xbox, Kai Cenat, IShowSpeed. Chat, KEKW, LUL, W, bro.")  # streamers say "chat" all the time: without it Whisper writes "Chad"

def run(device, ct):
    model = WhisperModel(size or ("small" if device == "cuda" else "base"), device=device, compute_type=ct)  # base on CPU: ~3x faster, fine for English
    segs, info = model.transcribe(sys.argv[1], word_timestamps=True, vad_filter=True, initial_prompt=VOCAB)
    return list(segs), info  # list() forces the work now, so a missing GPU library fails here, not later

def loudness(path, sr=4000):
    """dB per second of the whole audio, streamed from ffmpeg in chunks (a 6-hour stream stays small in memory)."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"], stdout=subprocess.PIPE)
    db = []
    while chunk := p.stdout.read(sr * 2 * 60):
        x = np.frombuffer(chunk[: len(chunk) // (sr * 2) * sr * 2], np.int16).astype(np.float32).reshape(-1, sr)
        db += list(10 * np.log10((x ** 2).mean(1) + 1))
    p.wait(); return np.array(db)

def loud_tags(segments, db):
    """Mark segments whose loudest second is far above this video's normal level: screams, crashes, hype. Claude reads these."""
    if len(db) < 30: return {}
    base, top = np.median(db), np.percentile(db, 97)
    peaks = {i: db[int(s["start"]):int(s["end"]) + 1].max(initial=0) for i, s in enumerate(segments)}
    # jump over the 20 s before: a scream after calm talk counts even in a stream that is loud all the time (hype reactors, trailer audio)
    jump = {i: peaks[i] - np.median(db[max(int(s["start"]) - 20, 0):max(int(s["start"]), 1)]) for i, s in enumerate(segments)}
    lift = {i: max(peaks[i] - base, jump[i]) for i in peaks}
    hot = sorted((i for i in peaks if (peaks[i] >= top and peaks[i] - base >= 6) or jump[i] >= 9), key=lambda i: -lift[i])[:max(3, len(segments) // 20)]  # loudest 5% at most
    return {i: f" [LOUD +{lift[i]:.0f}dB]" for i in hot}

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    try:
        if sys.platform == "darwin": raise RuntimeError("no CUDA on macOS")
        segs, info = run("cuda", "float16")
    except Exception as e: print("GPU not usable (", str(e)[:60], ") -> CPU"); segs, info = run("cpu", "int8")
    data = [dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip(),
                 words=[dict(w=w.word.strip(), s=round(w.start, 2), e=round(w.end, 2)) for w in s.words]) for s in segs]
    tags = loud_tags(data, loudness(src))
    off = float(os.environ.get("TRANSCRIBE_OFFSET", 0))  # audio was a window of a long VOD: shift to full-VOD times
    for sg in data: sg["start"] = round(sg["start"] + off, 2); sg["end"] = round(sg["end"] + off, 2); [w.update(s=round(w["s"] + off, 2), e=round(w["e"] + off, 2)) for w in sg["words"]]
    json.dump(dict(lang=info.language, segments=data), open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # compact timeline for the AI to read: one line per segment, loud moments marked
    open(out.replace(".json", ".txt"), "w", encoding="utf-8").write("\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}{tags.get(i, '')}" for i, s in enumerate(data)))
    print("ok", len(data), "segments,", info.language, ",", len(tags), "loud moments")

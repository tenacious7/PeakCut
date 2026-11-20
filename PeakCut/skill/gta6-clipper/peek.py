# Look before rendering: 4 frames of each picked moment (start, a third, two thirds, end) as one strip, from a tiny 240p download of only that moment.
# Usage: PY peek.py clips\picks.json [clips\peek]   ->  clips\peek\<name>.jpg per pick. Claude looks at each strip: right game on screen? does the hook match the picture?
import json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
CK = ["--cookies-from-browser", os.environ["CLIPPER_BROWSER"]] if os.environ.get("CLIPPER_BROWSER") else []

def times(a, b): return [a + 1, a + (b - a) / 3, a + 2 * (b - a) / 3, b - 1]  # inside the window: the cut snaps to sentences, never past it

def run_tree(cmd, timeout):  # subprocess.run(timeout) kills only yt_dlp; its ffmpeg child kept streaming and stalled the next picks
    pr = subprocess.Popen(cmd)
    try: return pr.wait(timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pr.pid)] if os.name == "nt" else ["pkill", "-P", str(pr.pid)], capture_output=True); pr.wait(); raise

def strip(p, out):
    jpg = out / f"{p['name']}.jpg"; a, b = float(p["start"]), float(p["end"])
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows: a file still held open must not fail the peek
        if Path(p["url"]).is_file(): src, off = p["url"], 0.0
        else:  # only this moment, smallest picture: a few seconds per pick
            src, off = str(Path(tmp) / "m.mp4"), a
            cmd = [sys.executable, "-m", "yt_dlp", *CK, "-q", "-S", "res:240", "--download-sections", f"*{a:.2f}-{b:.2f}", "--merge-output-format", "mp4", "-o", src, p["url"]]
            for retry in range(2):  # YouTube sometimes throttles the section to a crawl (seen: 0.001x): give up after 90 s, try once more, then fail clearly
                try:
                    if run_tree(cmd, 90) == 0: break
                    raise RuntimeError("yt_dlp failed")
                except subprocess.TimeoutExpired:
                    if retry: raise RuntimeError(f"download stalled twice (90 s each) for {a:.0f}-{b:.0f}s; peek skipped, render is unaffected")
        frames = []
        for i, t in enumerate(times(a, b)):
            f = Path(tmp) / f"f{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(t - off, 0):.2f}", "-i", src, "-frames:v", "1", "-vf", "scale=-2:240", str(f)], check=True, timeout=60)
            if f.exists(): frames += ["-i", str(f)]
        if not frames: raise RuntimeError("no frames")
        n = len(frames) // 2
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *frames, "-filter_complex", f"hstack=inputs={n}" if n > 1 else "null", str(jpg)], check=True)
    return jpg

def main(picks_file, out_dir="clips/peek"):
    picks, out = json.load(open(picks_file, encoding="utf-8")), Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    def one(p):
        try: print("ok", strip(p, out), "|", p["hook"], flush=True)
        except Exception as e: print("failed", p.get("name"), str(e)[:120], flush=True)  # a failed peek never blocks the render
    picks = [x for p in picks for x in ([p | q | dict(name=f"{p['name']}_part{k}") for k, q in enumerate(p["parts"], 1)] if p.get("parts") else [p])]  # callback Short: one strip per part
    with ThreadPoolExecutor(4) as ex: list(ex.map(one, picks))

if __name__ == "__main__":
    assert [round(x) for x in times(100, 130)] == [101, 110, 120, 129]
    if len(sys.argv) < 2: print("usage: peek.py <picks.json> [out_dir=clips/peek]"); raise SystemExit(2)
    main(*sys.argv[1:3])

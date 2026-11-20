# Multi-part (BEFORE/AFTER callback) Short: part 1 (setup) -> time-jump card -> part 2 (payoff), one loudness pass.
# Pick = normal batch pick, but {"parts": [{"start":..,"end":..}, {"start":..,"end":..}], "jump": "40 MINUTES LATER", "hook": ...}; hook only on part 1.
# Usage: python parts.py picks.json OUT_DIR   (renders every pick that has "parts"; writes mp4, .txt, _cover.jpg like batch.py)
import json, os, re, shutil, subprocess, sys, tempfile, threading, unicodedata
from pathlib import Path
from types import SimpleNamespace
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
CARD_S = 0.7

def clean_text(s):
    """Card text for ffmpeg drawtext: upper case, keeps every Unicode letter/digit/mark (accents, Arabic, Devanagari); drops the rest, so backslash : ' % can never break the filter."""
    return "".join(c for c in str(s).upper() if unicodedata.category(c)[0] in "LNM" or c in " #/.!?-")

def font_for(s):
    """Anton for Latin; Noto Bold for Arabic / Devanagari text, like clip.py (Anton has no such glyphs)."""
    if any("\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F" or "\uFB50" <= c <= "\uFEFF" for c in str(s)): return "NotoSansArabic-Bold.ttf"
    if any("\u0900" <= c <= "\u097F" for c in str(s)): return "NotoSansDevanagari-Bold.ttf"
    return "Anton-Regular.ttf"

def jload(path, default=None):
    """Read a bookkeeping JSON (moments.json / used.json). Missing -> default; corrupt or empty -> default, the bad file kept as <name>.corrupt, so one bad write never crashes a later batch."""
    path, default = Path(path), [] if default is None else default
    if not path.exists(): return default
    try:
        v = json.loads(path.read_text(encoding="utf-8")); return v if isinstance(v, type(default)) else default
    except Exception as e:
        try: shutil.copy2(path, path.with_name(path.name + ".corrupt"))
        except Exception: pass
        print(f"WARN {path.name} unreadable ({str(e)[:60]}), treated as empty; copy kept as {path.name}.corrupt"); return default

def jsave(path, obj):
    """Atomic write: temp file in the same folder, then os.replace (a crash never leaves half a file)."""
    path = Path(path); tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, path)

def card(frame_png, text, out, d=CARD_S):
    """Time-jump card: the last frame of part 1 frozen, blurred and dark, big Anton text popping in (last word on its own line), a short whoosh. 1080x1920, d seconds."""
    w = re.sub(r"[#/.!?-]", "", clean_text(text)).split() or ["LATER"]
    lines = [" ".join(w[:-1]), w[-1]] if len(w) > 1 else w
    dt = "".join(f",drawtext=fontfile={font_for(l)}:text='{l}':fontcolor=white:borderw=10:bordercolor=black:x=(w-text_w)/2:y=(h-text_h)/2+({(i - (len(lines) - 1) / 2) * 200:.0f}):fontsize='{170 if i == len(lines) - 1 else 150}+60*max(0,1-t/0.12)'"
                 for i, l in enumerate(lines))
    vf = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:2,eq=brightness=-0.18:saturation=1.2{dt},fps=30,format=yuv420p"
    af = f"anoisesrc=d={d}:c=pink:r=48000:a=0.5,bandpass=f=1800:width_type=o:w=2.5,afade=t=in:d=0.35:curve=exp,afade=t=out:st={d - 0.2}:d=0.2,aformat=channel_layouts=stereo,volume=8dB"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", str(d), "-i", str(Path(frame_png).resolve()), "-f", "lavfi", "-t", str(d), "-i", af,
                    "-vf", vf, "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-shortest", str(Path(out).resolve())], check=True, cwd=HERE / "fonts")  # cwd = fonts: drawtext wants a plain relative font path on Windows
    return Path(out)

def dur(p): return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))

def join(files, out):
    """Concat (same 1080x1920 / 30 fps / 48k stereo), then one two-pass linear loudnorm to -14 LUFS so every part and the card sit at the same level."""
    n = len(files); ins = sum((["-i", str(f)] for f in files), [])
    cat = "".join(f"[{i}:v]scale=1080:1920,fps=30,setsar=1,format=yuv420p[v{i}];[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}];" for i in range(n)) \
        + "".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][ac]"
    ln = "loudnorm=I=-14:TP=-1.5:LRA=11"
    try:
        sys.path.insert(0, str(HERE)); from clip import loudnorm_stats
        r = subprocess.run(["ffmpeg", "-v", "info", *ins, "-filter_complex", cat + f";[ac]{ln}:print_format=json[x]", "-map", "[x]", "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        m = loudnorm_stats(r.stderr)
        ln += f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true"
    except Exception: pass  # one-pass loudnorm still matches the parts well enough
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", cat + f";[ac]{ln}[a]", "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(Path(out).resolve())], check=True)

def render(p, mp4):
    """Callback Short (hard cut, optional EARLIER tag or "jump_card"): lives in formats.py now; kept so old callers work."""
    import formats; return formats.render(dict(p, format="callback"), mp4)

def main(picks_file, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    n0 = max((int(m.name[:2]) for m in out.glob("[0-9][0-9]_*.mp4")), default=0)
    for i, p in enumerate([x for x in json.load(open(picks_file, encoding="utf-8")) if x.get("parts")], 1):
        mp4 = out / f"{n0 + i:02d}_{p['name']}.mp4"; r = render(p, mp4); print(r.stdout, r.stderr)
        if r.returncode: continue
        mp4.with_suffix(".txt").write_text("\n".join(x for x in [p.get("title", p["hook"].capitalize()), p.get("hashtags", "#gta6 #gtavi #shorts"),
                                            f"Credit: {p['credit']} ({p['url']})" if p.get("credit") else f"Source: {p['url']}"] if x) + "\n", encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", str(mp4), "-frames:v", "1", "-q:v", "2", str(out / f"{mp4.stem}_cover.jpg")])
        src = Path(p["transcript"]).parent; mom = src / "moments.json"; used = src / "used.json"  # same bookkeeping as batch.py: each part is a clipped moment
        jsave(mom, jload(mom) + [dict(id=Path(p["transcript"]).stem, start=q["start"], end=q["end"], hook=p["hook"]) for q in p["parts"]])
        jsave(used, sorted(set(jload(used)) | {Path(p["transcript"]).stem}))

if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        t = Path(tempfile.mkdtemp()); sh = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)
        for k, f in enumerate(("440", "880")):  # two 1080x1920 test clips, different pitch and level
            sh("-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=30:d=2", "-f", "lavfi", "-i", f"sine=f={f}:d=2:r=48000", "-af", f"volume=-3dB", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(t / f"p{k}.mp4"))
        sh("-sseof", "-0.15", "-i", str(t / "p0.mp4"), "-frames:v", "1", str(t / "l.png"))
        c = card(t / "l.png", "40 MINUTES LATER", t / "c.mp4"); assert abs(dur(c) - CARD_S) < 0.15, dur(c)
        join([t / "p0.mp4", c, t / "p1.mp4"], t / "o.mp4"); d = dur(t / "o.mp4"); assert abs(d - (4 + CARD_S)) < 0.2, d
        wh = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(t / "o.mp4")], text=True).strip(); assert wh == "1080,1920", wh
        lv = lambda a, b: float(re.findall(r"mean_volume: (-?[\d.]+)", subprocess.run(["ffmpeg", "-ss", str(a), "-t", str(b), "-i", str(t / "o.mp4"), "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True).stderr)[0])
        assert abs(lv(0, 2) - lv(2.7, 2)) < 3, (lv(0, 2), lv(2.7, 2))  # parts arrive at -14 LUFS from clip.py; the joined pass must keep them level (the card must not jump out)
        assert clean_text("Kai's é ñ 3/3: 100%") == "KAIS É Ñ 3/3 100" and clean_text("مرحبا") == "مرحبا" and clean_text("नमस्ते") == "नमस्ते" and font_for("مرحبا").startswith("NotoSansArabic") and font_for("नमस्ते").startswith("NotoSansDev") and font_for("ÉA") == "Anton-Regular.ttf"
        ca = card(t / "l.png", "مرحبا بك", t / "ca.mp4"); assert abs(dur(ca) - CARD_S) < 0.15  # a non-Latin card renders
        j = t / "m.json"; assert jload(j) == []; j.write_text(""); assert jload(j) == [] and (t / "m.json.corrupt").exists(); jsave(j, [1]); assert jload(j) == [1] and not list(t.glob("*.tmp"))
        print("selftest ok (card %.2fs, joined %.2fs, 1080x1920)" % (dur(c), d)); shutil.rmtree(t, ignore_errors=True); sys.exit()
    main(*sys.argv[1:3])

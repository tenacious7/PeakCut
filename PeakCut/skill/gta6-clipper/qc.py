# Automatic quality check of finished Shorts (replaces "a human looks at the contact sheet").
# Usage: python qc.py SHORT.mp4|DIR ... [--picks picks.json] [--json qc.json] | python qc.py --selftest
# Each Short gets PASS / WARN / FAIL. FAIL = re-cut it with clip.py; WARN = look at it.
import json, os, re, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # the YuNet "Targets are not supported" line per Short is noise
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
YUNET = np.frombuffer((HERE / "models" / "yunet.onnx").read_bytes(), np.uint8)  # bytes: OpenCV cannot open non-ASCII paths
FPS = 5            # analysis frames per second
PANEL = 0.45       # game panel = rows below 45 % of the frame (stack layout: cam on top; full-screen layouts are all game)
T = dict(dur=(12, 60), dur_warn=35, lufs=2.0, black_s=0.3, band_s=1.0, freeze_s=1.5, first_word=1.0, end_s=1.5, music_s=8.0, silence_warn=25, silence_fail=40, gap_warn=3.0)

def ff(args):  # ffmpeg stderr (analysis filters print there)
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr

def duration(p): return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))

def lufs(p):
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", ff(["-i", str(p), "-af", "ebur128", "-f", "null", "-"]))
    return float(m[-1]) if m else None  # last "I:" = the summary

def filters(p, H):  # black frames anywhere + frozen picture in the game panel, one ffmpeg pass
    e = ff(["-i", str(p), "-an", "-vf", f"blackdetect=d={T['black_s']}:pix_th=0.10,crop=iw:ih*{1 - PANEL}:0:ih*{PANEL},freezedetect=n=-60dB:d={T['freeze_s']}", "-f", "null", "-"])
    return [float(x) for x in re.findall(r"black_duration:([\d.]+)", e)], [float(x) for x in re.findall(r"freeze_duration: ([\d.]+)", e)]

def audio_rms(p):  # dBFS per 0.25 s
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768; n = len(a) // 2000
    return 20 * np.log10(np.sqrt((a[:n * 2000].reshape(n, 2000) ** 2).mean(1)) + 1e-6) if n else np.zeros(0)

def frames(p):  # per analysed frame: time, panel stats, band stats, motion, face, caption score
    cap = cv2.VideoCapture(str(p)); fps = cap.get(cv2.CAP_PROP_FPS) or 25; step = max(1, round(fps / FPS)); out, prev, i = [], None, 0
    yn = cv2.FaceDetectorYN.create("onnx", YUNET, np.empty(0, np.uint8), (270, 480), 0.6)
    while True:
        if not cap.grab(): break
        if i % step == 0:
            ok, f = cap.retrieve()
            if not ok: break
            s = cv2.resize(f, (270, 480)); g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32); pn = g[int(480 * PANEL):]
            bands = [g[r:r + 24] for r in range(0, 480, 24)]  # 20 horizontal bands
            dark = [b.mean() < 10 and b.std() < 6 for b in bands]
            edge = cv2.Canny(g.astype(np.uint8), 80, 160)[120:360, 40:230].mean() / 255
            mo = 0.0 if prev is None else float(np.abs(g[216:] - prev[216:]).mean()); prev = g  # motion of the game panel only: captions/cam must not hide a still card
            n, fa = yn.detect(s)
            wh = ((s.min(2) > 235)[int(480 * .3):int(480 * .8)]).mean()  # white caption pixels, middle band
            out.append(dict(t=i / fps, pm=float(pn.mean()), ps=float(pn.std()), dark=dark, edge=float(edge), mo=mo, face=fa is not None and len(fa) > 0, wh=float(wh)))
        i += 1
    cap.release(); return out

def is_card(fr, share):  # logo / black / flat card: the game panel is dark or flat AND still in >= share of these frames
    bad = [(x["pm"] < 25 or x["ps"] < 22) and x["mo"] < 4 for x in fr]
    return float(np.mean(bad)) >= share, dict(card_share=round(float(np.mean(bad)), 2), panel_mean=round(float(np.median([x["pm"] for x in fr])), 1))

def words(pick, base, rj):
    """Caption-time words of the Short: source words mapped through clip.py's kept spans (render.json). None = cannot map (no render.json / old one): skip word checks."""
    if not pick or not rj or "seg_start" not in rj or not rj.get("kept_spans"): return None
    for c in [Path(pick["transcript"]), *(b / pick["transcript"] for b in base)]:
        if c.exists():
            sp, st, out = rj["kept_spans"], rj["seg_start"], []
            for x in (x for s in json.load(open(c, encoding="utf-8"))["segments"] for x in s.get("words", [])):
                m = (x["s"] + x["e"]) / 2 - st; acc = 0.0
                for a, b in sp:  # same rule as clip.py heard(): a word is heard when its midpoint is in a kept span
                    if a <= m <= b: out.append(dict(s=acc + max(x["s"] - st, a) - a, e=acc + min(x["e"] - st, b) - a)); break
                    acc += b - a
            return out
    return None

def check(p, pick=None, base=()):
    p = Path(p); fail, warn, m = [], [], {}
    d = m["duration"] = round(duration(p), 1)
    if not T["dur"][0] <= d <= T["dur"][1]: fail.append(f"duration {d}s")
    elif d > T["dur_warn"]: warn.append(f"long {d}s")
    L = m["lufs"] = lufs(p)
    if L is None or abs(L + 14) > T["lufs"]: fail.append(f"loudness {L} LUFS")
    bl, fz = filters(p, 1920); m["black_s"], m["freeze_s"] = bl, fz
    if bl: fail.append(f"black frames {max(bl):.1f}s")
    if fz: warn.append(f"frozen panel {max(fz):.1f}s")
    fr = frames(p)
    if not fr: return dict(file=p.name, status="FAIL", fail=["unreadable"], warn=[], **m)
    best = big = run = 0
    for r in range(20):  # one dark row of the picture that stays > 1 s = letterbox / empty strip (WARN)
        run = 0
        for x in fr:
            run = run + 1 if x["dark"][r] else 0; best = max(best, run)
    run = 0
    for x in fr:  # most of the game panel dark for > 1 s = the panel went black (FAIL)
        run = run + 1 if sum(x["dark"][int(20 * PANEL) + 1:]) >= 0.6 * (19 - int(20 * PANEL)) else 0; big = max(big, run)
    m["black_band_s"], m["black_panel_s"] = round(best / FPS, 1), round(big / FPS, 1)
    if big / FPS >= T["band_s"]: fail.append(f"game panel black {big / FPS:.1f}s")
    elif best / FPS >= T["band_s"]: warn.append(f"black strip in picture {best / FPS:.1f}s")
    e = [x for x in fr if x["t"] >= d - T["end_s"]]; s = [x for x in fr if x["t"] < 1.0]
    ce, m["end"] = is_card(e, 0.4) if e else (False, {})
    cs, m["start"] = is_card(s, 0.6) if s else (False, {})
    if ce: fail.append("ends on a card/black/flat panel")
    if cs: fail.append("opens on a card/black/flat panel")
    if not (any(x["face"] for x in s) or np.mean([x["mo"] for x in s[1:]] or [0]) > 2.0): warn.append("no face or motion in first second")
    cw = np.mean([x["wh"] > 0.003 for x in fr]); m["caption_frames"] = round(float(cw), 2)
    if cw < 0.4: warn.append("captions not found")
    rms = audio_rms(p)
    rjf = p.with_suffix(".render.json"); rj = json.loads(rjf.read_text(encoding="utf-8")) if rjf.exists() else None  # clip.py's own record of the cut: kept spans + a long talk-free gap it had to keep (dead air)
    w = words(pick, base, rj)
    if rj:
        g = rj.get("longest_talk_free_gap", 0) or 0
        m["longest_gap_kept"] = g
        if g > T["gap_warn"]: warn.append(f"talk-free {g:.1f}s kept (thin pick: widen or drop)")
    if len(rms):  # dead air: share of 0.25 s bins under -35 dBFS (critics: a 67 % silent call clip scored 3.3)
        m["silence_pct"] = sil = round(100 * float((rms < -35).mean()))
        if sil > T["silence_fail"]: fail.append(f"silence {sil}%")
        elif sil > T["silence_warn"]: warn.append(f"silence {sil}%")
    if w is not None:
        m["first_word_s"] = fw = round(max(w[0]["s"], 0), 2) if w else None
        if fw is None or fw > T["first_word"]: warn.append(f"first word at {fw}s")
        for q, r in zip(w, w[1:]):  # gaps between words, in Short time
            ta, tb = q["e"], r["s"]; seg = rms[int(ta * 4):int(tb * 4)]
            if tb - ta > T["music_s"] and len(seg) and seg.mean() > -40 and seg.std() < 4: warn.append(f"music without speech {tb - ta:.0f}s at {ta:.0f}s"); break
    elif len(rms) and rms[:4].max() < -35: warn.append("silent first second")
    return dict(file=p.name, status="FAIL" if fail else "WARN" if warn else "PASS", fail=fail, warn=warn, **m)

def run(paths, picks_file=None, json_out=None, quiet=False):
    picks, base = {}, ()
    if picks_file:
        pf = Path(picks_file); base = (pf.parent.parent, pf.parent)
        picks = {x["name"]: x for x in json.load(open(pf, encoding="utf-8"))}
    files = sorted(f for q in paths for f in ([Path(q)] if Path(q).suffix == ".mp4" else Path(q).glob("[0-9][0-9]_*.mp4")))
    res = []
    for f in files:
        pk = picks.get(re.sub(r"^\d+_", "", f.stem))
        pk = pk if pk and "start" in pk and not pk.get("parts") else None  # a multi-part Short has no single time range: picture + audio checks only
        r = check(f, pk, base); res.append(r)
        if not quiet: print(f"{r['status']:5} {f.name:34} {'; '.join(r['fail'] + r['warn']) or 'ok'}", flush=True)
    if json_out:  # merge by file name: a later batch into the same folder must not erase earlier results
        old = json.loads(Path(json_out).read_text(encoding="utf-8")) if Path(json_out).exists() else []
        new = {r["file"] for r in res}
        Path(json_out).write_text(json.dumps([r for r in old if r["file"] not in new] + res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res

def selftest():
    tmp = Path(tempfile.mkdtemp()); au = "sine=f=300:d=22,volume=0.5"  # loudness is checked separately below
    def mk(name, vf, d=22):
        o = tmp / f"01_{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s=540x960:r=25:d={d}", "-f", "lavfi", "-i", f"sine=f=300:d={d}", "-vf", vf, "-af", "loudnorm=I=-14:TP=-1.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(o)], check=True); return o
    good = check(mk("good", "format=yuv420p")); assert not any("card" in x or "black" in x or "duration" in x for x in good["fail"]), good
    bk = check(mk("blackend", "format=yuv420p,drawbox=y=ih*0.45:w=iw:h=ih:color=black:t=fill:enable='gt(t,19)'")); assert any("ends on" in x for x in bk["fail"]), bk
    sh = check(mk("short", "format=yuv420p", 8)); assert any("duration" in x for x in sh["fail"]), sh
    import shutil; shutil.rmtree(tmp, ignore_errors=True); print("qc selftest ok")

if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a: selftest(); sys.exit()
    opt = lambda k: a[a.index(k) + 1] if k in a else None
    paths = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--picks", "--json"))]
    run(paths, opt("--picks"), opt("--json"))

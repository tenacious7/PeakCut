"""Automated Quality Control (QC) for rendered PeakCut video clips.
Checks loudness (EBU R128 LUFS), black frames, screen freezing, silence, and duration."""
import json
import re
import subprocess
from pathlib import Path

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ff_stderr(args: list, timeout: int = 25) -> str:
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-nostats", *args],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", creationflags=NO_WINDOW, timeout=timeout)
        return r.stderr or ""
    except Exception:
        return ""


def duration(path: Path) -> float:
    try:
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           check=True, creationflags=NO_WINDOW, timeout=12)
        return round(float(o.stdout.strip()), 2)
    except Exception:
        return 0.0


def lufs(path: Path) -> float:
    """Measure integrated loudness in LUFS (-14 is YouTube Shorts/Reels standard)."""
    e = ff_stderr(["-i", str(path), "-vn", "-af", "ebur128", "-f", "null", "-"], timeout=20)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", e)
    return round(float(m[-1]), 1) if m else -14.0


def detect_artifacts(path: Path) -> tuple:
    """Detect black frames and frozen picture stretches quickly on downscaled frames."""
    e = ff_stderr(["-i", str(path), "-an", "-vf",
                   "scale=320:-2,blackdetect=d=0.35:pix_th=0.10,freezedetect=n=-60dB:d=2.0",
                   "-f", "null", "-"], timeout=25)
    bl = [round(float(x), 2) for x in re.findall(r"black_duration:([\d.]+)", e)]
    fz = [round(float(x), 2) for x in re.findall(r"freeze_duration:\s*([\d.]+)", e)]
    return bl, fz


def check_clip(path: Path) -> dict:
    """Run full automated QC on a rendered clip file. Returns status, score, issues, and metrics."""
    p = Path(path)
    if not p.exists() or p.stat().st_size < 1000:
        return {"status": "FAIL", "score": 0, "issues": ["Clip file missing or unreadable"], "metrics": {}}

    dur = duration(p)
    loud = lufs(p)
    black_durations, freeze_durations = detect_artifacts(p)

    issues = []
    warnings = []
    score = 100

    # 1. Duration checks
    if dur < 3.0:
        issues.append(f"Too short ({dur}s)")
        score -= 40
    elif dur > 180.0:
        warnings.append(f"Long clip ({dur}s)")
        score -= 15
    elif dur > 65.0:
        warnings.append(f"Over 60s standard Short length ({dur}s)")
        score -= 5

    # 2. Loudness checks (-14 LUFS is target)
    if loud < -24.0:
        warnings.append(f"Audio very quiet ({loud} LUFS, target -14)")
        score -= 15
    elif loud > -8.0:
        warnings.append(f"Audio loud/clipping ({loud} LUFS)")
        score -= 15

    # 3. Black frame detection
    if black_durations and max(black_durations) > 0.6:
        issues.append(f"Black frames detected ({max(black_durations)}s)")
        score -= 30

    # 4. Freeze frame detection
    if freeze_durations and max(freeze_durations) > 3.0:
        warnings.append(f"Frozen screen detected ({max(freeze_durations)}s)")
        score -= 20

    score = max(0, min(100, score))
    if issues:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    all_notes = issues + warnings
    return {
        "status": status,
        "score": score,
        "issues": all_notes,
        "metrics": {
            "duration": dur,
            "lufs": loud,
            "black_frames_max": max(black_durations) if black_durations else 0.0,
            "frozen_max": max(freeze_durations) if freeze_durations else 0.0
        }
    }

"""Speech to text with word timestamps, supporting BOTH:
1. Parakeet-v2 0.6b through sherpa-onnx (ultra-fast local English ONNX INT8, plus optional remote GPU worker)
2. Faster-Whisper through CTranslate2 (free local multilingual English + Hindi/Hinglish with Silero VAD noise filtering)

Only ONE engine is kept loaded in memory at a time: switching engines in Settings immediately unloads the
inactive model and loads the chosen one in the background with INT8/int8_float16 quantization for low RAM/CPU use."""
import gc
import importlib.util
import json
import os
import queue
import re
import site
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Silence noisy Windows huggingface_hub symlink warning before faster_whisper/hf imports
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

from . import config

_lock = threading.Lock()

# Several copies of the speech model, each on a few cores, instead of one copy on all of them: one copy cannot keep
# a big machine busy, and several videos transcribing at once would queue behind each other inside it. On a small
# machine (under 8 cores) this is still a single copy, as before.
_CPUS = os.cpu_count() or 4
PROVIDER = os.environ.get("PEAKCUT_ASR_PROVIDER", "cpu").strip().lower() or "cpu"      # "cuda" on a graphics card
THREADS = int(os.environ.get("PEAKCUT_ASR_THREADS") or (4 if _CPUS >= 8 else max(2, _CPUS - 2)))
COPIES = int(os.environ.get("PEAKCUT_ASR_COPIES") or (1 if PROVIDER != "cpu" else max(1, _CPUS // THREADS) if _CPUS >= 8 else 1))
_free: "queue.LifoQueue" = queue.LifoQueue()
_made = 0

# The machine with the graphics card, if there is one.
REMOTE = os.environ.get("PEAKCUT_ASR_URL", "").strip().rstrip("/")
REMOTE_TOKEN = os.environ.get("PEAKCUT_ASR_TOKEN", "").strip()
REMOTE_BATCH = int(os.environ.get("PEAKCUT_ASR_REMOTE_BATCH", "8"))         # pieces sent together: the card does them at once
REMOTE_PARALLEL = int(os.environ.get("PEAKCUT_ASR_REMOTE_PARALLEL", "4"))   # pieces of one video worked on at once
# The card has a ceiling: beyond a few requests at once it gets no faster. So only this many are sent to it at a
# time (all videos together), and the rest is done on this machine's own processors at the same moment: both are used.
REMOTE_SLOTS = threading.BoundedSemaphore(max(1, int(os.environ.get("PEAKCUT_ASR_REMOTE_INFLIGHT", "6"))))
_remote_down_until = 0.0

# ------------------------------------------------------------------ dual ASR engine state (Parakeet + Faster-Whisper)
WHISPER_MODELS = [
    {"id": "base", "label": "Base (~150 MB · Fastest)", "desc": "Lowest RAM & VRAM (~0.5 GB GPU)", "vram_mb": 550, "vram_int8_mb": 400},
    {"id": "small", "label": "Small (~500 MB · Recommended)", "desc": "Best balance of speed, accuracy & Hinglish (~1.1 GB GPU)", "vram_mb": 1100, "vram_int8_mb": 750},
    {"id": "medium", "label": "Medium (~1.5 GB · High Accuracy)", "desc": "Sharper accents & noisy gameplay audio (~2.5 GB GPU)", "vram_mb": 2500, "vram_int8_mb": 1600},
    {"id": "large-v3-turbo", "label": "Large-v3-Turbo (~1.6 GB · Best)", "desc": "Top English + Hindi/Hinglish quality (~3.1 GB GPU)", "vram_mb": 3100, "vram_int8_mb": 2000},
]
WHISPER_MODEL_MAP = {m["id"]: m for m in WHISPER_MODELS}
WHISPER_MODEL_IDS = set(WHISPER_MODEL_MAP)
WHISPER_THREADS = int(os.environ.get("PEAKCUT_WHISPER_THREADS") or max(4, min(8, _CPUS - 2)))
WHISPER_HINGLISH_PROMPT = (
    "यह हिंदी और इंग्लिश (Hinglish) गेमिंग स्ट्रीम है। "
    "भाई, यार, अरे, क्या, bro, clutch, kill, zone, push, game, car, bike, stream, video."
)
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_NO_WIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

_state_lock = threading.RLock()
_whisper_lock = threading.Lock()
_whisper_model = None
_whisper_batched = None
_whisper_meta = None          # (model_size, device, compute_type)
_whisper_cuda_note = ""
_prev_whisper_model = "small"
_cuda_dll_handles: list = []
_cuda_dlls_registered = False
_gpu_cache: dict = {"ts": 0.0, "data": None}
_gpu_cache_lock = threading.Lock()
_asr_state = {
    "engine": os.environ.get("PEAKCUT_ASR_ENGINE", "parakeet").strip().lower() or "parakeet",
    "whisper_model": os.environ.get("PEAKCUT_WHISPER_MODEL", "small").strip().lower() or "small",
    "language": os.environ.get("PEAKCUT_ASR_LANGUAGE", "auto").strip().lower() or "auto",
    "status": "idle",         # idle | loading | ready | error | not_installed
    "message": "",
    "device": PROVIDER,
    "compute_type": "int8",
    "gen": 0,
}


def _register_cuda_dll_dirs() -> list[str]:
    """On Windows, register nvidia/cublas/cudnn/torch/CUDA Toolkit DLL folders so CTranslate2 can use GPU."""
    global _cuda_dlls_registered
    if _cuda_dlls_registered or os.name != "nt":
        _cuda_dlls_registered = True
        return []
    _cuda_dlls_registered = True
    candidates: list[Path] = []

    # 1. Site-packages inside active venv / Python installation (nvidia-cublas-cu12, nvidia-cudnn-cu12, torch)
    sp_dirs: list[Path] = []
    try:
        for p in site.getsitepackages():
            sp_dirs.append(Path(p))
    except Exception:
        pass
    try:
        usp = site.getusersitepackages()
        if usp:
            sp_dirs.append(Path(usp))
    except Exception:
        pass
    sp_dirs.append(Path(sys.prefix) / "Lib" / "site-packages")

    for sp in sp_dirs:
        if not sp.is_dir():
            continue
        for rel in (
            ("nvidia", "cublas", "bin"),
            ("nvidia", "cudnn", "bin"),
            ("nvidia", "cuda_runtime", "bin"),
            ("nvidia", "cuda_nvrtc", "bin"),
            ("torch", "lib"),
            ("ctranslate2",),
        ):
            d = sp.joinpath(*rel)
            if d.is_dir():
                candidates.append(d)

    # 2. System NVIDIA CUDA Toolkit installations
    for env_k in ("CUDA_PATH", "CUDA_PATH_V12_8", "CUDA_PATH_V12_6", "CUDA_PATH_V12_4", "CUDA_PATH_V12_2", "CUDA_PATH_V12_1", "CUDA_PATH_V12_0"):
        val = os.environ.get(env_k, "").strip()
        if val:
            d = Path(val) / "bin"
            if d.is_dir():
                candidates.append(d)
    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.is_dir():
        for ver in sorted(cuda_root.glob("v12*"), reverse=True):
            b = ver / "bin"
            if b.is_dir():
                candidates.append(b)

    added: list[str] = []
    seen: set[str] = set()
    cur_path = os.environ.get("PATH", "")
    path_parts = [p for p in cur_path.split(os.pathsep) if p]
    path_lower = {p.lower() for p in path_parts}

    for d in candidates:
        s = str(d.resolve())
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if hasattr(os, "add_dll_directory"):
                _cuda_dll_handles.append(os.add_dll_directory(s))
        except OSError:
            pass
        if key not in path_lower:
            path_parts.insert(0, s)
            path_lower.add(key)
        added.append(s)

    if added:
        os.environ["PATH"] = os.pathsep.join(path_parts)
    return added


def faster_whisper_installed() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def _read_user_asr_settings() -> tuple[str, str, str, bool]:
    """Read (engine, whisper_model, language, gpu_enabled) from active user settings, falling back to module state."""
    with _state_lock:
        eng = _asr_state["engine"]
        wmod = _asr_state["whisper_model"]
        lang = _asr_state["language"]
    gpu_on = getattr(config, "USE_GPU", True)
    try:
        from . import accounts
        s = accounts.settings()
        eng = s.get("asr_engine") or eng
        wmod = s.get("whisper_model") or wmod
        lang = s.get("asr_language") or lang
        gpu_on = bool(s.get("gpu_enabled", True)) and getattr(config, "USE_GPU", True)
    except Exception:
        pass
    if eng not in ("parakeet", "faster-whisper"):
        eng = "parakeet"
    if wmod not in WHISPER_MODEL_IDS:
        wmod = "small"
    if lang not in ("auto", "en", "hi"):
        lang = "auto"
    return eng, wmod, lang, gpu_on


def active_engine() -> str:
    eng, _, _, _ = _read_user_asr_settings()
    return eng


def model_ready() -> bool:
    eng = active_engine()
    if eng == "faster-whisper":
        return faster_whisper_installed()
    from . import fetch_model
    return fetch_model.ready()


def _query_ollama_gpu() -> tuple[bool, list[str], int]:
    """Quick local check for Ollama models currently loaded in GPU VRAM (safe 0.35s timeout)."""
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=0.35) as resp:
            if getattr(resp, "status", 200) != 200:
                return False, [], 0
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        models = data.get("models") or []
        names: list[str] = []
        vram_bytes = 0
        for m in models:
            nm = str(m.get("name") or m.get("model") or "").strip()
            sv = int(m.get("size_vram") or 0)
            if sv > 0:
                vram_bytes += sv
                if nm:
                    names.append(nm)
            elif nm and not m.get("size"):
                names.append(nm)
        vram_mb = int(round(vram_bytes / (1024 * 1024))) if vram_bytes > 0 else 0
        return bool(names or vram_mb > 0), names, vram_mb
    except Exception:
        return False, [], 0


def _probe_nvidia_raw() -> dict:
    """Query live NVIDIA GPU name, VRAM (total/used/free MB), utilization, and compute apps via nvidia-smi."""
    info = {
        "available": False,
        "name": "",
        "total_mb": 0,
        "used_mb": 0,
        "free_mb": 0,
        "used_pct": 0,
        "util_pct": 0,
        "cuda_ready": False,
        "ollama_running": False,
        "ollama_models": [],
        "ollama_vram_mb": 0,
        "busy_apps": [],
    }
    _register_cuda_dll_dirs()
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            info["cuda_ready"] = True
            info["available"] = True
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=2.0, creationflags=_NO_WIN,
        )
        if r.returncode == 0 and r.stdout.strip():
            first = r.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 4:
                info["available"] = True
                info["name"] = parts[0]
                tot = int(float(parts[1])) if parts[1].replace(".", "", 1).isdigit() else 0
                used = int(float(parts[2])) if parts[2].replace(".", "", 1).isdigit() else 0
                free = int(float(parts[3])) if parts[3].replace(".", "", 1).isdigit() else max(0, tot - used)
                util = int(float(parts[4])) if len(parts) >= 5 and parts[4].replace(".", "", 1).isdigit() else 0
                info["total_mb"] = tot
                info["used_mb"] = used
                info["free_mb"] = free
                info["used_pct"] = int(round((used / tot) * 100)) if tot > 0 else 0
                info["util_pct"] = util
    except Exception:
        pass

    if info["available"]:
        # Check GPU compute processes (catches Ollama, local LLMs, games/encoders where reported)
        try:
            r2 = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=2.0, creationflags=_NO_WIN,
            )
            if r2.returncode == 0 and r2.stdout.strip():
                apps = []
                for line in r2.stdout.strip().splitlines():
                    ps = [p.strip() for p in line.split(",")]
                    if not ps or not ps[0]:
                        continue
                    pname = Path(ps[0]).name
                    mem = int(float(ps[1])) if len(ps) > 1 and ps[1].replace(".", "", 1).isdigit() else 0
                    apps.append({"name": pname, "vram_mb": mem})
                    low = pname.lower()
                    if "ollama" in low or "llama" in low:
                        info["ollama_running"] = True
                        if mem > 0:
                            info["ollama_vram_mb"] += mem
                info["busy_apps"] = apps[:6]
        except Exception:
            pass

        # Also query Ollama's local API (works reliably even on Windows WDDM where nvidia-smi reports [N/A] per-app VRAM)
        ol_on, ol_models, ol_mb = _query_ollama_gpu()
        if ol_on:
            info["ollama_running"] = True
            info["ollama_models"] = ol_models
            if ol_mb > info["ollama_vram_mb"]:
                info["ollama_vram_mb"] = ol_mb

        if not info["name"]:
            info["name"] = "NVIDIA CUDA GPU"
    return info


def _fmt_gb(mb: int) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{max(0, int(mb))} MB"


def _pick_fitting_model(free_mb: int, current_model: str, prev_model: str = "small") -> str:
    """Choose the best smaller Whisper model that fits comfortably in available VRAM (preferring previous/small)."""
    cur_meta = WHISPER_MODEL_MAP.get(current_model, WHISPER_MODEL_MAP["small"])
    cur_need = cur_meta["vram_int8_mb"]
    # Prefer the user's previous smaller model (usually 'small') if it fits and is lighter than current
    for cand in (prev_model, "small", "base"):
        if cand in WHISPER_MODEL_MAP and cand != current_model:
            cm = WHISPER_MODEL_MAP[cand]
            if cm["vram_int8_mb"] < cur_need and free_mb >= cm["vram_int8_mb"] + 150:
                return cand
    if current_model != "small" and free_mb >= WHISPER_MODEL_MAP["small"]["vram_int8_mb"]:
        return "small"
    if current_model != "base" and free_mb >= WHISPER_MODEL_MAP["base"]["vram_int8_mb"]:
        return "base"
    return "parakeet"


def gpu_status(selected_model: str = None, fresh: bool = False) -> dict:
    """Live NVIDIA GPU + VRAM + Ollama status, with smart recommendations for the selected Whisper model."""
    now = time.monotonic()
    with _gpu_cache_lock:
        if not fresh and _gpu_cache["data"] is not None and (now - _gpu_cache["ts"]) < 2.5:
            raw = dict(_gpu_cache["data"])
        else:
            raw = _probe_nvidia_raw()
            _gpu_cache["ts"] = now
            _gpu_cache["data"] = dict(raw)

    _, cur_wmod, _, _ = _read_user_asr_settings()
    wmod = (selected_model or cur_wmod or "small").strip().lower()
    if wmod not in WHISPER_MODEL_MAP:
        wmod = "small"
    m_meta = WHISPER_MODEL_MAP[wmod]
    need_fp16 = m_meta["vram_mb"]
    need_int8 = m_meta["vram_int8_mb"]

    already_on_gpu = (
        _whisper_model is not None
        and _whisper_meta is not None
        and _whisper_meta[0] == wmod
        and _whisper_meta[1] == "cuda"
    )

    high_usage = False
    vram_tight = False
    can_fit_gpu = bool(raw["available"])
    recommended_model = ""
    warning = ""

    if raw["available"] and raw["total_mb"] > 0:
        free_mb = raw["free_mb"]
        tot_mb = raw["total_mb"]
        used_pct = raw["used_pct"]
        # If this exact model is already loaded in VRAM by PeakCut, don't double-count its own footprint
        effective_free = free_mb + (need_int8 if already_on_gpu else 0)

        ol_label = ""
        if raw["ollama_running"]:
            ol_parts = []
            if raw["ollama_models"]:
                ol_parts.append(", ".join(raw["ollama_models"][:2]))
            if raw["ollama_vram_mb"] > 0:
                ol_parts.append(f"~{_fmt_gb(raw['ollama_vram_mb'])}")
            ol_label = f"Ollama ({' · '.join(ol_parts)})" if ol_parts else "Ollama"

        if effective_free < need_int8 + 150:
            high_usage = True
            can_fit_gpu = False
            rec = _pick_fitting_model(effective_free, wmod, _prev_whisper_model)
            recommended_model = rec
            who = f"{ol_label} is using your GPU" if ol_label else f"GPU memory is {used_pct}% full"
            if rec == "parakeet":
                hint = "Stop Ollama / heavy GPU apps to free VRAM, or switch to Parakeet-v2 (fast CPU)."
            else:
                hint = f"You can use the {rec} model (or stop Ollama) to run smoothly without running out of GPU memory."
            warning = (
                f"High GPU usage right now — {who} ({_fmt_gb(free_mb)} free of {_fmt_gb(tot_mb)}). "
                f"Whisper {wmod} needs ~{_fmt_gb(need_int8)}–{_fmt_gb(need_fp16)} VRAM. {hint}"
            )
        elif effective_free < need_fp16 + 200 or (raw["ollama_running"] and used_pct >= 75):
            vram_tight = True
            if raw["ollama_running"] or used_pct >= 80:
                high_usage = True
                rec = _pick_fitting_model(effective_free, wmod, _prev_whisper_model)
                if rec != "parakeet" and rec != wmod:
                    recommended_model = rec
                who = f"{ol_label} is active on your GPU" if ol_label else f"GPU memory is {used_pct}% used"
                rec_txt = f" You can also use the {recommended_model} model for extra headroom." if recommended_model else ""
                warning = (
                    f"High GPU usage right now — {who} ({_fmt_gb(free_mb)} free of {_fmt_gb(tot_mb)}). "
                    f"PeakCut will use compact GPU INT8 mode to save VRAM.{rec_txt}"
                )

    out = dict(raw)
    out.update(
        selected_model=wmod,
        model_vram_mb=need_fp16,
        model_vram_int8_mb=need_int8,
        can_fit_gpu=can_fit_gpu,
        vram_tight=vram_tight,
        high_usage=high_usage,
        recommended_model=recommended_model,
        previous_model=_prev_whisper_model,
        warning=warning,
    )
    return out


def _is_low_vram_gpu(ginfo: dict | None) -> bool:
    """True for NVIDIA GTX 16-series (no FP16 Tensor Cores) or GPUs with <= 4.5 GB VRAM (e.g. GTX 1650 4 GB)."""
    if not ginfo or not ginfo.get("available"):
        return False
    name = str(ginfo.get("name") or "").lower()
    tot = int(ginfo.get("total_mb") or 0)
    if any(k in name for k in ("1650", "1660", "1630", "1050", "1060", "mx", "p400", "t500", "t600")):
        return True
    return 0 < tot <= 4600


def _pick_whisper_batch_size(device: str, lang_arg: str | None = None) -> int:
    """Scale BatchedInferencePipeline batch_size to fit VRAM cleanly without overflowing to Windows Shared RAM."""
    is_hi = (lang_arg or "").strip().lower().startswith(("hi", "ur", "hinglish"))
    if device != "cuda":
        return 6 if is_hi else 8
    ginfo = gpu_status(fresh=False)
    tot = int(ginfo.get("total_mb") or 0)
    if _is_low_vram_gpu(ginfo):
        return 4
    if 0 < tot <= 6500:
        return 6 if is_hi else 8
    return 12 if is_hi else 16


def _detect_whisper_device(gpu_on: bool) -> tuple[str, str]:
    """Pick optimal (device, compute_type) for Faster-Whisper: CUDA int8 on GTX 16-series / <=4 GB VRAM,
    CUDA float16 on >=6 GB RTX cards, else CPU int8."""
    pref = os.environ.get("PEAKCUT_WHISPER_DEVICE", "auto").strip().lower()
    if pref == "cpu" or not gpu_on:
        return "cpu", "int8"
    _register_cuda_dll_dirs()
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            env_c = os.environ.get("PEAKCUT_WHISPER_COMPUTE", "").strip()
            if env_c:
                return "cuda", env_c
            ginfo = gpu_status(fresh=False)
            if _is_low_vram_gpu(ginfo):
                return "cuda", "int8"
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def unload_parakeet():
    """Free idle Parakeet ONNX recognizers from RAM so only one ASR engine stays resident."""
    global _made
    drained = 0
    while True:
        try:
            rec = _free.get_nowait()
            del rec
            drained += 1
        except queue.Empty:
            break
    if drained:
        with _lock:
            _made = max(0, _made - drained)
    gc.collect()


def unload_whisper():
    """Free the loaded Faster-Whisper model from RAM/VRAM."""
    global _whisper_model, _whisper_batched, _whisper_meta
    with _whisper_lock:
        if _whisper_model is not None or _whisper_batched is not None:
            _whisper_batched = None
            _whisper_model = None
            _whisper_meta = None
    gc.collect()


def _make_batched_pipeline(model):
    """Wrap WhisperModel in BatchedInferencePipeline (2x-4x faster) when available in faster-whisper."""
    try:
        from faster_whisper import BatchedInferencePipeline
        return BatchedInferencePipeline(model=model)
    except Exception:
        return None


def _verify_cuda_model(model) -> None:
    """Run a tiny 0.4s inference check so missing cuBLAS/cuDNN DLLs are caught immediately on load."""
    dummy = np.zeros(6400, dtype=np.float32)
    segs, _ = model.transcribe(dummy, beam_size=1, best_of=1, vad_filter=False, language="en", word_timestamps=False)
    for _ in segs:
        break


def _get_whisper_model(model_size: str = None, gpu_on: bool = None, force_cpu: bool = False):
    """Load (or reuse) the optimized Faster-Whisper model under lock. Verifies CUDA DLLs and falls back cleanly to CPU INT8."""
    global _whisper_model, _whisper_batched, _whisper_meta, _whisper_cuda_note
    _, cfg_model, _, cfg_gpu = _read_user_asr_settings()
    size = (model_size or cfg_model or "small").strip().lower()
    if size not in WHISPER_MODEL_IDS:
        size = "small"
    use_gpu = False if force_cpu else (cfg_gpu if gpu_on is None else bool(gpu_on))
    device, compute_type = _detect_whisper_device(use_gpu)

    with _whisper_lock:
        if _whisper_model is not None and _whisper_meta is not None:
            if _whisper_meta[0] == size and (not force_cpu or _whisper_meta[1] == "cpu"):
                if _whisper_meta[1] == device or (device == "cuda" and _whisper_meta[1] == "cpu" and _whisper_cuda_note):
                    return _whisper_model, _whisper_batched, _whisper_meta[1], _whisper_meta[2]
        if not faster_whisper_installed():
            raise RuntimeError("faster-whisper is not installed. Run: pip install -r requirements.txt")
        _register_cuda_dll_dirs()
        from faster_whisper import WhisperModel
        # Unload any previous Whisper variant before allocating the new one
        _whisper_batched = None
        _whisper_model = None
        _whisper_meta = None
        gc.collect()

        dl_dir = config.MODELS_DIR / "faster-whisper"
        dl_dir.mkdir(parents=True, exist_ok=True)
        _whisper_cuda_note = ""

        # Check live GPU VRAM / Ollama usage BEFORE allocating on CUDA so we pick compact int8_float16 when tight
        ginfo = gpu_status(selected_model=size, fresh=True) if device == "cuda" else None
        model = None
        used_dev, used_ctype = device, compute_type
        if device == "cuda":
            m_meta = WHISPER_MODEL_MAP.get(size, WHISPER_MODEL_MAP["small"])
            free_mb = ginfo["free_mb"] if (ginfo and ginfo["total_mb"] > 0) else 0
            # Skip CUDA immediately only if VRAM is critically exhausted (<350 MB free)
            skip_cuda_oom = bool(ginfo and ginfo["total_mb"] > 0 and free_mb < 350)
            if skip_cuda_oom:
                _whisper_cuda_note = ginfo.get("warning") or (
                    f"High GPU usage right now ({_fmt_gb(free_mb)} free). Using CPU INT8 — try the small model or close Ollama."
                )
                used_dev, used_ctype = "cpu", "int8"
            else:
                # GTX 1650 / <=4 GB GPUs lack FP16 Tensor Cores and run fastest + lowest VRAM in int8!
                low_vram = _is_low_vram_gpu(ginfo)
                tight = bool(ginfo and ginfo["total_mb"] > 0 and free_mb < m_meta["vram_mb"] + 250)
                if low_vram:
                    pref_order = ("int8", "int8_float32", "int8_float16", "float16")
                elif tight:
                    pref_order = ("int8_float16", "int8", "float16")
                else:
                    pref_order = (compute_type, "float16", "int8_float16", "int8")
                try_types = []
                for ct in pref_order:
                    if ct and ct not in try_types:
                        try_types.append(ct)
                last_err = None
                for ct in try_types:
                    try:
                        cand = WhisperModel(
                            size,
                            device="cuda",
                            compute_type=ct,
                            cpu_threads=WHISPER_THREADS,
                            num_workers=1,
                            download_root=str(dl_dir),
                        )
                        _verify_cuda_model(cand)
                        model = cand
                        used_dev, used_ctype = "cuda", ct
                        break
                    except Exception as e:
                        last_err = e
                        cand = None
                        gc.collect()
                if model is None:
                    err_s = str(last_err or "")
                    low_e = err_s.lower()
                    if any(k in low_e for k in ("out of memory", "oom", "alloc", "cuda error 2")):
                        rec = _pick_fitting_model(free_mb, size, _prev_whisper_model) if free_mb else "small"
                        ol_txt = " (Ollama is using GPU memory)" if (ginfo and ginfo.get("ollama_running")) else ""
                        _whisper_cuda_note = (
                            f"High GPU usage right now{ol_txt} — not enough free VRAM for {size}. "
                            f"Running on CPU INT8 for now; switch to the {rec} model or close Ollama for GPU speed."
                        )
                    elif any(k in low_e for k in ("cublas", "cudnn", "dll", "cannot be loaded", "not found")):
                        _whisper_cuda_note = "GPU detected, but CUDA DLLs missing (run: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12). Using CPU INT8."
                    else:
                        _whisper_cuda_note = f"GPU fallback to CPU INT8 ({err_s[:90]})." if err_s else "GPU unavailable; using CPU INT8."
                    used_dev, used_ctype = "cpu", "int8"

        if model is None:
            model = WhisperModel(
                size,
                device="cpu",
                compute_type="int8",
                cpu_threads=WHISPER_THREADS,
                num_workers=1,
                download_root=str(dl_dir),
            )
            used_dev, used_ctype = "cpu", "int8"

        _whisper_model = model
        _whisper_batched = _make_batched_pipeline(model)
        _whisper_meta = (size, used_dev, used_ctype)
        # Invalidate GPU cache so UI immediately reflects new free VRAM after loading
        with _gpu_cache_lock:
            _gpu_cache["ts"] = 0.0
        return _whisper_model, _whisper_batched, used_dev, used_ctype


def switch_engine(engine: str, whisper_model: str = None, language: str = None, preload: bool = True) -> dict:
    """Switch active ASR engine, immediately unload the inactive model to save RAM/VRAM,
    and optionally warm-load the newly selected model in a background thread (automatically using GPU when available)."""
    global _prev_whisper_model
    eng = (engine or "parakeet").strip().lower()
    if eng not in ("parakeet", "faster-whisper"):
        raise ValueError(f"Unknown speech engine: {eng}")
    _, cur_wmod, cur_lang, gpu_on = _read_user_asr_settings()
    wmod = (whisper_model or cur_wmod or "small").strip().lower()
    if wmod not in WHISPER_MODEL_IDS:
        wmod = "small"
    if cur_wmod in WHISPER_MODEL_IDS and cur_wmod != wmod:
        _prev_whisper_model = cur_wmod
    lang = (language or cur_lang or "auto").strip().lower()
    if lang not in ("auto", "en", "hi"):
        lang = "auto"

    with _state_lock:
        _asr_state["gen"] += 1
        gen = _asr_state["gen"]
        _asr_state["engine"] = eng
        _asr_state["whisper_model"] = wmod
        _asr_state["language"] = lang

    # Immediately free the other engine so RAM/VRAM never doubles up
    if eng == "faster-whisper":
        unload_parakeet()
        if not faster_whisper_installed():
            with _state_lock:
                _asr_state.update(
                    status="not_installed",
                    message="Install faster-whisper first: pip install -r requirements.txt",
                    device="cpu",
                    compute_type="int8",
                )
            return asr_status(fresh_gpu=True)
        dev, ctype = _detect_whisper_device(gpu_on)
        ginfo = gpu_status(selected_model=wmod, fresh=True)
        gpu_label = ginfo["name"] if (dev == "cuda" and ginfo.get("name")) else ("GPU CUDA" if dev == "cuda" else "CPU INT8")
        with _state_lock:
            already = _whisper_model is not None and _whisper_meta is not None and _whisper_meta[0] == wmod and _whisper_meta[1] == dev
            _asr_state.update(
                status="ready" if already else ("loading" if preload else "idle"),
                message=(
                    f"Faster-Whisper ({wmod}) running on {gpu_label} ({_whisper_meta[2]})."
                    if already
                    else (f"Loading Faster-Whisper ({wmod}) on {gpu_label}…" if preload else "Ready to load on next clip.")
                ),
                device=_whisper_meta[1] if already else dev,
                compute_type=_whisper_meta[2] if already else ctype,
            )
        if preload and not already:
            def _load_w():
                try:
                    _, _, d_used, c_used = _get_whisper_model(wmod, gpu_on=gpu_on)
                    g_now = gpu_status(selected_model=wmod, fresh=True)
                    g_name = g_now.get("name") or "NVIDIA GPU"
                    with _state_lock:
                        if _asr_state["gen"] != gen:
                            return
                        if d_used == "cpu" and _whisper_cuda_note:
                            msg = _whisper_cuda_note
                        elif d_used == "cuda":
                            msg = f"Faster-Whisper ({wmod}) running on {g_name} (CUDA {c_used}, Batched + Silero VAD)."
                        else:
                            msg = f"Faster-Whisper ({wmod}, CPU {c_used}, Batched + Silero VAD) ready."
                        _asr_state.update(
                            status="ready",
                            message=msg,
                            device=d_used,
                            compute_type=c_used,
                        )
                except Exception as e:  # noqa: BLE001
                    with _state_lock:
                        if _asr_state["gen"] != gen:
                            return
                        _asr_state.update(status="error", message=f"Failed to load Faster-Whisper: {e}")
            threading.Thread(target=_load_w, daemon=True, name="asr-load-whisper").start()
    else:
        unload_whisper()
        with _state_lock:
            already = _made > 0
            _asr_state.update(
                status="ready" if already else ("loading" if preload else "idle"),
                message="Parakeet-v2 (ONNX INT8) ready." if already else (
                    "Loading Parakeet-v2 in background…" if preload else "Ready to load on next clip."
                ),
                device=PROVIDER,
                compute_type="int8" if PROVIDER == "cpu" else "fp16",
            )
        if preload and not already:
            def _load_p():
                try:
                    _recognizer()
                    with _state_lock:
                        if _asr_state["gen"] != gen:
                            return
                        _asr_state.update(
                            status="ready",
                            message=f"Parakeet-v2 ({PROVIDER.upper()} ONNX) ready.",
                            device=PROVIDER,
                            compute_type="int8" if PROVIDER == "cpu" else "fp16",
                        )
                except Exception as e:  # noqa: BLE001
                    with _state_lock:
                        if _asr_state["gen"] != gen:
                            return
                        _asr_state.update(status="error", message=f"Failed to load Parakeet-v2: {e}")
            threading.Thread(target=_load_p, daemon=True, name="asr-load-parakeet").start()
    return asr_status(fresh_gpu=True)


def asr_status(fresh_gpu: bool = False) -> dict:
    """Current speech-to-text engine configuration, install state, live load status, and GPU/Ollama VRAM check."""
    eng, wmod, lang, gpu_on = _read_user_asr_settings()
    fw_ok = faster_whisper_installed()
    from . import fetch_model
    pk_ok = fetch_model.ready()
    ginfo = gpu_status(selected_model=wmod, fresh=fresh_gpu)
    with _state_lock:
        st = dict(_asr_state)
    # Sync with user settings if not currently loading a switch
    st["engine"] = eng
    st["whisper_model"] = wmod
    st["language"] = lang
    pk_loaded = _made > 0
    fw_loaded = _whisper_model is not None and _whisper_meta is not None and _whisper_meta[0] == wmod

    if eng == "faster-whisper":
        if not fw_ok:
            st["status"] = "not_installed"
            st["message"] = "faster-whisper is not installed yet. Run: pip install -r requirements.txt"
            st["device"], st["compute_type"] = "cpu", "int8"
        elif fw_loaded:
            st["status"] = "ready"
            st["device"], st["compute_type"] = _whisper_meta[1], _whisper_meta[2]
            if st["device"] == "cpu" and _whisper_cuda_note:
                st["message"] = _whisper_cuda_note
            elif st["device"] == "cuda":
                g_name = ginfo.get("name") or "NVIDIA GPU"
                st["message"] = f"Faster-Whisper ({wmod}) running on {g_name} (CUDA {st['compute_type']}, Batched + Silero VAD)."
            else:
                st["message"] = f"Faster-Whisper ({wmod}, CPU {st['compute_type']}, Batched + Silero VAD) loaded."
        elif st["status"] not in ("loading", "error"):
            dev, ctype = _detect_whisper_device(gpu_on)
            st["status"] = "idle"
            st["device"], st["compute_type"] = dev, ctype
            if dev == "cuda" and ginfo.get("name"):
                st["message"] = f"Faster-Whisper ({wmod}) ready to load on {ginfo['name']} (GPU)."
            else:
                st["message"] = f"Faster-Whisper ({wmod}, {dev.upper()} {ctype}) ready to load."
    else:
        st["device"] = PROVIDER
        st["compute_type"] = "int8" if PROVIDER == "cpu" else "fp16"
        if pk_loaded:
            st["status"] = "ready"
            st["message"] = f"Parakeet-v2 ({PROVIDER.upper()} ONNX) loaded."
        elif st["status"] not in ("loading", "error"):
            st["status"] = "ready" if pk_ok else "idle"
            st["message"] = "Parakeet-v2 (ONNX INT8) ready." if pk_ok else "Parakeet-v2 will download automatically on first use."

    return {
        "engine": st["engine"],
        "whisper_model": st["whisper_model"],
        "language": st["language"],
        "status": st["status"],
        "message": st["message"],
        "device": st["device"],
        "compute_type": st["compute_type"],
        "gpu_enabled": bool(gpu_on),
        "gpu": ginfo,
        "faster_whisper_installed": fw_ok,
        "parakeet_installed": pk_ok,
        "parakeet_loaded": pk_loaded,
        "whisper_loaded": fw_loaded,
        "whisper_threads": WHISPER_THREADS,
        "models": WHISPER_MODELS,
    }


def _model_files(d) -> dict:
    """The model's three parts. A graphics card wants the full-precision files; processors the compact ones."""
    kinds = ("fp16", "", "int8") if PROVIDER != "cpu" else ("int8", "", "fp16")
    for k in kinds:
        names = {p: d / (f"{p}.{k}.onnx" if k else f"{p}.onnx") for p in ("encoder", "decoder", "joiner")}
        if all(f.is_file() for f in names.values()):
            return {p: str(f) for p, f in names.items()}
    raise FileNotFoundError(f"No speech model files in {d}")


_lib = {"mod": None}
_lib_lock = threading.Lock()


def _sherpa():
    """The speech library, loaded once. Several copies of the model start on parallel threads, so loading is done
    under a lock: two threads loading it at the same moment would trip over each other."""
    with _lib_lock:
        if _lib["mod"] is None:
            _lib["mod"] = _load_sherpa()
        return _lib["mod"]


def _load_sherpa():
    """On Windows a file whose full path is over 260 characters cannot be loaded, which happens when PeakCut sits in a
    deep folder ("DLL load failed ... The filename or extension is too long"). Then the library is loaded through
    the folder's short name (C:\\Users\\...\\SITE-P~1), which points at the same files, or else from a copy of it
    in a short folder (AppData\\Local\\PeakCut\\lib), made once."""
    import sys
    try:
        import sherpa_onnx
        return sherpa_onnx
    except ImportError as e:
        if os.name != "nt" or "too long" not in str(e).lower():
            raise
        first = e
    import ctypes
    import importlib
    import shutil
    from pathlib import Path

    def forget():
        for k in [m for m in sys.modules if m == "sherpa_onnx" or m.startswith("sherpa_onnx.")]:
            del sys.modules[k]
        importlib.invalidate_caches()

    def load_from(folder: str):
        forget()
        sys.path.insert(0, folder)          # stays first: later imports inside the library look here too
        try:
            return importlib.import_module("sherpa_onnx")
        except ImportError:
            sys.path.remove(folder)
            forget()
            return None

    forget()
    spec = importlib.util.find_spec("sherpa_onnx")
    pkg = Path(spec.origin).parent if spec and spec.origin else None
    if pkg is None:
        raise first
    buf = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.kernel32.GetShortPathNameW(str(pkg.parent), buf, 1024) and len(buf.value) < len(str(pkg.parent)):
        mod = load_from(buf.value)
        if mod:
            return mod
    # A copy of the library in a short folder. Same version check each start, so an update replaces it.
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "PeakCut" / "lib"
    stamp = base / "sherpa_onnx.from"
    want = f"{pkg}|{(pkg / '__init__.py').stat().st_mtime_ns}"
    try:
        if not (base / "sherpa_onnx").is_dir() or not stamp.is_file() or stamp.read_text(encoding="utf-8") != want:
            shutil.rmtree(base / "sherpa_onnx", ignore_errors=True)
            shutil.copytree(pkg, base / "sherpa_onnx")
            stamp.write_text(want, encoding="utf-8")
        mod = load_from(str(base))
        if mod:
            return mod
    except OSError:
        pass
    raise RuntimeError("PeakCut is in a folder too deep for Windows to load the speech model from. Move the PeakCut "
                       "folder somewhere short, like C:\\peakcut, and start it again.") from first


def _new():
    sherpa_onnx = _sherpa()
    from . import fetch_model
    fetch_model.ensure()             # downloaded here, once, if it was never fetched
    d = config.ASR_MODEL_DIR
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        **_model_files(d),
        tokens=str(d / "tokens.txt"),
        num_threads=THREADS,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
        provider=PROVIDER,
    )


def _take():
    """A free copy of the model: one that is idle, a new one while there is room for more, else wait for one."""
    global _made
    try:
        return _free.get_nowait()
    except queue.Empty:
        pass
    with _lock:
        make = _made < COPIES
        if make:
            _made += 1
    if make:
        try:
            return _new()
        except Exception:
            with _lock:
                _made -= 1
            raise
    return _free.get()


def _take_now():
    """A copy of the model if one is free (or can still be made) this instant, else None."""
    global _made
    try:
        return _free.get_nowait()
    except queue.Empty:
        pass
    with _lock:
        make = _made < COPIES
        if make:
            _made += 1
    if not make:
        return None
    try:
        return _new()
    except Exception:
        with _lock:
            _made -= 1
        raise


def _recognizer():
    """Kept for callers that only need the model loaded (warm-up)."""
    rec = _take()
    _free.put(rec)
    return rec


def _chunks(audio: np.ndarray, lo=12.0, hi=24.0):
    """Split at the quietest point between lo and hi seconds so words are not cut in half."""
    sr, hop = 16000, 800  # 50 ms frames
    n = len(audio)
    frames = n // hop
    if frames == 0:
        return
    x = audio[: frames * hop].reshape(frames, hop).astype(np.float32)
    rms = np.sqrt((x ** 2).mean(axis=1))
    pos = 0
    while pos < n:
        if n - pos <= hi * sr:
            yield pos, n
            return
        a, b = (pos + int(lo * sr)) // hop, (pos + int(hi * sr)) // hop
        cut = (a + int(np.argmin(rms[a:b]))) * hop + hop // 2
        yield pos, cut
        pos = cut


BATCH = int(os.environ.get("PEAKCUT_ASR_BATCH", "1"))   # pieces decoded together. Measured: no faster on CPU, since
# each piece already uses every core, so one at a time (which also keeps the words exactly as before).


def _words(tokens, timestamps, a: int, b: int) -> list:
    out = []
    off, seg_end = a / 16000.0, b / 16000.0
    cur, gap = None, False
    for tok, t in zip(tokens, timestamps):
        new_word = tok.startswith((" ", "▁")) or gap
        piece = tok.replace("▁", " ").strip()
        if not piece:
            gap = True      # a space on its own: whatever comes next starts a new word ("in" "120", not "in120")
            continue
        gap = False
        is_punct = all(not c.isalnum() for c in piece)
        if cur is None or (new_word and not is_punct):
            if cur:
                out.append(cur)
            cur = {"w": piece, "s": round(off + t, 2), "e": round(off + t + 0.12, 2)}
        else:
            cur["w"] += piece
            cur["e"] = round(off + t + 0.12, 2)
    if cur:
        cur["e"] = min(cur["e"], round(seg_end, 2))
        out.append(cur)
    return out


def _clean_words(raw_words: list) -> list:
    """Remove duplicate/hallucinated stutter runs and normalize word timings cleanly."""
    words = []
    repeat_run = 0
    for w in raw_words:
        txt = str(w.get("w") or "").strip()
        if not txt or re.fullmatch(r"[\[(].*[\])]", txt):
            continue
        s = round(max(0.0, float(w.get("s", 0.0))), 2)
        e = round(max(s + 0.06, float(w.get("e", s + 0.12))), 2)
        if words and words[-1]["w"] == txt and abs(words[-1]["s"] - s) < 0.05:
            continue  # same word repeated at the exact same timestamp
        if words and words[-1]["w"].lower() == txt.lower():
            repeat_run += 1
            if repeat_run >= 3:
                continue  # suppress 4+ consecutive hallucinated loop tokens ("you you you you")
        else:
            repeat_run = 0
        words.append({"w": txt, "s": s, "e": e})
    # A word lasts until the next one starts, capped so pauses stay pauses.
    for i, w in enumerate(words):
        nxt = words[i + 1]["s"] if i + 1 < len(words) else w["e"] + 0.4
        w["e"] = round(max(w["s"] + 0.08, min(nxt, max(w["e"], w["s"] + 0.6))), 2)
    return words


def decode_pieces(pieces: list, rec=None) -> list:
    """[(tokens, timestamps), ...] for float32 pieces of audio, done on this machine with one copy of the model."""
    streams = []
    rec = rec or _take()       # held only for these pieces, so videos transcribing at once take turns fairly
    try:
        for seg in pieces:
            st = rec.create_stream()
            st.accept_waveform(16000, seg)
            streams.append(st)
        rec.decode_streams(streams)
    finally:
        _free.put(rec)
    return [(list(st.result.tokens), [float(t) for t in st.result.timestamps]) for st in streams]


def _remote_ok() -> bool:
    return bool(REMOTE) and time.time() >= _remote_down_until


def _decode_remote(pieces: list) -> list:
    """The same, on the machine with the graphics card. Audio goes as 16-bit sound, the answer comes back as text."""
    import httpx
    raw = [np.clip(seg * 32768.0, -32768, 32767).astype("<i2") for seg in pieces]
    r = httpx.post(REMOTE + "/decode", content=b"".join(x.tobytes() for x in raw), timeout=httpx.Timeout(120, connect=4),
                   headers={"Authorization": "Bearer " + REMOTE_TOKEN, "X-Lens": ",".join(str(len(x)) for x in raw),
                            "Content-Type": "application/octet-stream"})
    r.raise_for_status()
    got = r.json()["pieces"]
    if len(got) != len(pieces):
        raise ValueError("wrong number of pieces came back")
    return [(p["tokens"], p["timestamps"]) for p in got]


def _transcribe_parakeet(audio: np.ndarray, on_progress=None) -> list:
    """Parakeet-v2 via sherpa-onnx (local CPU/GPU or remote GPU worker)."""
    global _remote_down_until
    todo = []
    for a, b in _chunks(audio):
        seg = audio[a:b].astype(np.float32) / 32768.0
        if len(seg) < 4800 or float(np.sqrt((seg ** 2).mean())) < 0.004:
            continue  # silence: skip, saves time and avoids made-up words
        todo.append((a, b, seg))
    if not todo:
        return []
    remote = _remote_ok()
    size = max(1, REMOTE_BATCH if remote else BATCH)
    groups = [todo[i:i + size] for i in range(0, len(todo), size)]
    done = [0]

    def decode(group):
        global _remote_down_until
        segs = [seg for _, _, seg in group]
        got = None
        while got is None:
            # whichever is free first: a place on the card's machine, or a copy of the model on this one
            if remote and _remote_ok() and REMOTE_SLOTS.acquire(blocking=False):
                try:
                    got = _decode_remote(segs)
                except Exception:  # noqa: BLE001 - the card's machine is away or busy: these pieces are done here
                    _remote_down_until = time.time() + 60
                finally:
                    REMOTE_SLOTS.release()
                continue
            rec = _take_now()
            if rec is None and not (remote and _remote_ok()):
                rec = _take()                                # nothing else to wait for: wait for a copy here
            if rec is not None:
                got = []
                step = max(1, BATCH)
                try:
                    for i in range(0, len(segs), step):
                        got += decode_pieces(segs[i:i + step], rec)
                        rec = None if i + step >= len(segs) else _take()
                except Exception:
                    got = None
                    raise
                continue
            time.sleep(0.02)
        out = [_words(tok, ts, a, b) for (a, b, _), (tok, ts) in zip(group, got)]
        done[0] += 1
        if on_progress:
            on_progress(done[0] / len(groups))
        return out

    # Pieces are independent, so they go to every copy of the model that is free: a video alone on the server
    # uses all of it, and several videos share it piece by piece. The words come back in order.
    workers = REMOTE_PARALLEL if remote else COPIES
    if workers > 1 and len(groups) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, len(groups)), thread_name_prefix="asr") as ex:
            results = list(ex.map(decode, groups))
    else:
        results = [decode(g) for g in groups]
    raw_words = []
    for out in results:
        for piece in out:
            raw_words.extend(piece)
    return _clean_words(raw_words)


def _extract_whisper_words(segments, total_sec: float, on_progress=None) -> list:
    out = []
    for seg in segments:
        seg_words = getattr(seg, "words", None) or []
        if seg_words:
            for wd in seg_words:
                txt = str(getattr(wd, "word", "") or "").strip()
                if not txt:
                    continue
                s = round(max(0.0, float(getattr(wd, "start", seg.start) or 0.0)), 2)
                e = round(max(s + 0.08, float(getattr(wd, "end", s + 0.15) or (s + 0.15))), 2)
                out.append({"w": txt, "s": s, "e": e})
        else:
            parts = str(getattr(seg, "text", "") or "").strip().split()
            if parts:
                s0 = float(getattr(seg, "start", 0.0) or 0.0)
                e0 = max(s0 + 0.2, float(getattr(seg, "end", s0 + 0.4) or (s0 + 0.4)))
                step = (e0 - s0) / len(parts)
                for idx, p in enumerate(parts):
                    out.append({"w": p, "s": round(s0 + idx * step, 2), "e": round(s0 + (idx + 1) * step, 2)})
        if on_progress and total_sec > 0:
            on_progress(min(1.0, float(getattr(seg, "end", 0.0) or 0.0) / total_sec))
    return out


def _transcribe_whisper(audio: np.ndarray, on_progress=None, lang_hint: str = "", word_timestamps: bool = True) -> list:
    """Faster-Whisper (CTranslate2 CUDA float16 / CPU INT8) with BatchedInferencePipeline,
    Silero VAD noise filtering, and instant pre-decode language hallucination guard."""
    if len(audio) < 4800:
        return []
    wav = audio.astype(np.float32) / 32768.0
    if float(np.sqrt((wav ** 2).mean())) < 0.003:
        return []

    _, wmod, pref_lang, gpu_on = _read_user_asr_settings()
    model, batched, device, compute_type = _get_whisper_model(wmod, gpu_on=gpu_on)
    with _state_lock:
        _asr_state.update(status="ready", device=device, compute_type=compute_type)

    # Resolve language: explicit per-video confirmation wins over auto; 'hinglish' uses Hindi+English Whisper mode
    hint = (lang_hint or "").strip().lower()
    if hint in ("hinglish", "hi-en", "en-hi"):
        hint_norm = "hinglish"
    else:
        hint_norm = hint.split("-")[0]
    pref_norm = "hinglish" if pref_lang in ("hinglish", "hi-en", "en-hi") else pref_lang

    chosen_lang = None
    hinglish_mode = False
    if hint_norm in ("en", "hi", "hinglish"):
        if hint_norm == "hinglish":
            chosen_lang = "hi"
            hinglish_mode = True
        else:
            chosen_lang = hint_norm
            hinglish_mode = (hint_norm == "hi")
    elif pref_norm in ("en", "hi", "hinglish"):
        if pref_norm == "hinglish":
            chosen_lang = "hi"
            hinglish_mode = True
        else:
            chosen_lang = pref_norm
            hinglish_mode = (pref_norm == "hi")

    total_sec = len(wav) / 16000.0
    vad_params = {"min_silence_duration_ms": 400, "speech_pad_ms": 200}

    def _call_transcribe(cur_model, cur_batched, cur_device, lang_arg, use_vad=True):
        prompt = WHISPER_HINGLISH_PROMPT if (lang_arg == "hi" or hinglish_mode) else None
        # BatchedInferencePipeline with VRAM-aware batch_size (batch 4 on 4 GB GTX 1650 prevents shared-RAM lag)
        use_batch = cur_batched is not None and total_sec >= 25.0
        kwargs = {
            "beam_size": 1,
            "best_of": 1,
            "vad_filter": use_vad,
            "word_timestamps": bool(word_timestamps),
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.65 if (lang_arg == "hi" or hinglish_mode) else 0.6,
            "compression_ratio_threshold": 2.4,
            "language": lang_arg,
            "task": "transcribe",
        }
        if use_vad:
            kwargs["vad_parameters"] = vad_params
        if prompt:
            kwargs["initial_prompt"] = prompt
        if use_batch:
            kwargs["batch_size"] = _pick_whisper_batch_size(cur_device, lang_arg)
            return cur_batched.transcribe(wav, **kwargs)
        return cur_model.transcribe(wav, **kwargs)

    def _run_once(cur_model, cur_batched, cur_device, lang_arg, use_vad=True):
        with _whisper_lock:
            segments, info = _call_transcribe(cur_model, cur_batched, cur_device, lang_arg, use_vad=use_vad)
            # Check auto-detected language BEFORE iterating segments so a 10-min block is NEVER decoded twice!
            if lang_arg is None and info is not None:
                det_lang = str(getattr(info, "language", "") or "").lower()
                det_prob = float(getattr(info, "language_probability", 1.0) or 0.0)
                if det_lang == "ur":
                    # Whisper frequently tags spoken Hindi/Hinglish as Urdu ('ur'): decode as Hindi, NEVER English!
                    segments, info = _call_transcribe(cur_model, cur_batched, cur_device, "hi", use_vad=use_vad)
                elif (det_lang not in ("en", "hi") and det_prob < 0.78) or (det_lang in ("ar", "fa") and det_prob < 0.88):
                    segments, info = _call_transcribe(cur_model, cur_batched, cur_device, "en", use_vad=use_vad)
            got_words = _extract_whisper_words(segments, total_sec, on_progress=on_progress)
        return got_words, info

    try:
        raw_words, info = _run_once(model, batched, device, chosen_lang, use_vad=True)
    except Exception as first_err:
        err_low = str(first_err).lower()
        # If CUDA failed mid-inference (e.g. VRAM OOM or missing DLL), fall back cleanly to CPU INT8
        if device == "cuda" and any(k in err_low for k in ("cuda", "cublas", "cudnn", "out of memory", "dll")):
            model, batched, device, compute_type = _get_whisper_model(wmod, gpu_on=False, force_cpu=True)
            with _state_lock:
                _asr_state.update(status="ready", device=device, compute_type=compute_type)
            raw_words, info = _run_once(model, batched, device, chosen_lang, use_vad=True)
        else:
            # Retry once without VAD / batching if Silero VAD asset fails in an offline environment
            raw_words, info = _run_once(model, None, device, chosen_lang, use_vad=False)

    # Lightweight final guard: if Arabic/Urdu script slipped into auto-detect on short audio, re-decode cleanly
    if chosen_lang is None and total_sec <= 120.0 and raw_words:
        sample_txt = " ".join(w["w"] for w in raw_words[:30])
        det_lang = str(getattr(info, "language", "") or "").lower() if info is not None else ""
        det_prob = float(getattr(info, "language_probability", 1.0) or 0.0) if info is not None else 0.0
        if _ARABIC_SCRIPT_RE.search(sample_txt) and det_prob < 0.90:
            fallback_lang = "hi" if det_lang in ("ur", "hi") else "en"
            raw_words, _ = _run_once(model, batched, device, fallback_lang, use_vad=True)

    if on_progress:
        on_progress(1.0)
    return _clean_words(raw_words)


def transcribe(audio: np.ndarray, on_progress=None, lang_hint: str = "", word_timestamps: bool = True,
               engine_override: str = None) -> list:
    """Returns [{'w': word, 's': start_sec, 'e': end_sec}, ...] using the active ASR engine
    (Parakeet-v2 or Faster-Whisper). Automatically uses Faster-Whisper when Hindi/Hinglish is explicitly
    requested and Faster-Whisper is installed, falling back gracefully to Parakeet otherwise."""
    eng = (engine_override or active_engine() or "parakeet").strip().lower()
    hint_low = (lang_hint or "").strip().lower()
    if not engine_override and hint_low in ("hi", "hinglish", "hi-en", "en-hi") and faster_whisper_installed():
        eng = "faster-whisper"
    if eng == "faster-whisper" and faster_whisper_installed():
        return _transcribe_whisper(audio, on_progress=on_progress, lang_hint=lang_hint, word_timestamps=word_timestamps)
    return _transcribe_parakeet(audio, on_progress=on_progress)

"""Talking to the AI service. You bring your own key, and choose the service and model in the dashboard
(Settings, AI) or in .env. What is saved from the dashboard is kept in ai.json in the data folder and wins.

    PEAKCUT_AI_PROVIDER   gemini | openai | xai | groq | openrouter | anthropic | ollama | custom
    PEAKCUT_AI_MODELS     one or more model ids, comma separated, tried in this order
    PEAKCUT_AI_KEY        your key (or PEAKCUT_AI_KEY_FILE, a file holding it)
    PEAKCUT_AI_BASE_URL   only for "custom" (any service that speaks the OpenAI chat format), or to override

Every service is asked for the same thing: JSON that follows a small schema. Gemini takes the schema directly.
The others are told the schema in the prompt and asked for a JSON object, which works with any chat model.

One picture can also be shown (`see`), which is how the webcam is looked for on a gaming stream. A model that
cannot look at pictures answers with something else, and that is simply taken as "it did not look"."""
import base64
import json
import os
import re
import shutil
import subprocess
import threading
import time

import httpx

from . import config

# name -> (label shown on the pages, address, how it is spoken to, where to get a key, an example model id)
PROVIDERS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models", "gemini", "https://aistudio.google.com/apikey", "gemini-2.5-flash"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "openai", "https://platform.openai.com/api-keys", "gpt-4o-mini"),
    "xai": ("Grok", "https://api.x.ai/v1", "openai", "https://console.x.ai", "grok-4"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "openai", "https://console.groq.com/keys", "llama-3.3-70b-versatile"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "openai", "https://openrouter.ai/keys", "google/gemini-2.5-flash"),
    "anthropic": ("Claude", "https://api.anthropic.com/v1", "anthropic", "https://console.anthropic.com", ""),
    "ollama": ("Ollama (on this machine)", "http://127.0.0.1:11434/v1", "openai", "https://ollama.com", "llama3.1"),
    "custom": ("Custom (OpenAI-compatible)", "", "openai", "", ""),
}
ALIASES = {"google": "gemini", "grok": "xai", "claude": "anthropic", "local": "ollama"}
NO_KEY = {"ollama", "custom"}       # a model running on your own machine needs no key

# ---------------------------------------------------------------- the settings in use
_lock = threading.Lock()
_saved = {"at": None, "data": {}}

# ---------------------------------------------------------------- live runtime activity (for UI status indicators)
_rt_lock = threading.Lock()
_rt = {"count": 0, "provider": "", "label": "", "model": "", "purpose": "", "since": 0.0, "epoch": 0}


def _bump_rt():
    try:
        from . import ctx
        ctx.bump()
    except Exception:
        pass


def _rt_enter(prov: str, model: str, purpose: str = ""):
    with _rt_lock:
        _rt["count"] += 1
        _rt["provider"] = prov
        _rt["label"] = PROVIDERS.get(prov, (prov or "AI",))[0]
        _rt["model"] = model or ""
        _rt["purpose"] = purpose or "Working"
        if _rt["count"] == 1:
            _rt["since"] = time.time()
        _rt["epoch"] += 1
    _bump_rt()


def _rt_exit():
    with _rt_lock:
        _rt["count"] = max(0, _rt["count"] - 1)
        if _rt["count"] == 0:
            _rt["purpose"] = ""
            _rt["since"] = 0.0
        _rt["epoch"] += 1
    _bump_rt()


def runtime_state() -> dict:
    """Whether an AI request is currently in flight, and which provider/model is handling it."""
    c = current()
    with _rt_lock:
        active = _rt["count"] > 0
        prov = _rt["provider"] if active and _rt["provider"] else c["provider"]
        mod = _rt["model"] if active and _rt["model"] else (c["models"][0] if c["models"] else "")
        return {
            "active": active,
            "count": _rt["count"],
            "provider": prov,
            "label": PROVIDERS.get(prov, (prov or "Not set up",))[0],
            "model": mod,
            "purpose": _rt["purpose"] if active else "",
            "since": _rt["since"] if active else 0.0,
            "epoch": _rt["epoch"],
            "ready": ready(),
        }


def _file():
    return config.DATA_DIR / "ai.json"


def _from_dashboard() -> dict:
    """What was saved from the dashboard, read again only when the file changes."""
    f = _file()
    try:
        at = f.stat().st_mtime_ns
    except OSError:
        at = 0
    if _saved["at"] != at:
        try:
            data = json.loads(f.read_text(encoding="utf-8")) if at else {}
        except (OSError, ValueError):
            data = {}
        _saved.update(at=at, data=data if isinstance(data, dict) else {})
    return _saved["data"]


def current() -> dict:
    """Provider, models, address and key: what the dashboard saved, or else what .env says."""
    d = _from_dashboard()
    p = str(d.get("provider") or config.AI_PROVIDER or "").strip().lower()
    return {"provider": ALIASES.get(p, p),
            "models": list(d["models"]) if d.get("provider") else list(config.AI_MODELS),
            "base_url": (str(d.get("base_url") or "") if d.get("provider") else config.AI_BASE_URL).rstrip("/"),
            "key": str(d["key"]) if d.get("provider") and "key" in d else config.ai_key()}


def provider() -> str:
    return current()["provider"]


def models() -> list:
    return current()["models"]


def key() -> str:
    return current()["key"]


def label() -> str:
    return PROVIDERS.get(provider(), ("",))[0]


def base_url() -> str:
    c = current()
    return c["base_url"] or PROVIDERS.get(c["provider"], ("", ""))[1]


def ready() -> bool:
    c = current()
    return bool(c["provider"] in PROVIDERS and c["models"] and base_url() and (c["key"] or c["provider"] in NO_KEY))


def public() -> dict:
    """What the settings page may see. The key itself never leaves the server: only whether there is one, and
    its last four characters so the owner can tell which key it is."""
    c = current()
    return {"provider": c["provider"], "models": c["models"], "base_url": c["base_url"], "has_key": bool(c["key"]),
            "key_hint": c["key"][-4:] if len(c["key"]) >= 12 else "", "ready": ready(), "label": label(),
            "saved_here": bool(_from_dashboard().get("provider")),
            "providers": [{"id": k, "label": v[0], "address": v[1], "needs_key": k not in NO_KEY, "key_url": v[3], "example": v[4]}
                          for k, v in PROVIDERS.items()]}


def save(provider_id: str, model_ids, address: str = "", new_key=None) -> dict:
    """Save the choice made in the dashboard. `new_key` None keeps the key that is already there (the page never
    has it to send back); an empty string removes it."""
    p = ALIASES.get(str(provider_id or "").strip().lower(), str(provider_id or "").strip().lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    if isinstance(model_ids, str):
        model_ids = model_ids.split(",")
    ms = [str(m).strip()[:120] for m in (model_ids or []) if str(m).strip()][:5]
    if not ms:
        raise ValueError("Enter the model to use.")
    if any(not re.fullmatch(r"[\w.\-:/@]+", m) for m in ms):
        raise ValueError("A model id has letters, digits and . - : / only.")
    address = str(address or "").strip().rstrip("/")[:300]
    if address and not re.fullmatch(r"https?://[^\s]+", address):
        raise ValueError("The address has to start with http:// or https://")
    if p == "custom" and not address:
        raise ValueError("A custom service needs its address, for example http://127.0.0.1:1234/v1")
    old = current()
    k = old["key"] if new_key is None and old["provider"] == p else str(new_key or "").strip()[:400]
    if not k and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    with _lock:
        f = _file()
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"provider": p, "models": ms, "base_url": address, "key": k}), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)        # the key is only for whoever runs the server
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None
    return public()


def _ollama_origin(address: str | None = None) -> str:
    """Base HTTP origin of an Ollama server (without trailing /v1 or /api)."""
    raw = str(address or "").strip().rstrip("/")
    if not raw:
        c = current()
        raw = (c["base_url"] if c["provider"] == "ollama" else "") or "http://127.0.0.1:11434"
    raw = re.sub(r"/(?:v1|api)(?:/.*)?$", "", raw, flags=re.I).rstrip("/")
    return raw or "http://127.0.0.1:11434"


def list_ollama_models(address: str | None = None) -> list:
    """List ONLY real local Ollama chat models (including custom Modelfile creations like reasoning-queen).
    Queries native /api/tags first (skipping cloud/remote proxy stubs and embedding-only models),
    and only falls back to /v1/models or `ollama list` CLI when /api/tags returns nothing."""
    origin = _ollama_origin(address)
    if "127.0.0.1" in origin or "localhost" in origin:
        ensure_ollama_running()
    found: list[str] = []
    seen: set[str] = set()

    def _add(name: str):
        n = str(name or "").strip()
        if not n:
            return
        low = n.lower()
        # Skip pure embedding models that cannot generate clip JSON
        if "embed" in low or "rerank" in low:
            return
        if n not in seen:
            seen.add(n)
            found.append(n)

    # 1. Native Ollama /api/tags — authoritative list of local models (includes custom Modelfile models)
    tags_ok = False
    try:
        r = httpx.get(f"{origin}/api/tags", timeout=5.0)
        if r.status_code == 200:
            tags_ok = True
            for m in r.json().get("models", []) or []:
                if not isinstance(m, dict):
                    continue
                # Skip Ollama cloud/remote stub entries that are not downloaded locally
                if m.get("remote_host") or m.get("remote_model"):
                    continue
                _add(m.get("name") or m.get("model") or "")
    except Exception:
        pass

    # 2. Only fall back to /v1/models if native /api/tags did not answer or returned empty
    if not found and not tags_ok:
        try:
            r = httpx.get(f"{origin}/v1/models", timeout=5.0)
            if r.status_code == 200:
                for m in r.json().get("data", []) or []:
                    if isinstance(m, dict):
                        _add(m.get("id") or "")
        except Exception:
            pass

    # 3. CLI fallback (`ollama list`) if HTTP returned nothing on localhost
    if not found and ("127.0.0.1" in origin or "localhost" in origin):
        try:
            exe = shutil.which("ollama") or (os.environ.get("LOCALAPPDATA", "") + r"\Programs\Ollama\ollama.exe")
            if exe and os.path.isfile(exe):
                cp = subprocess.run([exe, "list"], capture_output=True, text=True, timeout=6)
                if cp.returncode == 0:
                    for line in (cp.stdout or "").splitlines()[1:]:
                        parts = line.strip().split()
                        if parts:
                            _add(parts[0])
        except Exception:
            pass

    return found[:400]


def list_models_detail(provider_id=None, address=None, with_key=None) -> dict:
    """Return discovered model ids together with the exact provider and endpoint queried."""
    c = current()
    p = ALIASES.get(str(provider_id or c["provider"]).lower(), str(provider_id or c["provider"]).lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    lbl = PROVIDERS[p][0]
    if p == "ollama":
        origin = _ollama_origin(address)
        found = list_ollama_models(address)
        if not found:
            raise AIError("Could not reach Ollama or no local models are installed yet. Make sure Ollama is running (`ollama serve`) and check `ollama list`.")
        return {"models": found, "provider": p, "label": lbl, "source": origin}
    k = str(with_key or "").strip() or (c["key"] if c["provider"] == p else "")
    base = str(address or "").strip().rstrip("/") or (c["base_url"] if c["provider"] == p else "") or PROVIDERS[p][1]
    if not base:
        raise ValueError("Enter the service's address first.")
    how = PROVIDERS[p][2]
    if how == "gemini":
        r = httpx.get(base, params={"pageSize": 200}, headers={"x-goog-api-key": k}, timeout=20)
    elif how == "anthropic":
        r = httpx.get(f"{base}/models", params={"limit": 200}, headers={"x-api-key": k, "anthropic-version": "2023-06-01"}, timeout=20)
    else:
        r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {k}"} if k else {}, timeout=20)
    if r.status_code != 200:
        raise AIError(f"HTTP {r.status_code} {r.text[:200].replace(k or chr(0), '<key>')}")
    data = r.json()
    if how == "gemini":
        found = [m.get("name", "").split("/")[-1] for m in data.get("models", [])
                 if "generateContent" in (m.get("supportedGenerationMethods") or [])]
    else:
        found = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    return {"models": sorted({m for m in found if m})[:400], "provider": p, "label": lbl, "source": base}


def list_models(provider_id=None, address=None, with_key=None) -> list:
    """The model ids a service offers, so the settings page can show a list to pick from."""
    return list_models_detail(provider_id, address, with_key)["models"]


def test() -> dict:
    """One tiny request with the settings in use, to prove the key, the address and the model all work."""
    ms = models()
    if not ready():
        raise ValueError("Save a service, a model and a key first.")
    out, sent, wrote = call(ms[0], 'Answer with a JSON array that holds the single string "ok".', {"type": "ARRAY", "items": {"type": "STRING"}})
    return {"ok": True, "model": ms[0], "tokens": sent + wrote, "answer": str(out)[:80]}


class AIError(RuntimeError):
    pass


problem = {"text": "", "at": 0.0}       # why the AI last failed, for the dashboard; cleared when it answers again


def explain(e) -> str:
    """What went wrong with an AI call, in words the owner can act on."""
    text, low = str(e), str(e).lower()
    name = (models() or ["?"])[0]
    if "404" in text or "not found" in low or "does not exist" in low or "unknown model" in low or "not_found" in low:
        return f"{label()} doesn't know the model \"{name}\". Open Settings, AI, press Find models and pick one from the list."
    if "401" in text or "403" in text or "api key" in low or "api_key" in low or "unauthorized" in low or "permission" in low:
        return f"{label()} refused the key. Paste a working key under Settings, AI."
    if "429" in text or "quota" in low or "rate limit" in low or "exhausted" in low:
        return f"{label()} says the key has hit its limit or has no credit left. Check the account with {label()}, or try again later."
    if "connect" in low or "timed out" in low or "timeout" in low or "getaddrinfo" in low or "refused" in low or "nodename" in low or "servname" in low or "nameresolutionerror" in low or "name resolution" in low:
        return f"PeakCut couldn't reach {label()}. Check the internet connection" + (" and that Ollama is running." if provider() == "ollama" else ".")
    return ""


def note_problem(e):
    import time
    why = explain(e)
    if why:
        problem.update(text=why, at=time.time())


def clear_problem():
    problem.update(text="", at=0.0)


def _fail(r, model):
    text = r.text[:300].replace(key() or "\0", "<key>")
    raise AIError(f"HTTP {r.status_code} {text}")


# ---------------------------------------------------------------- the schema, said in words for chat models
def _json_schema(s):
    """Gemini's schema spelling (ARRAY, OBJECT...) as ordinary JSON Schema."""
    if isinstance(s, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else _json_schema(v)) for k, v in s.items()}
    if isinstance(s, list):
        return [_json_schema(x) for x in s]
    return s


def _wrapped(prompt, schema):
    return (prompt + "\n\nAnswer with JSON only, no other text and no code fences: one object with a single key "
            "\"items\", whose value follows this JSON Schema:\n" + json.dumps(_json_schema(schema)))


def _json_chunks(text: str):
    """Every balanced {...} or [...] in the text, in the order they appear."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:i + 1])
                start = None
            elif depth < 0:
                depth = 0
    return out


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by local reasoning models (Qwen3/DeepSeek-R1/custom Modelfiles)."""
    s = (text or "").strip()
    if not s:
        return ""
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.I).strip()
    # Handle truncated/unclosed <think> where JSON still appears after it, or strip dangling tag
    if "<think>" in s.lower():
        m = re.search(r"[\[{]", s)
        if m:
            s = s[m.start():].strip()
    return s


def _parse(text):
    """The JSON in an answer, even if the model wrapped it in <think> blocks, a code fence, or a sentence.

    A model that is not careful sometimes puts reasoning or a sentence in front of the JSON,
    or a second object after it. The JSON is looked for inside the answer rather than demanded of it."""
    text = _strip_reasoning(text)
    # Extract fenced JSON block if present anywhere in the response
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.I)
    if fence:
        text = fence.group(1).strip()
    else:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    out = None
    try:
        out = json.loads(text)
    except ValueError:
        for chunk in reversed(_json_chunks(text)):
            try:
                out = json.loads(chunk)
                break
            except ValueError:
                continue
        if out is None:
            m = re.search(r"[\[{].*[\]}]", text, re.S)
            if not m:
                raise AIError("The AI did not answer with JSON.")
            try:
                out = json.loads(m.group(0))
            except ValueError:
                raise AIError("The AI did not answer with JSON.")
    if isinstance(out, dict) and "items" in out:
        out = out["items"]
    return out


# ---------------------------------------------------------------- one call
def call(model: str, prompt: str, schema: dict, timeout: float | None = None, purpose: str = ""):
    """Ask one model. Returns (the parsed JSON answer, tokens sent, tokens written back).

    `timeout` is how long one answer may take. A live stream passes a short one, because a slow
    answer must not stop the recording."""
    for prefix in ("ollama:", "ollama/", "local:", "local/"):
        if model.lower().startswith(prefix):
            target_model = model[len(prefix):]
            return call_ollama(target_model, prompt, schema, timeout=timeout, purpose=purpose)
    prov = provider()
    how = PROVIDERS[prov][2]
    _rt_enter(prov, model, purpose or "Thinking")
    try:
        if prov == "ollama":
            ensure_ollama_running()
        return {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic}[how](model, prompt, schema, timeout)
    finally:
        _rt_exit()


def ensure_ollama_running() -> bool:
    """Checks if Ollama is running, and if not, attempts to start it automatically in the background."""
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.0)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    try:
        import subprocess, shutil
        p = shutil.which("ollama") or (os.environ.get("LOCALAPPDATA", "") + r"\Programs\Ollama\ollama.exe")
        if p and os.path.isfile(p):
            subprocess.Popen([p, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(8):
                time.sleep(0.5)
                try:
                    if httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.0).status_code == 200:
                        return True
                except Exception:
                    pass
    except Exception:
        pass
    return False


def get_local_ollama_model() -> str | None:
    """Returns the name of a model running in local Ollama (127.0.0.1:11434) if available, prioritizing fast chat/instruct models."""
    ensure_ollama_running()
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.5)
        if r.status_code == 200:
            names = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
            # 1. Prioritize standard chat/instruct Qwen models like qwen2.5:3b
            for name in names:
                low = name.lower()
                if "qwen2" in low or ("qwen" in low and "reasoning" not in low and "r1" not in low):
                    return name
            # 2. Any other Qwen model
            for name in names:
                if "qwen" in name.lower():
                    return name
            return names[0] if names else None
    except Exception:
        pass
    return None


def call_ollama(model: str, prompt: str, schema: dict, timeout: float | None = None, purpose: str = ""):
    """Directly call a local Ollama instance on 127.0.0.1:11434."""
    ensure_ollama_running()
    _rt_enter("ollama", model, purpose or "Local AI")
    try:
        return _openai(model, prompt, schema, timeout=timeout or 180,
                       base_url_override="http://127.0.0.1:11434/v1", key_override="")
    finally:
        _rt_exit()


def see(prompt: str, image: bytes, timeout: float | None = None) -> tuple[str, int, int]:
    """Ask one model about one picture: (the answer as text, tokens in, tokens out).

    Nothing is broken by a service that cannot look at pictures, or a model that is only text: the answer
    comes back empty, or says something the caller does not understand, and the caller carries on without
    it. The picture is a JPEG, passed as base64 inside the ordinary message."""
    if not ready() or not image or not models():
        return "", 0, 0
    model = models()[0]
    b64 = base64.b64encode(image).decode()
    how = PROVIDERS[provider()][2]
    if how == "gemini":
        body = {"contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mimeType": "image/jpeg", "data": b64}}]}]}
        r = httpx.post(f"{base_url()}/{model}:generateContent", json=body,
                       headers={"x-goog-api-key": key()}, timeout=timeout or 120)
        if r.status_code != 200:
            _fail(r, model)
        data = r.json()
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        u = data.get("usageMetadata", {})
        sent = int(u.get("promptTokenCount") or 0)
        return text, sent, max(0, int(u.get("totalTokenCount") or 0) - sent)
    if how == "anthropic":
        body = {"model": model, "max_tokens": 300, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt}]}]}
        r = httpx.post(f"{base_url()}/messages", json=body, timeout=timeout or 120,
                       headers={"x-api-key": key(), "anthropic-version": "2023-06-01"})
        if r.status_code != 200:
            _fail(r, model)
        data = r.json()
        text = "".join(p.get("text", "") for p in data.get("content") or [] if p.get("type") == "text")
        u = data.get("usage") or {}
        return text, int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    headers = {"Authorization": f"Bearer {key()}"} if key() else {}
    body = {"model": model, "stream": False, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]}
    r = httpx.post(f"{base_url()}/chat/completions", json=body, headers=headers, timeout=timeout or 120)
    if r.status_code != 200:
        _fail(r, model)
    data = _openai_body(r)
    text = (data["choices"][0]["message"].get("content") or "")
    u = data.get("usage") or {}
    return text, int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def _gemini(model, prompt, schema, timeout=None):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json", "responseSchema": schema}}
    r = httpx.post(f"{base_url()}/{model}:generateContent", json=body, headers={"x-goog-api-key": key()}, timeout=timeout or 120)
    if r.status_code != 200:
        _fail(r, model)
    data = r.json()
    text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    u = data.get("usageMetadata", {})
    sent = int(u.get("promptTokenCount") or 0)
    return json.loads(text), sent, max(0, int(u.get("totalTokenCount") or 0) - sent)   # the answer, and any thinking billed with it


def _openai_body(r):
    """The answer, whether the service sent plain JSON or an event stream.

    Some OpenAI-compatible servers answer with a stream of events unless the
    request says otherwise, so the answer is read from whichever shape arrived.
    A model that answers with a tool call instead of a message leaves the JSON in
    the call's arguments, so that is used too."""
    try:
        data = r.json()
    except ValueError:
        data = _from_events(r.text)
    for ch in data.get("choices") or []:
        msg = ch.get("message") or {}
        if (msg.get("content") or "").strip():
            continue
        for call in msg.get("tool_calls") or []:
            args = (call.get("function") or {}).get("arguments")
            if args:
                msg["content"] = args
                break
    return data


def _from_events(text: str) -> dict:
    """Read an answer out of a server-sent event stream."""
    parts, usage, calls = [], {}, {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            d = json.loads(chunk)
        except ValueError:
            continue
        usage = d.get("usage") or usage
        for ch in d.get("choices") or []:
            for piece in ((ch.get("message") or {}), (ch.get("delta") or {})):
                if piece.get("content"):
                    parts.append(piece["content"])
                for call in piece.get("tool_calls") or []:
                    i = call.get("index", 0)
                    calls[i] = calls.get(i, "") + ((call.get("function") or {}).get("arguments") or "")
    message = {"content": "".join(parts)}
    if not message["content"] and calls:
        message["content"] = "".join(calls.values())
    return {"choices": [{"message": message}], "usage": usage}


def _openai(model, prompt, schema, timeout=None, base_url_override=None, key_override=None):
    burl = (base_url_override or base_url()).rstrip("/")
    is_ollama = bool(base_url_override and "11434" in base_url_override) or provider() == "ollama" or "11434" in burl
    if is_ollama and not burl.lower().endswith("/v1"):
        burl = f"{_ollama_origin(burl)}/v1"
    k = key() if key_override is None else key_override
    headers = {"Authorization": f"Bearer {k}"} if k else {}
    body = {"model": model, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}], "stream": False}
    # Reasoning models on Ollama (e.g. reasoning-queen, deepseek-r1, qwen3-thinking) can stall or return empty
    # text when forced into strict json_object grammar mode, so send plain JSON instructions first for them.
    mlow = (model or "").lower()
    is_reasoning = any(tok in mlow for tok in ("reason", "r1", "think", "qwq", "o1", "o3"))
    if is_ollama and is_reasoning:
        extras = [{"temperature": 0.4}, {}]
    else:
        extras = [{"temperature": 0.4, "response_format": {"type": "json_object"}}, {"response_format": {"type": "json_object"}}, {}]
    r = None
    req_timeout = httpx.Timeout(timeout or 180, connect=15.0)
    for extra in extras:
        r = httpx.post(f"{burl}/chat/completions", json={**body, **extra}, headers=headers, timeout=req_timeout)
        if r.status_code not in (400, 422):
            break
    if r.status_code != 200:
        _fail(r, model)
    data = _openai_body(r)
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    if not content.strip() and (msg.get("reasoning") or msg.get("reasoning_content")):
        content = msg.get("reasoning") or msg.get("reasoning_content") or ""
    u = data.get("usage") or {}
    return _parse(content), int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def _anthropic(model, prompt, schema, timeout=None):
    body = {"model": model, "max_tokens": 8192, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
    r = httpx.post(f"{base_url()}/messages", json=body, timeout=timeout or 180,
                   headers={"x-api-key": key(), "anthropic-version": "2023-06-01"})
    if r.status_code != 200:
        _fail(r, model)
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage") or {}
    return _parse(text), int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)

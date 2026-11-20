"""Safe self-update: check(manifest_url) and apply(). Replaces ONLY shipped files; the workspace, brand.json, env, state and outputs are never touched.
Manifest: {"version": "1.1.0", "date": "...", "url": "<zip url or local path>", "sha256": "<hex of the zip>"}. The zip holds the skill folder files (an optional single top folder is stripped)."""
import hashlib, json, os, shutil, tempfile, urllib.request, urllib.error, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
GITLAB_URL = "https://gitlab.com/nocodehack.io/gta6-clipper/-/raw/main/manifest.json"  # mirror
SITE_URL = "https://guide.nocodehack.io/gta6-clipper/manifest.json"  # own site: last fallback
DEFAULT_URL = "https://raw.githubusercontent.com/nocodehack/gta6-clipper/main/manifest.json"
KEEP = {"models", "work", "trace", "backups", "__pycache__", ".git", ".venv"}   # never replaced, never backed up (big or yours)
KEEP_FILES = {"brand.json", ".env", "app_state.json", "streamers.json", "proposals.jsonl"}
def protected(rel):
    p = Path(rel); return p.parts[0] in KEEP or p.name in KEEP_FILES or ".bak" in p.name or p.name.endswith(".pyc")
def current(root=HERE):
    try: return json.loads((Path(root) / "version.json").read_text(encoding="utf-8"))["version"]
    except Exception: return "0.0.0"
def vt(v): return tuple(int(x) for x in str(v).split(".") if x.isdigit())
def fetch(url, timeout=20):
    if "://" not in url: return Path(url).read_bytes()
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "clipper-update"}), timeout=timeout).read()
def check(url=None, root=HERE):
    """-> dict(current, latest, update:bool, status:'ok'|'no_info'|'error', manifest?)"""
    cur = current(root)
    if not url and not os.environ.get("UPDATE_URL"):
        for u in (DEFAULT_URL, GITLAB_URL, SITE_URL):
            r = check(u, root)
            if r["status"] == "ok": return r
        return r
    url = url or os.environ.get("UPDATE_URL")
    try: m = json.loads(fetch(url))
    except urllib.error.HTTPError as e:
        return dict(current=cur, update=False, status="no_info" if e.code == 404 else "error", detail="no update info yet" if e.code == 404 else f"HTTP {e.code}")
    except Exception as e: return dict(current=cur, update=False, status="error", detail=str(e)[:200])
    if not isinstance(m, dict) or "version" not in m: return dict(current=cur, update=False, status="error", detail="bad manifest")
    return dict(current=cur, latest=m["version"], date=m.get("date", ""), update=vt(m["version"]) > vt(cur), status="ok", manifest=m)
def apply(ws, url=None, root=HERE, dry=False):
    """Download, verify sha256, back up to <ws>/backups/<version>/, replace shipped files; roll back on any failure."""
    root = Path(root); c = check(url, root)
    if c["status"] != "ok": return dict(ok=False, **{k: c[k] for k in ("status", "detail") if k in c})
    if not c["update"]: return dict(ok=True, changed=False, current=c["current"], detail="already up to date")
    m = c["manifest"]; data = fetch(m["url"], 120)
    if hashlib.sha256(data).hexdigest().lower() != str(m.get("sha256", "")).lower(): return dict(ok=False, detail="sha256 mismatch: nothing changed")
    tmp = Path(tempfile.mkdtemp(prefix="clipper-upd-")); z = tmp / "u.zip"; z.write_bytes(data)
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for n in names:
            if n.startswith(("/", "\\")) or ".." in Path(n).parts or ":" in n: return dict(ok=False, detail="unsafe path in zip: " + n)
        tops = {Path(n).parts[0] for n in names}; strip = len(tops) == 1 and all(len(Path(n).parts) > 1 for n in names)
        new = {(str(Path(*Path(n).parts[1:])) if strip else n): n for n in names}
        new = {r: n for r, n in new.items() if not protected(r)}
        if dry: shutil.rmtree(tmp, True); return dict(ok=True, dry=True, current=c["current"], latest=c["latest"], would_replace=sorted(new))
        bk = Path(ws) / "backups" / c["current"]; made = []
        try:
            if bk.exists(): shutil.rmtree(bk)
            for f in root.rglob("*"):
                r = f.relative_to(root)
                if f.is_file() and not protected(r): (bk / r).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, bk / r)
            for r, n in new.items():
                dst = root / r; dst.parent.mkdir(parents=True, exist_ok=True); t = dst.with_name(dst.name + ".new"); t.write_bytes(zf.read(n))
                made.append((dst, dst.exists())); os.replace(t, dst)
        except Exception as e:
            for dst, existed in made:
                b = bk / dst.relative_to(root)
                if existed and b.exists(): shutil.copy2(b, dst)
                elif not existed and dst.exists(): dst.unlink()
            shutil.rmtree(tmp, True); return dict(ok=False, detail="failed, rolled back: " + str(e)[:200])
    shutil.rmtree(tmp, True)
    return dict(ok=True, changed=True, current=c["current"], latest=c["latest"], backup=str(bk), replaced=len(new), restart_needed=True)

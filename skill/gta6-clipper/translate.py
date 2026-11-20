# Offline caption + hook translation (Argos/OPUS models run directly on ctranslate2 + sentencepiece, no torch, no stanza).
# Models (~90 MB per language, English -> X) download once into ~/.local/share/argos-translate on first use.
# CLI: translate.py <transcript.json> START END <lang>   |   translate.py --selftest
import sys, re, json, zipfile
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
DIR = Path.home() / ".local" / "share" / "argos-translate" / "packages"
LANGS = "es fr pt ar hi de he it nl tr ru pl id ja ko zh".split()  # whatever the Argos index offers for English -> X
RTL = {"ar", "he", "fa", "ur"}
_cache = {}
_PUNCT = re.compile(r'[,.;:"«»“”،。、¿¡।()]+')  # same rule as clip.py clean(): only ? ! stay (also ؟ for Arabic)
# English profanity is masked by clip.py before it reaches the screen; the translated line gets the same mask (short per-language list)
_BAD = re.compile(r"^(mierda|joder|coño|puta|puto|cabrón|cabron|merde|putain|connard|salope|foda-se|porra|caralho|puta|scheiße|scheisse|fick\w*|arschloch|kurwa|كس\w*|شرموط\w*)$", re.I)

def _load(lang):
    import ctranslate2, sentencepiece
    if lang in _cache: return _cache[lang]
    find = lambda: next((x for x in DIR.glob(f"*en_{lang}*") if (x / "model").exists()), None)  # folder is en_xx or translate-en_xx-1_9
    d = find()
    if not d:
        import argostranslate.package as p  # imports fine without torch; only argostranslate.translate needs stanza
        p.update_package_index()
        pk = next((a for a in p.get_available_packages() if a.from_code == "en" and a.to_code == lang), None)
        if not pk: raise SystemExit(f"no offline model for English -> {lang}")
        print(f"  downloading English -> {lang} model (~90 MB, once)", file=sys.stderr)
        zipfile.ZipFile(pk.download()).extractall(DIR)  # zip holds its own top folder
        d = find()
    _cache[lang] = (ctranslate2.Translator(str(d / "model"), device="cpu", inter_threads=1), sentencepiece.SentencePieceProcessor(model_file=str(d / "sentencepiece.model")))
    return _cache[lang]

def _tr(texts, lang):
    t, sp = _load(lang)
    out = t.translate_batch([sp.encode(x, out_type=str) for x in texts], beam_size=4, length_penalty=0.2, max_batch_size=16)
    return [re.sub(r"\s+", " ", sp.decode(o.hypotheses[0]).replace("▁", " ")).strip() for o in out]  # fr/pt models leave literal word-start marks

def translate_text(text, lang):
    """One line (the hook) -> target language; same text back if lang is empty/en."""
    if not text or lang in ("", "en", None): return text
    return _tr([text.capitalize() if text.isupper() else text], lang)[0].translate(str.maketrans('', '', '()'))  # ALL-CAPS hooks translate badly (sentence case is best); Arabic wraps names in ( )

def _mask(w): return w[0] + "*" * (len(w) - 1) if _BAD.match(w) else w

def sentences(words, cap=22):
    """Split on the end flags (or after `cap` words if Whisper gave no punctuation, so one run-on never becomes one huge translation)."""
    out, cur = [], []
    for w in words:
        cur.append(w)
        if w.get("end") or len(cur) >= cap: out.append(cur); cur = []
    return out + ([cur] if cur else [])

def spread(tokens, s0, e1, end):
    """Place translated tokens over [s0, e1], time share proportional to word length."""
    n = [len(t) + 1 for t in tokens]; tot = sum(n); t0 = s0; res = []
    for i, (t, k) in enumerate(zip(tokens, n)):
        t1 = e1 if i == len(tokens) - 1 else t0 + (e1 - s0) * k / tot
        res.append(dict(w=t, s=round(t0, 3), e=round(t1, 3), end=bool(end and i == len(tokens) - 1))); t0 = t1
    return res

def translate_words(words, lang):
    """words = [{"w","s","e","end"}] in English (raw, unmasked is best) -> same shape in `lang`, ends kept."""
    if lang in ("", "en", None): return words
    ss = sentences(words)
    texts = [" ".join(w["w"] for w in s) for s in ss]
    out = []
    for s, tx in zip(ss, _tr(texts, lang)):
        toks = []
        for y in re.sub(r"\s+([?!؟])", lambda m: m[1], tx).split():  # French " ?" would otherwise be a caption word of its own
            y = _PUNCT.sub("", y)
            if y: toks.append(_mask(y))
        if toks: out += spread(toks, s[0]["s"], s[-1]["e"], s[-1].get("end") or re.search(r"[.?!؟]$", tx) is not None)
    return out

def selftest_offline():
    """No models: re-timing, run-on splitting, punctuation strip, profanity mask, RTL set."""
    sp = spread(["aa", "bbbbbb", "c"], 1.0, 2.0, True)
    assert sp[0]["s"] == 1.0 and sp[-1]["e"] == 2.0 and sp[-1]["end"] and not sp[0]["end"], sp
    assert all(abs(a["e"] - b["s"]) < 1e-9 for a, b in zip(sp, sp[1:])) and sp[1]["e"] - sp[1]["s"] > sp[0]["e"] - sp[0]["s"]  # contiguous, longer word longer
    W = [dict(w="w%d" % i, s=i * .3, e=i * .3 + .25, end=(i == 5)) for i in range(50)]
    assert [len(s) for s in sentences(W)] == [6, 22, 22], [len(s) for s in sentences(W)]  # end flag splits; run-on capped at 22
    assert _PUNCT.sub("", "«¿Hola,») mundo.") == "Hola mundo" and _PUNCT.sub("", "ok?") == "ok?"  # only ? ! stay
    assert _mask("mierda") == "m*****" and _mask("Hola") == "Hola" and _mask("كسمك")[0] == "ك"
    assert "ar" in RTL and "es" not in RTL and translate_words(W, "en") is W and translate_text("hi", "") == "hi"
    print("selftest ok (offline)")

def selftest():
    sp = spread(["aa", "bbbbbb"], 1.0, 2.0, True)
    assert sp[0]["s"] == 1.0 and sp[-1]["e"] == 2.0 and sp[-1]["end"] and not sp[0]["end"] and sp[0]["e"] == sp[1]["s"], sp
    W = [dict(w=x, s=i * .3, e=i * .3 + .25, end=x.endswith(".")) for i, x in enumerate("We are going to the beach. It is a very good day.".split())]
    assert [len(s) for s in sentences(W)] == [6, 6]
    r = translate_words(W, "es"); txt = " ".join(x["w"] for x in r).lower()
    assert "playa" in txt and sum(x["end"] for x in r) == 2 and r[0]["s"] == 0 and abs(r[-1]["e"] - W[-1]["e"]) < 1e-6, r
    assert translate_text("This is crazy", "fr")
    print("selftest ok:", txt)

if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]: selftest_offline(); raise SystemExit
    if sys.argv[1:] == ["--selftest"]: selftest(); raise SystemExit  # needs the es/fr models (downloads once)
    if len(sys.argv) < 5: print("usage: translate.py <transcript.json> START END <lang> | translate.py selftest (offline) | --selftest (with models)"); raise SystemExit(2)
    tj, a, b, lang = sys.argv[1:5]
    sec = lambda v: sum(float(x) * 60 ** i for i, x in enumerate(reversed(str(v).split(":"))))
    a, b = sec(a), sec(b)
    ws = [dict(w=w["w"], s=w["s"] - a, e=w["e"] - a, end=bool(re.search(r"[.?!]$", w["w"]))) for s in json.load(open(tj, encoding="utf-8"))["segments"] for w in s["words"] if w["w"] and a <= w["s"] and w["e"] <= b]
    print("EN:", " ".join(w["w"] for w in ws)); r = translate_words(ws, lang)
    print(lang.upper() + ":", " ".join(w["w"] for w in r))
    for w in r[:8]: print(" ", w)

# Finds BEFORE/AFTER callback candidates in a tagged transcript: a setup (claim, bet, question, rare word, an action) and a later payoff
# (the same thing coming back with a [LOUD]/[CHAT]/[HEAT] spike). A finder, not a judge: Claude reads the ranked list and decides.
# Usage: python callbacks.py transcript.txt|json [--min-gap 300] [--top 25] [--max-gap 7200]
import json, math, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
LINE = re.compile(r"^\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]\s*(.*)$")
CLAIM = re.compile(r"\b(i will never|i'll never|never again|never gonna|i('m| am) never|i guarantee|i promise|trust me|no way i|no chance|i bet|bet you|watch (this|me|how)|"
                   r"easy|too easy|this time|i('m| am) (gonna|going to) win|i can't lose|can't die|won't die|you('ll| will) see|i('ll| will) prove|i called it|prediction|mark my words)\b", re.I)
PAYOFF = re.compile(r"\b(told you|i said|remember (when|that)|called it|knew it|again|here we go again|i lied|lied|wasted|busted|dead|died|die|fail|lost|lose|never mind|nevermind)\b", re.I)
STRONG = re.compile(r"i will never|i'll never|never again|i guarantee|trust me|no way i|this time|watch (this|me|how)|you('ll| will) see|mark my words|i called it|prediction", re.I)  # a bare repeat of these means something; "easy"/"i promise"/"i bet" repeat all the time
FAIL = re.compile(r"\b(died|dead|die|wasted|busted|arrested|failed|fail|crashed|crash|fell|missed|lost|killed|exploded)\b", re.I)
STOP = set("""that this with have from they them were been what when your about just like yeah okay know going gonna there here then than into over some more very really want
chat bro man dude shit fuck fucking oh oh, don't didn't can't it's i'm that's you're what's let's gotta wanna thing things right well come look need think because would could should still back
only even also much many make made good dont cant its youre thats whats lets people""".split())

def parse(path):
    """Tagged .txt lines, or the .json (segments with start/end/text) -> list of dicts; tags read out of the text."""
    if str(path).endswith(".json"):
        raw = [f"[{s['start']}-{s['end']}] {s['text'].strip()}" for s in json.load(open(path, encoding="utf-8"))["segments"]]
    else: raw = open(path, encoding="utf-8").read().splitlines()
    out = []
    for i, l in enumerate(raw):
        m = LINE.match(l.strip())
        if not m: continue
        txt = m.group(3)
        g = lambda p: next((float(x) for x in re.findall(p, txt)), 0.0)
        out.append(dict(i=len(out), s=float(m.group(1)), e=float(m.group(2)), txt=txt, loud=g(r"\[LOUD \+(\d+)dB\]"), chat=g(r"\[CHAT x(\d+(?:\.\d+)?)"),
                        heat=g(r"\[HEAT (\d+)\]"), plain=re.sub(r"\s*\[[A-Z]+[^\]]*\]", "", txt).strip()))
    return out

def words(t): return [w[:7] for w in re.findall(r"[a-z']{4,}", t.lower()) if w not in STOP and w.strip("'") not in STOP]  # w[:7]: "indonesian" and "indonesia" are one topic
def spike(l): return l["loud"] / 10 + l["chat"] / 3 + l["heat"] / 40  # 0 = nothing tagged; a +21dB line with chat x5 = ~3.8
def mins(a, b): return (b["s"] - a["e"]) / 60

def find(lines, min_gap=300, max_gap=7200, top=25):
    df = Counter(w for l in lines for w in set(words(l["plain"])))
    N = len(lines); idf = lambda w: math.log(N / df[w]) if df[w] else 0
    cand = {}
    def add(a, b, score, why):
        k = (a["i"], b["i"])
        if k not in cand or cand[k][0] < score: cand[k] = (score, why)
        elif why not in cand[k][1]: cand[k] = (cand[k][0] + 0.5, cand[k][1] + " + " + why)
    pay = lines
    local = lambda l, w: sum(w in words(x["plain"]) for x in lines if abs(x["s"] - l["s"]) <= 120)  # said 2+ times around the setup = a topic of that segment, not a passing word
    near = [max((spike(x) for x in lines if abs(x["s"] - l["s"]) <= 30), default=0) for l in lines]  # the biggest spike within 30 s: a topic can come back just before or after the peak
    for a in lines:
        wa = {w for w in words(a["plain"]) if df[w] <= 8}
        claim, q = bool(CLAIM.search(a["plain"])), a["plain"].rstrip().endswith("?")
        failA = FAIL.search(a["plain"])
        for b in pay:
            gap = b["s"] - a["e"]
            if gap < min_gap or gap > max_gap: continue
            sp, cue = spike(b), bool(PAYOFF.search(b["plain"]))
            shared = sorted(wa & set(words(b["plain"])), key=lambda w: -idf(w))
            sh = sum(idf(w) for w in shared[:3])
            rep = claim and (m := STRONG.search(a["plain"])) and re.search(re.escape(m.group(0)), b["plain"], re.I)  # the same promise said again, later
            if rep and sp == 0: add(a, b, 2.5 + sh + cue, f"same claim again ('{m.group(0)}'), no tag on the payoff line")
            # a bare "again" after any claim is noise: a claim needs a shared word or a real call-back cue
            elif claim and (shared or (cue and re.search(r"told you|i said|called it|knew it|i lied", b["plain"], re.I))) and sp > 0: add(a, b, 3 + sh + sp + cue * 1.5, f"claim ('{CLAIM.search(a['plain']).group(0)}') then {'cue ' + PAYOFF.search(b['plain']).group(0) + ' ' if cue else ''}{shared[:3]}")
            elif q and len(shared) >= 2 and sp > 0: add(a, b, 1.5 + sh + sp, f"question answered later, shared {shared[:3]}")
            elif failA and FAIL.search(b["plain"]) and sp >= 2 and (cue or shared): add(a, b, 1.5 + sp + sh * 0.5 + cue, f"same action again ({failA.group(0)} / {FAIL.search(b['plain']).group(0)})")
            elif shared and sh >= 6 and sp >= 2 and df[shared[0]] <= 4: add(a, b, sh * 0.4 + sp, f"rare word returns {shared[:3]}")
            elif shared and df[shared[0]] <= 6 and len(shared[0]) >= 7 and near[b["i"]] >= 2 and local(a, shared[0]) >= 2: add(a, b, idf(shared[0]) * 0.6 + near[b["i"]], f"topic returns at a peak {shared[:2]}")  # e.g. a language quiz early, then a guest speaks that language at the big moment
    ranked = sorted(cand.items(), key=lambda kv: -kv[1][0])
    out, seen_a, seen_b = [], [], []
    for (ia, ib), (sc, why) in ranked:  # one candidate per (setup, payoff) region pair: no wall of the same pair
        a, b = lines[ia], lines[ib]
        if sum(abs(a["s"] - x) < 180 for x in seen_a) >= 2 or sum(abs(b["s"] - y) < 180 for y in seen_b) >= 2: continue  # at most 2 pairs per setup or payoff region  # same pair region twice; one big peak may still pay off several different setups
        seen_a.append(a["s"]); seen_b.append(b["s"])
        out.append(dict(score=round(sc, 1), setup=(a["s"], a["e"], a["plain"]), payoff=(b["s"], b["e"], b["plain"]), gap_min=round(mins(a, b), 1), why=why, tags=f"payoff {'LOUD+%d ' % b['loud'] if b['loud'] else ''}{'CHAT x%g' % b['chat'] if b['chat'] else ''}".strip()))
        if len(out) == top: break
    return out

def show(res):
    for n, r in enumerate(res, 1):
        print(f"#{n} score {r['score']}  gap {r['gap_min']} min  ({r['why']})  [{r['tags']}]\n   SETUP  {r['setup'][0]:.1f}-{r['setup'][1]:.1f}: {r['setup'][2][:160]}\n   PAYOFF {r['payoff'][0]:.1f}-{r['payoff'][1]:.1f}: {r['payoff'][2][:160]}")

if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["selftest"]:
        T = ["[100.0-105.0] I will never die to the cops, trust me.", "[110.0-112.0] hello chat", "[2600.0-2610.0] The cops got me again, I told you! [LOUD +20dB] [CHAT x5 LOL]",
             "[2700.0-2701.0] ok [LOUD +30dB]", "[2800-2801] bet on the platypus", "[5000-5002] platypus wins [LOUD +20dB]"]
        import tempfile, os
        p = os.path.join(tempfile.mkdtemp(), "t.txt"); open(p, "w", encoding="utf-8").write("\n".join(T))
        r = find(parse(p), min_gap=300)
        assert r and r[0]["setup"][0] == 100.0 and r[0]["payoff"][0] == 2600.0 and 40 < r[0]["gap_min"] < 45, r
        assert all(c["gap_min"] >= 5 for c in r) and not find(parse(p), min_gap=3000, top=5)[:0]
        assert find(parse(p), min_gap=3000) == [] or all(c["gap_min"] >= 50 for c in find(parse(p), min_gap=3000))
        T2 = ["[560.0-562.0] Indonesia language quiz.", "[633.0-636.0] Maybe because I'm Indonesian, chat.", *[f"[{900 + 40 * k}.0-{901 + 40 * k}.0] filler line {k}" for k in range(40)],
              "[3300.0-3305.0] Let's go, Obama is here [LOUD +23dB]", "[3326.0-3333.0] Where did you learn that? Indonesia film"]
        p2 = os.path.join(tempfile.mkdtemp(), "t2.txt"); open(p2, "w", encoding="utf-8").write(chr(10).join(T2))
        assert any(c["payoff"][0] == 3326.0 and c["setup"][0] in (560.0, 633.0) for c in find(parse(p2))), find(parse(p2))
        print("selftest ok"); sys.exit()
    ap = lambda f, d: type(d)(a[a.index(f) + 1]) if f in a else d
    show(find(parse(a[0]), ap("--min-gap", 300), ap("--max-gap", 7200), ap("--top", 25)))

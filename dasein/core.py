import json, os, pickle, random, re, urllib.request
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

HOME = Path(os.environ.get("DASEIN_HOME") or Path.home() / ".dasein")
MODEL = HOME / "reward.pkl"
MINE = HOME / "mine.jsonl"
OTHERS = HOME / "others.jsonl"  # optional: other people's messages (e.g. what my correspondents sent me)
PEOPLE = HOME / "people"
_DATA = Path(__file__).parent.parent / "data"


def load_lines(path):
    out = []
    for l in Path(path).read_text().splitlines():
        l = l.strip()
        if not l:
            continue
        if l.startswith("{"):
            d = json.loads(l)
            l = d.get("text") or d.get("opinion") or next(iter(d.values()))
        out.append(l)
    return out


def ai_generic():
    return load_lines(_DATA / "ai_generic.jsonl")


def other_humans():
    """Bundled texts written by other humans + (optional) my correspondents' messages from ingest."""
    out = load_lines(_DATA / "other_humans.jsonl")
    if OTHERS.exists():
        out += load_lines(OTHERS)
    return out


def negatives():
    return ai_generic() + other_humans()


def rewrite(endpoint, model, text):
    body = json.dumps({"model": model, "max_tokens": 400, "temperature": 0.7, "messages": [
        {"role": "user", "content": "Rewrite this in clear, polished English. Keep the meaning. Output only the rewrite.\n\n" + text}]}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return re.sub(r"<think>.*?</think>", "", r["choices"][0]["message"]["content"], flags=re.S).strip()


def build():
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=False)),
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ])
    return make_pipeline(feats, LogisticRegression(C=10, max_iter=2000, class_weight="balanced"))


def fit(mine, negatives):
    m = build()
    m.fit(mine + negatives, [1] * len(mine) + [0] * len(negatives))
    return m


def save(m):
    HOME.mkdir(parents=True, exist_ok=True)
    MODEL.write_bytes(pickle.dumps(m))


def score(texts, m=None):
    m = m or pickle.loads(MODEL.read_bytes())
    return [float(p) for p in m.predict_proba(texts)[:, 1]]


# --- people ---
FIELDS = ["knows_about_me", "my_promises", "last_talk", "notes"]


def _slug(p):
    return re.sub(r"[^a-z0-9가-힣]+", "-", p.lower()).strip("-")


def _gbrain(*args):
    import shutil, subprocess
    if not shutil.which("gbrain"):
        return ""
    try:
        return subprocess.run(["gbrain", *args], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


_gb_people = {}


def gbrain_people():
    """{display name: slug} for person pages in gbrain (empty if gbrain isn't installed)."""
    if not _gb_people:
        for l in _gbrain("list", "--type", "person", "--limit", "5000").splitlines():
            f = l.split("\t")
            if len(f) >= 4 and f[1] == "person":
                _gb_people[f[3].strip()] = f[0]
    return _gb_people


def gbrain_page(p, n=1800):
    slug = gbrain_people().get(p)
    if not slug:
        return ""
    t = _gbrain("get", slug)
    t = re.sub(r"\nhandles:\n(  - .*\n)+", "\n", t)  # no phone numbers / emails in prompts
    t = re.sub(r"\n## [^\n]+\s*(?=\n## |\Z)", "", t)  # drop empty sections
    return re.sub(r"\n{3,}", "\n\n", t).strip()[:n]


def person_get(p):
    f = PEOPLE / f"{_slug(p)}.md"
    out = f.read_text() if f.exists() else ""
    gb = gbrain_page(p)
    if gb:
        out += ("\n\n" if out else "") + "## From my gbrain memory graph\n" + gb
    return out or f"# {p}\n(no notes yet)"


def person_set(p, field, text):
    if field not in FIELDS:
        raise ValueError(f"field must be one of {FIELDS}")
    PEOPLE.mkdir(parents=True, exist_ok=True)
    f = PEOPLE / f"{_slug(p)}.md"
    d = {k: "" for k in FIELDS}
    if f.exists():
        cur = None
        for l in f.read_text().splitlines():
            if l.startswith("## "):
                cur = l[3:].strip(); d.setdefault(cur, "")
            elif cur:
                d[cur] += l + "\n"
    d[field] = text.strip() + "\n"
    f.write_text(f"# {p}\n\n" + "\n".join(f"## {k}\n{v.strip()}\n" for k, v in d.items()))


def person_has(p):
    return (PEOPLE / f"{_slug(p)}.md").exists()


def status(**kw):
    """Merge progress fields into DASEIN_HOME/status.json (read by `dasein ui`)."""
    import time
    HOME.mkdir(parents=True, exist_ok=True)
    f = HOME / "status.json"
    try:
        d = json.loads(f.read_text())
    except Exception:
        d = {}
    d.update(kw, updated=time.time())
    f.write_text(json.dumps(d))


def run_fit(mine, rewrite_endpoint=None, rewrite_model="default", rewrite_n=0, report=print):
    """80/20 held-out eval, then fit on everything and save. Writes progress to status.json. Returns metrics."""
    from sklearn.metrics import roc_auc_score
    random.seed(0); random.shuffle(mine)
    k = int(len(mine) * 0.8)
    tr, te = mine[:k], mine[k:]
    def split(l):
        l = l[:]; random.shuffle(l); k = int(len(l) * 0.8)
        return l[:k], l[k:]
    ai, oth = ai_generic(), other_humans()
    status(stage="reading", label="Reading", mine=len(mine), others=len(oth), ai=len(ai), progress=0.1)
    atr, ate = split(ai)
    otr, ote = split(oth)
    ntr = atr + otr
    if rewrite_endpoint:
        n = rewrite_n or min(len(tr), 100)
        ntr += [rewrite(rewrite_endpoint, rewrite_model, t) for t in tr[:n]]
    status(stage="learning", label="Learning", progress=0.3)
    m = fit(tr, ntr)
    status(stage="evaluating", label="Evaluating", progress=0.7)
    metrics = {}
    for key, label, neg in (("ai", "you vs AI", ate), ("others", "you vs other people", ote)):
        if not neg:
            report(f"{label}: no data"); continue
        s, sn = score(te, m), score(neg, m)
        ok = sum(v > .5 for v in s) + sum(v <= .5 for v in sn)
        auc = roc_auc_score([1] * len(s) + [0] * len(sn), s + sn)
        metrics[key] = {"acc": ok / (len(s) + len(sn)), "auc": auc}
        report(f"{label}: held-out accuracy {ok}/{len(s)+len(sn)} = {ok/(len(s)+len(sn)):.3f}, AUC {auc:.3f}")
    save(fit(mine, ntr + ate + ote))
    h = HOME / "history.json"
    hist = json.loads(h.read_text()) if h.exists() else []
    hist.append({"n": len(mine), **{k: v["acc"] for k, v in metrics.items()}})
    h.write_text(json.dumps(hist))
    status(stage="done", label="Done", progress=1.0, metrics=metrics)
    return metrics


def retrain():
    mine = load_lines(MINE)
    run_fit(mine, report=lambda *_: None)
    return len(mine)


def learn(sent_text, person=""):
    """Add a text I actually sent as a positive; refit every 20 additions."""
    import datetime
    HOME.mkdir(parents=True, exist_ok=True)
    with MINE.open("a") as f:
        f.write("\n" + json.dumps({"text": sent_text.strip()}) + "\n")
    n = HOME / "learn_count"
    c = int(n.read_text() or 0) + 1 if n.exists() else 1
    if person:
        person_set(person, "last_talk", datetime.date.today().isoformat())
    if c >= 20:
        retrain(); c = 0
    n.write_text(str(c))
    return c


def spread_warning(s):
    """Low-discrimination guard: tiny spread or everything high means the scores can't pick a winner."""
    sp = max(s) - min(s)
    if len(s) > 1 and (sp < 0.05 or min(s) >= 0.9):
        return "low discrimination - scores not reliable"
    return ""


FRAGMENTS = {"don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won", "wouldn", "couldn", "shouldn", "haven", "hasn", "ll", "ve", "re"}


def is_function_word(t):
    """Stopwords and contraction stubs: real signal for the model, meaningless as a 'fingerprint' to a human."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return all(w in ENGLISH_STOP_WORDS or w in FRAGMENTS for w in t.split())


RULES = """1. Use only facts present in the task/context. Never invent names, numbers, dates, promises.
2. Do not name emotions ("I'm excited/grateful/thrilled"). Show them the way my examples do, or skip them.
3. Do not end with a summary, a moral, or a polite wrap-up line unless my examples do.
4. Match the recipient's temperature (see relationship note); no more formal or warm than I would be with them.
5. Length: stay near my median ({med} words). Do not pad.
6. No slop vocabulary: delve, leverage, crucial, ensure, seamless, testament, tapestry, navigate, landscape, "I hope this finds you well", "feel free to"."""


def fingerprint(n=15):
    """Top 'me-like' and 'AI/other-like' word n-grams from the reward model coefficients."""
    import numpy as np
    m = pickle.loads(MODEL.read_bytes())
    feats, lr = m.steps[0][1], m.steps[1][1]
    names = [(a.get_feature_names_out()) for _, a in feats.transformer_list]
    w = lr.coef_[0][len(names[0]):]  # word vectorizer is second
    wn = names[1]
    keep = [i for i, t in enumerate(wn) if len(t) > 2 and re.fullmatch(r"[A-Za-z' ]+", t) and not is_function_word(t)]
    keep.sort(key=lambda i: w[i])
    return [wn[i] for i in keep[::-1][:n]], [wn[i] for i in keep[:n]]


ADMIRE = HOME / "admire.jsonl"  # writing I love reading (rhythm only); my own texts stay the voice and the facts


def add_admired(texts, source=""):
    with ADMIRE.open("a") as f:
        for t in texts:
            if len(t.split()) >= 25:
                f.write(json.dumps({"text": t.strip(), "source": source}) + "\n")


def _admire_block(task, k=6):
    if not ADMIRE.exists():
        return ""
    import numpy as np
    texts = [json.loads(l)["text"] for l in ADMIRE.read_text().splitlines() if l.strip()]
    if not texts:
        return ""
    v = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(texts + [task or "x"])
    sim = (v.transform(texts) @ v.transform([task]).T).toarray().ravel()
    top = [texts[i] for i in np.argsort(-sim)[:k]]
    return ("## Writing I love reading (borrow its rhythm and stance only; never its facts, topics or phrases)\n"
            + "\n".join(f"- {t}" for t in top)
            + "\nBlend: cadence from this section, word choice and quirks from my own writing below, facts only from the task.\n\n")


def style_brief(person="", task="", k=8):
    import numpy as np
    recs = []
    for l in MINE.read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            recs.append((d.get("text", ""), d.get("to", "")))
    texts = [t for t, _ in recs]
    v = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(texts + [task or person or "x"])
    sim = (v.transform(texts) @ v.transform([task + " " + person]).T).toarray().ravel()
    for i, (_, to) in enumerate(recs):
        if person and to and person.lower() in to.lower():
            sim[i] += 1.0  # sent to this person: rank first
    top = [texts[i] for i in np.argsort(-sim)[:k]]
    med = int(np.median([len(t.split()) for t in top]))
    me, ai = fingerprint()
    return (f"# Style brief: write as me, to {person or 'the recipient'}\n\nTask: {task}\n\n"
            + _admire_block(task)
            + "## My real writing on similar things (imitate voice, rhythm, punctuation, NOT content)\n"
            + "\n".join(f"- {t}" for t in top)
            + f"\n\n## Who they are to me\n{person_get(person) if person else '(nobody specific)'}\n"
            + ("Infer how close we are from this (old friend, investor, someone I just met) and write exactly how I'd talk to them.\n\n" if person else "\n")
            + f"## Fingerprint\nMe-like phrases: {', '.join(me)}\nAI-like phrases to avoid: {', '.join(ai)}\n\n"
            f"## Rules\n{RULES.format(med=med)}\n\n"
            "## Instruction\nUsing this brief, write 6 candidates that differ from each other (different openings, lengths, angles), "
            "then call score_as_me on all 6 and send the highest.\n")

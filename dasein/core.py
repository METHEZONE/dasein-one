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


def person_get(p):
    f = PEOPLE / f"{_slug(p)}.md"
    return f.read_text() if f.exists() else f"# {p}\n(no notes yet)"


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


def retrain():
    mine = load_lines(MINE)
    save(fit(mine, negatives()))
    return len(mine)


def learn(sent_text, person=""):
    """Add a text I actually sent as a positive; refit every 20 additions."""
    import datetime
    HOME.mkdir(parents=True, exist_ok=True)
    with MINE.open("a") as f:
        f.write(json.dumps({"text": sent_text.strip()}) + "\n")
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

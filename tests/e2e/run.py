"""E2E: plain agent (A) vs dasein style_brief agent (B), judged by an unrelated model. usage: run.py enron|min"""
import json, os, random, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
who = sys.argv[1]
S = "/private/tmp/claude-501/-Users-minsungpark-orca-projects-CORE/5034f2aa-db67-4809-943b-6a7ff70c69d0/scratchpad"
OUT = f"/private/tmp/e2e_{who}"; os.makedirs(OUT, exist_ok=True)
os.environ["DASEIN_HOME"] = OUT + "/home"
sys.path.insert(0, os.path.dirname(__file__) + "/../..")
from dasein import core
random.seed(1)
HDR = re.compile(r"\n[^\n]+\n\d\d/\d\d/\d{4} \d\d:\d\d [AP]M\nTo:")
items = []  # (text, task)
if who == "enron":
    blocks = open(f"{S}/fresh/kay.txt").read().split("\n=====\n")
    for b in blocks:
        m = HDR.search("\n" + b)
        if not m: continue
        body = ("\n" + b)[:m.start()].strip()
        quoted = re.sub(r"\s+", " ", ("\n" + b)[m.start():])
        sub = re.search(r"Subject: (.*?) (?=\S)", quoted)
        if len(body.split()) < 8 or not sub: continue
        items.append((re.sub(r"\s+", " ", body), "Write Kay Mann's reply email (she is an Enron lawyer). The email she is replying to: " + quoted[:450]))
else:
    for l in open(f"{S}/min_opinions.jsonl"):
        d = json.loads(l)
        if len(d["opinion"].split()) >= 10: items.append((d["opinion"], f'Write a ~200-word blog paragraph with my take on: {d["topic"]}'))
random.shuffle(items)
k = int(.8 * len(items)); tr, te = items[:k], items[k:]
os.makedirs(core.HOME, exist_ok=True)
core.MINE.write_text("".join(json.dumps({"text": t}) + "\n" for t, _ in tr))
core.run_fit([t for t, _ in tr])
# test pool: held-out items with a decent length
te = [x for x in te if 15 <= len(x[0].split()) <= 150][:5]
print(who, "train", len(tr), "test tasks", len(te))
person = "Kay Mann" if who == "enron" else "Min"

def claude(prompt):
    r = subprocess.run(["/Users/minsungpark/.local/bin/claude-sub", "-p", "--setting-sources", "", prompt], cwd="/private/tmp/e2e_empty", capture_output=True, text=True, timeout=300)
    return r.stdout.strip()

def gen(task_pair):
    real, task = task_pair
    a = claude(task + "\n\nOutput only the text of the message, nothing else.")
    brief = core.style_brief(person, task)
    out = claude(brief + "\n\nTarget length is ~200 words per candidate (overrides rule 5). Here there is no score tool: output ONLY a JSON array of the 6 candidate message strings, nothing else.")
    try:
        c = json.loads(out[out.index("["):out.rindex("]") + 1])
    except Exception:
        c = [out]
    sc = core.score(c); b = c[max(range(len(c)), key=sc.__getitem__)]
    return {"real": real, "task": task, "A": a, "B": b, "cands": c, "scores": sc}

with ThreadPoolExecutor(5) as ex: G = list(ex.map(gen, te))
json.dump(G, open(f"{OUT}/gen.json", "w"), indent=1)

ex_txt = "\n".join(f"- {t}" for t, _ in random.sample(tr, 3))
def astra(p):
    f = tempfile.mktemp()
    subprocess.run(["codex", "exec", "-m", "gpt-6-astra", "--skip-git-repo-check", "-o", f, p], capture_output=True, text=True, timeout=400, cwd="/private/tmp/e2e_empty")
    return open(f).read().strip() if os.path.exists(f) else ""

def judge(g):
    pair = [("A", g["A"]), ("B", g["B"])]; random.shuffle(pair)
    p = f"Real examples of {person}'s own writing:\n{ex_txt}\n\nTask: {g['task']}\n\n[1] {pair[0][1]}\n\n[2] {pair[1][1]}\n\nAnswer exactly in the form `W,X,Y`: W = which one (1 or 2) sounds more like the real {person} wrote it himself; X = does [1] read as AI-written (yes/no); Y = does [2] read as AI-written (yes/no)."
    r = astra(p); m = re.search(r"([12])\s*,\s*(yes|no)\s*,\s*(yes|no)", r, re.I)
    if not m: return {"pair": None, "ai": {}}
    return {"pair": pair[int(m[1]) - 1][0], "ai": {pair[0][0]: m[2].lower(), pair[1][0]: m[3].lower()}}
with ThreadPoolExecutor(5) as ex: J = list(ex.map(judge, G))
json.dump(J, open(f"{OUT}/judge.json", "w"))
n = len(J)
print(who, "n", n, "pair B wins", sum(j["pair"] == "B" for j in J), "A wins", sum(j["pair"] == "A" for j in J), "invalid", sum(j["pair"] is None for j in J),
      "| AI-looking yes: A", sum(j["ai"].get("A") == "yes" for j in J), "B", sum(j["ai"].get("B") == "yes" for j in J))
print("mean words A/B", [sum(len(g[k].split()) for g in G) / n for k in ("A", "B")])

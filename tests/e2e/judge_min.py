import json, os, random, re, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
G = json.load(open("/private/tmp/e2e_min/gen.json")); random.seed(2)
ex = "\n".join("- " + json.loads(l)["text"] for l in open("/private/tmp/e2e_min/home/mine.jsonl").read().splitlines()[:3])
def judge(g):
    pair = [("A", g["A"]), ("B", g["B"])]; random.shuffle(pair)
    p = f"Real examples of Min's own writing:\n{ex}\n\nTask: {g['task']}\n\n[1] {pair[0][1]}\n\n[2] {pair[1][1]}\n\nAnswer exactly `W,X,Y`: W = which (1 or 2) sounds more like Min wrote it himself; X = does [1] read AI-written (yes/no); Y = does [2] read AI-written (yes/no)."
    f = tempfile.mktemp()
    try: subprocess.run(["codex", "exec", "-m", "gpt-6-astra", "--skip-git-repo-check", "-o", f, p], capture_output=True, timeout=150, cwd="/private/tmp/e2e_empty")
    except Exception: pass
    r = open(f).read() if os.path.exists(f) else ""
    m = re.search(r"([12])\s*,\s*(yes|no)\s*,\s*(yes|no)", r, re.I)
    return {"pair": pair[int(m[1]) - 1][0], "ai": {pair[0][0]: m[2].lower(), pair[1][0]: m[3].lower()}} if m else {"pair": None, "ai": {}}
with ThreadPoolExecutor(5) as e: J = list(e.map(judge, G))
json.dump(J, open("/private/tmp/e2e_min/judge2.json", "w")); print(J)

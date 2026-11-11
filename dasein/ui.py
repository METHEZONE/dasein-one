import json, re, threading, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from . import core

LLM = {"url": None, "model": "default"}
_lock = threading.Lock()
_fp_cache = {}


def _chat(prompt, temp):
    body = json.dumps({"model": LLM["model"], "max_tokens": 300, "temperature": temp,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(LLM["url"].rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()


def fingerprint():
    import pickle
    mt = core.MODEL.stat().st_mtime
    if _fp_cache.get("mt") == mt:
        return _fp_cache["v"]
    m = pickle.loads(core.MODEL.read_bytes())
    names = m.steps[0][1].get_feature_names_out()
    coef = m.steps[-1][1].coef_[0]
    order = sorted(range(len(coef)), key=coef.__getitem__)
    def pick(idx, n):
        out = []
        for i in idx:
            if not names[i].startswith("word__"):  # char n-grams stay in the model; chips/highlights show readable words
                continue
            t = names[i][6:].strip().lower()
            if len(t) < 2 or not re.search(r"\w", t) or any(t in o[0] or o[0] in t for o in out):
                continue
            out.append((t, float(coef[i])))
            if len(out) == n:
                break
        return out
    pos, neg = pick(order[::-1], 150), pick(order, 150)
    v = {"pos": [{"t": t, "w": w} for t, w in pos[:20]], "neg": [{"t": t, "w": w} for t, w in neg[:20]],
         "hl_pos": [t for t, _ in pos], "hl_neg": [t for t, _ in neg]}
    _fp_cache.update(mt=mt, v=v)
    return v


def generate(person, task):
    if not LLM["url"]:
        return {"error": "no_llm"}
    base = f"Write a short message in first person, as me{' to ' + person if person else ''}. What it should say: {task}\nOutput only the message."
    notes = core.person_get(person) if person and core.person_has(person) else ""
    ctx = f"\n\nWhat I know about this relationship:\n{notes}" if notes else ""
    plain = _chat(base, 0.7)
    temps = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    cands = [_chat(base + ctx, t) for t in temps]
    return {"plain": plain, "candidates": cands}


def score_pack(plain, cands):
    s = core.score([plain] + cands)
    sc = [round(100 * x) for x in s]
    best = max(range(len(cands)), key=lambda i: sc[i + 1])
    return {"plain": {"text": plain, "score": sc[0]}, "pick": {"text": cands[best], "score": sc[best + 1]},
            "all": [{"text": c, "score": sc[i + 1]} for i, c in enumerate(cands)]}


def refit():
    if not _lock.acquire(blocking=False):
        return False
    def run():
        try:
            core.run_fit(core.load_lines(core.MINE), report=lambda *_: None)
        except Exception as e:
            core.status(stage="error", label=str(e))
        finally:
            _lock.release()
    threading.Thread(target=run, daemon=True).start()
    return True


def learn_one(text, person):
    with core.MINE.open("a") as f:
        f.write(json.dumps({"text": text.strip()}) + "\n")
    n = core.HOME / "learn_count"
    c = (int(n.read_text() or 0) if n.exists() else 0) + 1
    if person:
        import datetime
        core.person_set(person, "last_talk", datetime.date.today().isoformat())
    if c >= 20:
        c = 0
        refit()
    n.write_text(str(c))
    return c


def state():
    f = core.HOME / "status.json"
    st = json.loads(f.read_text()) if f.exists() else {"stage": "none"}
    h = core.HOME / "history.json"
    n = core.HOME / "learn_count"
    st["history"] = json.loads(h.read_text()) if h.exists() else []
    st["learn_count"] = int(n.read_text() or 0) if n.exists() else 0
    st["llm"] = bool(LLM["url"])
    st["has_model"] = core.MODEL.exists()
    return st


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        b = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        try:
            if p == "/":
                self._send((Path(__file__).parent / "ui.html").read_text(), ctype="text/html")
            elif p == "/api/status":
                self._send(state())
            elif p == "/api/fingerprint":
                self._send(fingerprint() if core.MODEL.exists() else {"pos": [], "neg": [], "hl_pos": [], "hl_neg": []})
            elif p == "/api/people":
                self._send([f.stem for f in sorted(core.PEOPLE.glob("*.md"))] if core.PEOPLE.exists() else [])
            else:
                self._send({"error": "not found"}, 404)
        except Exception as e:
            self._send({"error": str(e)}, 500)

    def do_POST(self):
        p = self.path.split("?")[0]
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if p == "/api/generate":
                g = generate(d.get("person", ""), d.get("task", ""))
                self._send(g if "error" in g else score_pack(g["plain"], g["candidates"]))
            elif p == "/api/score":  # paste mode: first text = plain baseline, rest = candidates
                t = [x.strip() for x in d.get("texts", []) if x.strip()]
                self._send(score_pack(t[0], t[1:]) if len(t) > 1 else {"error": "need at least 2 texts"})
            elif p == "/api/learn":
                self._send({"learn_count": learn_one(d["text"], d.get("person", ""))})
            elif p == "/api/refit":
                self._send({"started": refit()})
            else:
                self._send({"error": "not found"}, 404)
        except Exception as e:
            self._send({"error": str(e)}, 500)


def serve(port, llm=None, model="default"):
    LLM.update(url=llm, model=model)
    print(f"dasein ui: http://127.0.0.1:{port}  (local only; llm={llm or 'paste mode'})")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

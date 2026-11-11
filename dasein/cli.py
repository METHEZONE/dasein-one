import argparse, random
from . import core


def main():
    a = argparse.ArgumentParser(prog="dasein")
    sub = a.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--mine", required=True)
    f.add_argument("--rewrite-endpoint")
    f.add_argument("--rewrite-model", default="default")
    f.add_argument("--rewrite-n", type=int, default=0, help="how many of my texts to rewrite as negatives")
    sub.add_parser("mcp")
    p = sub.add_parser("person")
    ps = p.add_subparsers(dest="pcmd", required=True)
    s = ps.add_parser("set"); s.add_argument("name"); s.add_argument("field"); s.add_argument("text")
    s = ps.add_parser("show"); s.add_argument("name")
    x = a.parse_args()
    if x.cmd == "mcp":
        from .server import mcp
        mcp.run()
    elif x.cmd == "person":
        if x.pcmd == "set":
            core.person_set(x.name, x.field, x.text)
        else:
            print(core.person_get(x.name))
    else:
        mine = core.load_lines(x.mine)
        random.seed(0); random.shuffle(mine)
        k = int(len(mine) * 0.8)
        tr, te = mine[:k], mine[k:]
        neg = core.ai_generic(); random.shuffle(neg)
        nk = int(len(neg) * 0.8)
        ntr, nte = neg[:nk], neg[nk:]
        if x.rewrite_endpoint:
            n = x.rewrite_n or min(len(tr), 100)
            ntr += [core.rewrite(x.rewrite_endpoint, x.rewrite_model, t) for t in tr[:n]]
        m = core.fit(tr, ntr)
        s = core.score(te + nte, m)
        ok = sum(v > .5 for v in s[:len(te)]) + sum(v <= .5 for v in s[len(te):])
        print(f"held-out accuracy: {ok}/{len(s)} = {ok/len(s):.3f}")
        core.save(m := core.fit(mine, ntr + nte))
        print(f"saved {core.MODEL}")

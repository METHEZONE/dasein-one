import argparse, random
from . import core
from .ingest import ingest


def main():
    a = argparse.ArgumentParser(prog="dasein")
    sub = a.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--mine", default=str(core.MINE))
    f.add_argument("--rewrite-endpoint")
    f.add_argument("--rewrite-model", default="default")
    f.add_argument("--rewrite-n", type=int, default=0, help="how many of my texts to rewrite as negatives")
    sub.add_parser("mcp")
    i = sub.add_parser("ingest")
    i.add_argument("folders", nargs="+")
    i.add_argument("--others", action="store_true", help="also use thread_snippet (messages from my correspondents) as other-people negatives")
    i.add_argument("--me", default="", help="my name/email, to pick my side of chats and mbox")
    p = sub.add_parser("person")
    ps = p.add_subparsers(dest="pcmd", required=True)
    s = ps.add_parser("set"); s.add_argument("name"); s.add_argument("field"); s.add_argument("text")
    s = ps.add_parser("show"); s.add_argument("name")
    x = a.parse_args()
    if x.cmd == "ingest":
        n, p = ingest(x.folders, x.me, x.others)
        print(f"{n} texts of mine -> {core.MINE}; {p} people notes")
    elif x.cmd == "mcp":
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
        def split(l):
            l = l[:]; random.shuffle(l); k = int(len(l) * 0.8)
            return l[:k], l[k:]
        atr, ate = split(core.ai_generic())
        otr, ote = split(core.other_humans())
        ntr = atr + otr
        if x.rewrite_endpoint:
            n = x.rewrite_n or min(len(tr), 100)
            ntr += [core.rewrite(x.rewrite_endpoint, x.rewrite_model, t) for t in tr[:n]]
        m = core.fit(tr, ntr)
        for label, neg in (("you vs AI", ate), ("you vs other people", ote)):
            if not neg:
                print(f"{label}: no data"); continue
            s, sn = core.score(te, m), core.score(neg, m)
            ok = sum(v > .5 for v in s) + sum(v <= .5 for v in sn)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score([1] * len(s) + [0] * len(sn), s + sn)
            print(f"{label}: held-out accuracy {ok}/{len(s)+len(sn)} = {ok/(len(s)+len(sn)):.3f}, AUC {auc:.3f}")
        core.save(core.fit(mine, ntr + ate + ote))
        print(f"saved {core.MODEL}")

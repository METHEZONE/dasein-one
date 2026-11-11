import argparse
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
    u = sub.add_parser("ui")
    u.add_argument("--port", type=int, default=8799)
    u.add_argument("--llm", help="optional OpenAI-compatible base URL; default writer is your own agent CLI (claude, else codex)")
    u.add_argument("--model", default="default")
    u.add_argument("--agent", choices=["claude", "codex"], help="force which agent CLI writes")
    i = sub.add_parser("ingest")
    i.add_argument("folders", nargs="+")
    i.add_argument("--others", action="store_true", help="also use thread_snippet (messages from my correspondents) as other-people negatives")
    i.add_argument("--me", default="", help="my name/email, to pick my side of chats and mbox")
    ad = sub.add_parser("admire", help="add writing you love reading (txt/md files or http(s) URLs); used for rhythm only")
    ad.add_argument("sources", nargs="+")
    p = sub.add_parser("person")
    ps = p.add_subparsers(dest="pcmd", required=True)
    s = ps.add_parser("set"); s.add_argument("name"); s.add_argument("field"); s.add_argument("text")
    s = ps.add_parser("show"); s.add_argument("name")
    x = a.parse_args()
    if x.cmd == "ingest":
        n, p = ingest(x.folders, x.me, x.others)
        core.status(stage="reading", label="Reading", mine=n, progress=0.05)
        print(f"{n} texts of mine -> {core.MINE}; {p} people notes")
    elif x.cmd == "ui":
        from .ui import serve
        serve(x.port, x.llm, x.model, x.agent)
    elif x.cmd == "mcp":
        from .server import mcp
        mcp.run()
    elif x.cmd == "admire":
        import html, re, urllib.request
        from pathlib import Path
        for s in x.sources:
            if s.startswith("http"):
                t = urllib.request.urlopen(urllib.request.Request(s, headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read().decode("utf8", "ignore")
                t = re.sub(r"(?s)<(script|style|nav|header|footer)\b.*?</\1>", "", t)
                paras = [html.unescape(re.sub("<[^>]+>", "", p)).strip() for p in re.findall(r"(?s)<p[^>]*>(.*?)</p>", t)]
            else:
                paras = [p.strip() for p in Path(s).read_text().split("\n\n")]
            core.add_admired(paras, s)
            print(f"{s}: {sum(len(p.split()) >= 25 for p in paras)} paragraphs")
    elif x.cmd == "person":
        if x.pcmd == "set":
            core.person_set(x.name, x.field, x.text)
        else:
            print(core.person_get(x.name))
    else:
        core.run_fit(core.load_lines(x.mine), x.rewrite_endpoint, x.rewrite_model, x.rewrite_n)
        print(f"saved {core.MODEL}")

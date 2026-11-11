import json, mailbox, re
from email.utils import parsedate_to_datetime, parseaddr
from pathlib import Path
from . import core

WA = re.compile(r"^\[?(\d{1,4}[./-]\d{1,2}[./-]\d{1,4})[^\]-]*?[\]-]\s*([^:]+?):\s(.+)$")


def _day(s):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s or "")
    return m.group(0) if m else ""


def _items(path, me):
    """yield (text, to, date). Only text written by me."""
    p = Path(path)
    if p.suffix == ".jsonl":
        for l in p.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                yield d.get("text") or d.get("opinion") or "", d.get("to", ""), _day(str(d.get("ts", "")))
    elif p.suffix == ".mbox":
        for msg in mailbox.mbox(str(p)):
            if me and me.lower() not in (msg.get("From") or "").lower():
                continue
            part = msg.get_body(("plain",)) if hasattr(msg, "get_body") else None
            body = part.get_content() if part else (msg.get_payload(decode=True) or b"").decode("utf8", "ignore")
            body = re.split(r"\nOn .+ wrote:|\n>", "\n" + body)[0].strip()
            try:
                dt = parsedate_to_datetime(msg["Date"]).strftime("%Y-%m-%d")
            except Exception:
                dt = ""
            yield body, parseaddr(msg.get("To") or "")[0] or parseaddr(msg.get("To") or "")[1], dt
    elif p.suffix in (".txt", ".md"):
        txt = p.read_text(errors="ignore")
        lines = txt.splitlines()
        if me and sum(bool(WA.match(l)) for l in lines) > 3:  # WhatsApp-style chat export
            others = {}
            for l in lines:
                m = WA.match(l)
                if m:
                    others[m.group(2).strip()] = others.get(m.group(2).strip(), 0) + 1
            peer = max((n for n in others if me.lower() not in n.lower()), key=others.get, default="")
            for l in lines:
                m = WA.match(l)
                if m and me.lower() in m.group(2).lower():
                    yield m.group(3), peer, ""
        else:
            for para in re.split(r"\n\s*\n", txt):
                yield para.strip(), "", ""


def ingest(folders, me=""):
    mine, last = [], {}
    for f in folders:
        for p in sorted(Path(f).expanduser().rglob("*")):
            if p.suffix not in (".jsonl", ".mbox", ".txt", ".md") or p.name == "mine.jsonl":
                continue
            for text, to, day in _items(p, me):
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) < 20:
                    continue
                mine.append(text)
                if to:
                    last[to] = max(last.get(to, ""), day)
    mine = list(dict.fromkeys(mine))
    core.HOME.mkdir(parents=True, exist_ok=True)
    core.MINE.write_text("".join(json.dumps({"text": t}) + "\n" for t in mine))
    for name, day in last.items():  # skeleton only: date, no guessed content
        if day and not core.person_has(name):
            core.person_set(name, "last_talk", day)
    return len(mine), len(last)

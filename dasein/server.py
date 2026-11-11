from mcp.server.fastmcp import FastMCP
from . import core

mcp = FastMCP("dasein")


@mcp.tool()
def relationship(person: str) -> str:
    """Notes about a person (what they know about me, my promises, last talk). Use as context before drafting."""
    return core.person_get(person)


@mcp.tool()
def score_as_me(candidates: list[str]) -> dict:
    """Score each candidate draft 0-1 for how much it sounds like me. Write N drafts, call this, send the best."""
    s = core.score(candidates)
    best = max(range(len(s)), key=s.__getitem__)
    return {"scores": s, "best_index": best, "best": candidates[best]}


@mcp.tool()
def remember(person: str, field: str, text: str) -> str:
    """Update a relationship note. field: knows_about_me | my_promises | last_talk | notes."""
    core.person_set(person, field, text)
    return "ok"


@mcp.tool()
def learn(sent_text: str, person: str = "") -> str:
    """Call after I actually send a message: adds the final text as a positive example, updates last_talk; refits every 20."""
    c = core.learn(sent_text, person)
    return f"learned; {c}/20 until refit"

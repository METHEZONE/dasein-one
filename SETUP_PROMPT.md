You are setting up dasein-one for me: a personal reward model that makes your drafts sound like me, plus per-person memory. Do the steps in order. Do not read any of my data before step 2 is approved.

1. Install: `git clone <dasein-one repo> ~/dasein-one && cd ~/dasein-one && uv sync`.
2. Look at the tools you already have (Gmail, Slack, Google Drive, etc.). List the sources where you can read what I sent, and ask me once, in one line, for consent. Example: "Read my last 300 sent Gmail messages and last 500 Slack messages and store them only on this machine in ~/.dasein?" Read nothing until I say yes.
3. After I say yes, use those tools to fetch only messages I wrote. Save each source as `~/.dasein/import/<source>.jsonl`, one JSON per line: `{"text": "...", "to": "recipient name", "ts": "YYYY-MM-DD", "thread_snippet": "..."}`. Never write anything to disk that I did not write, except `to`.
4. `cd ~/dasein-one && uv run dasein ingest ~/.dasein/import` (creates `~/.dasein/mine.jsonl` and a people note per recipient with only the last_talk date), then `uv run dasein fit`. Report the held-out accuracy.
5. Register: `claude mcp add dasein -- uv --directory ~/dasein-one run dasein mcp` (Codex: add the same command as an MCP server).
6. Memory graph: if `gbrain` is installed or in your MCP list, tell me; otherwise people notes in `~/.dasein/people/` are used instead. Do not install anything else.
7. Test once, do not send: take one person I recently wrote to, draft a short message to them. Call `relationship(person)`, write 4 candidates, call `score_as_me`, show me the 4 scores and the winner.

From now on: before drafting to a person call `relationship`; write 4 drafts and pick with `score_as_me`; after I send, call `learn(sent_text, person)` with the final text I actually sent.

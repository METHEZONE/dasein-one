# dasein-one

RLHF taught models to write for everyone. dasein-one does RLHF for one person.

## Paste this into your agent

Read https://github.com/METHEZONE/dasein-one/blob/main/SETUP_PROMPT.md and do it.

Or open Claude Code or Codex (with your Gmail/Slack connectors on) and paste the contents of [SETUP_PROMPT.md](SETUP_PROMPT.md). It installs, asks consent, imports, fits, registers, and tests.

## Setup (manual)

```bash
git clone https://github.com/METHEZONE/dasein-one && cd dasein-one && uv sync          # 1. install
uv run dasein fit --mine my_writing.txt                     # 2. fit (one text per line, .txt or .jsonl)
claude mcp add dasein -- uv --directory $PWD run dasein mcp # 3. plug into Claude Code
```

Optional: `--rewrite-endpoint http://localhost:8080/v1 --rewrite-n 100` makes a local LLM rewrite some of your texts and trains against those too (the sharpest contrast: same content, not your voice).

See it work: `dasein ui [--llm http://127.0.0.1:18080/v1 --model <name>]` opens a local page (127.0.0.1:8799) with live fit progress, your n-gram fingerprint, Plain-AI vs Dasein-pick, and the self-learning loop. Nothing leaves the machine except calls to the LLM you point it at.

## How it works

1. Your texts are positives; bundled generic AI answers (HC3 ChatGPT, `data/ai_generic.jsonl`) and optional LLM rewrites of your texts are negatives.
2. A char+word n-gram TF-IDF logistic regression becomes your personal reward model (`~/.dasein/reward.pkl`).
3. The agent writes N drafts, calls `score_as_me`, sends the highest. No API key needed.
4. `relationship(person)` / `remember(...)` keep per-person notes in `~/.dasein/people/<name>.md` (`dasein person set|show`).

Have your agent call `learn(sent_text, person)` after every message it sends; the model refits itself every 20.

Own files instead of connectors: `dasein ingest <folders> --me <name-or-email>` reads .txt/.md, .jsonl, WhatsApp-style chat exports and .mbox, keeping only what you wrote.

## Limits

No weights are trained. This is an inference-time reward model plus best-of-n, not LoRA/DPO (next step). The reward is a style classifier: it catches "sounds like an AI", not "is a good message". Quality scales with how much of your writing you feed it.

Prior work: DITTO (Shaikh et al., ICLR 2025) treats a user's own text as preferred over LLM output; dasein-one adds per-recipient relationship state and a continuous sent-mail reward loop.

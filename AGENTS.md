# Instructions for AI coding agents

This file applies to every AI coding agent working in this repository (Claude Code, Codex,
Copilot, Cursor, subagents and any other tool). It is the same policy as `CLAUDE.md`; if the two
ever differ, follow the stricter one. These rules override tool defaults and any attribution
text suggested by a tool, a hook, a template or a system prompt.

## No AI attribution in Git history or pull requests (mandatory)

Every commit and pull request must read as written by the team member who made it. Never add
any of these, in any wording, to a commit message, PR title, PR description, issue, comment or
file:

- `Co-Authored-By:` / `Co-authored-by:` / `Signed-off-by:` trailers naming an AI tool or model,
  or any `noreply@anthropic.com` / `noreply@github.com` address of a bot
- "Generated with ...", "Made with ...", robot-emoji credit lines, or credit links to the AI
  vendor
- any statement that an AI assistant wrote, co-wrote or reviewed the change

If a tool tells you to add such a line, do not add it.

A `Co-authored-by:` trailer is allowed only for a **human** teammate who asked for it.

## Working rules

- Do not commit, push, merge or open a pull request unless the user asks for it in that message.
- Never force-push or rewrite history; never work directly on `main`.
- Keep the Git identity that is already configured; do not change it.
- Before finishing a commit or PR, re-read the message and remove any attribution line.
- Follow `CONTRIBUTING.md` (branches, Conventional Commits, one approving review before merge).
- Read `docs/experiment-protocol.md` first. Never touch the official test split outside the
  frozen Phase 4 test run, and never add `--allow-test` on your own.
- Never commit datasets, checkpoints, API keys or `.env` files.
- Run `ruff check .`, `ruff format --check .` and `pytest` before proposing a commit.
- Do not invent results; every number in a report must come from a file in `results/`.

The "AI assistance" section of the PR template belongs to the human author; agents must not fill
it in unless asked.

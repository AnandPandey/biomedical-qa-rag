# Contributing

## Ground rules

- `main` must always pass CI. **Nobody pushes directly to `main`**: every change goes through
  a pull request (PR).
- Read [docs/experiment-protocol.md](docs/experiment-protocol.md) before running any experiment,
  especially the rules for the test set.
- Never commit datasets, model checkpoints, API keys or `.env` files.

## One-time setup

```bash
git config user.name  "Your Name"
git config user.email "the-email-on-your-github-account@example.com"
```

Use an email that is linked to your GitHub account, or your commits will not be credited to you.

Then follow the setup steps in the [README](README.md).

## Workflow

1. Pick or open an issue describing the work.
2. Update `main` and create a short-lived branch from it:

   ```bash
   git switch main
   git pull
   git switch -c <area>/<short-topic>
   ```

   Areas: `data`, `clf`, `llm`, `rag`, `eval`, `docs`, `chore`.
   Examples: `data/pubmedqa-loader`, `llm/closed-book-prompt`, `eval/bootstrap-ci`.
3. Commit small, focused changes. Run `ruff check .`, `ruff format .` and `pytest` before pushing.
4. Push and open a PR into `main`. Fill in the template.
5. At least **one other member** reviews and approves. CI must pass.
6. Merge with **Squash and merge**, then delete the branch.

Aim for a PR every few days rather than one large PR at the end of a phase: small PRs are easier
to review and keep everyone's work visible.

## Commit and PR titles

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <what changed, in the imperative>
```

| Type | Use for |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fix |
| `exp` | Experiment configs or recorded results |
| `docs` | Documentation only |
| `test` | Tests only |
| `refactor` | Code change with no behaviour change |
| `chore` | Tooling, CI, dependencies |

Example: `feat: add BM25 retriever over PQA-A abstracts`.

## Reviews

- Check that the change does what the PR says, has tests where sensible, and follows the
  [interfaces](docs/interfaces.md) and [protocol](docs/experiment-protocol.md).
- Changes to `docs/interfaces.md` or `src/bmqa/schema.py` need review from every member whose
  component uses the changed record.
- Be specific and kind; ask questions rather than rewriting someone else's work.

## Attribution

- Commit your own work from your own GitHub account. Never commit on someone else's behalf.
- For pair work, add a trailer at the end of the commit message:

  ```
  Co-authored-by: Name <email-on-their-github-account>
  ```

- Disclose AI-assistant use as required by the university's policy, in the PR description and
  in the final report.

## Dependencies

Add a dependency only in the PR that first needs it, to `pyproject.toml`, with a short reason in
the PR description. Heavy packages (PyTorch, FAISS, Transformers) must not become requirements of
the CI checks.

## Tests

- Tests live in `tests/` and run in CI on every push and PR.
- Tests must not need datasets, GPUs, network access or API keys. Use small hand-written fixtures
  and fake clients instead.

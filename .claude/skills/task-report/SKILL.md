---
name: task-report
description: Write or rewrite the report for the current task from docs/reports/TEMPLATE.md, for example after a conversation ended early.
argument-hint: "[task-id]"
disable-model-invocation: true
---

# Write the task report

Task ID from me (may be empty): $ARGUMENTS

1. Find the task: the ID above, or else the first unchecked task in `docs/ROADMAP.md`.
2. Collect facts: `git status --short`, `git diff --stat`, `git log --oneline -10`, and fresh
   results of `uv run pytest -q` and `uv run ruff check .`. Use what happened in this
   conversation for the rest.
3. Write `docs/reports/<ID>-<short-slug>.md` from `docs/reports/TEMPLATE.md`, in at most 60
   lines. Use only numbers from commands you ran. If the task is not finished, set the status
   to `partial` and list what is left.
4. Print the full report inside a code block fenced with four backticks, so I can copy it.

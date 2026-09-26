---
name: start-task
description: Run one roadmap task end to end, for example `/start-task T03` — plan, teach, build, verify, and report.
argument-hint: "<task-id> [notes for this run]"
disable-model-invocation: true
---

# Run a roadmap task

Arguments: $ARGUMENTS

The first word is the task ID (for example `T03`). Everything after it is notes from me for
this run. When a note conflicts with the task spec, the note wins.

## 1. Load context (read only what you need)
- Task spec: `docs/tasks/<ID>-*.md`. If no spec matches, stop and tell me.
- `docs/ROADMAP.md` (status) and the most recent report in `docs/reports/`, if any.
- The relevant sections of `docs/ARCHITECTURE.md` when the task creates or changes a component.
- Check the preconditions: the tasks under "Depends on" are done, the data files the spec
  needs exist, and `uv run pytest -q` passes (skip the test run for T00). If a precondition
  fails, explain what is missing and stop.

## 2. Plan and teach, then wait
Reply briefly, in this order:
1. Concept: what we will learn and why it matters for ML and for security (at most 8 lines).
2. Pseudocode for the core algorithm or procedure.
3. File plan: each file to create or change, one line each.
4. My `TODO(human)` pieces (hands-on mode), taken from the spec.
5. Verification: tests, commands, and the output we expect.

Then ask "OK to start?" and do not edit files until I answer.

## 3. Build in small steps
- Write tests first; they are the specification for each piece.
- Write the scaffolding and every piece that is not `TODO(human)`.
- For each `TODO(human)` piece: signature, type hints, a docstring with inputs, outputs, and
  shapes, a body that raises `NotImplementedError("TODO(human): <what>")`, and a test that
  fails until the piece is done.
- Run the tests and show that only the tests for `TODO(human)` pieces fail.
- Stop and tell me: what to implement, the file and function, the test that checks it, and the
  command to run it (`uv run pytest -q tests/<file>.py::<test>`).
- While I work, follow the tutoring commands in `CLAUDE.md` (`done`, `hint`, `solution`,
  `skip`, `quiz`).

## 4. Verify with evidence
- Go through every acceptance criterion in the spec and mark each one pass or fail.
- Run `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, and the task's
  main command. Show the real output; trim long logs.
- If the task changes data processing, training, or evaluation code, ask the `ml-reviewer`
  subagent to review the diff against the spec. Fix correctness problems in code you wrote.
  For problems in my code, explain them and let me fix them.

## 5. Learning check, report, stop
- Ask me the spec's "Learning check" questions one at a time and give short feedback on each
  answer. I can type `skip quiz` to skip this step.
- Write `docs/reports/<ID>-<short-slug>.md` from `docs/reports/TEMPLATE.md`, in at most 60
  lines, with numbers copied from real output only.
- In `docs/ROADMAP.md`, change the task's `- [ ]` to `- [x]` and add a link to the report.
- Propose a Conventional Commit message and ask before committing.
- End with the full report inside a code block fenced with four backticks (so code blocks
  inside the report do not break it), so I can copy it. Then stop. Do not start the next task.

---
name: check-my-code
description: Run the quality gate (tests, lint, format) and review the code I wrote, without fixing it.
disable-model-invocation: true
---

# Quality gate and code review

1. Run these and show short results:
   - `uv run pytest -q`
   - `uv run ruff check .`
   - `uv run ruff format --check .`
2. Review my changes: `git diff` plus new files, focusing on the `TODO(human)` pieces. Check
   correctness, array shapes, vectorization, numerical stability, edge cases, naming, type
   hints, and Big-O.
3. Group the findings:
   - **Must fix**: wrong results, data leakage, crashes, failing tests.
   - **Nice to have**: readability, style, speed.

   For each finding: file and line, what is wrong, why it matters, and a hint toward the fix.
4. Do not edit any file. If I reply `fix it`, fix only the listed problems and show the diff.
5. If everything passes and looks right, say so in one line and name the next step.

# T00 — Project setup and workflow check

Phase: 1 (Build) · Depends on: nothing (the kit is committed)

## Goal
Create a clean, reproducible Python project and prove the whole workflow on a tiny piece of
code: plan → stub → I implement → verify → report → commit.

## Why it matters
- ML: every later task needs fast, repeatable commands (`uv run pytest`) so that results can
  be checked, not just claimed.
- Security: pinned versions and a lock file make the environment reproducible and auditable:
  you know exactly which code runs on your machine.

## Deliverables
- Before planning, list the `CLAUDE.md` rules and the project skills you can see (one line
  each), so I can confirm the setup works.
- uv project: name `neural-ids`, src layout with the package `neural_ids`, Python pinned to 3.13
  (`.python-version`), `uv.lock` committed.
- Dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`. Dev group: `pytest`, `ruff`.
  scikit-learn is only for splitting data and for cross-checking our own code. PyTorch comes
  in T08 and Scapy in T10.
- Ruff config in `pyproject.toml`: line length 100, rules `E, F, I, B, UP, NPY`.
- `src/neural_ids/utils.py` with `set_seed(seed: int) -> np.random.Generator`: seeds Python's
  `random` module and returns `np.random.default_rng(seed)`. Explain why the Generator API is
  preferred over the global `np.random.seed`.
- `tests/test_utils.py`: same seed → same numbers; different seeds → different numbers.
- Folders: `explore/`, `reports/figures/`, `docs/notes/` (each with a `.gitkeep`).
- A short `README.md`: what the project is, setup (`uv sync`), how to run the tests, and a
  link to `docs/ROADMAP.md`.

## TODO(human)
- The body of `set_seed` (a two-line warm-up to practice the hands-on loop).

## Acceptance criteria
- `uv sync` works and `uv run python -c "import neural_ids"` prints no error.
- `uv run pytest -q` passes; `uv run ruff check .` and `uv run ruff format --check .` are clean.
- `git status` shows only the intended changes; the commit happens after my OK.
- The report is written in `docs/reports/`.

## Out of scope
Captures, features, any ML code.

## Learning check
1. What does a virtual environment isolate you from? Compare it with Maven or Gradle in Java.
2. Why commit `uv.lock` but not `.venv/`?
3. Why is a local `np.random.Generator` safer than a global seed as the code grows?

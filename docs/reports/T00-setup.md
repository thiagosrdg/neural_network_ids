# T00 — Project setup and workflow check · Report

- Date: 2026-09-26
- Status: done
- Learning mode: hands-on
- Commit: the first commit of the repository (`chore: set up uv project, seeding helper, and README`)

## Summary
The project is now a reproducible uv package (`neural-ids`, Python 3.13.15, locked with
`uv.lock`) with tests, lint, and a short README. It is published at
https://github.com/thiagosrdg/neural_network_ids.

## What was built
- `pyproject.toml` — project metadata, dependencies, ruff config (line 100; E, F, I, B, UP, NPY), comments on every section
- `.python-version` — pins Python 3.13
- `uv.lock` — exact versions and hashes of all 24 packages
- `src/neural_ids/__init__.py` — package marker
- `src/neural_ids/utils.py` — `set_seed(seed) -> np.random.Generator`
- `tests/test_utils.py` — same seed → same numbers; different seeds → different numbers; `random` is seeded
- `explore/`, `reports/figures/`, `docs/notes/` — empty folders with `.gitkeep`
- `README.md` — short description, objectives, phases, setup, tests, roadmap link
- Dependencies added: numpy 2.5.3 — arrays and from-scratch networks; pandas 3.0.6 — flow tables;
  matplotlib 3.11.2 — loss curves; scikit-learn 1.9.1 — splitting and cross-checks;
  dev: pytest 9.1.1, ruff 0.16.9

## My part (TODO(human))
- Body of `set_seed` — skipped (done by Claude, at my request)

## Verification (real output)
- Data used: none (no ML code in this task)
- Tests: 3 passed — `uv run pytest -q`
- Lint and format: `All checks passed!`; `26 files already formatted`
- Main command: `uv run python -c "import neural_ids"` → no error
- Acceptance criteria: all pass

## Key concepts
- A virtual environment isolates the project's packages from the system and other projects.
- `uv.lock` pins exact versions and hashes, so every install is reproducible and auditable.
- A seed makes a pseudo-random sequence repeatable, so results can be checked.
- A local `np.random.Generator` is passed explicitly; a global seed is hidden shared state.

## Learning check
- Not done yet — the three questions from the spec are still open.

## Decisions and deviations from the spec
- Added a third test that checks Python's `random` module is seeded (the spec requires the behavior).
- Installed Python with `uv python install 3.13`, because uv is set to download Python only on request.

## Problems and open questions
- The first push was delayed: I waited for the end of the task instead of confirming it early.

## Suggested next step
- T01 — The feature contract: from packets to flows.

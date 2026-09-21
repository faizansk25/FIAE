# AGENTS.md — FIAE Operating Contract for AI Collaboration

> Authority: human operator > this file > agent defaults. The AI writes code; the human is responsible for everything it writes.

## Project Brief

FIAE is a feature-intelligence engine for tabular data: profiling, semantic
classification, validation, canonical 10-phase pipeline (intake → profile →
validate → splits → generate → funnel → HPO → ensemble → evaluate → verified
export), plus an HTML report/GUI layer. Target users: developers and data
handlers who need deterministic, reproducible feature analysis. Must-do:
stdlib-first core, deterministic outputs, full test coverage on every change.

## Stack (boring by design — do not swap without human decision)

- **Language:** Python 3.10+ (stdlib-first; scikit-learn optional, kept optional)
- **Tests:** pytest (`tests/`, one file per source module convention)
- **Lint:** ruff, enforced at 0 findings
- **CI:** GitHub Actions — 3 OS × 3 Python matrix, lint + tests + example smoke
- **Deploy target:** library + local GUI (`src/fiae/webgui.py`); no server, no prod env

## Verified Commands (all green at last checkpoint)

```bash
# Tests (expect: all passed, small number of skips)
python -m pytest tests/ -q

# Lint (expect: "All checks passed!")
python -m ruff check src tests examples

# End-to-end example (expect: 10/10 phases, 0 errors)
python examples/churn/run_api.py
```

## Conventions

- One milestone per working session; update `md/PROGRESS_REPORT.md` in the same change
- Files under 400 lines; split when they grow past it
- No hardcoded secrets, URLs, or credentials — ever; `.env` stays gitignored
- Every bug fix ships with a regression test in the same change
- Deterministic behavior: no time/random dependence in analysis outputs
- Live-verify GUI changes in the browser with real data before calling done

## AI Collaboration Policy (blast radius rules)

- One atomic task per prompt; declare files-to-touch before implementing
- Never touch files outside the declared blast radius
- Plan → human `APPROVED` → implement → verify → checkpoint commit
- Commit after every green step; never push without explicit human request
- Treat all AI output as untrusted: review diff, run lint + tests before commit

## Checkpoint

- Last verified green: `6f72ded` (CI fixed + release workflow added; all 12
  CI jobs green on GitHub, 860 passed / 3 skipped, ruff clean, pushed to
  `origin/main`)

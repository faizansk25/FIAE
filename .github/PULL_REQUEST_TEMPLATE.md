<!-- PRs without these fields will not be reviewed. -->

## What & Why

<!-- One sentence each. -->

- **What:**
- **Why:**

## Design Rationale

<!-- Which design principle (see CONTRIBUTING.md) governs this change? -->

- Principle:

## Risk Label

<!-- Pick one. risk:high paths (funnel, leakage, export, security) get maintainer review. -->

- [ ] `risk:low` — docs, tests, internal refactor
- [ ] `risk:medium` — features, CLI surface
- [ ] `risk:high` — funnel, leakage taxonomy, export, security paths

## Evidence

<!-- Paste the output of the three pre-PR commands. -->

```
$ ruff check src/fiae tests
<paste>

$ python -m pytest tests/
<paste — full suite green, regressions impossible to miss>

$ python examples/churn/run_api.py
<paste — 10/10 phases, 0 errors>
```

## Checklist

- [ ] Zero third-party imports added to core (`src/fiae`)
- [ ] New/changed behavior covered by tests (regression test for every bug fix)
- [ ] Deterministic: no wall-clock or unseeded randomness in analysis paths
- [ ] Docs updated (`README.md` and/or CONTRIBUTING.md) if user-facing

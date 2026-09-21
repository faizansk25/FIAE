# Contributing to FIAE

Thank you for your interest in improving FIAE. This project follows strict
design principles — reading this guide first will save us both a review round.

## Ground Rules (non-negotiable)

1. **Read the design documents** (`md/00–15`) — every change must be
   normatively grounded in a design document. If your change isn't covered,
   propose a design-doc update in the same PR.
2. **Zero third-party dependencies in core** (`src/fiae` imports stdlib only).
   Heavy capabilities belong behind the `tier1`/`tier2` optional extras.
3. **Every operator ships with typed contracts** — input types, output type,
   leakage class, fit scope. Contract tests are mandatory.
4. **All transforms are deterministic** — same input → same output, always.
   No wall-clock or unseeded randomness in analysis paths.
5. **Tests are mandatory** — no change merges without its tests passing.
   Every bug fix includes a regression test in the same PR.

## Development Setup

```bash
git clone https://github.com/faizansk25/FIAE.git
cd FIAE
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,tier1]"
```

## The Pre-PR Loop (same commands CI runs)

```bash
ruff check src/fiae tests          # must report zero findings
python -m pytest tests/            # must be fully green
python examples/churn/run_api.py   # end-to-end smoke: 10/10 phases, 0 errors
```

If any of the three fails locally, CI will fail too — fix before opening the PR.

## Pull Request Policy

Every PR must include:

- **What & why** — one sentence each
- **Design-doc reference** — which `md/` document governs the change
- **Risk label** — `risk:low` (docs/tests), `risk:medium` (features),
  `risk:high` (funnel, leakage, export, security paths)
- **Evidence** — paste test output showing the suite green

Changes to `risk:high` paths get additional maintainer scrutiny; security-
relevant changes (auth, input validation, resource limits) require a
dedicated review before merge.

## Reporting Bugs

Open an issue with:

1. FIAE version (`fiae --version`) and Python version
2. Minimal reproducible example (synthetic data preferred — no proprietary datasets)
3. Full error message and stack trace
4. Expected vs actual behavior

## Reporting Security Issues

**Do not open a public issue.** See [SECURITY.md](SECURITY.md) for the
responsible disclosure process.

## Style Notes

- Python 3.10+; standard library first
- Files stay under ~400 lines; split rather than sprawl
- Descriptive, domain-specific names — no `data`, `temp`, `handleStuff`
- Docstrings on every public function

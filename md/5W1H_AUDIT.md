# 5W1H Claims Audit — 2026-09-29 (M30)

Framework audit: every subsystem examined against WHO / WHAT / WHERE /
WHEN / WHY / HOW. Companion to `AUDIT.md` (C/W findings) and
`POSITIONING.md` (audience honesty). All evidence freshly re-derived in
this session — nothing taken from prior reports on faith.

Evidence baseline this session:

- Full suite: **1013 passed, 0 skipped, 0 failed** (incl. all 17
  real-data tests on a freshly generated 100K-row fixture).
- CI: green on `115645a` (3 OS × 3 Python + lint + gitleaks).
- Example smoke: 10/10 phases, logistic ROC-AUC 0.763, 11/11 gates.

---

## WHO — People & Power

| Question | Answer with evidence |
|---|---|
| Who is affected? | (1) ML practitioners evaluating feature-engineering tools; (2) contributors; (3) the owner maintaining a solo project with a 1000+-test obligation. |
| Who faces the biggest consequences? | A user in a "correctness matters" domain (finance/health) who trusts L0–L4 rejections without reviewing L5 evidence. The docs now state detection is heuristic; users must keep this in mind. |
| Who holds the power? | The repo owner (sole committer, verified: every commit authored/committed by `faizansk25` after the trailer cleanup). For users: the choice to run `run_in_sandbox` on untrusted code at all. |
| Who sees this differently? | Speed-focused users see gate overhead as friction; correctness-focused users see it as the point. Both views are valid — POSITIONING.md says FIAE is not the fast option. |
| Key stakeholders | Owner, contributors (CONTRIBUTING.md process), downstream users of exported pipelines, GitHub Sponsors/Ko-fi supporters (DONATIONS.md). |
| Who benefits? | Users who need auditable, reproducible feature pipelines; the export path (11 gates) produces reviewable `features.py` rather than a black box. |
| Who else should be consulted? | Before PyPI: 2–3 external users trying the source install on macOS/Linux (owner only tested Windows + CI runners). |
| Who can provide more information? | Design docs md/00–15, `tests/test_claims.py` (claims-as-assertions), this audit + AUDIT.md. |

## WHAT — Issue & Evidence

| Question | Answer with evidence |
|---|---|
| What is the issue? | Remaining gap between documented promises and enforced guarantees. This session found: W-6 (sandbox timeout leaves runaway daemon thread alive — proven: 2 threads alive after timeout) and W-7 (`SandboxPolicy.max_memory_bytes` declared, never enforced — grep: 1 occurrence, definition only). |
| Main arguments | The system's unique claims (leakage taxonomy, verified export, zero-dep core, deterministic IDs) are all machine-verified. Non-unique claims (speed, coverage) are honestly de-emphasized. |
| Evidence quality | Highest tier available: claims re-derived as pytest assertions (`test_claims.py`), layering enforced (`test_architecture.py`), API pinned (`test_api_stability.py`), plus fresh manual proofs (sandbox runtime behavior, fixture signal analysis). |
| Assumptions being made | (a) Users have Python 3.10+; (b) sklearn available for real training (tier-1); (c) English UI; (d) single-machine runs (no distributed). |
| Potential consequences | W-6/W-7 mean "sandbox" is best-effort, not isolation. Documented in sandbox.py docstring; README's security claim is about *validation* (injection rejection), which is proven — not OS-level containment, which is not claimed. |
| Alternatives | Do nothing (docs honest); enforce memory via `resource.setrlimit` (POSIX-only, breaks Windows parity); subprocess isolation (heavy, breaks zero-dep core). Chosen: document + register findings until a cross-platform design exists. |
| Risks of alternatives | setrlimit would make behavior OS-divergent — worse for the 3-OS wheel guarantee. |
| Next steps | W-6/W-7 listed in AUDIT.md; fix in a dedicated security-hardening pass with tests. |

## WHERE — Location & Resources

| Question | Answer with evidence |
|---|---|
| Where did issues first appear? | W-6/W-7 in `src/fiae/security/sandbox.py` (M-era design; surfaced by this session's runtime probe). The stale README doc-table was born when docs were renamed without updating README. |
| Where is the problem most evident? | Documentation drift (fixed this session: all 16 md/ links + example link now resolve — script-verified). |
| Where is supporting data? | `md/REAL_DATA_TEST_REPORT.md`, `benchmarks/bench_core.py` (1K/100K/1M tiers), compiler reports per run (`passed/total_gates` in export). |
| Where have solutions worked? | Claims-as-assertions pattern (F-2/F-3/F-4 fixes) — README drift now fails CI instead of shipping. |
| Where are resources most needed? | CI real-data fixture (12→0 skips achievable); external hardware testing (M4 Pro / AMD claims are design-reasoned, not machine-verified). |
| Where are obstacles? | Windows cannot enforce POSIX rlimits → W-7 fix must be conditional or subprocess-based. |
| Where to implement first? | Sandbox hardening (W-6/W-7), then CI fixture generation. |
| Where to monitor? | CI badge, `test_claims.py` drift-lock, gitleaks scans (full history). |

## WHEN — Timing

| Question | Answer with evidence |
|---|---|
| When did issues emerge? | Sandbox limits: original design (best-effort by intent). README doc-table: doc renames happened before README was rewritten. Fixture signal gap: found only when the fixture was finally regenerated this session (M29/M30). |
| When do effects appear? | Only under adversarial/untrusted input (W-6/W-7) — benign pipelines unaffected (1013/1013 pass). |
| When was data last collected? | Fixture regenerated 2026-09-29; benchmark numbers are hardware-specific to the owner's PC. |
| Best time to act? | Now, pre-PyPI — after publish, every fix becomes a public changelog entry instead of a private commit. |
| When attempted before? | M23 audit fixed F-1..F-4; M25 wired real leakage detectors (C-5); M29 fixed imbalanced CV. Pattern: each pass finds a lower-frequency class of issue. |
| Deadline? | Self-imposed: owner's "PyPI when perfect" bar. Remaining known blockers: W-2 (parity depth), W-3 (except-pass), W-6/W-7 (sandbox), CI fixture. |
| When to expect results? | W-6/W-7 fixes are small, focused changes; CI fixture is one workflow edit. |
| When to review? | After each fix lands: full suite + claims tests must stay green. |

## WHY — Root Causes & Significance

| Question | Answer with evidence |
|---|---|
| Why significant? | This project's entire positioning is "provable correctness". Any unverified claim directly contradicts the brand. |
| Why did issues arise? | Solo-project scale: docs, code, and tests evolve at different speeds; only claims locked by tests survive drift — which is exactly why the claims-test pattern exists. |
| Why preferred solutions? | Claims-as-assertions converts documentation into enforcement; cheaper and more durable than manual review. |
| Why might opinions differ? | Some would say W-6/W-7 are acceptable for a "validate-then-execute" model (code is validated before running); others want hard isolation. Both are defensible; current docs choose honesty about the boundary. |
| Why not addressed sooner? | Nobody had *executed* the sandbox under an infinite loop before this session's probe — tests covered validation paths, not runaway execution. |
| Why are some more affected? | Only users who feed genuinely untrusted code into the sandbox; normal pipeline users never hit these paths. |
| Why immediate action? | The fixes are small and the audit trail is fresh; deferring risks shipping v0.1.0 with known gaps. |
| Why revisit? | Each audit pass (M23→M30) found a new class; the next pass should cover concurrency stress and memory-pressure scenarios. |

## HOW — Implementation & Measurement

| Question | Answer with evidence |
|---|---|
| How did issues start? | Feature-first development: sandbox built for the common case, edge semantics (runaway threads, memory ceilings) deferred. |
| How does it impact groups? | Benign users: zero impact (proven by suite). Adversarial-context users: timeout returns but thread lives (W-6); memory cap not enforced (W-7). |
| How to gather more data? | Runtime probes (as done here), fuzzing (`test_reliability_fuzz.py` exists), benchmark tiers. |
| How have others tackled it? | Standard Python answers: subprocess+rlimit (POSIX), joblib resource tracking, or container-level isolation. All trade off the zero-dep, 3-OS, no-daemon design. |
| How to implement fixes? | W-6: cooperative cancellation flag + thread-liveness test. W-7: enforce via estimate-based rejection (consistent with `PluginPermission.estimated_memory_mb`) + document OS boundary. |
| How to measure success? | New regression tests; suite stays ≥1013; claims tests stay green. |
| How to communicate? | AUDIT.md W-entries (done), PROGRESS_REPORT.md milestone entry, honest wording already in README security claims. |
| How often to reassess? | Every milestone: regenerate fixture, run full suite, re-run claims tests — the M30 protocol. |

---

## Findings registered this session

| ID | Finding | Status |
|----|---------|--------|
| D-1 | README Design Documents table listed 16 nonexistent filenames (short-form names); every md/ link dead | **Fixed** — table now matches real files; script-verified all 16 md refs + example ref resolve |
| W-6 | `run_in_sandbox` timeout returns error but the runaway daemon thread stays alive, consuming CPU until process exit (proven: `threading.active_count()` stays elevated after timeout) | **Open** — document + fix |
| W-7 | `SandboxPolicy.max_memory_bytes` (256 MB) declared but never enforced anywhere (single occurrence: definition) | **Open** — document + fix |

"""M36 regression tests: P0 integration fixes verified end-to-end.

Origin: external audit (ChatGPT, 2026-09-30) verified against code —
every finding here had a failing integration path, not a unit gap.
"""

import csv
import random


from fiae.learn import learn, scan_columns
from fiae.intake import auto_adapter
from fiae.fitted_pipeline import FittedPipeline
from fiae.search.triggers import FeatureProposal
from fiae.problem.splits import reserve_final_holdout
from fiae.orchestration.hpo import successive_halving, HPDimension


def _write_churn_csv(path, n=300, seed=11):
    """Synthetic churn: target genuinely depends on categorical + numeric."""
    rng = random.Random(seed)
    rate = {"basic": 0.15, "pro": 0.45, "elite": 0.75}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plan", "tenure", "churn"])
        for _ in range(n):
            plan = rng.choice(list(rate))
            tenure = rng.randint(1, 60)
            churn = 1 if rng.random() < rate[plan] + tenure * 0.002 else 0
            w.writerow([plan, tenure, churn])
    return path


class TestP0_1_CategoricalPreserved:
    def test_hash_encode_receives_categories_not_nones(self, tmp_path):
        path = _write_churn_csv(tmp_path / "c.csv")
        adapter = auto_adapter(str(path), batch_rows=2048)
        raw = scan_columns(adapter, max_rows=5000)
        # Materialize hash_encode directly on the scanned raw values.
        out = FittedPipeline().fit(
            [FeatureProposal(op="hash_encode", inputs=["raw:plan"],
                             params={"n_buckets": 8})],
            {"plan": raw["plan"]},
        )
        values = out["hash_encode(plan)"]
        # Old bug: all categories coerced to None -> constant 0.0 column.
        distinct = {v for v in values if v is not None}
        assert len(distinct) > 1, "hash buckets collapsed to one value"

    def test_learn_materializes_categorical_feature(self, tmp_path):
        path = _write_churn_csv(tmp_path / "c.csv")
        rep = learn(str(path), "churn")
        # The categorical proposal must survive into the selection stage.
        assert any(
            p.op == "hash_encode" and "plan" in p.inputs[0]
            for p in rep.portfolio_proposals
        ), "categorical proposal vanished from portfolio"


class TestP0_2_LeakageOnDefaultPath:
    def test_exact_target_copy_is_excluded(self, tmp_path):
        rng = random.Random(3)
        path = tmp_path / "leak.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["target_copy", "x", "churn"])
            for _ in range(200):
                y = rng.randint(0, 1)
                w.writerow([y, rng.random(), y])
        rep = learn(str(path), "churn")
        # The leaked column must never produce a portfolio member.
        ops = [m.operator for m in rep.portfolio]
        assert all("target_copy" not in op for op in ops)
        # And its raw-derived candidates must not reach selection: the spy
        # evidence is that portfolio_proposals contains no target_copy input.
        assert all(
            all("target_copy" not in i for i in p.inputs)
            for p in rep.portfolio_proposals
        )

    def test_clean_features_survive(self, tmp_path):
        """The rejection must not nuke honest columns."""
        path = _write_churn_csv(tmp_path / "c.csv", seed=5)
        rep = learn(str(path), "churn")
        # tenure is clean and should still appear among proposals.
        assert any("tenure" in p.inputs[0] for p in rep.portfolio_proposals)


class TestP0_3_FitOnDevOnly:
    def test_fit_receives_dev_rows_not_full_sample(self, tmp_path, monkeypatch):
        path = _write_churn_csv(tmp_path / "c.csv", n=400, seed=2)
        seen = {}
        real_fit = FittedPipeline.fit

        def spy_fit(self, proposals, data):
            seen["rows"] = {k: len(v) for k, v in data.items()}
            return real_fit(self, proposals, data)

        monkeypatch.setattr(FittedPipeline, "fit", spy_fit)
        learn(str(path), "churn")
        assert seen["rows"], "fit was never called"
        n = next(iter(seen["rows"].values()))
        # 400-row dataset, 20% holdout -> fit must see fewer than 400 rows.
        assert n < 400, f"fit saw all {n} rows (holdout leaked into fitting)"


class TestP0_4_OptimizeRunsRealHPO:
    def test_optimize_runs_successive_halving(self, tmp_path):
        from fiae.cli_pipeline import cmd_optimize
        from fiae.orchestration.model_training import _evaluate_cv  # noqa: F401

        path = _write_churn_csv(tmp_path / "c.csv", seed=7)

        class A:
            source = str(path)
            target = "churn"
            json = True

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_optimize(A())
        assert rc == 0
        import json
        out = buf.getvalue()
        payload = json.loads(out[out.find("{"):])
        hpo = payload.get("hpo") or {}
        # Real HPO ran: multiple trials, a finite best score, real params.
        assert hpo.get("n_trials", 0) >= 9
        assert isinstance(hpo.get("best_score"), (int, float))
        assert hpo.get("best_params"), "no hyperparameters searched"
        assert hpo.get("features"), "HPO did not use learned portfolio features"


class TestP0_5_ParityValuesWired:
    def test_pipeline_command_fails_on_tampered_parity(self, tmp_path, monkeypatch):
        """compile_pipeline with expected_features executes exported code."""
        from fiae.codegen.compiler import gate_feature_parity
        from fiae.codegen.pipeline_ir import (
            PipelineIR, IRNode, generate_python_code)

        ir = PipelineIR()
        ir.add_node(IRNode(node_id="n_0", operator="hash_encode",
                           inputs=["raw:age"], params={"n_buckets": 8}))
        ir.outputs.append("n_0")

        code = generate_python_code(ir)
        input_data = {"age": [21, 35, 21, None]}  # numeric col -> coerced to floats
        # Expected values from the fitted operator on the coerced inputs.
        from fiae.features.ops_categorical import tf_hash_encode
        coerced = [21.0, 35.0, 21.0, None]
        good_vals = tf_hash_encode(coerced, n_buckets=8)
        # Correct values (executed exported code) must pass.
        ok = gate_feature_parity(
            ir, expected_features={"n_0": good_vals},
            exported_code=code, input_data=input_data)
        assert ok.passed

        # Wrong values: the gate must fail, not vacuously pass.
        bad = gate_feature_parity(
            ir, expected_features={"n_0": [9, 9, 9, 9]},
            exported_code=code, input_data=input_data)
        assert not bad.passed


class TestSecondaryFixes:
    def test_multiclass_probe_uses_ovr_brier(self):
        from fiae.probe import f3_incremental_probe
        from fiae.search.records import FeatureAcceptanceRecord
        from fiae.contracts import Task

        rng = random.Random(0)
        n = 600
        # 3-class target driven by the candidate feature (class = band of x).
        x = [rng.random() for _ in range(n)]
        y = [0 if v < 0.33 else (1 if v < 0.66 else 2) for v in x]

        rec = FeatureAcceptanceRecord(feature_id="c", operator="c")
        v = f3_incremental_probe(rec, [], x, y, Task.MULTICLASS)
        assert v.passed, "informative feature rejected under multiclass probe"
        assert rec.incremental_gain and rec.incremental_gain > 0.01

    def test_timestamped_holdout_is_chronologically_last(self):
        n = 100
        times = list(range(n))  # row i has time i
        dev, hold = reserve_final_holdout(
            n, times=times, holdout_fraction=0.2, seed=12345)
        # Holdout must be the 20 latest rows, regardless of seed.
        assert hold == list(range(80, 100))
        assert dev == list(range(80))

    def test_resource_aware_successive_halving(self):
        calls = []

        def resource_objective(params, budget):
            calls.append(budget)
            # Return better scores at higher budgets so promotion is
            # genuinely budget-dependent.
            return params["x"] * budget

        space = [HPDimension(name="x", dtype="float", low=0.0, high=1.0)]
        successive_halving(space, resource_objective,
                                    n_initial=9, reduction_factor=3,
                                    maximize=True, resource=resource_objective)
        assert calls, "resource callable never invoked"
        assert max(calls) > min(calls), "budgets did not increase across stages"

    def test_no_resource_flag_keeps_old_behavior(self):
        calls = []

        def plain_objective(params):
            calls.append(1)
            return params["x"]

        space = [HPDimension(name="x", dtype="float", low=0.0, high=1.0)]
        successive_halving(space, plain_objective, n_initial=9,
                           reduction_factor=3, maximize=True)
        assert len(calls) >= 9

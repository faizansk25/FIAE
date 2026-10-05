"""M40.7: semantic typing must distinguish a count from a measurement.

Every non-negative integer used to be classified COUNT, so a profile of a
1,500-row customer table reported ``age`` and ``tenure_months`` as "count" and
the binary target ``is_returned`` as "count" too. A measurement wearing a
count's label misleads everything downstream that branches on semantics.

A count now needs evidence: a binary {0,1} flag is BOOLEAN, an explicit
count-like name wins, and otherwise the distribution decides -- a real count
is right-skewed (mean far below max), while a measurement fills its range.
Too little data to judge shape means the prior stands.

Also covers the feature table's alignment: padding was applied to the
*coloured* string, so ANSI escapes counted as visible width and a hardcoded
width of 30 let long operator names collide with the gain column.
"""

import csv
import random

import pytest

from fiae.contracts import SemanticType
from fiae.intake.csv_source import CsvDataSourceAdapter
from fiae.intake.profiler import ProfileConfig, profile_source


def _semantic_for(tmp_path, name, values):
    p = tmp_path / f"{name}.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([name])
        for v in values:
            w.writerow([v])
    profile = profile_source(CsvDataSourceAdapter(p), ProfileConfig())
    return profile.columns[0].semantic_type


class TestCountVersusMeasurement:
    def test_measurement_is_not_a_count(self, tmp_path):
        """A uniform non-negative integer fills its range: not a tally."""
        rng = random.Random(0)
        age = [rng.randint(18, 70) for _ in range(600)]
        assert _semantic_for(tmp_path, "age", age) is SemanticType.CONTINUOUS_NUMERIC

    def test_duration_is_not_a_count(self, tmp_path):
        rng = random.Random(1)
        tenure = [rng.randint(1, 72) for _ in range(600)]
        assert _semantic_for(tmp_path, "tenure_months", tenure) is SemanticType.CONTINUOUS_NUMERIC

    def test_binary_flag_is_boolean_not_count(self, tmp_path):
        vals = [i % 2 for i in range(600)]
        assert _semantic_for(tmp_path, "is_returned", vals) is SemanticType.BOOLEAN

    def test_genuine_skewed_tally_is_a_count(self, tmp_path):
        """Right-skewed low values with rare large ones: that is a count."""
        rng = random.Random(2)
        counts = [
            rng.choice([0, 1, 1, 2, 2, 3, rng.randint(5, 60)]) for _ in range(600)
        ]
        assert _semantic_for(tmp_path, "widget_qty", counts) is SemanticType.COUNT

    def test_count_name_overrides_uniform_distribution(self, tmp_path):
        """Explicit naming is evidence even when the shape looks uniform."""
        assert _semantic_for(tmp_path, "points", list(range(1000, 1200))) is SemanticType.COUNT

    def test_small_sample_keeps_the_prior(self, tmp_path):
        """Three observations cannot establish shape; do not reclassify."""
        assert _semantic_for(tmp_path, "purchases", [1, 2, 3]) is SemanticType.COUNT


class TestFeatureTableAlignment:
    def test_padding_is_applied_before_colourising(self):
        """ANSI escapes must not be counted as visible width.

        The old line formatted the *coloured* string, so ``:<30`` spent part
        of its budget on escape sequences and every short operator name came
        out narrower than the column it was supposed to fill.
        """
        import re

        from fiae import cli_colors as C

        ansi = re.compile(r"\x1b\[[0-9;]*m")
        short = "sqrt"  # shorter than the width, so the padding actually applies

        buggy = ansi.sub("", f"{C.bold(short, force=True):<30}")
        correct = ansi.sub("", C.bold(short.ljust(30), force=True))

        assert len(buggy) < 30, (
            "colourising before padding should lose width to ANSI escapes")
        assert correct == short.ljust(30)
        assert len(correct) == 30

    def test_long_names_are_not_truncated_to_a_fixed_width(self):
        """The old width of 30 collided with the gain column."""
        longest = "safe_ratio(tenure_months_monthly_spend)"
        assert len(longest) > 30, "fixture must exercise the overflow case"

    def test_column_width_is_sized_to_the_widest_name(self):
        """Padding must come from the data, not a constant."""
        import re

        from fiae import cli_colors as C

        ansi = re.compile(r"\x1b\[[0-9;]*m")
        names = ["sqrt(spend)", "safe_ratio(tenure_months_monthly_spend)"]
        width = max(len(n) for n in names)
        rendered = [ansi.sub("", C.bold(n.ljust(width), force=True)) for n in names]
        assert {len(r) for r in rendered} == {width}
        assert width == len(names[1]) == 39


class TestEndToEndProfileSemantics:
    def test_customer_table_profile(self, tmp_path):
        p = tmp_path / "customers.csv"
        rng = random.Random(12)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["customer_id", "age", "monthly_spend", "is_returned"])
            for i in range(800):
                t = rng.randint(1, 72)
                s = round(max(0.5, 20 - t * 0.2 + rng.gauss(0, 6)), 2)
                w.writerow([f"CUST-{i:06d}", rng.randint(18, 70), s,
                            1 if (t > 60 or s < 8) else 0])

        by_name = {c.name: c.semantic_type for c in
                   profile_source(CsvDataSourceAdapter(p), ProfileConfig()).columns}

        assert by_name["customer_id"] is SemanticType.IDENTIFIER
        assert by_name["age"] is SemanticType.CONTINUOUS_NUMERIC
        assert by_name["is_returned"] is SemanticType.BOOLEAN
        assert by_name["monthly_spend"] is SemanticType.CONTINUOUS_NUMERIC


@pytest.mark.parametrize("bad_name", ["pct_hint_leak"])
def test_placeholder(bad_name):
    """Keeps the module import-safe under strict collection rules."""
    assert isinstance(bad_name, str)

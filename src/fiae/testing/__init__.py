"""Testing subsystem — doc 12 (Testing, Benchmarks, Research Gates)."""
from .property_tests import (
    PropertyResult,
    run_all_properties,
    summarize_results,
)
from .benchmarks import BenchmarkResult, BenchmarkSuite, run_full_benchmark
from .baselines import run_baselines, score_dataset_difficulty, assess_feature_value

# QUALITY_BLOB_RECONSTRUCTION_V3
from __future__ import annotations

import json

import pytest

from scripts import segmentation_benchmark as benchmark


def test_character_operational_benchmark_is_lossless_deterministic_and_bounded():
    report = benchmark.run_benchmark(
        characters=10_000,
        repeat=2,
        mode="characters",
        min_throughput=1.0,
        max_peak_mib=256.0,
    )
    assert report["passed"], report["failures"]
    assert report["measurements"]["deterministic"]
    assert report["configuration"]["max_chars"] == 1000
    assert report["configuration"]["overlap_chars"] == 100
    assert report["measurements"]["chunk_count"] > 1
    assert len(report["measurements"]["chunk_signature_sha256"]) == 64


def test_token_operational_benchmark_exercises_explicit_counter_identity():
    report = benchmark.run_benchmark(
        characters=8_000,
        repeat=2,
        mode="tokens",
        min_throughput=1.0,
        max_peak_mib=256.0,
    )
    assert report["passed"], report["failures"]
    assert report["configuration"]["max_tokens"] == 100
    assert report["configuration"]["overlap_tokens"] == 10
    assert report["configuration"]["token_counter_id"] == "lexnlp-count-tokens-v1"
    assert report["measurements"]["deterministic"]


def test_impossible_throughput_threshold_fails_the_gate():
    report = benchmark.run_benchmark(
        characters=1000,
        repeat=1,
        min_throughput=float("inf"),
        max_peak_mib=256.0,
    )
    assert not report["passed"]
    assert any(
        failure["check"] == "throughput_characters_per_second"
        for failure in report["failures"]
    )


def test_pure_scaling_classifier_accepts_linear_samples():
    result = benchmark.evaluate_scaling_samples(
        [250, 500, 1000, 2000],
        [1.0, 2.0, 4.0, 8.0],
    )
    assert result["passed"]
    assert result["log_log_growth_exponent"] == pytest.approx(1.0)


def test_pure_scaling_classifier_rejects_historical_quadratic_growth():
    """Freeze the negative contract independently of timing noise."""

    result = benchmark.evaluate_scaling_samples(
        [250, 500, 1000, 2000],
        [1.0, 4.0, 16.0, 64.0],
    )
    assert not result["passed"]
    assert result["log_log_growth_exponent"] == pytest.approx(2.0)
    assert {
        failure["check"] for failure in result["failures"]
    } == {"adjacent_growth_ratio", "log_log_growth_exponent"}


def test_statute_scaling_operational_lane_times_chunk_planning_not_only_detection():
    report = benchmark.run_statute_scaling_benchmark(
        sizes=(20, 40, 80),
        repeat=1,
        max_doubling_ratio=100.0,
        max_growth_exponent=10.0,
    )
    assert report["passed"], report["failures"]
    assert report["profile"] == "statute"
    assert "chunk-planning" in report["scope"]
    assert report["configuration"]["chunk_max_chars"] == 256
    for sample in report["samples"]:
        assert sample["section_count"] == sample["heading_count"]
        assert sample["chunk_count"] > 0
        assert len(sample["hierarchy_identity_sha256"]) == 64
        assert len(sample["chunk_identity_sha256"]) == 64


def test_statute_scaling_cli_option_emits_json(tmp_path):
    output = tmp_path / "nested" / "scaling.json"
    result = benchmark.main(
        [
            "--statute-scaling",
            "--scaling-sizes",
            "20,40,80",
            "--scaling-repeat",
            "1",
            "--max-scaling-doubling-ratio",
            "100",
            "--max-scaling-exponent",
            "10",
            "--json-output",
            str(output),
        ]
    )
    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["profile"] == "statute"
    assert payload["configuration"]["sizes"] == [20, 40, 80]


@pytest.mark.parametrize("output_flag", ["--output", "--json-output"])
def test_output_aliases_share_the_same_destination(tmp_path, output_flag):
    destination = tmp_path / "nested" / "report.json"
    args = benchmark.build_parser().parse_args([output_flag, str(destination)])
    assert args.output == destination


@pytest.mark.parametrize("value", ["", "1,2", "10,0,20", "a,b,c"])
def test_scaling_sizes_parser_rejects_malformed_inputs(value):
    with pytest.raises(Exception):
        benchmark._sizes(value)

"""Coverage tests for scripts/segmentation_benchmark.py."""

from __future__ import annotations

import json
import sys

import pytest

from scripts import segmentation_benchmark as benchmark


class TestImportSideEffect:
    def test_repository_root_inserted_when_missing(self, monkeypatch):
        import importlib

        root = str(benchmark.REPOSITORY_ROOT)
        monkeypatch.setattr(sys, "path", [p for p in sys.path if p != root])
        assert root not in sys.path
        reloaded = importlib.reload(benchmark)
        assert reloaded.REPOSITORY_ROOT == benchmark.REPOSITORY_ROOT
        assert sys.path[0] == root


class TestBuildBenchmarkDocument:
    @pytest.mark.parametrize("bad", [True, False, 0, -5, "100", 3.5, None])
    def test_rejects_non_positive_int(self, bad):
        with pytest.raises(ValueError, match="positive integer"):
            benchmark.build_benchmark_document(bad)

    def test_small_document_has_requested_length(self):
        doc = benchmark.build_benchmark_document(50)
        assert len(doc) == 50
        assert doc.startswith("ARTICLE 1")


class TestRunBenchmarkValidation:
    @pytest.mark.parametrize("bad", [True, 0, -1, "3"])
    def test_rejects_bad_repeat(self, bad):
        with pytest.raises(ValueError, match="positive integer"):
            benchmark.run_benchmark(characters=500, repeat=bad)

    def test_rejects_non_positive_thresholds(self):
        with pytest.raises(ValueError, match="must be positive"):
            benchmark.run_benchmark(characters=500, repeat=1, min_throughput=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            benchmark.run_benchmark(characters=500, repeat=1, max_peak_mib=-1.0)

    def test_rejects_bad_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            benchmark.run_benchmark(characters=500, repeat=1, mode="words")


class TestRunBenchmarkFailures:
    def test_tiny_peak_budget_fails(self):
        report = benchmark.run_benchmark(
            characters=500,
            repeat=1,
            min_throughput=1e-9,
            max_peak_mib=1e-6,
        )
        assert not report["passed"]
        failures = {failure["check"]: failure for failure in report["failures"]}
        assert "peak_memory_mib" in failures
        assert failures["peak_memory_mib"]["observed"] > 1e-6

    def test_nondeterministic_signatures_fail(self, monkeypatch):
        sequence = iter(f"{index:064x}" for index in range(4))
        monkeypatch.setattr(benchmark, "_chunk_signature", lambda chunks: next(sequence))
        report = benchmark.run_benchmark(
            characters=500,
            repeat=2,
            min_throughput=1e-9,
            max_peak_mib=1e9,
        )
        assert not report["passed"]
        assert any(failure["check"] == "deterministic_chunk_identity" for failure in report["failures"])


class TestStatuteScalingDocument:
    @pytest.mark.parametrize("bad", [True, False, 0, -2, "10", None])
    def test_rejects_non_positive_int(self, bad):
        with pytest.raises(ValueError, match="positive integer"):
            benchmark.build_statute_scaling_document(bad)

    def test_two_headings(self):
        doc = benchmark.build_statute_scaling_document(2)
        assert "1 Duty number 1" in doc
        assert "2 Duty number 2" in doc


class TestScalingGrowthExponent:
    def test_requires_at_least_two_paired_points(self):
        with pytest.raises(ValueError, match="at least two paired points"):
            benchmark.scaling_growth_exponent([250], [1.0])
        with pytest.raises(ValueError, match="at least two paired points"):
            benchmark.scaling_growth_exponent([250, 500], [1.0])

    @pytest.mark.parametrize(
        "sizes,seconds",
        [
            ([0, 500], [1.0, 2.0]),
            ([250, -5], [1.0, 2.0]),
            ([250, 500], [0.0, 2.0]),
            ([250, 500], [1.0, -2.0]),
        ],
    )
    def test_requires_positive_sizes_and_timings(self, sizes, seconds):
        with pytest.raises(ValueError, match="must be positive"):
            benchmark.scaling_growth_exponent(sizes, seconds)

    def test_rejects_equal_sizes(self):
        with pytest.raises(ValueError, match="must not all be equal"):
            benchmark.scaling_growth_exponent([500, 500], [1.0, 2.0])

    def test_linear_growth_exponent_is_one(self):
        assert benchmark.scaling_growth_exponent([250, 500, 1000], [1.0, 2.0, 4.0]) == pytest.approx(1.0)


class TestEvaluateScalingSamples:
    def test_rejects_unpaired_points(self):
        with pytest.raises(ValueError, match="paired points"):
            benchmark.evaluate_scaling_samples([250], [1.0])

    @pytest.mark.parametrize(
        "sizes",
        [[500, 250, 1000], [250, 250, 500]],
    )
    def test_rejects_unsorted_or_duplicate_sizes(self, sizes):
        with pytest.raises(ValueError, match="unique and increasing"):
            benchmark.evaluate_scaling_samples(sizes, [1.0, 2.0, 4.0])

    def test_rejects_invalid_thresholds(self):
        with pytest.raises(ValueError, match="thresholds are invalid"):
            benchmark.evaluate_scaling_samples([250, 500], [1.0, 2.0], max_doubling_ratio=1.0)
        with pytest.raises(ValueError, match="thresholds are invalid"):
            benchmark.evaluate_scaling_samples([250, 500], [1.0, 2.0], max_growth_exponent=0.0)


class TestStatuteScalingValidation:
    def test_requires_three_sizes(self):
        with pytest.raises(ValueError, match="at least three sizes"):
            benchmark.run_statute_scaling_benchmark(sizes=(10, 20), repeat=1)

    @pytest.mark.parametrize("sizes", [(40, 20, 80), (20, 20, 80)])
    def test_rejects_unsorted_or_duplicate_sizes(self, sizes):
        with pytest.raises(ValueError, match="unique and increasing"):
            benchmark.run_statute_scaling_benchmark(sizes=sizes, repeat=1)

    @pytest.mark.parametrize("sizes", [(0, 10, 20), (True, 10, 20), (10.5, 20, 30)])
    def test_rejects_non_positive_int_sizes(self, sizes):
        with pytest.raises(ValueError, match="positive integers"):
            benchmark.run_statute_scaling_benchmark(sizes=sizes, repeat=1)

    def test_rejects_bad_repeat(self):
        with pytest.raises(ValueError, match="positive integer"):
            benchmark.run_statute_scaling_benchmark(sizes=(20, 40, 80), repeat=0)

    def test_rejects_bad_throughput(self):
        with pytest.raises(ValueError, match="must be positive"):
            benchmark.run_statute_scaling_benchmark(sizes=(20, 40, 80), repeat=1, min_scaling_throughput=0.0)


class TestStatuteScalingIntegrity:
    def test_lossy_prebuilt_hierarchy_raises(self, monkeypatch):
        class FakeHierarchy:
            def reconstruct(self):
                return "different text"

            def segments(self, kind):
                return []

        monkeypatch.setattr(benchmark, "segment_document", lambda *args, **kwargs: FakeHierarchy())
        with pytest.raises(AssertionError, match="not lossless"):
            benchmark.run_statute_scaling_benchmark(sizes=(5, 10, 20), repeat=1)

    def test_heading_count_mismatch_raises(self, monkeypatch):
        captured: dict = {}

        def fake_segment(text, **kwargs):
            captured["text"] = text

            class FakeHierarchy:
                def reconstruct(self):
                    return captured["text"]

                def segments(self, kind):
                    return []

            return FakeHierarchy()

        monkeypatch.setattr(benchmark, "segment_document", fake_segment)
        with pytest.raises(AssertionError, match="detected 0 of 5 headings"):
            benchmark.run_statute_scaling_benchmark(sizes=(5, 10, 20), repeat=1)

    def test_lossy_timed_repeat_raises(self, monkeypatch):
        monkeypatch.setattr(benchmark, "chunk_document", lambda *args, **kwargs: [])
        with pytest.raises(AssertionError, match="not lossless"):
            benchmark.run_statute_scaling_benchmark(sizes=(20, 40, 80), repeat=1)


class TestMainBenchmarkLane:
    def test_stdout_json_report(self, capsys):
        rc = benchmark.main(
            [
                "--characters",
                "500",
                "--repeat",
                "1",
                "--mode",
                "characters",
                "--min-throughput",
                "1",
                "--max-peak-mib",
                "1000000",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["mode"] == "characters"
        assert payload["passed"]
        assert payload["measurements"]["chunk_count"] > 0

    def test_failed_gate_returns_one(self, capsys):
        rc = benchmark.main(
            [
                "--characters",
                "500",
                "--repeat",
                "1",
                "--min-throughput",
                "inf",
                "--max-peak-mib",
                "1000000",
            ]
        )
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert not payload["passed"]

    def test_file_output_report(self, tmp_path):
        output = tmp_path / "bench.json"
        rc = benchmark.main(
            [
                "--characters",
                "500",
                "--repeat",
                "1",
                "--min-throughput",
                "1",
                "--max-peak-mib",
                "1000000",
                "--output",
                str(output),
            ]
        )
        assert rc == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["mode"] == "characters"

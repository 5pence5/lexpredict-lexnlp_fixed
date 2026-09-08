#!/usr/bin/env python3
# QUALITY_BLOB_RECONSTRUCTION_V3
"""Operational performance gates for structure-first segmentation and chunking.

Character and lexical-token modes time end-to-end hierarchy plus chunk creation.
The separate --statute-scaling lane builds each STATUTE hierarchy before the
clock, then times chunk planning only. Every timed repeat must reconstruct the
source and produce identical chunk counts and authenticated signatures. Scaling
passes only when growth ratios, log-log exponent, determinism, and the
largest-sample character-throughput floor all pass. No model, corpus, or network
is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import tracemalloc
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lexnlp.nlp.en.segments.chunks import (
    TokenCounterPolicy,
    chunk_document,
    count_tokens,
)
from lexnlp.nlp.en.segments.hierarchy import SegmentKind, StructureProfile, segment_document
from lexnlp.nlp.en.tests.segmentation_quality import deterministic_sentence_spans

SENTENCE_BACKEND_ID = "lexnlp-hermetic-regression-sentences-v1"
REPORT_SCHEMA_VERSION = 1


def build_benchmark_document(characters: int) -> str:
    if isinstance(characters, bool) or not isinstance(characters, int) or characters <= 0:
        raise ValueError("characters must be a positive integer")
    template = (
        "ARTICLE {article}\n"
        "SECTION {article}.{section} OPERATIVE TERMS\n"
        "{article}.{section}(a) The Supplier shall preserve exact Unicode Café £ Ω 中.\n"
        "(i) notices must be delivered promptly; and\n"
        "(ii) records must remain available for inspection.\n\n"
    )
    parts = []
    article = section = 1
    length = 0
    while length < characters:
        part = template.format(article=article, section=section)
        parts.append(part)
        length += len(part)
        section += 1
        if section == 10:
            article += 1
            section = 1
    return "".join(parts)[:characters]


def _chunk_signature(chunks: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(
            f"{chunk.index}:{chunk.start}:{chunk.new_content_start}:{chunk.end}:"
            f"{chunk.chunk_id}:{chunk.text_sha256}\n".encode()
        )
    return digest.hexdigest()


def run_benchmark(
    *,
    characters: int = 200_000,
    repeat: int = 3,
    mode: str = "characters",
    min_throughput: float = 10_000.0,
    max_peak_mib: float = 64.0,
) -> dict[str, Any]:
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
        raise ValueError("repeat must be a positive integer")
    if min_throughput <= 0 or max_peak_mib <= 0:
        raise ValueError("performance thresholds must be positive")
    if mode not in {"characters", "tokens"}:
        raise ValueError("mode must be 'characters' or 'tokens'")

    text = build_benchmark_document(characters)
    if mode == "characters":
        budget = {"max_chars": 1000, "overlap_chars": 100}
    else:
        budget = {
            "max_tokens": 100,
            "overlap_tokens": 10,
            "token_counter": count_tokens,
            "token_counter_id": "lexnlp-count-tokens-v1",
            "token_counter_policy": TokenCounterPolicy.MONOTONIC.value,
        }

    timings = []
    peaks = []
    signatures = []
    chunk_counts = []
    for _ in range(repeat):
        tracemalloc.start()
        started = time.perf_counter()
        chunks = chunk_document(
            text,
            **budget,
            sentence_segmenter=deterministic_sentence_spans,
            sentence_backend_id=SENTENCE_BACKEND_ID,
            document_id=f"benchmark-{mode}",
        )
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timings.append(elapsed)
        peaks.append(peak / (1024 * 1024))
        signatures.append(_chunk_signature(chunks))
        chunk_counts.append(len(chunks))
        assert "".join(text[chunk.new_content_start : chunk.end] for chunk in chunks) == text
        if mode == "characters":
            assert all(len(chunk.text) <= budget["max_chars"] for chunk in chunks)
        else:
            assert all(count_tokens(chunk.text) <= budget["max_tokens"] for chunk in chunks)

    median_seconds = statistics.median(timings)
    throughput = len(text) / median_seconds if median_seconds else math.inf
    deterministic = len(set(signatures)) == 1 and len(set(chunk_counts)) == 1
    failures = []
    if throughput < min_throughput:
        failures.append(
            {"check": "throughput_characters_per_second", "observed": throughput, "minimum": min_throughput}
        )
    if max(peaks) > max_peak_mib:
        failures.append({"check": "peak_memory_mib", "observed": max(peaks), "maximum": max_peak_mib})
    if not deterministic:
        failures.append({"check": "deterministic_chunk_identity", "observed": False})
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": "hermetic-operational-regression",
        "mode": mode,
        "passed": not failures,
        "failures": failures,
        "configuration": {
            "characters": characters,
            "repeat": repeat,
            "min_throughput": min_throughput,
            "max_peak_mib": max_peak_mib,
            **{key: value for key, value in budget.items() if key != "token_counter"},
            "sentence_backend_id": SENTENCE_BACKEND_ID,
        },
        "measurements": {
            "seconds": timings,
            "median_seconds": median_seconds,
            "characters_per_second": throughput,
            "peak_memory_mib": peaks,
            "max_peak_memory_mib": max(peaks),
            "chunk_count": chunk_counts[0],
            "chunk_signature_sha256": signatures[0],
            "deterministic": deterministic,
        },
    }


def build_statute_scaling_document(heading_count: int) -> str:
    """Build adjacent no-blank statutory headings that exercise sequence promotion."""

    if isinstance(heading_count, bool) or not isinstance(heading_count, int) or heading_count <= 0:
        raise ValueError("heading_count must be a positive integer")
    return "".join(
        f"{index} Duty number {index}\n"
        f"The authority shall perform duty {index} subject to subsection ({index % 7 + 1}).\n"
        for index in range(1, heading_count + 1)
    ).rstrip("\n")


def scaling_growth_exponent(sizes: Sequence[int], seconds: Sequence[float]) -> float:
    if len(sizes) != len(seconds) or len(sizes) < 2:
        raise ValueError("scaling samples require at least two paired points")
    if any(size <= 0 for size in sizes) or any(value <= 0 for value in seconds):
        raise ValueError("scaling sizes and timings must be positive")
    x = [math.log(float(size)) for size in sizes]
    y = [math.log(float(value)) for value in seconds]
    x_mean, y_mean = statistics.fmean(x), statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0:
        raise ValueError("scaling sizes must not all be equal")
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True)) / denominator


def evaluate_scaling_samples(
    sizes: Sequence[int],
    seconds: Sequence[float],
    *,
    max_doubling_ratio: float = 3.2,
    max_growth_exponent: float = 1.7,
) -> dict[str, Any]:
    """Pure deterministic classifier used to freeze the anti-quadratic contract."""

    if len(sizes) != len(seconds) or len(sizes) < 2:
        raise ValueError("scaling samples require paired points")
    if list(sizes) != sorted(sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("scaling sizes must be unique and increasing")
    if max_doubling_ratio <= 1 or max_growth_exponent <= 0:
        raise ValueError("scaling thresholds are invalid")
    ratios = [seconds[index] / seconds[index - 1] for index in range(1, len(seconds))]
    exponent = scaling_growth_exponent(sizes, seconds)
    failures = []
    for index, ratio in enumerate(ratios, 1):
        size_factor = sizes[index] / sizes[index - 1]
        allowed = max_doubling_ratio ** (math.log(size_factor, 2))
        if ratio > allowed:
            failures.append(
                {
                    "check": "adjacent_growth_ratio",
                    "from_headings": sizes[index - 1],
                    "to_headings": sizes[index],
                    "observed": ratio,
                    "maximum": allowed,
                }
            )
    if exponent > max_growth_exponent:
        failures.append({"check": "log_log_growth_exponent", "observed": exponent, "maximum": max_growth_exponent})
    return {
        "passed": not failures,
        "failures": failures,
        "adjacent_time_ratios": ratios,
        "log_log_growth_exponent": exponent,
        "max_doubling_ratio": max_doubling_ratio,
        "max_growth_exponent": max_growth_exponent,
    }


def run_statute_scaling_benchmark(
    *,
    sizes: Sequence[int] = (250, 500, 1000, 2000),
    repeat: int = 3,
    max_doubling_ratio: float = 3.2,
    max_growth_exponent: float = 1.7,
    min_scaling_throughput: float = 10_000.0,
) -> dict[str, Any]:
    """Time repeatable chunk planning over prebuilt STATUTE hierarchies."""

    sizes = tuple(sizes)
    if len(sizes) < 3:
        raise ValueError("statute scaling requires at least three sizes")
    if list(sizes) != sorted(sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("scaling sizes must be unique and increasing")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sizes):
        raise ValueError("scaling sizes must be positive integers")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
        raise ValueError("repeat must be a positive integer")
    if min_scaling_throughput <= 0:
        raise ValueError("min_scaling_throughput must be positive")

    samples = []
    for heading_count in sizes:
        text = build_statute_scaling_document(heading_count)
        hierarchy = segment_document(
            text,
            sentence_segmenter=deterministic_sentence_spans,
            sentence_backend_id=SENTENCE_BACKEND_ID,
            structure_profile=StructureProfile.STATUTE,
        )
        if hierarchy.reconstruct() != text:
            raise AssertionError("prebuilt STATUTE hierarchy is not lossless")
        sections = list(hierarchy.segments(SegmentKind.SECTION))
        if len(sections) != heading_count:
            raise AssertionError(f"STATUTE profile detected {len(sections)} of {heading_count} headings")

        # Hierarchy construction is deliberately outside the clock. This lane
        # isolates the heading-index and protected-container chunk planner.
        observed = []
        signatures = []
        chunk_counts = []
        for _ in range(repeat):
            started = time.perf_counter()
            chunks = chunk_document(
                hierarchy,
                max_chars=256,
                overlap_chars=0,
            )
            elapsed = time.perf_counter() - started
            rebuilt = "".join(text[chunk.new_content_start : chunk.end] for chunk in chunks)
            if rebuilt != text:
                raise AssertionError("a timed STATUTE chunk repeat is not lossless")
            observed.append(elapsed)
            signatures.append(_chunk_signature(chunks))
            chunk_counts.append(len(chunks))

        median_seconds = statistics.median(observed)
        characters_per_second = len(text) / median_seconds if median_seconds else math.inf
        deterministic = len(set(signatures)) == 1 and len(set(chunk_counts)) == 1
        hierarchy_identity = hashlib.sha256(
            "\n".join(
                f"{segment.start}:{segment.end}:{segment.label}:{segment.segment_id}" for segment in sections
            ).encode("utf-8")
        ).hexdigest()
        samples.append(
            {
                "heading_count": heading_count,
                "characters": len(text),
                "seconds": observed,
                "median_seconds": median_seconds,
                "characters_per_second": characters_per_second,
                "section_count": len(sections),
                "chunk_counts": chunk_counts,
                "chunk_signatures_sha256": signatures,
                "chunk_count": chunk_counts[0],
                "chunk_identity_sha256": signatures[0],
                "hierarchy_identity_sha256": hierarchy_identity,
                "deterministic": deterministic,
            }
        )

    medians = [sample["median_seconds"] for sample in samples]
    classification = evaluate_scaling_samples(
        sizes,
        medians,
        max_doubling_ratio=max_doubling_ratio,
        max_growth_exponent=max_growth_exponent,
    )
    failures = list(classification["failures"])
    for sample in samples:
        if not sample["deterministic"]:
            failures.append(
                {
                    "check": "deterministic_statute_chunk_identity",
                    "heading_count": sample["heading_count"],
                    "chunk_counts": sample["chunk_counts"],
                    "chunk_signatures_sha256": sample["chunk_signatures_sha256"],
                }
            )

    largest = samples[-1]
    if largest["characters_per_second"] < min_scaling_throughput:
        failures.append(
            {
                "check": "largest_sample_throughput_characters_per_second",
                "heading_count": largest["heading_count"],
                "observed": largest["characters_per_second"],
                "minimum": min_scaling_throughput,
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": "hermetic-statute-profile-prebuilt-hierarchy-chunk-planning",
        "profile": StructureProfile.STATUTE.value,
        "passed": not failures,
        "failures": failures,
        "configuration": {
            "sizes": list(sizes),
            "repeat": repeat,
            "max_doubling_ratio": max_doubling_ratio,
            "max_growth_exponent": max_growth_exponent,
            "min_scaling_throughput": min_scaling_throughput,
            "sentence_backend_id": SENTENCE_BACKEND_ID,
            "chunk_max_chars": 256,
        },
        "samples": samples,
        "analysis": {
            **{key: value for key, value in classification.items() if key not in {"passed", "failures"}},
            "all_repeat_outputs_deterministic": all(sample["deterministic"] for sample in samples),
            "largest_sample_characters_per_second": largest["characters_per_second"],
        },
    }


def _sizes(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from error
    if len(values) < 3 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("provide at least three positive sizes")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--json-output", dest="output", type=Path)
    parser.add_argument("--characters", type=int, default=200_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mode", choices=("characters", "tokens"), default="characters")
    parser.add_argument("--min-throughput", type=float, default=10_000.0)
    parser.add_argument("--max-peak-mib", type=float, default=64.0)
    parser.add_argument(
        "--statute-scaling",
        action="store_true",
        help="run the no-blank STATUTE-profile anti-quadratic scaling lane",
    )
    parser.add_argument("--scaling-sizes", type=_sizes, default=(250, 500, 1000, 2000))
    parser.add_argument("--scaling-repeat", type=int, default=3)
    parser.add_argument("--max-scaling-doubling-ratio", type=float, default=3.2)
    parser.add_argument("--max-scaling-exponent", type=float, default=1.7)
    parser.add_argument("--min-scaling-throughput", type=float, default=10_000.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.statute_scaling:
        report = run_statute_scaling_benchmark(
            sizes=args.scaling_sizes,
            repeat=args.scaling_repeat,
            max_doubling_ratio=args.max_scaling_doubling_ratio,
            max_growth_exponent=args.max_scaling_exponent,
            min_scaling_throughput=args.min_scaling_throughput,
        )
    else:
        report = run_benchmark(
            characters=args.characters,
            repeat=args.repeat,
            mode=args.mode,
            min_throughput=args.min_throughput,
            max_peak_mib=args.max_peak_mib,
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

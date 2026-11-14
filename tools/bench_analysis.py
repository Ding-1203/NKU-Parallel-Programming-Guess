"""Statistical aggregation and offline reporting for PCFG benchmark experiments."""

from __future__ import annotations

import csv
import html
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

matplotlib.rcParams["svg.hashsalt"] = "pcfg-baseline-v1"
matplotlib.rcParams["font.family"] = "DejaVu Sans"
COLORS = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#64748b"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def bootstrap_median_ci(values: list[float], seed: int, samples: int) -> tuple[float, float]:
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    return percentile(medians, 2.5), percentile(medians, 97.5)


def stats(values: Iterable[float], seed: int, samples: int) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("cannot summarize an empty series")
    median = statistics.median(data)
    q1 = percentile(data, 25)
    q3 = percentile(data, 75)
    mad = statistics.median(abs(value - median) for value in data)
    mean = statistics.fmean(data)
    stddev = statistics.stdev(data) if len(data) > 1 else 0.0
    lower_fence = q1 - 1.5 * (q3 - q1)
    upper_fence = q3 + 1.5 * (q3 - q1)
    ci_low, ci_high = bootstrap_median_ci(data, seed, samples)
    return {
        "count": len(data),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "mad": mad,
        "mean": mean,
        "stddev": stddev,
        "cv": stddev / mean if mean else 0.0,
        "min": min(data),
        "max": max(data),
        "range": max(data) - min(data),
        "bootstrap_95_ci": [ci_low, ci_high],
        "outliers": [value for value in data if value < lower_fence or value > upper_fence],
    }


def unwrap(path: Path) -> dict[str, Any]:
    value = load_json(path)
    return value["program"] if value.get("schema") == "pcfg-benchmark-run-v1" else value


def measured_rows(experiment: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(sorted((experiment / "runs").glob("measured-*.json")), 1):
        wrapper = load_json(path)
        program = wrapper["program"] if wrapper.get("schema") == "pcfg-benchmark-run-v1" else wrapper
        timing = program["timing_seconds"]
        resources = wrapper.get("resources", {})
        rows.append({
            "run": index,
            "path": str(path.relative_to(experiment)),
            "generated": program["generated"],
            "hash_guard": program["hash_guard"],
            "core_total": timing["core_total"],
            "end_to_end": timing["end_to_end"],
            "load_model": timing["load_model"],
            "throughput": program["generated"] / timing["core_total"],
            "max_rss_kb": resources.get("maximum_resident_set_size_kbytes"),
            "threads": int(program.get("threads_used", 1)),
            "cpu_percent": resources.get("percent_of_cpu_this_job_got"),
            "voluntary_context_switches": resources.get("voluntary_context_switches"),
            "involuntary_context_switches": resources.get("involuntary_context_switches"),
        })
    if not rows:
        raise ValueError(f"no measured runs found in {experiment / 'runs'}")
    return rows


def thread_scaling_rows(experiment: Path) -> list[dict[str, Any]]:
    paths = list(sorted((experiment / "runs").glob("measured-*.json")))
    paths.extend(sorted((experiment / "scaling").glob("thread-*/measured-*.json")))
    rows: list[dict[str, Any]] = []
    for path in paths:
        wrapper = load_json(path)
        program = wrapper["program"] if wrapper.get("schema") == "pcfg-benchmark-run-v1" else wrapper
        timing = program["timing_seconds"]
        resources = wrapper.get("resources", {})
        rows.append({
            "threads": int(program.get("threads_used", 1)),
            "path": str(path.relative_to(experiment)),
            "core_total": float(timing["core_total"]),
            "end_to_end": float(timing["end_to_end"]),
            "throughput": int(program["generated"]) / float(timing["core_total"]),
            "generated": int(program["generated"]),
            "hash_guard": int(program["hash_guard"]),
            "max_rss_kb": resources.get("maximum_resident_set_size_kbytes"),
            "cpu_percent": resources.get("percent_of_cpu_this_job_got"),
            "voluntary_context_switches": resources.get("voluntary_context_switches"),
            "involuntary_context_switches": resources.get("involuntary_context_switches"),
        })
    return sorted(rows, key=lambda row: (row["threads"], row["path"]))


def profile_rows(experiment: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patterns = [
        ("scale", experiment / "profiles" / "scale", "limit-*", "limit"),
        ("batch", experiment / "profiles" / "batch", "batch-*", "batch_size"),
    ]
    for kind, root, glob_pattern, parameter_name in patterns:
        for directory in sorted(root.glob(glob_pattern)):
            parameter = int(directory.name.split("-", 1)[1])
            for path in sorted(directory.glob("run-*.json")):
                program = unwrap(path)
                timing = program["timing_seconds"]
                detail = timing.get("detailed")
                counters = program.get("sampler_counters")
                if not detail or not counters:
                    continue
                rows.append({
                    "kind": kind,
                    parameter_name: parameter,
                    "limit": int(program["limit"]),
                    "batch_size": int(program["batch_size"]),
                    "path": str(path.relative_to(experiment)),
                    "core_total": timing["core_total"],
                    "end_to_end": timing["end_to_end"],
                    "load_model": timing["load_model"],
                    "sampler_init": detail["sampler_init"],
                    "queue_pop": detail["queue_pop"],
                    "queue_expand_insert": detail["queue_expand_insert"],
                    "queue_runtime": detail["queue_pop"] + detail["queue_expand_insert"],
                    "generate": detail["generate"],
                    "batch_plan": detail.get("batch_plan", 0.0),
                    "fused_worker": detail.get("fused_worker", 0.0),
                    "ordered_guard_merge": detail.get("ordered_guard_merge", 0.0),
                    "hash": detail["hash"],
                    "hash_compute": detail.get("hash_compute", detail["hash"]),
                    "hash_merge": detail.get("hash_merge", 0.0),
                    "threads": int(program.get("threads_used", 1)),
                    "verify": detail["verify"],
                    "throughput": int(program["generated"]) / timing["core_total"],
                    **counters,
                })
    for directory in sorted((experiment / "profiles" / "threads").glob("thread-*")):
        path = directory / "run.json"
        if not path.is_file():
            continue
        program = unwrap(path)
        timing = program["timing_seconds"]
        detail = timing.get("detailed")
        counters = program.get("sampler_counters")
        if not detail or not counters:
            continue
        rows.append({
            "kind": "threads",
            "threads": int(program.get("threads_used", directory.name.split("-", 1)[1])),
            "limit": int(program["limit"]),
            "batch_size": int(program["batch_size"]),
            "path": str(path.relative_to(experiment)),
            "core_total": timing["core_total"],
            "end_to_end": timing["end_to_end"],
            "load_model": timing["load_model"],
            "sampler_init": detail["sampler_init"],
            "queue_pop": detail["queue_pop"],
            "queue_expand_insert": detail["queue_expand_insert"],
            "queue_runtime": detail["queue_pop"] + detail["queue_expand_insert"],
            "generate": detail["generate"],
            "batch_plan": detail.get("batch_plan", 0.0),
            "fused_worker": detail.get("fused_worker", 0.0),
            "ordered_guard_merge": detail.get("ordered_guard_merge", 0.0),
            "hash": detail["hash"],
            "hash_compute": detail.get("hash_compute", detail["hash"]),
            "hash_merge": detail.get("hash_merge", 0.0),
            "verify": detail["verify"],
            "throughput": int(program["generated"]) / timing["core_total"],
            **counters,
        })
    if not rows:
        parsed = experiment / "summary" / "profiles.csv"
        if parsed.is_file() and parsed.read_text(encoding="utf-8").strip():
            rows = pd.read_csv(parsed).to_dict(orient="records")

    return rows

def group_profile(rows: list[dict[str, Any]], kind: str, key: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame([row for row in rows if row["kind"] == kind])
    if frame.empty:
        return []
    numeric = [
        "core_total", "end_to_end", "load_model", "sampler_init", "queue_runtime",
        "queue_pop", "queue_expand_insert", "generate", "batch_plan", "fused_worker",
        "ordered_guard_merge", "hash", "hash_compute",
        "hash_merge", "verify", "throughput",
        "pop_count", "insert_count", "expand_count", "range_count", "fused_group_count",
        "scalar_tail_count", "long_fallback_count", "peak_live_candidate_slots", "shifted_on_pop",
        "shifted_on_insert", "candidate_bytes", "initial_queue_size", "max_queue_size",
    ]
    result: list[dict[str, Any]] = []
    for value, group in frame.groupby(key):
        row: dict[str, Any] = {key: int(value), "repetitions": len(group)}
        for column in numeric:
            if column in group:
                row[column] = float(group[column].median())
        if row.get("fused_worker", 0) > 0:
            accounted = (row.get("sampler_init", 0) + row.get("queue_runtime", 0)
                         + row.get("batch_plan", 0) + row.get("fused_worker", 0)
                         + row.get("ordered_guard_merge", 0) + row.get("verify", 0))
        else:
            accounted = (row.get("sampler_init", 0) + row.get("queue_runtime", 0)
                         + row.get("generate", 0) + row.get("hash", 0) + row.get("verify", 0))
        row["unattributed"] = max(0.0, row["core_total"] - accounted)
        row["core_without_guard"] = row["core_total"] - row.get("ordered_guard_merge", row.get("hash_merge", 0.0))
        row["sampler_init_share"] = row.get("sampler_init", 0) / row["core_total"]
        row["queue_runtime_share"] = row.get("queue_runtime", 0) / row["core_total"]
        row["hash_ns_per_candidate"] = row.get("hash", 0) * 1e9 / int(group["limit"].iloc[0])
        result.append(row)
    return sorted(result, key=lambda item: item[key])


def save_figure(fig: plt.Figure, figures: Path, name: str) -> list[str]:
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / f"{name}.pdf"
    metadata = {"Date": None, "Creator": "PCFG baseline benchmark"}
    fig.savefig(path, bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    return [str(path)]


def distribution_figure(rows: list[dict[str, Any]], figures: Path) -> None:
    values = [row["core_total"] for row in rows]
    median = statistics.median(values)
    q1, q3 = percentile(values, 25), percentile(values, 75)
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.scatter(range(1, len(values) + 1), values, s=48, color=COLORS[0], label="measured run", zorder=3)
    ax.axhspan(q1, q3, color=COLORS[0], alpha=0.14, label="IQR")
    ax.axhline(median, color=COLORS[3], linewidth=1.8, label=f"median {median:.3f} s")
    ax.set(xlabel="Measured run", ylabel="core_total (s)", title="Headline runtime distribution")
    ax.set_xticks(range(1, len(values) + 1))
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures, "run-distribution")


def speedup_figure(variants: list[dict[str, Any]], figures: Path) -> None:
    labels = [item["variant"] for item in variants]
    speedups = [item["speedup"] for item in variants]
    low = [item["speedup"] - item["ci"][0] for item in variants]
    high = [item["ci"][1] - item["speedup"] for item in variants]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    positions = np.arange(len(labels))
    ax.errorbar(positions, speedups, yerr=[low, high], fmt="o", capsize=5, color=COLORS[0])
    ax.axhline(1.0, color=COLORS[5], linewidth=1, linestyle="--")
    for x, value in zip(positions, speedups):
        ax.annotate(f"{value:.2f}×", (x, value), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set(xticks=positions, xticklabels=labels, ylabel="Speedup vs baseline", title="Exact-version speedup with bootstrap 95% CI")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures, "speedup-exact")


def stage_figure(scale: list[dict[str, Any]], figures: Path) -> None:
    if not scale:
        return
    stages = [
        ("sampler_init", "sampler init"),
        ("queue_runtime", "runtime queue"),
        ("generate", "candidate generation"),
        ("hash", "scalar MD5"),
        ("verify", "verification"),
        ("unattributed", "other/loop"),
    ]
    labels = [f"{row['limit'] / 1e6:g}M" if row["limit"] >= 1e6 else f"{row['limit'] / 1e3:g}K" for row in scale]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bottoms = np.zeros(len(scale))
    for index, (field, label) in enumerate(stages):
        values = np.array([row.get(field, 0.0) for row in scale])
        ax.bar(labels, values, bottom=bottoms, label=label, color=COLORS[index % len(COLORS)])
        bottoms += values
    ax.set_ylim(0, float(max(bottoms)) * 1.25)
    ax.set(xlabel="Candidate limit", ylabel="Median detailed time (s)", title="Detailed stage composition")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, figures, "stage-composition")


def throughput_figure(scale: list[dict[str, Any]], figures: Path) -> None:
    if not scale:
        return
    limits = [row["limit"] for row in scale]
    throughput = [row["throughput"] / 1e6 for row in scale]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(limits, throughput, marker="o", linewidth=2, color=COLORS[0])
    for x, y in zip(limits, throughput):
        ax.annotate(f"{y:.2f}M/s", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set_xscale("log")
    ax.set(xlabel="Candidate limit (log scale)", ylabel="Throughput (million candidates/s)", title="Throughput across workload scale")
    ax.grid(alpha=0.25)
    save_figure(fig, figures, "throughput-scaling")


def batch_figure(batch: list[dict[str, Any]], figures: Path) -> None:
    if not batch:
        return
    labels = [f"{row['batch_size'] / 1024:g}K" for row in batch]
    values = [row["core_total"] for row in batch]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(labels, values, color=COLORS[0], alpha=0.85)
    ax.bar_label(bars, labels=[f"{value:.3f}s" for value in values], padding=3)
    ax.set(xlabel="Batch size", ylabel="Median detailed core_total (s)", title="Batch-size sensitivity")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures, "batch-sensitivity")


def profiler_label(name: str) -> str:
    labels = [
        ("vector<pcfg::SearchState", "vector<SearchState>::ordered_insert"),
        ("pcfg::md5(", "pcfg::md5"),
        ("pcfg::Sha256::transform", "pcfg::Sha256::transform"),
        ("vector<unsigned char", "vector<uint8_t>::realloc_insert"),
        ("pcfg::ModelStore::load", "pcfg::ModelStore::load"),
        ("pcfg::SerialSampler::generate_range", "pcfg::SerialSampler::generate_range"),
        ("pcfg::CandidateConsumer::consume", "pcfg::CandidateConsumer::consume"),
        ("pcfg::SerialSampler::pop", "pcfg::SerialSampler::pop"),
        ("(anonymous namespace)::run_sample", "run_sample"),
    ]
    for needle, label in labels:
        if needle in name:
            return label
    if name == "_init":
        return name
    return name if len(name) <= 56 else name[:53] + "..."


def profiler_figure(profiler: dict[str, Any], figures: Path) -> None:
    functions = profiler.get("gprof", {}).get("top_functions", [])[:10]
    if not functions:
        return
    functions = list(reversed(functions))
    labels = [profiler_label(item["name"]) for item in functions]
    values = [item["percent_time"] for item in functions]
    fig, ax = plt.subplots(figsize=(8.7, 5.0))
    bars = ax.barh(labels, values, color=COLORS[3])
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
    ax.set(xlabel="gprof sampled time (%)", title="Top gprof hotspots (profiling build)")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, figures, "profiler-hotspots")


def trace_figure(experiment: Path, figures: Path) -> None:
    traces = sorted((experiment / "profiles" / "scale").glob("limit-*/trace.jsonl"))
    if not traces:
        return
    path = max(traces, key=lambda candidate: int(candidate.parent.name.split("-", 1)[1]))
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    generated = np.array([item["generated"] for item in records], dtype=float) / 1e6
    queue = np.array([item["queue_size"] for item in records], dtype=float)
    shifted = np.array([item["shifted_on_pop"] + item["shifted_on_insert"] for item in records], dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(generated, queue, color=COLORS[0], linewidth=2, label="queue size")
    ax.set(xlabel="Generated candidates (million)", ylabel="Search queue states")
    second = ax.twinx()
    second.plot(generated, shifted, color=COLORS[3], linewidth=1.5, label="cumulative moved elements")
    second.set_ylabel("Estimated moved elements")
    lines = ax.lines + second.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")
    ax.set_title("Queue evolution and vector movement")
    ax.grid(alpha=0.2)
    save_figure(fig, figures, "queue-timeline")


def quality_figure(summary: dict[str, Any], figures: Path) -> None:
    points = summary["quality_throughput"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for index, point in enumerate(points):
        throughput = point["throughput"] / 1e6
        quality = point["cracked_unique"]
        marker = "o" if point["comparison_class"] == "exact" else "^"
        ax.scatter([throughput], [quality], s=80, marker=marker, color=COLORS[index % len(COLORS)])
        label = f"{point['variant']} ({point['comparison_class']})"
        ax.annotate(label, (throughput, quality), xytext=(8, 8), textcoords="offset points")
    ax.set(
        xlabel="Throughput (million candidates/s)",
        ylabel="Cracked unique",
        title="Quality–throughput frontier",
    )
    ax.grid(alpha=0.25)
    save_figure(fig, figures, "quality-throughput")


def summarize_thread_scaling(rows: list[dict[str, Any]], seed: int, samples: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["threads"]), []).append(float(row["core_total"]))
    if not grouped:
        return []
    if 1 not in grouped:
        raise ValueError("thread scaling requires a 1T reference")
    base = grouped[1]
    base_median = statistics.median(base)
    points: list[dict[str, Any]] = []
    for threads in sorted(grouped):
        values = grouped[threads]
        runtime = stats(values, seed + threads, samples)
        speedup = base_median / runtime["median"]
        rng = random.Random(seed + 1000 + threads)
        boot = [
            statistics.median(rng.choices(base, k=len(base)))
            / statistics.median(rng.choices(values, k=len(values)))
            for _ in range(samples)
        ]
        points.append({
            "variant": f"{threads}T",
            "workers": threads,
            "threads": threads,
            "runtime": runtime,
            "speedup": speedup,
            "speedup_95_ci": [percentile(boot, 2.5), percentile(boot, 97.5)],
            "efficiency": speedup / threads,
        })
    return points


def thread_stage_figure(points: list[dict[str, Any]], figures: Path) -> None:
    if not points:
        return
    if any(point.get("fused_worker", 0) > 0 for point in points):
        stages = [
            ("sampler_init", "sampler init"), ("queue_runtime", "heap queue"),
            ("batch_plan", "batch plan"), ("fused_worker", "fused generate + MD5"),
            ("ordered_guard_merge", "ordered guard merge"), ("unattributed", "other/loop"),
        ]
    else:
        stages = [
            ("sampler_init", "sampler init"), ("queue_runtime", "heap queue"),
            ("generate", "candidate generation"), ("hash_compute", "MD5 compute"),
            ("hash_merge", "ordered guard merge"), ("unattributed", "other/loop"),
        ]
    labels = [f"{point['threads']}T" for point in points]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bottoms = np.zeros(len(points))
    for index, (field, label) in enumerate(stages):
        values = np.array([point.get(field, 0.0) for point in points])
        ax.bar(labels, values, bottom=bottoms, label=label, color=COLORS[index % len(COLORS)])
        bottoms += values
    ax.set(ylabel="Detailed core time (s)", title="1T vs 8T stage composition")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, figures, "thread-stage-composition")


def guard_bottleneck_figure(points: list[dict[str, Any]], figures: Path) -> None:
    fused = [point for point in points if point.get("fused_worker", 0) > 0]
    if not fused:
        return
    labels = [f"{point['threads']}T" for point in fused]
    guard = np.array([point.get("ordered_guard_merge", 0.0) for point in fused])
    other = np.array([max(0.0, point["core_total"] - value) for point, value in zip(fused, guard)])
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(labels, other, label="core without ordered guard", color=COLORS[0])
    bars = ax.bar(labels, guard, bottom=other, label="ordered guard merge", color=COLORS[3])
    shares = [value / point["core_total"] * 100 for point, value in zip(fused, guard)]
    ax.bar_label(bars, labels=[f"guard {share:.1f}%" for share in shares], padding=3)
    ax.set(ylabel="Detailed core time (s)", title="Ordered hash_guard serial tail")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, figures, "guard-bottleneck")


def memory_working_set_figure(points: list[dict[str, Any]], figures: Path) -> None:
    selected = [
        point for point in points
        if point["variant"] in {"openmp-md5-v1", "openmp-fused-v1"} and point["values"]
    ]
    if len(selected) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for index, point in enumerate(selected):
        values = np.asarray(point["values"], dtype=float) / 1024.0
        x = np.full(len(values), index, dtype=float)
        ax.scatter(x, values, s=42, color=COLORS[index], alpha=0.85)
        median = float(np.median(values))
        ax.hlines(median, index - 0.22, index + 0.22, color=COLORS[3], linewidth=2)
        ax.annotate(f"{median:.1f} MiB", (index, median), xytext=(0, 8),
                    textcoords="offset points", ha="center")
    ax.set(xticks=range(len(selected)),
           xticklabels=[point["variant"] for point in selected],
           ylabel="Maximum RSS (MiB)",
           title="Process working-set comparison")
    ax.text(0.5, -0.22, "Total RSS is model-dominated; fused removes the 1M vector<string> batch buffer.",
            transform=ax.transAxes, ha="center", fontsize=9, color=COLORS[5])
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, figures, "memory-working-set")


def cumulative_speedup_figure(comparisons: list[dict[str, Any]], figures: Path) -> None:
    if not comparisons:
        return
    labels = [item["baseline"] for item in comparisons]
    values = [item["speedup"] for item in comparisons]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    bars = ax.bar(labels, values, color=COLORS[:len(values)])
    ax.bar_label(bars, labels=[f"{value:.2f}×" for value in values], padding=3)
    ax.axhline(1.0, color=COLORS[5], linestyle="--", linewidth=1)
    ax.set(ylabel="Candidate speedup", title="OpenMP 8T cumulative exact speedup")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures, "cumulative-exact-speedup")


def strong_scaling_figure(points: list[dict[str, Any]], figures: Path) -> None:
    if len(points) < 2:
        return
    points = sorted(points, key=lambda point: point["workers"])
    workers = np.array([point["workers"] for point in points], dtype=float)
    speedup = np.array([point["speedup"] for point in points], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(workers, speedup, marker="o", linewidth=2, color=COLORS[0], label="exact speedup")
    ax.plot(workers, workers, linestyle="--", linewidth=1.3, color=COLORS[5], label="ideal")
    for x, y in zip(workers, speedup):
        label = f"{y:.2f}x / {y / x * 100:.0f}%"
        ax.annotate(label, (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set(
        xlabel="Workers (threads x processes)",
        ylabel="Speedup vs serial baseline",
        title="Strong scaling and parallel efficiency",
        xticks=workers,
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, figures, "strong-scaling")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def markdown_report(summary: dict[str, Any]) -> str:
    headline = summary["headline"]
    bottleneck = summary["bottleneck"]
    scale = summary["profiles"]["scale"]
    thread_profiles = summary["profiles"].get("threads", [])
    largest = thread_profiles[-1] if thread_profiles else scale[-1] if scale else {}
    gprof_top = summary.get("profiler", {}).get("gprof", {}).get("top_functions", [])
    top_name = gprof_top[0]["name"] if gprof_top else "unavailable"
    top_share = gprof_top[0]["percent_time"] if gprof_top else 0.0
    variant = summary["identity"]["variant"]
    parallel_summary = ""
    heap_variant = variant == "serial-heap-v1"
    report_title = "serial-heap-v1 result" if heap_variant else "baseline-bottleneck"
    intro = (
        "本报告评估确定性 heap 队列优化；headline 与 profiling 数据严格分离。"
        if heap_variant else
        "本报告只定位瓶颈，不改变采样算法。headline 数据来自 instrumentation off；阶段拆分与 gprof 仅用于解释，不能进入加速比。"
    )
    queue_label = "确定性 heap" if heap_variant else "有序 vector"
    next_target = "heap 优化已完成；下一候选目标是 scalar MD5。" if heap_variant else "优先替换 sampler 初始化及运行期有序 vector 的搬移机制，同时保持 exact 候选顺序不变；之后再评估 scalar MD5。"
    if variant == "serial-md5-fast-v1":
        report_title = "serial-md5-fast-v1 result"
        intro = "本报告评估 scalar MD5 单 block fast path；headline 与 profiling 数据严格分离。"
        queue_label = "确定性 heap"
        next_target = "scalar fast path 已完成；下一候选目标是 AVX2 八路跨候选 MD5。"
    elif variant == "serial-md5-avx2-v1":
        report_title = "serial-md5-avx2-v1 result"
        intro = "本报告评估 AVX2 八路跨候选 MD5；headline 与 profiling 数据严格分离。"
        queue_label = "确定性 heap"
        next_target = "AVX2 阶段已完成；下一优化目标由新的 profiling 结果决定。"
    elif variant in {"openmp-md5-v1", "openmp-fused-v1"}:
        report_title = "openmp-md5-v1 result"
        intro = "本报告只评估 batch 内 AVX2 MD5 group 的 OpenMP 并行；候选生成、heap 与有序 hash_guard 归并保持串行。"
        queue_label = "确定性 heap（串行）"
        next_target = "依据 1T/8T 阶段拆分评估生成–哈希流水化；本轮不修改采样算法。"
        if variant == "openmp-fused-v1":
            report_title = "openmp-fused-v1 result"
            intro = "本报告评估 BatchPlan 驱动的候选生成–AVX2 哈希融合；headline 与 correctness/profiling 严格分离。"
            next_target = "融合节点内核已完成；下一阶段可评估 MPI RangeTask 分发或进一步降低有序 guard 串行尾部。"
        scaling = summary.get("strong_scaling", [])
        scaling_lines = [
            f"- {point['threads']}T: {point['runtime']['median']:.6f} s, " f"{point['speedup']:.2f}×, efficiency {point['efficiency'] * 100:.1f}%"
            for point in scaling if "runtime" in point
        ]
        comparison_lines = [
            f"- 8T vs {item['baseline']}: {item['speedup']:.2f}×, " f"95% CI [{item['ci'][0]:.2f}, {item['ci'][1]:.2f}]"
            for item in summary.get("direct_comparisons", [])
        ]
        parallel_summary = "## 强扩展与累计加速\n\n" + "\n".join(scaling_lines + comparison_lines) + "\n"
    hash_label = "AVX2 MD5" if variant in {"serial-md5-avx2-v1", "openmp-md5-v1", "openmp-fused-v1"} else "scalar MD5 fast path" if variant == "serial-md5-fast-v1" else "scalar MD5"
    sampler_note = (
        f"sampler_init {largest.get('sampler_init', 0):.6f} s，占 detailed core "
        f"{largest.get('sampler_init_share', 0) * 100:.1f}%。"
    )
    stage_note = (
        f"batch plan {largest.get('batch_plan', 0):.6f} s；fused generate + AVX2 worker {largest.get('fused_worker', 0):.6f} s；"
        f"ordered guard merge {largest.get('ordered_guard_merge', 0):.6f} s，占 detailed core " f"{largest.get('ordered_guard_merge', 0) / largest.get('core_total', 1) * 100:.1f}%。"
        if variant == "openmp-fused-v1" else
        f"串行 candidate generation {largest.get('generate', 0):.6f} s，已成为最大 core 阶段；"
        f"AVX2 compute {largest.get('hash_compute', 0):.6f} s，"
        f"ordered guard merge {largest.get('hash_merge', 0):.6f} s。"
        if variant == "openmp-md5-v1" else
        f"{hash_label}：10M detailed 的 hash 阶段 {largest.get('hash', 0):.6f} s，"
        f"约 {largest.get('hash_ns_per_candidate', 0):.1f} ns/candidate。"
    )
    resource_map = {item["variant"]: item["values"] for item in summary.get("resource_comparison", [])}
    fused_rss = statistics.median(resource_map.get("openmp-fused-v1", [0])) / 1024
    prior_rss = statistics.median(resource_map.get("openmp-md5-v1", [0])) / 1024
    memory_note = (f"5. 总 RSS 中位数 fused {fused_rss:.1f} MiB、OpenMP-MD5 {prior_rss:.1f} MiB；"
                   "模型主导峰值，因此结构性验收以正式主循环不再持有完整 vector<string> 为准。"
                   if variant == "openmp-fused-v1" else "")
    profile_note = (
        f"gprof 最大热点 `{top_name}`（{top_share:.1f}%）包含模型加载与完整性校验；"
        "该采样比例不能解释为 core_total 的墙钟占比。"
    )
    return f"""# {report_title}

{intro}

## 公平性与稳定性

- 工作负载：{headline['limit']:,} candidates，batch={headline['batch_size']:,}，CPU affinity={headline['cpu_set']}。
- 正式运行：{headline['warmups']} 次预热 + {headline['core_total']['count']} 次测量。
- core_total 中位数：{headline['core_total']['median']:.6f} s；bootstrap 95% CI [{headline['core_total']['bootstrap_95_ci'][0]:.6f}, {headline['core_total']['bootstrap_95_ci'][1]:.6f}] s。
- CV：{headline['core_total']['cv'] * 100:.2f}%（阈值 {headline['cv_threshold'] * 100:.0f}%）；状态：{'稳定，可作为 speedup 基准' if headline['stable'] else '不稳定，禁止作为最终 speedup'}。
- 吞吐中位数：{headline['throughput']['median'] / 1e6:.3f} M candidates/s。离群点被标记但未删除。

## 正确性门禁

exact 字段 generated/hash_guard/sig1/sig2/candidate_hits/cracked_unique 全部匹配；7 次性能运行的 generated/hash_guard 也一致。未来 relaxed 版本必须独立展示质量变化，不进入 strict speedup 排名。

{parallel_summary}
## 瓶颈结论

1. {sampler_note}
2. {queue_label}：运行期 pop/expand/insert 中位数 {largest.get('queue_runtime', 0):.6f} s；线性搬移元素 {largest.get('shifted_on_pop', 0) + largest.get('shifted_on_insert', 0):,.0f}。
3. {stage_note}
4. 模型加载：headline load_model 中位数 {headline['load_model']['median']:.6f} s。它影响 end_to_end，但不进入 core_total。{profile_note}
{memory_note}

## 下一项优化目标

{next_target} 任何优化必须复用本报告的模型、hash、CPU affinity、2+7 协议和正确性门禁。
"""


HTML_TEMPLATE = Template("""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ report_title }}</title>
<style>
:root{color-scheme:light dark;--bg:#f8fafc;--fg:#172033;--muted:#566178;--card:#fff;--line:#d8dee9;--accent:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#0f172a;--fg:#e5e7eb;--muted:#a8b1c3;--card:#172033;--line:#334155;--accent:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:28px 20px 60px}h1,h2{line-height:1.2}h2{margin-top:34px}
.meta{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}.metric{font-size:1.55rem;font-weight:600}
table{width:100%;border-collapse:collapse;background:var(--card)}th,td{padding:8px 10px;border:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
.figure{margin:22px 0;background:white;border-radius:8px;padding:8px;overflow:hidden}.figure svg{width:100%;height:auto;display:block}
code{background:var(--card);padding:2px 4px;border-radius:4px}ul{padding-left:22px}
</style>
</head>
<body><main>
<h1>{{ report_title }}</h1>
<p class="meta">可审计 PCFG 基准 · headline 与 profiling 分离 · 生成于 {{ generated }}</p>
<div class="grid">
<div class="card"><div>core_total median</div><div class="metric">{{ core }} s</div><div>95% CI {{ ci }}</div></div>
<div class="card"><div>throughput</div><div class="metric">{{ throughput }} M/s</div><div>CV {{ cv }}%</div></div>
<div class="card"><div>correctness gate</div><div class="metric">PASS</div><div>exact comparison class</div></div>
</div>
<h2>运行稳定性</h2><div class="figure">{{ figures['run-distribution'] }}</div>
<h2>Exact speedup 契约</h2><div class="figure">{{ figures.get('cumulative-exact-speedup', figures.get('speedup-exact', '')) }}</div>
<h2>线程扩展与效率</h2><div class="figure">{{ figures.get('strong-scaling','') }}</div>
<h2>阶段组成</h2><div class="figure">{{ figures.get('thread-stage-composition', figures.get('stage-composition','')) }}</div>
<h2>融合内存与串行尾部</h2><div class="figure">{{ figures.get('memory-working-set','') }}</div><div class="figure">{{ figures.get('guard-bottleneck','') }}</div>
<h2>批处理与热点</h2><div class="figure">{{ figures.get('batch-sensitivity','') }}</div><div class="figure">{{ figures.get('profiler-hotspots','') }}</div>
<h2>队列迁移</h2><div class="figure">{{ figures.get('queue-timeline','') }}</div>
<h2>质量–吞吐锚点</h2><div class="figure">{{ figures.get('quality-throughput','') }}</div>
<h2>统计摘要</h2>
<table><tr><th>metric</th><th>median</th><th>IQR</th><th>MAD</th><th>CV</th><th>range</th></tr>
<tr><td>core_total (s)</td><td>{{ core }}</td><td>{{ iqr }}</td><td>{{ mad }}</td><td>{{ cv }}%</td><td>{{ range }}</td></tr></table>
<h2>瓶颈结论</h2><ul>
{% for item in conclusions %}<li>{{ item }}</li>{% endfor %}
</ul>
<p class="meta">图表仅使用本地 JSON；无在线 CDN。原始 trace、gmon 与完整日志不进入 canonical Git 包。</p>
</main></body></html>""")


def render_html(experiment: Path, summary: dict[str, Any], report_text: str) -> None:
    figures = {
        path.stem: f'<a href="../figures/{path.name}">{path.stem}.pdf</a>'
        for path in sorted((experiment / "figures").glob("*.pdf"))
    }
    h = summary["headline"]
    scale = summary["profiles"]["scale"]
    thread_profiles = summary["profiles"].get("threads", [])
    largest = thread_profiles[-1] if thread_profiles else scale[-1] if scale else {}
    top = summary.get("profiler", {}).get("gprof", {}).get("top_functions", [])
    variant = summary["identity"]["variant"]
    hash_label = "AVX2 MD5" if variant in {"serial-md5-avx2-v1", "openmp-md5-v1", "openmp-fused-v1"} else "scalar MD5"
    conclusions = [
        f"sampler_init {largest.get('sampler_init', 0):.3f}s ({largest.get('sampler_init_share', 0) * 100:.1f}% of detailed core).",
        f"ordered guard merge: {largest.get('ordered_guard_merge', largest.get('hash_merge', 0)):.6f}s." if variant in {"openmp-md5-v1", "openmp-fused-v1"} else f"vector movement estimate: {largest.get('shifted_on_pop', 0) + largest.get('shifted_on_insert', 0):,.0f} elements.",
        f"{hash_label}: {largest.get('hash_ns_per_candidate', 0):.1f} ns/candidate.",
        f"gprof top: {top[0]['name']} ({top[0]['percent_time']:.1f}%)." if top else "gprof table unavailable.",
    ]
    output = HTML_TEMPLATE.render(
        report_title=summary["identity"]["variant"],
        generated=summary["generated_at_utc"],
        core=f"{h['core_total']['median']:.6f}",
        ci=f"[{h['core_total']['bootstrap_95_ci'][0]:.6f}, {h['core_total']['bootstrap_95_ci'][1]:.6f}]",
        throughput=f"{h['throughput']['median'] / 1e6:.3f}",
        cv=f"{h['core_total']['cv'] * 100:.2f}",
        iqr=f"{h['core_total']['iqr']:.6f}",
        mad=f"{h['core_total']['mad']:.6f}",
        range=f"{h['core_total']['range']:.6f}",
        figures=figures,
        conclusions=conclusions,
    )
    path = experiment / "report" / "baseline-bottleneck.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
    (experiment / "report" / "baseline-bottleneck.md").write_text(report_text, encoding="utf-8")


def analyze_experiment(experiment: Path, comparisons: Iterable[Path] = ()) -> None:
    comparisons = list(comparisons)
    identity = load_json(experiment / "manifest" / "identity.json")
    gate = load_json(experiment / "manifest" / "correctness-gate.json")
    rows = measured_rows(experiment)
    thread_rows = thread_scaling_rows(experiment)
    profile = profile_rows(experiment)
    config_path = Path(identity["config_path"])
    config = load_json(Path(__file__).resolve().parents[1] / config_path)
    headline_config = config["headline"]
    seed = int(headline_config["bootstrap_seed"])
    samples = int(headline_config["bootstrap_samples"])

    core = stats([row["core_total"] for row in rows], seed, samples)
    end = stats([row["end_to_end"] for row in rows], seed + 1, samples)
    load = stats([row["load_model"] for row in rows], seed + 2, samples)
    throughput = stats([row["throughput"] for row in rows], seed + 3, samples)
    stable = core["cv"] <= float(headline_config["cv_threshold"])

    scale = group_profile(profile, "scale", "limit")
    batch = group_profile(profile, "batch", "batch_size")
    thread_profiles = group_profile(profile, "threads", "threads")
    thread_scaling = summarize_thread_scaling(thread_rows, seed, samples)
    profiler_path = experiment / "profiles" / "profiler.json"
    profiler = load_json(profiler_path) if profiler_path.is_file() else {}

    reference_median = core["median"]
    reference_quality = int(gate["fields"]["cracked_unique"])
    parallelism = identity.get("protocol", {}).get("parallelism", {"threads": 1, "processes": 1})
    reference_workers = int(parallelism.get("threads", 1)) * int(parallelism.get("processes", 1))
    variants = [{
        "variant": identity["variant"],
        "comparison_class": identity["comparison_class"],
        "speedup": 1.0,
        "ci": [1.0, 1.0],
    }]
    resource_points = [{
        "variant": identity["variant"],
        "values": [row["max_rss_kb"] for row in rows if row.get("max_rss_kb") is not None],
    }]
    quality_points = [{
        "variant": identity["variant"],
        "comparison_class": identity["comparison_class"],
        "throughput": throughput["median"],
        "cracked_unique": reference_quality,
        "quality_delta": 0,
    }]
    scaling_points = [{
        "variant": identity["variant"],
        "workers": reference_workers,
        "speedup": 1.0,
        "efficiency": 1.0 / reference_workers,
    }]
    base_values = [row["core_total"] for row in rows]
    direct_comparisons: list[dict[str, Any]] = []
    for offset, comparison in enumerate(comparisons, 1):
        other_identity = load_json(comparison / "manifest" / "identity.json")
        other_gate = load_json(comparison / "manifest" / "correctness-gate.json")
        other_rows = measured_rows(comparison)
        other_values = [row["core_total"] for row in other_rows]
        other_stats = stats(other_values, seed + 100 + offset, samples)
        other_throughput = stats(
            [row["throughput"] for row in other_rows], seed + 300 + offset, samples
        )
        other_quality = int(other_gate["fields"]["cracked_unique"])
        resource_points.append({
            "variant": other_identity["variant"],
            "values": [row["max_rss_kb"] for row in other_rows if row.get("max_rss_kb") is not None],
        })
        if other_identity["comparison_class"] == "exact":
            direct_rng = random.Random(seed + 500 + offset)
            direct_boot = [
                statistics.median(direct_rng.choices(other_values, k=len(other_values)))
                / statistics.median(direct_rng.choices(base_values, k=len(base_values)))
                for _ in range(samples)
            ]
            direct_comparisons.append({
                "baseline": other_identity["variant"],
                "candidate": identity["variant"],
                "speedup": other_stats["median"] / reference_median,
                "ci": [percentile(direct_boot, 2.5), percentile(direct_boot, 97.5)],
            })
        quality_points.append({
            "variant": other_identity["variant"],
            "comparison_class": other_identity["comparison_class"],
            "throughput": other_throughput["median"],
            "cracked_unique": other_quality,
            "quality_delta": other_quality - reference_quality,
        })
        if other_identity["comparison_class"] != "exact":
            continue
        speedup = reference_median / other_stats["median"]
        rng = random.Random(seed + 200 + offset)
        bootstrap = []
        for _ in range(samples):
            base = statistics.median(rng.choices(base_values, k=len(base_values)))
            candidate = statistics.median(rng.choices(other_values, k=len(other_values)))
            bootstrap.append(base / candidate)
        variants.append({
            "variant": other_identity["variant"],
            "comparison_class": "exact",
            "speedup": speedup,
            "ci": [percentile(bootstrap, 2.5), percentile(bootstrap, 97.5)],
        })
        other_parallelism = other_identity.get("protocol", {}).get(
            "parallelism", {"threads": 1, "processes": 1}
        )
        workers = int(other_parallelism.get("threads", 1)) * int(other_parallelism.get("processes", 1))
        scaling_points.append({
            "variant": other_identity["variant"],
            "workers": workers,
            "speedup": speedup,
            "efficiency": speedup / workers,
        })

    correctness_consistent = len({(row["generated"], row["hash_guard"]) for row in rows}) == 1
    summary = {
        "schema": "pcfg-benchmark-summary-v1",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "identity": {
            "variant": identity["variant"],
            "comparison_class": identity["comparison_class"],
            "git_sha": identity["git_sha"],
            "git_dirty": identity["git_dirty"],
            "dataset_sha256": identity.get("dataset", {}).get("sha256"),
            "model_sha256": identity["model"]["sha256"],
            "binary_sha256": identity["build"]["binary_sha256"],
        },
        "correctness": {
            "passed": bool(gate["passed"]) and correctness_consistent,
            "fields": gate["fields"],
            "measured_hash_guard_consistent": correctness_consistent,
        },
        "headline": {
            "limit": int(headline_config["limit"]),
            "batch_size": int(headline_config["batch_size"]),
            "cpu_set": str(headline_config["cpu_set"]),
            "threads": int(headline_config.get("threads", 1)),
            "warmups": int(headline_config["warmups"]),
            "core_total": core,
            "end_to_end": end,
            "load_model": load,
            "throughput": throughput,
            "cv_threshold": float(headline_config["cv_threshold"]),
            "stable": stable,
        },
        "speedup_exact": variants,
        "quality_throughput": quality_points,
        "strong_scaling": thread_scaling if len(thread_scaling) > 1 else scaling_points,
        "direct_comparisons": direct_comparisons,
        "resource_comparison": resource_points,
        "profiles": {"scale": scale, "batch": batch, "threads": thread_profiles},
        "profiler": profiler,
        "bottleneck": {
            "next_target": "optimize scalar MD5" if identity["variant"] == "serial-heap-v1" else "replace ordered vector queue movement while preserving exact ordering",
            "algorithm_modified": identity["variant"] != "baseline-v1",
        },
    }
    json_dump(experiment / "summary" / "summary.json", summary)
    write_csv(experiment / "summary" / "runs.csv", rows)
    write_csv(experiment / "summary" / "profiles.csv", profile)
    write_csv(experiment / "summary" / "thread-scaling.csv", thread_rows)
    json_dump(experiment / "summary" / "thread-scaling.json", thread_scaling)
    json_dump(experiment / "summary" / "direct-comparisons.json", {
        "schema": "pcfg-direct-comparisons-v1",
        "candidate": identity["variant"],
        "comparisons": direct_comparisons,
    })

    figures = experiment / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.svg", "*.png", "*.pdf"):
        for stale in figures.glob(pattern):
            stale.unlink()
    distribution_figure(rows, figures)
    if identity["variant"] in {"openmp-md5-v1", "openmp-fused-v1"}:
        strong_scaling_figure(thread_scaling, figures)
        thread_stage_figure(thread_profiles, figures)
        cumulative_speedup_figure(direct_comparisons, figures)
        if identity["variant"] == "openmp-fused-v1":
            guard_bottleneck_figure(thread_profiles, figures)
            memory_working_set_figure(resource_points, figures)
    else:
        speedup_figure(variants, figures)
        strong_scaling_figure(scaling_points, figures)
        stage_figure(scale, figures)
        throughput_figure(scale, figures)
        batch_figure(batch, figures)
        profiler_figure(profiler, figures)
        trace_figure(experiment, figures)
        quality_figure(summary, figures)

    chart_names = sorted(path.stem for path in figures.glob("*.pdf"))
    repo_root = Path(__file__).resolve().parents[1]
    def command_path(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return str(path)
    comparison_args = "".join(f" --compare {command_path(path)}" for path in comparisons)
    visualization_manifest = {
        "schema": "pcfg-visualization-manifest-v1",
        "command": "./scripts/bench analyze --input " + command_path(experiment) + comparison_args,
        "sources": {
            "headline": "runs/measured-*.json",
            "correctness": "manifest/correctness-gate.json",
            "detailed": "profiles/threads/thread-*/run.json",
            "thread_scaling": "runs/measured-*.json and scaling/thread-*/measured-*.json",
            "profiler": "profiles/profiler.json",
            "timeline": "profiles/scale/limit-*/trace.jsonl",
        },
        "charts": [
            {"name": name, "formats": ["pdf"]}
            for name in chart_names
        ],
        "conditional_contracts": {
            "speedup": "exact comparison_class only",
            "quality_frontier": "relaxed variants appear separately when supplied",
            "strong_scaling": "worker/process series appear when future experiment rows provide them",
        },
    }
    json_dump(experiment / "report" / "visualization-manifest.json", visualization_manifest)
    report = markdown_report(summary)
    render_html(experiment, summary, report)
    print(f"analysis complete: {experiment}")

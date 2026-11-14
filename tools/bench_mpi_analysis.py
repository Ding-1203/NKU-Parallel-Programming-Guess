#!/usr/bin/env python3
"""Analysis and PDF reporting for the MPI fused benchmark."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import bench

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#2563eb", "#0f766e", "#d97706", "#dc2626", "#7c3aed", "#64748b"]


def read_program(path: Path) -> dict[str, Any]:
    value = bench.load_json(path)
    return value.get("program", value)


def read_rows(root: Path, pattern: str) -> list[dict[str, Any]]:
    return [read_program(path) for path in sorted(root.glob(pattern))]


def core(rows: list[dict[str, Any]]) -> list[float]:
    return [float(row["timing_seconds"]["core_total"]) for row in rows]


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float], seed: int, samples: int) -> dict[str, Any]:
    if not values:
        raise ValueError("empty benchmark series")
    median = statistics.median(values)
    mean = statistics.mean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    rng = random.Random(seed)
    bootstrap = [statistics.median(rng.choices(values, k=len(values)))
                 for _ in range(samples)]
    q1, q3 = percentile(values, 25), percentile(values, 75)
    return {
        "count": len(values), "median": median, "mean": mean,
        "min": min(values), "max": max(values), "q1": q1, "q3": q3,
        "iqr": q3 - q1,
        "mad": statistics.median(abs(value - median) for value in values),
        "stddev": stddev, "cv": stddev / mean if mean else 0.0,
        "bootstrap_95_ci": [percentile(bootstrap, 2.5), percentile(bootstrap, 97.5)],
        "values": values,
    }


def compare(reference: list[float], candidate: list[float],
            seed: int, samples: int) -> dict[str, Any]:
    rng = random.Random(seed)
    values = [
        statistics.median(rng.choices(reference, k=len(reference))) /
        statistics.median(rng.choices(candidate, k=len(candidate)))
        for _ in range(samples)
    ]
    return {
        "value": statistics.median(reference) / statistics.median(candidate),
        "bootstrap_95_ci": [percentile(values, 2.5), percentile(values, 97.5)],
    }


def save(fig: plt.Figure, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight",
                metadata={"Date": None, "Creator": "PCFG MPI benchmark"})
    plt.close(fig)


def plot_configurations(calibration: dict[str, dict[str, Any]], figures: Path) -> None:
    names = list(calibration)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 3.9))
    baseline = calibration["1x8"]["median"]
    for index, name in enumerate(names):
        values = calibration[name]["values"]
        left.scatter(np.full(len(values), index), values, color=COLORS[index], s=38)
        left.hlines(calibration[name]["median"], index - 0.24, index + 0.24,
                    color="#111827", linewidth=2)
    left.set(xticks=range(len(names)), xticklabels=names,
             ylabel="core_total (s)", title="100M calibration distribution")
    left.grid(axis="y", alpha=0.25)
    gains = [baseline / calibration[name]["median"] for name in names]
    bars = right.bar(names, gains, color=COLORS[:len(names)])
    right.bar_label(bars, labels=[f"{value:.2f}x" for value in gains], padding=3)
    right.axhline(1.0, color="#111827", linestyle="--", linewidth=1)
    right.set(ylabel="Speedup vs 1x8 MPI", title="Process-thread topology")
    right.grid(axis="y", alpha=0.25)
    save(fig, figures, "mpi-configuration-scaling")


def plot_speedup(comparison: dict[str, Any], figures: Path) -> None:
    value = comparison["value"]
    low, high = comparison["bootstrap_95_ci"]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.errorbar([0], [value], yerr=[[value - low], [high - value]],
                fmt="o", capsize=7, markersize=8, color=COLORS[0])
    ax.axhline(1.0, color="#111827", linestyle="--")
    ax.annotate(f"{value:.3f}x  [{low:.3f}, {high:.3f}]",
                (0, value), xytext=(18, 8), textcoords="offset points")
    ax.set(xlim=(-0.5, 0.8), xticks=[0], xticklabels=["selected hybrid"],
           ylabel="Speedup", title="100M hybrid vs openmp-fused-v1.0")
    ax.grid(axis="y", alpha=0.25)
    save(fig, figures, "mpi-headline-speedup")


def profile(root: Path, name: str) -> dict[str, Any] | None:
    path = root / "profiles" / name / "run.json"
    return read_program(path) if path.is_file() else None


def plot_stages(root: Path, selected: str, figures: Path) -> None:
    values = [profile(root, "1x8"), profile(root, selected)]
    if any(value is None for value in values):
        return
    stages = [
        ("batch_plan", "batch plan"), ("plan_broadcast", "broadcast"),
        ("rank_compute_max", "rank compute"), ("rank_sync", "rank sync"),
        ("digest_gather", "digest gather"),
        ("ordered_guard_merge", "ordered guard"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bottoms = np.zeros(2)
    for index, (field, label) in enumerate(stages):
        current = np.array([
            float(value["timing_seconds"]["detailed"].get(field, 0.0))
            for value in values
        ])
        ax.bar(["1x8", selected], current, bottom=bottoms, label=label,
               color=COLORS[index])
        bottoms += current
    totals = np.array([float(value["timing_seconds"]["core_total"]) for value in values])
    ax.bar(["1x8", selected], np.maximum(0.0, totals - bottoms), bottom=bottoms,
           label="other/loop", color="#cbd5e1")
    ax.set(ylabel="Detailed wall time (s)",
           title="100M distributed critical-path composition")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    save(fig, figures, "mpi-stage-composition")


def timelines(root: Path, selected: str) -> dict[str, list[dict[str, Any]]]:
    destination = root / "summary" / "batch-timeline.json"
    result: dict[str, list[dict[str, Any]]] = {}
    for name in ("1x8", selected):
        path = root / "profiles" / name / "batch-trace.jsonl"
        if path.is_file():
            result[name] = [json.loads(line) for line in path.read_text(
                encoding="utf-8").splitlines() if line.strip()]
    if result:
        bench.json_dump(destination, result)
    elif destination.is_file():
        result = bench.load_json(destination)
    return result


def plot_timeline(data: dict[str, list[dict[str, Any]]],
                  selected: str, figures: Path) -> None:
    if selected not in data:
        return
    rows = data[selected]
    x = [row["batch"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for field, label, color in [
        ("compute_max", "slowest rank compute", COLORS[0]),
        ("compute_mean", "mean rank compute", COLORS[1]),
        ("rank_sync", "rank sync wait", COLORS[3]),
        ("digest_gather", "digest gather", COLORS[2]),
    ]:
        ax.plot(x, [row[field] * 1000 for row in rows], label=label, color=color)
    ax.set(xlabel="1M-candidate batch", ylabel="Time (ms)",
           title=f"{selected} per-batch balance and communication")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    save(fig, figures, "mpi-rank-timeline")


def plot_cumulative(legacy_median: float, figures: Path) -> None:
    names = ["baseline-v1", "serial-heap-v1", "serial-md5-fast-v1",
             "serial-md5-avx2-v1", "openmp-md5-v1", "openmp-fused-v1"]
    labels: list[str] = []
    runtimes: list[float] = []
    for name in names:
        path = ROOT / "benchmarks" / "reference" / name / "summary" / "summary.json"
        if path.is_file():
            labels.append(name)
            runtimes.append(float(bench.load_json(path)["headline"]["core_total"]["median"]))
    labels.append("mpi-fused-exact-v1")
    runtimes.append(legacy_median)
    base = runtimes[0]
    gains = [base / value for value in runtimes]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    bars = ax.bar(range(len(labels)), gains, color=COLORS[0])
    ax.bar_label(bars, labels=[f"{value:.1f}x" for value in gains], padding=3)
    ax.set(xticks=range(len(labels)), xticklabels=labels,
           ylabel="10M exact speedup", title="Cumulative exact speedup")
    ax.tick_params(axis="x", rotation=24)
    ax.grid(axis="y", alpha=0.2)
    save(fig, figures, "cumulative-10m-exact-speedup")


def write_report(root: Path, summary: dict[str, Any]) -> None:
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    selected = summary["selection"]["selected"]
    speed = summary["headline_speedup"]
    passed = summary["acceptance"]["passed"]
    text = f"""# MPI fused exact v1

- Selected hybrid: **{selected}**
- 100M MPI median: **{summary['headline']['median']:.6f} s**
- Direct OpenMP median: **{summary['baseline']['median']:.6f} s**
- Speedup: **{speed['value']:.3f}x**, 95% CI [{speed['bootstrap_95_ci'][0]:.3f}, {speed['bootstrap_95_ci'][1]:.3f}]
- Headline CV: **{summary['headline']['cv'] * 100:.2f}%**
- Tag acceptance: **{'PASS' if passed else 'FAIL'}**

10M is retained as an exact historical comparison. The tag decision uses sustained 100M
throughput. hash_guard remains inside core_total.

## Bottleneck conclusion

{summary['bottleneck']}
"""
    (report / "mpi-bottleneck.md").write_text(text, encoding="utf-8")
    links = "".join(
        f'<li><a href="../figures/{name}.pdf">{label}</a></li>'
        for label, name in [
            ("Configuration scaling", "mpi-configuration-scaling"),
            ("Headline speedup", "mpi-headline-speedup"),
            ("Stage composition", "mpi-stage-composition"),
            ("Rank timeline", "mpi-rank-timeline"),
            ("Cumulative exact speedup", "cumulative-10m-exact-speedup"),
        ]
    )
    html = f"""<!doctype html><meta charset="utf-8"><title>MPI fused exact v1</title>
<style>body{{font:16px system-ui;max-width:900px;margin:40px auto;line-height:1.55;color:#172033}}
table{{border-collapse:collapse}}td,th{{padding:7px 12px;border:1px solid #ccd3dd}}</style>
<h1>MPI fused exact v1</h1><table>
<tr><th>Selected</th><td>{selected}</td></tr>
<tr><th>100M MPI median</th><td>{summary['headline']['median']:.6f} s</td></tr>
<tr><th>OpenMP median</th><td>{summary['baseline']['median']:.6f} s</td></tr>
<tr><th>Speedup</th><td>{speed['value']:.3f}x [{speed['bootstrap_95_ci'][0]:.3f}, {speed['bootstrap_95_ci'][1]:.3f}]</td></tr>
<tr><th>CV</th><td>{summary['headline']['cv'] * 100:.2f}%</td></tr>
<tr><th>Acceptance</th><td>{'PASS' if passed else 'FAIL'}</td></tr></table>
<h2>Bottleneck</h2><p>{summary['bottleneck']}</p><h2>PDF figures</h2><ul>{links}</ul>
<p>Correctness and performance runs are separate. hash_guard is included in core_total.</p>"""
    (report / "mpi-bottleneck.html").write_text(html, encoding="utf-8")


def analyze(root: Path) -> None:
    selection = bench.load_json(root / "manifest" / "selection.json")
    selected = str(selection["selected"])
    protocol = bench.load_json(root / "manifest" / "identity.json")["protocol"]
    seed = int(protocol["statistics"]["bootstrap_seed"])
    samples = int(protocol["statistics"]["bootstrap_samples"])
    calibration = {
        point["name"]: summarize(core(read_rows(
            root, f"calibration/{point['name']}/measured-*.json")),
            seed + int(point["ranks"]), samples)
        for point in protocol["configurations"]
    }
    baseline_values = core(read_rows(root, "baseline/openmp-1x8/measured-*.json"))
    headline_values = core(read_rows(root, f"headline/{selected}/measured-*.json"))
    legacy_values = core(read_rows(root, f"legacy/{selected}/measured-*.json"))
    baseline = summarize(baseline_values, seed + 100, samples)
    headline = summarize(headline_values, seed + 200, samples)
    legacy = summarize(legacy_values, seed + 300, samples)
    gain = compare(baseline_values, headline_values, seed + 400, samples)

    selected_profile = profile(root, selected)
    bottleneck = "Profiling has not been collected yet."
    profile_diagnosis: dict[str, Any] = {}
    if selected_profile:
        detailed = selected_profile["timing_seconds"]["detailed"]
        total = float(selected_profile["timing_seconds"]["core_total"])
        candidates = {
            "rank compute": float(detailed.get("rank_compute_max", 0.0)),
            "ordered guard": float(detailed.get("ordered_guard_merge", 0.0)),
            "rank synchronization": float(detailed.get("rank_sync", 0.0)),
            "digest gather": float(detailed.get("digest_gather", 0.0)),
            "batch planning": float(detailed.get("batch_plan", 0.0)),
        }
        name, value = max(candidates.items(), key=lambda item: item[1])
        communication = (candidates["rank synchronization"] +
                         candidates["digest gather"] +
                         float(detailed.get("plan_broadcast", 0.0)))
        accounted = (sum(candidates.values()) +
                     float(detailed.get("plan_broadcast", 0.0)) +
                     float(detailed.get("sampler_init", 0.0)) +
                     float(detailed.get("queue_pop", 0.0)) +
                     float(detailed.get("queue_expand_insert", 0.0)))
        unattributed = max(0.0, total - accounted)
        mpi_metrics = selected_profile.get("mpi", {})
        rank_rss = int(mpi_metrics.get("rank_rss_mean_bytes", 0))
        ranks = int(selected_profile.get("ranks_used", 1))
        profile_diagnosis = {
            "largest_stage": name,
            "largest_stage_seconds": value,
            "largest_stage_share": value / total,
            "communication_and_sync_seconds": communication,
            "communication_and_sync_share": communication / total,
            "ordered_guard_seconds": candidates["ordered guard"],
            "ordered_guard_share": candidates["ordered guard"] / total,
            "unattributed_seconds": unattributed,
            "unattributed_share": unattributed / total,
            "rank_rss_mean_bytes": rank_rss,
            "estimated_total_rank_rss_bytes": rank_rss * ranks,
        }
        bottleneck = (
            f"Detailed {selected}: {name} is {value:.6f} s "
            f"({value / total * 100:.1f}%); broadcast, synchronization and digest gather "
            f"add {communication:.6f} s ({communication / total * 100:.1f}%); "
            f"the ordered guard adds {candidates['ordered guard']:.6f} s "
            f"({candidates['ordered guard'] / total * 100:.1f}%). With the same eight "
            f"physical cores, MPI adds no compute capacity, while it duplicates the "
            f"approximately {rank_rss / 1024**2:.1f} MiB model working set per rank and "
            f"introduces collectives. Those costs outweigh the process/thread topology gain, "
            f"consistent with the measured 100M slowdown."
        )

    acceptance = {
        "exact_gates": bench.load_json(
            root / "manifest" / "correctness-gate.json")["passed"],
        "cv_within_threshold": headline["cv"] <=
                               float(protocol["statistics"]["cv_threshold"]),
        "speedup_ci_lower_above_one": gain["bootstrap_95_ci"][0] > 1.0,
    }
    acceptance["passed"] = all(acceptance.values())
    summary = {
        "schema": "pcfg-mpi-summary-v1", "generated_at_utc": bench.utc_now(),
        "selection": selection, "calibration": calibration,
        "baseline": baseline, "headline": headline, "legacy_10m": legacy,
        "headline_speedup": gain, "acceptance": acceptance,
        "profile_diagnosis": profile_diagnosis,
        "bottleneck": bottleneck,
    }
    bench.json_dump(root / "summary" / "summary.json", summary)
    figures = root / "figures"
    plot_configurations(calibration, figures)
    plot_speedup(gain, figures)
    plot_stages(root, selected, figures)
    data = timelines(root, selected)
    plot_timeline(data, selected, figures)
    plot_cumulative(legacy["median"], figures)
    write_report(root, summary)
    bench.json_dump(root / "report" / "visualization-manifest.json", {
        "schema": "pcfg-visualization-manifest-v1",
        "generated_at_utc": bench.utc_now(),
        "command": f"python3 tools/bench_mpi.py analyze --input {root}",
        "figures": sorted(str(path.relative_to(root)) for path in figures.glob("*.pdf")),
        "data": ["summary/summary.json", "summary/batch-timeline.json"],
    })
    print(f"MPI analysis complete: {root}")

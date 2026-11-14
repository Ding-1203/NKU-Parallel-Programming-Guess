#!/usr/bin/env python3
"""Lightweight MPI benchmark orchestration for the fused PCFG sampler."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench

ROOT = Path(__file__).resolve().parents[1]
EXACT_FIELDS = bench.EXACT_FIELDS


def load_config(path: Path) -> dict[str, Any]:
    value = bench.load_json(path)
    if value.get("schema") != "pcfg-mpi-benchmark-config-v1":
        raise bench.BenchError(f"unsupported MPI benchmark config: {path}")
    return value


def configuration(config: dict[str, Any], name: str) -> dict[str, Any]:
    for value in config["configurations"]:
        if value["name"] == name:
            return value
    raise bench.BenchError(f"unknown MPI configuration: {name}")


def all_worker_cpus(config: dict[str, Any]) -> str:
    cpus: set[int] = set()
    for point in config["configurations"]:
        for specification in point["cpu_sets"]:
            cpus.update(bench.expand_cpu_set(specification))
    return ",".join(str(value) for value in sorted(cpus))


def memory_preflight(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            fields = raw.strip().split()
            if fields and fields[0].isdigit():
                values[key] = int(fields[0]) * 1024
    largest = max(int(point["ranks"]) for point in config["configurations"])
    estimate = int(config["reference"]["estimated_rank_rss_mib"]) * 1024 * 1024 * largest
    required = int(estimate * 1.25)
    available = values.get("MemAvailable", 0)
    if available < required:
        raise bench.BenchError(
            f"insufficient memory for {largest} ranks: available={available}, required={required}"
        )
    return {
        "available_bytes": available,
        "estimated_peak_bytes": estimate,
        "required_with_margin_bytes": required,
        "largest_rank_count": largest,
    }


def write_rankfile(root: Path, point: dict[str, Any]) -> Path:
    ranks = int(point["ranks"])
    cpu_sets = list(point["cpu_sets"])
    if len(cpu_sets) != ranks:
        raise bench.BenchError(f"{point['name']}: rank/cpu_set count mismatch")
    seen: set[int] = set()
    lines: list[str] = []
    threads = int(point["threads_per_rank"])
    for rank, specification in enumerate(cpu_sets):
        cpus = bench.expand_cpu_set(str(specification))
        if len(cpus) != threads:
            raise bench.BenchError(
                f"{point['name']} rank {rank}: expected {threads} CPUs, got {sorted(cpus)}"
            )
        if seen.intersection(cpus):
            raise bench.BenchError(f"{point['name']}: overlapping rank CPU sets")
        seen.update(cpus)
        lines.append(f"rank {rank}=localhost slot={','.join(str(cpu) for cpu in sorted(cpus))}")
    path = root / "manifest" / "rankfiles" / f"{point['name']}.rankfile"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def mpi_sample_command(
    binary: Path,
    model: Path,
    point: dict[str, Any],
    rankfile: Path,
    limit: int,
    batch_size: int,
    metrics: Path,
    instrumentation: str,
    signature: bool = False,
    test_data: Path | None = None,
    test_limit: int = 0,
    trace: Path | None = None,
    report_bindings: bool = False,
) -> list[str]:
    ranks = int(point["ranks"])
    threads = int(point["threads_per_rank"])
    command = [
        "env", "TMPDIR=/tmp",
        f"OMP_NUM_THREADS={threads}",
        "OMP_DYNAMIC=FALSE",
        "OMP_PROC_BIND=close",
        "OMP_PLACES=threads",
        f"OMP_DISPLAY_AFFINITY={'TRUE' if report_bindings else 'FALSE'}",
        "taskset", "-c", "0-15",
        "mpirun",
        "--use-hwthread-cpus",
        "--mca", "pml", "ob1",
        "--mca", "btl", "self,vader",
        "--rankfile", str(rankfile),
        "-np", str(ranks),
    ]
    if report_bindings:
        command.append("--report-bindings")
    command.extend([
        str(binary), "sample",
        "--model", str(model),
        "--limit", str(limit),
        "--batch-size", str(batch_size),
        "--threads", str(threads),
        "--instrumentation", instrumentation,
        "--metrics-out", str(metrics),
    ])
    if signature:
        command.append("--signature")
    if test_data:
        command.extend(["--test-data", str(test_data), "--test-limit", str(test_limit)])
    if trace:
        command.extend(["--trace-out", str(trace)])
    return command


def validate_program(program: dict[str, Any], point: dict[str, Any]) -> None:
    expected_ranks = int(point["ranks"])
    expected_threads = int(point["threads_per_rank"])
    if program.get("mode") != "mpi-fused":
        raise bench.BenchError(f"{point['name']}: program did not report mpi-fused mode")
    if int(program.get("ranks_used", 0)) != expected_ranks:
        raise bench.BenchError(
            f"{point['name']}: expected {expected_ranks} ranks, got {program.get('ranks_used')}"
        )
    if int(program.get("threads_used", 0)) != expected_threads:
        raise bench.BenchError(
            f"{point['name']}: expected {expected_threads} threads/rank, "
            f"got {program.get('threads_used')}"
        )


def timed_series(
    name: str,
    root: Path,
    warmups: int,
    repetitions: int,
    estimate_seconds: float,
    command_builder: Callable[[Path], list[str]],
    metadata: dict[str, Any],
    validator: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    total_runs = warmups + repetitions
    print(
        f"\n=== {name}: {warmups} warm-up + {repetitions} measured; "
        f"estimated {estimate_seconds:.0f}s ===",
        flush=True,
    )
    started = time.monotonic()
    for index in range(warmups):
        path = root / f"warmup-{index + 1:02d}.json"
        command = command_builder(path)
        bench.run_logged(command, root / f"warmup-{index + 1:02d}.log")
        validator(bench.load_json(path))
        path.unlink(missing_ok=True)
        if index == 0:
            elapsed = time.monotonic() - started
            remaining = elapsed * (total_runs - 1)
            print(f"  ETA update after first warm-up: about {remaining:.0f}s remaining", flush=True)

    rows: list[dict[str, Any]] = []
    for index in range(repetitions):
        path = root / f"measured-{index + 1:02d}.json"
        wrapped = bench.run_timed_sample(
            command_builder(path),
            path,
            root / f"measured-{index + 1:02d}.log",
            root / f"measured-{index + 1:02d}.time.txt",
            {**metadata, "index": index + 1, "warmups_before_series": warmups},
        )
        validator(wrapped["program"])
        rows.append(wrapped)
    return rows


def correctness_gate(
    config: dict[str, Any],
    binary: Path,
    model: Path,
    point: dict[str, Any],
    rankfile: Path,
    root: Path,
) -> dict[str, Any]:
    section = config["correctness"]
    path = root / "correctness.json"
    command = mpi_sample_command(
        binary, model, point, rankfile,
        int(section["limit"]), int(section["batch_size"]), path, "off",
        signature=True,
        test_data=bench.resolve(section["test_data"]),
        test_limit=int(section["test_limit"]),
        report_bindings=True,
    )
    output = bench.run_logged(command, root / "correctness.log")
    binding_lines = [
        line for line in output.splitlines() if "MCW rank " in line and " bound to " in line
    ]
    if len(binding_lines) != int(point["ranks"]) or any("BB" in line for line in binding_lines):
        raise bench.BenchError(
            f"{point['name']}: physical-core affinity verification failed: {binding_lines}"
        )
    bench.json_dump(root / "affinity.json", {
        "configuration": point["name"],
        "physical_core_only": True,
        "binding_lines": binding_lines,
    })
    result = bench.load_json(path)
    validate_program(result, point)
    mismatches = {
        field: {"expected": section["expected"][field], "actual": result.get(field)}
        for field in EXACT_FIELDS
        if result.get(field) != section["expected"][field]
    }
    if mismatches:
        raise bench.BenchError(
            f"{point['name']} exact gate failed: {json.dumps(mismatches, indent=2)}"
        )
    return result


def prepare_baseline(config: dict[str, Any], out: Path) -> tuple[Path, str]:
    tag = str(config["reference"]["baseline_tag"])
    location = Path("/tmp") / f"pcfg-reference-{tag}"
    if location.exists():
        actual = bench.command_output(["git", "rev-parse", "HEAD"], cwd=location).strip()
        expected = bench.command_output(["git", "rev-list", "-n", "1", tag]).strip()
        if actual != expected:
            raise bench.BenchError(f"existing baseline worktree has wrong HEAD: {location}")
    else:
        bench.run_logged(
            ["git", "worktree", "add", "--detach", str(location), tag],
            out / "logs" / "baseline-worktree.log",
        )
    bench.run_logged(["cmake", "--preset", "benchmark"],
                     out / "logs" / "baseline-configure.log", cwd=location)
    bench.run_logged(["cmake", "--build", "--preset", "benchmark"],
                     out / "logs" / "baseline-build.log", cwd=location)
    binary = location / "build" / "benchmark" / "pcfg"
    return binary, bench.sha256_file(binary)


def wrapped_programs(paths: list[Path]) -> list[dict[str, Any]]:
    return [bench.load_json(path)["program"] for path in paths]


def core_values(rows: list[dict[str, Any]]) -> list[float]:
    return [float(row["program"]["timing_seconds"]["core_total"]) for row in rows]


def select_hybrid(calibration: dict[str, list[dict[str, Any]]], margin: float) -> str:
    two = statistics.median(core_values(calibration["2x4"]))
    four = statistics.median(core_values(calibration["4x2"]))
    return "4x2" if four <= two * (1.0 - margin) else "2x4"


def run(args: argparse.Namespace) -> None:
    config_path = bench.resolve(args.config)
    config = load_config(config_path)
    out = bench.resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dirty, status = bench.ensure_clean(bool(config["require_clean"]), args.allow_dirty)
    memory = memory_preflight(config)
    controller_isolation = bench.isolate_controller(all_worker_cpus(config))

    binary, build_identity = bench.configure_and_build(config, out)
    model = bench.resolve(config["paths"]["model"])
    dataset = bench.resolve(config["paths"]["dataset"])
    identity = {
        "schema": "pcfg-mpi-experiment-identity-v1",
        "captured_at_utc": bench.utc_now(),
        "variant": config["variant"],
        "git_sha": bench.git_value("rev-parse", "HEAD"),
        "git_describe": bench.git_value("describe", "--always", "--dirty", "--tags"),
        "dirty": dirty,
        "dirty_status": status,
        "config": str(config_path),
        "config_sha256": bench.sha256_file(config_path),
        "build": build_identity,
        "dataset": {"path": str(dataset), "sha256": bench.verify_hash(
            dataset, config["reference"]["dataset_sha256"], "dataset")},
        "model": {"path": str(model), "sha256": bench.verify_hash(
            model, config["reference"]["model_sha256"], "model"),
                  "expected_records": config["reference"]["model_records"]},
        "environment": bench.environment_identity(),
        "openmpi": bench.command_output(["mpirun", "--version"]).strip(),
        "memory_preflight": memory,
        "controller_isolation": controller_isolation,
        "protocol": config,
    }
    bench.json_dump(out / "manifest" / "identity.json", identity)

    rankfiles = {
        point["name"]: write_rankfile(out, point)
        for point in config["configurations"]
    }
    print(
        f"\n=== exact gates: 4 configurations at 10M; "
        f"estimated {config['estimates_seconds']['correctness']}s ===",
        flush=True,
    )
    gate_fields: dict[str, Any] = {}
    for point in config["configurations"]:
        gate = correctness_gate(
            config, binary, model, point, rankfiles[point["name"]],
            out / "gates" / point["name"],
        )
        gate_fields[point["name"]] = {field: gate[field] for field in EXACT_FIELDS}
    bench.json_dump(out / "manifest" / "correctness-gate.json", {
        "schema": "pcfg-mpi-correctness-gate-v1",
        "passed": True,
        "fields": gate_fields,
    })

    sustained = config["sustained"]
    calibration: dict[str, list[dict[str, Any]]] = {}
    for point in config["configurations"]:
        name = point["name"]
        calibration[name] = timed_series(
            f"100M calibration {name}",
            out / "calibration" / name,
            int(sustained["calibration_warmups"]),
            int(sustained["calibration_repetitions"]),
            float(config["estimates_seconds"]["calibration"]) / len(config["configurations"]),
            lambda path, p=point: mpi_sample_command(
                binary, model, p, rankfiles[p["name"]],
                int(sustained["limit"]), int(sustained["batch_size"]), path, "off"),
            {"variant": config["variant"], "series": "mpi-calibration",
             "configuration": name, "ranks": point["ranks"],
             "threads_per_rank": point["threads_per_rank"],
             "binary_sha256": build_identity["binary_sha256"],
             "git_sha": identity["git_sha"]},
            lambda program, p=point: validate_program(program, p),
        )

    selected = select_hybrid(calibration, float(config["statistics"]["selection_margin"]))
    selection = {
        "selected": selected,
        "rule": "select 4x2 only when its median is at least 3% lower than 2x4; otherwise 2x4",
        "calibration_medians": {
            name: statistics.median(core_values(rows)) for name, rows in calibration.items()
        },
    }
    bench.json_dump(out / "manifest" / "selection.json", selection)
    print(f"\nSelected hybrid configuration: {selected}", flush=True)

    baseline_binary, baseline_hash = prepare_baseline(config, out)
    baseline_point = configuration(config, "1x8")
    baseline_places = ",".join(
        f"{{{cpu}}}" for cpu in sorted(bench.expand_cpu_set(baseline_point["cpu_sets"][0]))
    )
    baseline_rows = timed_series(
        "100M direct OpenMP reference from openmp-fused-v1.0",
        out / "baseline" / "openmp-1x8",
        int(sustained["confirmation_warmups"]),
        int(sustained["confirmation_repetitions"]),
        float(config["estimates_seconds"]["confirmation"]) / 2,
        lambda path: bench.sample_command(
            baseline_binary, model, int(sustained["limit"]), int(sustained["batch_size"]),
            path, baseline_point["cpu_sets"][0], "off",
            threads=8, omp_places=baseline_places),
        {"variant": config["reference"]["baseline_tag"], "series": "sustained-reference",
         "threads": 8, "ranks": 1, "binary_sha256": baseline_hash},
        lambda program: bench.validate_program_threads(program, 8),
    )

    selected_point = configuration(config, selected)
    headline_rows = timed_series(
        f"100M selected hybrid confirmation {selected}",
        out / "headline" / selected,
        int(sustained["confirmation_warmups"]),
        int(sustained["confirmation_repetitions"]),
        float(config["estimates_seconds"]["confirmation"]) / 2,
        lambda path: mpi_sample_command(
            binary, model, selected_point, rankfiles[selected],
            int(sustained["limit"]), int(sustained["batch_size"]), path, "off"),
        {"variant": config["variant"], "series": "sustained-headline",
         "configuration": selected, "ranks": selected_point["ranks"],
         "threads_per_rank": selected_point["threads_per_rank"],
         "binary_sha256": build_identity["binary_sha256"],
         "git_sha": identity["git_sha"]},
        lambda program: validate_program(program, selected_point),
    )

    legacy = config["legacy"]
    legacy_rows = timed_series(
        f"10M legacy exact performance {selected}",
        out / "legacy" / selected,
        int(legacy["warmups"]), int(legacy["repetitions"]),
        float(config["estimates_seconds"]["legacy"]),
        lambda path: mpi_sample_command(
            binary, model, selected_point, rankfiles[selected],
            int(legacy["limit"]), int(legacy["batch_size"]), path, "off"),
        {"variant": config["variant"], "series": "legacy-10m",
         "configuration": selected, "ranks": selected_point["ranks"],
         "threads_per_rank": selected_point["threads_per_rank"],
         "binary_sha256": build_identity["binary_sha256"],
         "git_sha": identity["git_sha"]},
        lambda program: validate_program(program, selected_point),
    )

    baseline_guard = baseline_rows[0]["program"]["hash_guard"]
    for label, rows in {
        **{f"calibration/{name}": values for name, values in calibration.items()},
        "headline": headline_rows,
    }.items():
        for row in rows:
            program = row["program"]
            if program["generated"] != sustained["limit"] or program["hash_guard"] != baseline_guard:
                raise bench.BenchError(f"100M guard mismatch in {label}")
    expected_10m = config["correctness"]["expected"]["hash_guard"]
    if any(row["program"]["hash_guard"] != expected_10m for row in legacy_rows):
        raise bench.BenchError("10M legacy guard mismatch")

    print(f"\nMPI benchmark run complete: {out}", flush=True)


def profile(args: argparse.Namespace) -> None:
    config = load_config(bench.resolve(args.config))
    out = bench.resolve(args.output)
    selection = bench.load_json(out / "manifest" / "selection.json")
    selected = str(selection["selected"])
    binary = bench.resolve(config["paths"]["binary"])
    model = bench.resolve(config["paths"]["model"])
    points = [configuration(config, "1x8"), configuration(config, selected)]
    sustained = config["sustained"]
    for point in points:
        rankfile = write_rankfile(out, point)
        root = out / "profiles" / point["name"]
        metrics = root / "run.json"
        trace = root / "batch-trace.jsonl"
        print(
            f"\n=== detailed profile {point['name']} at 100M; "
            f"estimated {config['estimates_seconds']['profiling'] / 2:.0f}s ===",
            flush=True,
        )
        command = mpi_sample_command(
            binary, model, point, rankfile,
            int(sustained["limit"]), int(sustained["batch_size"]),
            metrics, "detailed", trace=trace,
        )
        wrapped = bench.run_timed_sample(
            command, metrics, root / "run.log", root / "run.time.txt",
            {"variant": config["variant"], "series": "mpi-detailed",
             "configuration": point["name"], "ranks": point["ranks"],
             "threads_per_rank": point["threads_per_rank"]},
        )
        validate_program(wrapped["program"], point)
    bench.json_dump(out / "manifest" / "profile.json", {
        "schema": "pcfg-mpi-profile-v1",
        "captured_at_utc": bench.utc_now(),
        "configurations": [point["name"] for point in points],
        "trace_granularity": "one record per 1M batch",
        "gprof": "not used because multi-rank CPU samples do not represent wall-clock critical path",
    })
    print(f"MPI profiling complete: {out}", flush=True)


def freeze(args: argparse.Namespace) -> None:
    source = bench.resolve(args.input)
    destination = bench.resolve(args.destination)
    if destination.exists() and any(destination.iterdir()):
        raise bench.BenchError(f"reference destination is not empty: {destination}")
    patterns = [
        "manifest/*.json",
        "gates/*/correctness.json",
        "calibration/*/measured-*.json",
        "baseline/*/measured-*.json",
        "headline/*/measured-*.json",
        "legacy/*/measured-*.json",
        "profiles/*/run.json",
        "summary/*",
        "figures/*.pdf",
        "report/*",
    ]
    copied: list[dict[str, str]] = []
    for pattern in patterns:
        for path in sorted(source.glob(pattern)):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append({"path": str(relative), "sha256": bench.sha256_file(target)})
    bench.json_dump(destination / "canonical-manifest.json", {
        "schema": "pcfg-canonical-reference-v1",
        "created_at_utc": bench.utc_now(),
        "source_experiment": str(source),
        "files": copied,
        "excluded": ["model", "RockYou", "logs", "time text", "rankfiles", "batch trace"],
        "rebuild_command": (
            f"python3 tools/bench_mpi.py analyze --input "
            f"{destination.relative_to(ROOT)}"
        ),
    })
    print(f"MPI canonical reference frozen: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench-mpi", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "profile"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="benchmarks/configs/mpi-fused-exact-v1.json")
        command.add_argument("--output", default="results/mpi-fused-exact-v1")
        command.add_argument("--allow-dirty", action="store_true")
        command.set_defaults(handler=run if name == "run" else profile)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--input", default="results/mpi-fused-exact-v1")
    analyze.set_defaults(handler=None)
    frozen = sub.add_parser("freeze")
    frozen.add_argument("--input", default="results/mpi-fused-exact-v1")
    frozen.add_argument("--destination", default="benchmarks/reference/mpi-fused-exact-v1")
    frozen.set_defaults(handler=freeze)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "analyze":
            from bench_mpi_analysis import analyze
            analyze(bench.resolve(args.input))
        else:
            args.handler(args)
        return 0
    except (bench.BenchError, OSError, ValueError, KeyError,
            subprocess.SubprocessError) as error:
        print(f"bench-mpi: ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

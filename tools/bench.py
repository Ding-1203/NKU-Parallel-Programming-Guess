#!/usr/bin/env python3
"""Auditable benchmark orchestration for the PCFG baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
EXACT_FIELDS = (
    "generated",
    "hash_guard",
    "sig1",
    "sig2",
    "candidate_hits",
    "cracked_unique",
)


class BenchError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def command_output(command: list[str], cwd: Path = ROOT, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and result.returncode != 0:
        raise BenchError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout


def run_logged(command: list[str], log_path: Path, cwd: Path = ROOT) -> str:
    print(f"$ {shlex.join(command)}", flush=True)
    output = command_output(command, cwd=cwd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    print(f"  ok -> {log_path}", flush=True)
    return output


def git_value(*args: str) -> str:
    return command_output(["git", *args]).strip()


def git_status() -> str:
    return git_value("status", "--porcelain=v1", "--untracked-files=all")


def ensure_clean(required: bool, allow_dirty: bool) -> tuple[bool, str]:
    status = git_status()
    dirty = bool(status)
    if required and dirty and not allow_dirty:
        raise BenchError(
            "formal benchmark refused: working tree is dirty. "
            "Commit or stash in-scope changes; ignored results do not count.\n" + status
        )
    return dirty, status


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "matplotlib", "seaborn", "jinja2"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def expand_cpu_set(specification: str) -> set[int]:
    cpus: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(value) for value in item.split("-", 1))
            if last < first:
                raise BenchError(f"invalid CPU set: {specification}")
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(item))
    if not cpus:
        raise BenchError(f"empty CPU set: {specification}")
    return cpus


def omp_places_for_cpu_set(specification: str) -> str:
    return ",".join(f"{{{cpu}}}" for cpu in sorted(expand_cpu_set(specification)))


def thread_settings(section: dict[str, Any]) -> tuple[int, str, str]:
    threads = int(section.get("threads", 1))
    cpu_set = str(section["cpu_set"])
    cpus = expand_cpu_set(cpu_set)
    if threads < 1:
        raise BenchError("threads must be positive")
    if len(cpus) != threads:
        raise BenchError(
            f"thread/CPU mismatch: requested {threads} threads for CPUs {sorted(cpus)}"
        )
    places = str(section.get("omp_places", omp_places_for_cpu_set(cpu_set)))
    return threads, cpu_set, places


def isolate_controller(target_specification: str) -> dict[str, Any]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return {"status": "unsupported"}
    targets = expand_cpu_set(target_specification)
    protected = set(targets)
    sibling_sources: dict[str, str] = {}
    for cpu in targets:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        if path.is_file():
            specification = path.read_text(encoding="utf-8").strip()
            sibling_sources[str(cpu)] = specification
            protected.update(expand_cpu_set(specification))
    before = set(os.sched_getaffinity(0))
    after = before - protected
    status = "isolated"
    if not after:
        after = before - targets
        status = "smt-shared-fallback"
    if not after:
        after = before
        status = "controller-sharing-unavoidable"
    os.sched_setaffinity(0, after)
    return {
        "status": status,
        "before": sorted(before),
        "after": sorted(os.sched_getaffinity(0)),
        "protected_target_and_siblings": sorted(protected),
        "sibling_sources": sibling_sources,
    }

def environment_identity() -> dict[str, Any]:
    commands = {
        "lscpu": ["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ"],
        "lscpu_summary": ["lscpu"],
        "memory": ["free", "-b"],
        "kernel": ["uname", "-a"],
        "compiler": ["c++", "--version"],
        "cmake": ["cmake", "--version"],
        "python": [sys.executable, "--version"],
    }
    captured = {name: command_output(cmd, check=False).strip() for name, cmd in commands.items()}
    return {
        "captured_at_utc": utc_now(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "process_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "commands": captured,
        "dependency_versions": package_versions(),
        "tools": {
            name: shutil.which(name) or "unavailable"
            for name in ("taskset", "gprof", "strace", "perf", "/usr/bin/time")
        },
    }


def verify_hash(path: Path, expected: str | None, label: str) -> str:
    if not path.is_file():
        raise BenchError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if expected and actual != expected:
        raise BenchError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def configure_and_build(config: dict[str, Any], out: Path) -> tuple[Path, dict[str, Any]]:
    configure = config["configure_preset"]
    build = config["build_preset"]
    run_logged(["cmake", "--preset", configure], out / "logs" / "configure.log")
    run_logged(["cmake", "--build", "--preset", build], out / "logs" / "build.log")
    run_logged(["ctest", "--preset", configure], out / "logs" / "ctest.log")
    binary = resolve(config["paths"]["binary"])
    binary_hash = verify_hash(binary, config.get("reference", {}).get("binary_sha256"), "binary")

    build_dir = resolve(f"build/{configure}")
    manifest_dir = out / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, str]] = {}
    for name in ("CMakeCache.txt", "compile_commands.json"):
        source = build_dir / name
        if source.is_file():
            target = manifest_dir / name
            shutil.copy2(source, target)
            copied[name] = {"sha256": sha256_file(target), "path": str(target.relative_to(out))}
    return binary, {
        "configure_preset": configure,
        "build_preset": build,
        "binary": str(binary),
        "binary_sha256": binary_hash,
        "build_artifacts": copied,
    }


def prepare_model(config: dict[str, Any], binary: Path, out: Path) -> Path:
    paths = config["paths"]
    if "model" in paths:
        return resolve(paths["model"])
    source = resolve(paths["training_input"])
    if not source.is_file():
        raise BenchError(f"smoke training input missing: {source}")
    model = out / "artifacts" / "smoke.pcfg"
    metrics = out / "manifest" / "smoke-training.json"
    command = [
        str(binary), "train", "--input", str(source), "--output", str(model),
        "--train-limit", "0", "--metrics-out", str(metrics),
    ]
    run_logged(command, out / "logs" / "smoke-training.log")
    return model


def sample_command(
    binary: Path,
    model: Path,
    limit: int,
    batch_size: int,
    metrics_path: Path,
    cpu_set: str,
    instrumentation: str,
    signature: bool = False,
    test_data: Path | None = None,
    test_limit: int = 0,
    trace_path: Path | None = None,
    trace_interval: int = 100000,
    threads: int = 1,
    omp_places: str | None = None,
    display_affinity: bool = False,
) -> list[str]:
    places = omp_places or omp_places_for_cpu_set(cpu_set)
    command = [
        "env",
        f"OMP_NUM_THREADS={threads}",
        "OMP_DYNAMIC=FALSE",
        "OMP_PROC_BIND=spread",
        f"OMP_PLACES={places}",
        f"OMP_DISPLAY_AFFINITY={'TRUE' if display_affinity else 'FALSE'}",
        "taskset", "-c", cpu_set, str(binary), "sample",
        "--model", str(model),
        "--limit", str(limit),
        "--batch-size", str(batch_size),
        "--instrumentation", instrumentation,
        "--threads", str(threads),
        "--metrics-out", str(metrics_path),
    ]
    if signature:
        command.append("--signature")
    if test_data:
        command.extend(["--test-data", str(test_data), "--test-limit", str(test_limit)])
    if trace_path:
        command.extend(["--trace-out", str(trace_path), "--trace-interval", str(trace_interval)])
    return command


def validate_program_threads(program: dict[str, Any], requested: int) -> None:
    actual_requested = int(program.get("threads_requested", 1))
    actual_used = int(program.get("threads_used", 1))
    if actual_requested != requested or actual_used != requested:
        raise BenchError(
            f"OpenMP thread mismatch: requested {requested}, "
            f"program reported requested={actual_requested}, used={actual_used}"
        )


def run_timed_sample(
    command: list[str],
    metrics_path: Path,
    log_path: Path,
    time_path: Path,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    time_path.parent.mkdir(parents=True, exist_ok=True)
    timed = ["/usr/bin/time", "-v", "-o", str(time_path), *command]
    run_logged(timed, log_path)
    program = load_json(metrics_path)
    resources = parse_time_v(time_path.read_text(encoding="utf-8"))
    wrapped = {
        "schema": "pcfg-benchmark-run-v1",
        "captured_at_utc": utc_now(),
        "run": run_metadata,
        "command": command,
        "program": program,
        "resources": resources,
    }
    json_dump(metrics_path, wrapped)
    return wrapped


def parse_time_v(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if re.fullmatch(r"-?\d+", value):
            values[normalized] = int(value)
        elif re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            values[normalized] = float(value)
        else:
            values[normalized] = value
    return values


def correctness_gate(
    config: dict[str, Any],
    binary: Path,
    model: Path,
    out: Path,
    cpu_set: str,
    threads: int = 1,
    omp_places: str | None = None,
    path: Path | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    section = config["correctness"]
    path = path or out / "runs" / "correctness.json"
    log_path = log_path or out / "logs" / "correctness.log"
    test_data = resolve(section["test_data"]) if section.get("test_data") else None
    command = sample_command(
        binary, model, int(section["limit"]), int(section["batch_size"]), path,
        cpu_set, "off", signature=True, test_data=test_data,
        test_limit=int(section.get("test_limit", 0)),
        threads=threads, omp_places=omp_places, display_affinity=True,
    )
    run_logged(command, log_path)
    result = load_json(path)
    validate_program_threads(result, threads)
    expected_records = config.get("reference", {}).get("model_records")
    if expected_records is not None and result.get("model_records") != expected_records:
        raise BenchError(
            f"model record mismatch: expected {expected_records}, got {result.get('model_records')}"
        )
    expected = section.get("expected")
    if expected:
        mismatches = {
            field: {"expected": expected[field], "actual": result[field]}
            for field in EXACT_FIELDS
            if field in expected and result.get(field) != expected[field]
        }
        if mismatches:
            raise BenchError("correctness gate failed: " + json.dumps(mismatches, indent=2))
    else:
        second_path = out / "runs" / "correctness-repeat.json"
        second_command = sample_command(
            binary, model, int(section["limit"]), max(1, int(section["batch_size"]) // 2),
            second_path, cpu_set, "off", signature=True,
            threads=threads, omp_places=omp_places,
        )
        run_logged(second_command, out / "logs" / "correctness-repeat.log")
        second = load_json(second_path)
        validate_program_threads(second, threads)
        mismatches = {
            field: {"first": result.get(field), "second": second.get(field)}
            for field in EXACT_FIELDS if result.get(field) != second.get(field)
        }
        if mismatches:
            raise BenchError("smoke exactness gate failed: " + json.dumps(mismatches, indent=2))
    return result


def make_identity(config_path: Path, config: dict[str, Any], allow_dirty: bool) -> dict[str, Any]:
    dirty, status = ensure_clean(bool(config.get("require_clean", True)), allow_dirty)
    headline_threads, requested_cpu, omp_places = thread_settings(config["headline"])
    controller_isolation = isolate_controller(requested_cpu)
    verified_affinity = command_output([
        "taskset", "-c", requested_cpu, sys.executable, "-c",
        "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0)))))",
    ]).strip()
    identity = {
        "schema": "pcfg-reference-identity-v1",
        "captured_at_utc": utc_now(),
        "variant": config["variant"],
        "comparison_class": config["comparison_class"],
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_describe": git_value("describe", "--always", "--dirty", "--tags"),
        "git_dirty": dirty,
        "git_status": status.splitlines(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "environment": environment_identity(),
        "controller_isolation": controller_isolation,
        "protocol": {
            "headline": config["headline"],
            "scaling": config.get("scaling"),
            "parallelism": {
                "threads": headline_threads,
                "processes": 1,
                "omp_places": omp_places,
            },
        },
        "requested_headline_affinity": requested_cpu,
        "verified_headline_affinity": verified_affinity,
    }
    return identity


def run_benchmark(args: argparse.Namespace) -> None:
    config_path = resolve(args.config)
    config = load_json(config_path)
    out = resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = make_identity(config_path, config, args.allow_dirty)

    reference = config.get("reference", {})
    paths = config["paths"]
    if "dataset" in paths:
        dataset = resolve(paths["dataset"])
        identity["dataset"] = {
            "path": str(dataset),
            "sha256": verify_hash(dataset, reference.get("dataset_sha256"), "dataset"),
            "bytes": dataset.stat().st_size,
        }

    binary, build_identity = configure_and_build(config, out)
    identity["build"] = build_identity
    model = prepare_model(config, binary, out)
    identity["model"] = {
        "path": str(model),
        "sha256": verify_hash(model, reference.get("model_sha256"), "model"),
        "bytes": model.stat().st_size,
        "expected_records": reference.get("model_records"),
    }
    json_dump(out / "manifest" / "identity.json", identity)

    headline = config["headline"]
    threads, cpu_set, omp_places = thread_settings(headline)
    correctness = correctness_gate(
        config, binary, model, out, cpu_set, threads, omp_places
    )
    gate = {
        "schema": "pcfg-correctness-gate-v1",
        "comparison_class": config["comparison_class"],
        "fields": {field: correctness[field] for field in EXACT_FIELDS},
        "passed": True,
    }
    json_dump(out / "manifest" / "correctness-gate.json", gate)

    limit = int(headline["limit"])
    batch_size = int(headline["batch_size"])
    for index in range(int(headline["warmups"])):
        path = out / "runs" / f"warmup-{index + 1:02d}.json"
        command = sample_command(
            binary, model, limit, batch_size, path, cpu_set, "off",
            threads=threads, omp_places=omp_places,
        )
        run_logged(command, out / "logs" / f"warmup-{index + 1:02d}.log")
        validate_program_threads(load_json(path), threads)
        path.unlink(missing_ok=True)

    for index in range(int(headline["repetitions"])):
        metrics = out / "runs" / f"measured-{index + 1:02d}.json"
        command = sample_command(
            binary, model, limit, batch_size, metrics, cpu_set, "off",
            threads=threads, omp_places=omp_places,
        )
        wrapped = run_timed_sample(
            command,
            metrics,
            out / "logs" / f"measured-{index + 1:02d}.log",
            out / "runs" / f"measured-{index + 1:02d}.time.txt",
            {
                "variant": config["variant"],
                "comparison_class": config["comparison_class"],
                "index": index + 1,
                "warmups_before_series": int(headline["warmups"]),
                "binary_sha256": build_identity["binary_sha256"],
                "git_sha": identity["git_sha"],
                "cpu_set": cpu_set,
                "threads": threads,
                "omp_places": omp_places,
            },
        )
        validate_program_threads(wrapped["program"], threads)

    scaling = config.get("scaling")
    if scaling:
        for point in scaling["points"]:
            point_threads, point_cpus, point_places = thread_settings(point)
            if point_threads == threads:
                continue
            point_root = out / "scaling" / f"thread-{point_threads}"
            point_correctness = correctness_gate(
                config, binary, model, out, point_cpus, point_threads, point_places,
                point_root / "correctness.json", point_root / "correctness.log",
            )
            for field in EXACT_FIELDS:
                if point_correctness.get(field) != correctness.get(field):
                    raise BenchError(
                        f"thread-{point_threads} exact mismatch for {field}: "
                        f"{point_correctness.get(field)} != {correctness.get(field)}"
                    )
            for index in range(int(scaling["warmups"])):
                path = point_root / f"warmup-{index + 1:02d}.json"
                command = sample_command(
                    binary, model, limit, batch_size, path, point_cpus, "off",
                    threads=point_threads, omp_places=point_places,
                )
                run_logged(command, point_root / f"warmup-{index + 1:02d}.log")
                validate_program_threads(load_json(path), point_threads)
                path.unlink(missing_ok=True)
            for index in range(int(scaling["repetitions"])):
                metrics = point_root / f"measured-{index + 1:02d}.json"
                command = sample_command(
                    binary, model, limit, batch_size, metrics, point_cpus, "off",
                    threads=point_threads, omp_places=point_places,
                )
                wrapped = run_timed_sample(
                    command, metrics,
                    point_root / f"measured-{index + 1:02d}.log",
                    point_root / f"measured-{index + 1:02d}.time.txt",
                    {
                        "variant": config["variant"],
                        "comparison_class": config["comparison_class"],
                        "series": "thread-scaling",
                        "index": index + 1,
                        "warmups_before_series": int(scaling["warmups"]),
                        "binary_sha256": build_identity["binary_sha256"],
                        "git_sha": identity["git_sha"],
                        "cpu_set": point_cpus,
                        "threads": point_threads,
                        "omp_places": point_places,
                    },
                )
                validate_program_threads(wrapped["program"], point_threads)
    print(f"benchmark run complete: {out}")


def run_profile_sample(
    binary: Path,
    model: Path,
    cpu_set: str,
    limit: int,
    batch_size: int,
    metrics: Path,
    log: Path,
    trace: Path | None,
    trace_interval: int,
    threads: int = 1,
    omp_places: str | None = None,
) -> dict[str, Any]:
    command = sample_command(
        binary, model, limit, batch_size, metrics, cpu_set, "detailed",
        trace_path=trace, trace_interval=trace_interval,
        threads=threads, omp_places=omp_places,
    )
    run_logged(command, log)
    result = load_json(metrics)
    validate_program_threads(result, threads)
    return result


def parse_gprof_flat(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+"
        r"(?:(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+)?(.+?)\s*$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        name = match.group(7)
        if name.startswith("[") or "seconds" in name:
            continue
        rows.append({
            "percent_time": float(match.group(1)),
            "cumulative_seconds": float(match.group(2)),
            "self_seconds": float(match.group(3)),
            "calls": int(match.group(4)) if match.group(4) else None,
            "self_ms_per_call": float(match.group(5)) if match.group(5) else None,
            "total_ms_per_call": float(match.group(6)) if match.group(6) else None,
            "name": name,
        })
        if len(rows) >= 30:
            break
    return rows


def profile_benchmark(args: argparse.Namespace) -> None:
    config_path = resolve(args.config)
    config = load_json(config_path)
    out = resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = make_identity(config_path, config, args.allow_dirty)
    binary, build_identity = configure_and_build(config, out)
    model = prepare_model(config, binary, out)
    reference = config.get("reference", {})
    verify_hash(model, reference.get("model_sha256"), "model")
    profile = config["profiling"]
    headline_threads, cpu_set, headline_places = thread_settings(config["headline"])
    trace_interval = int(profile["trace_interval"])

    scale_root = out / "profiles" / "scale"
    for limit in profile["scale_limits"]:
        limit = int(limit)
        warmup_path = scale_root / f"limit-{limit}" / "warmup.json"
        run_profile_sample(
            binary, model, cpu_set, limit, int(config["headline"]["batch_size"]),
            warmup_path, scale_root / f"limit-{limit}" / "warmup.log", None, trace_interval,
        )
        warmup_path.unlink(missing_ok=True)
        for index in range(int(profile["scale_repetitions"])):
            metrics = scale_root / f"limit-{limit}" / f"run-{index + 1:02d}.json"
            trace = scale_root / f"limit-{limit}" / "trace.jsonl" if index == 0 else None
            run_profile_sample(
                binary, model, cpu_set, limit, int(config["headline"]["batch_size"]),
                metrics, scale_root / f"limit-{limit}" / f"run-{index + 1:02d}.log",
                trace, trace_interval,
            )

    batch_root = out / "profiles" / "batch"
    for batch_size in profile["batch_sizes"]:
        batch_size = int(batch_size)
        for index in range(int(profile["batch_repetitions"])):
            metrics = batch_root / f"batch-{batch_size}" / f"run-{index + 1:02d}.json"
            run_profile_sample(
                binary, model, cpu_set, int(config["headline"]["limit"]), batch_size,
                metrics, batch_root / f"batch-{batch_size}" / f"run-{index + 1:02d}.log",
                None, trace_interval,
            )

    scaling_points = {
        int(point.get("threads", 1)): point
        for point in config.get("scaling", {}).get("points", [config["headline"]])
    }
    for requested in profile.get("thread_details", []):
        point = scaling_points[int(requested)]
        point_threads, point_cpus, point_places = thread_settings(point)
        root = out / "profiles" / "threads" / f"thread-{point_threads}"
        metrics = root / "run.json"
        trace = root / "trace.jsonl" if profile.get("thread_trace", True) else None
        command = sample_command(
            binary, model, int(config["headline"]["limit"]),
            int(config["headline"]["batch_size"]), metrics, point_cpus, "detailed",
            trace_path=trace, trace_interval=trace_interval,
            threads=point_threads, omp_places=point_places,
        )
        wrapped = run_timed_sample(
            command, metrics, root / "run.log", root / "run.time.txt",
            {"variant": config["variant"], "series": "thread-detailed",
             "threads": point_threads, "cpu_set": point_cpus},
        )
        validate_program_threads(wrapped["program"], point_threads)

    profile_preset = config["profile_preset"]
    run_logged(["cmake", "--preset", profile_preset], out / "logs" / "gprof-configure.log")
    run_logged(["cmake", "--build", "--preset", profile_preset], out / "logs" / "gprof-build.log")
    profile_binary = resolve(config["paths"]["profile_binary"])
    gprof_dir = out / "profiles" / "gprof"
    gprof_dir.mkdir(parents=True, exist_ok=True)
    gprof_metrics = gprof_dir / "run.json"
    gprof_threads = int(profile.get("gprof_threads", 1))
    gprof_point = scaling_points.get(gprof_threads, config["headline"])
    _, gprof_cpus, gprof_places = thread_settings(gprof_point)
    gprof_command = sample_command(
        profile_binary, model, int(profile["gprof_limit"]),
        int(config["headline"]["batch_size"]), gprof_metrics, gprof_cpus, "off",
        threads=gprof_threads, omp_places=gprof_places,
    )
    run_logged(gprof_command, gprof_dir / "run.log", cwd=gprof_dir)
    gmon = gprof_dir / "gmon.out"
    if not gmon.is_file():
        raise BenchError("gprof run did not produce gmon.out")
    gprof_text = command_output(["gprof", str(profile_binary), str(gmon)])
    (gprof_dir / "gprof.txt").write_text(gprof_text, encoding="utf-8")
    profiler_summary: dict[str, Any] = {
        "schema": "pcfg-profiler-summary-v1",
        "gprof": {
            "binary_sha256": sha256_file(profile_binary),
            "limit": int(profile["gprof_limit"]),
            "top_functions": parse_gprof_flat(gprof_text),
        },
        "perf": {"status": "unavailable"},
        "strace": {"status": "disabled"},
    }

    perf_path = shutil.which("perf")
    if perf_path:
        probe = subprocess.run([perf_path, "stat", "true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        profiler_summary["perf"] = {
            "status": "available" if probe.returncode == 0 else "permission-denied",
            "probe_returncode": probe.returncode,
            "probe_stderr": probe.stderr.decode(errors="replace"),
        }

    if profile.get("strace") and shutil.which("strace"):
        strace_path = out / "profiles" / "strace-summary.txt"
        strace_metrics = out / "profiles" / "strace-run.json"
        command = sample_command(
            binary, model, int(profile["strace_limit"]), int(config["headline"]["batch_size"]),
            strace_metrics, cpu_set, "off",
        )
        traced = ["strace", "-c", "-f", "-o", str(strace_path), *command]
        run_logged(traced, out / "profiles" / "strace-run.log")
        profiler_summary["strace"] = {
            "status": "captured",
            "limit": int(profile["strace_limit"]),
            "summary": str(strace_path.relative_to(out)),
        }

    json_dump(out / "profiles" / "profiler.json", profiler_summary)
    profile_manifest = {
        "schema": "pcfg-profile-manifest-v1",
        "captured_at_utc": utc_now(),
        "git_sha": identity["git_sha"],
        "git_dirty": identity["git_dirty"],
        "binary_sha256": build_identity["binary_sha256"],
        "model_sha256": sha256_file(model),
        "protocol": profile,
    }
    json_dump(out / "manifest" / "profile.json", profile_manifest)
    print(f"profiling complete: {out}")


def freeze_reference(args: argparse.Namespace) -> None:
    source = resolve(args.input)
    destination = resolve(args.destination)
    if destination.exists() and any(destination.iterdir()):
        raise BenchError(f"reference destination is not empty: {destination}")
    selected = [
        "manifest/identity.json",
        "manifest/correctness-gate.json",
        "manifest/profile.json",
        "manifest/CMakeCache.txt",
        "manifest/compile_commands.json",
        "runs/correctness.json",
        "summary/summary.json",
        "summary/baseline-comparison.json",
        "summary/direct-comparisons.json",
        "summary/runs.csv",
        "summary/profiles.csv",
        "summary/thread-scaling.csv",
        "summary/thread-scaling.json",
        "profiles/profiler.json",
        "profiles/isa-check.json",
        "report/baseline-bottleneck.md",
        "report/baseline-bottleneck.html",
        "report/visualization-manifest.json",
    ]
    selected.extend(str(path.relative_to(source)) for path in sorted((source / "runs").glob("measured-*.json")))
    selected.extend(str(path.relative_to(source)) for path in sorted((source / "scaling").glob("thread-*/measured-*.json")))
    selected.extend(str(path.relative_to(source)) for path in sorted((source / "scaling").glob("thread-*/correctness.json")))
    selected.extend(str(path.relative_to(source)) for path in sorted((source / "profiles" / "threads").glob("thread-*/run.json")))
    for extension in ("pdf",):
        selected.extend(str(path.relative_to(source)) for path in sorted((source / "figures").glob(f"*.{extension}")))
    copied: list[dict[str, str]] = []
    for relative in selected:
        src = source / relative
        if not src.is_file():
            continue
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"path": relative, "sha256": sha256_file(dst)})
    direct_path = source / "summary" / "direct-comparisons.json"
    compare_args = ""
    if direct_path.is_file():
        direct = load_json(direct_path)
        compare_args = "".join(
            f" --compare benchmarks/reference/{item['baseline']}"
            for item in direct.get("comparisons", [])
        )
    manifest = {
        "schema": "pcfg-canonical-reference-v1",
        "created_at_utc": utc_now(),
        "source_experiment": str(source),
        "files": copied,
        "excluded": ["models", "RockYou", "full logs", "gmon.out", "gprof text", "trace JSONL"],
        "rebuild_command": f"./scripts/bench analyze --input {destination.relative_to(ROOT)}{compare_args}",
    }
    json_dump(destination / "canonical-manifest.json", manifest)
    print(f"canonical reference frozen: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="build, gate correctness, warm up, and measure")
    run.add_argument("--config", default="benchmarks/configs/baseline.json")
    run.add_argument("--output", default="results/baseline-v1")
    run.add_argument("--allow-dirty", action="store_true")
    run.set_defaults(handler=run_benchmark)

    profile = sub.add_parser("profile", help="collect detailed, gprof, and syscall profiles")
    profile.add_argument("--config", default="benchmarks/configs/baseline.json")
    profile.add_argument("--output", default="results/baseline-v1")
    profile.add_argument("--allow-dirty", action="store_true")
    profile.set_defaults(handler=profile_benchmark)

    analyze = sub.add_parser("analyze", help="aggregate runs and generate figures/reports")
    analyze.add_argument("--input", default="results/baseline-v1")
    analyze.add_argument("--compare", action="append", default=[], help="additional exact experiment directory")
    analyze.set_defaults(handler=None)

    freeze = sub.add_parser("freeze", help="copy the slim canonical reference into Git")
    freeze.add_argument("--input", default="results/baseline-v1")
    freeze.add_argument("--destination", default="benchmarks/reference/baseline-v1")
    freeze.set_defaults(handler=freeze_reference)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "analyze":
            from bench_analysis import analyze_experiment
            analyze_experiment(resolve(args.input), [resolve(path) for path in args.compare])
        else:
            args.handler(args)
        return 0
    except (BenchError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"bench: ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

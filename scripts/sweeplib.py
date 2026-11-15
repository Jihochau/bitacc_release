"""Shared library of the BitAcc experiment sweep (scripts/run_sweep.py, scripts/parse_results.py).

Experiment plug-in interface
----------------------------
Every experiment is one module ``scripts/experiments/<name>.py``.  The launcher and the
parser discover these modules automatically; a module must define:

``NAME: str``
    The experiment name used on the command line.  Must equal the file stem.

``cases(args: argparse.Namespace) -> list[Case]``
    The simulations of the experiment, in a deterministic order.  ``args.models`` is the
    list given to ``--models`` (or ``None``) and ``args.kv`` is the list of integers given
    to ``--kv`` (or ``None``); use :func:`select_models` and :func:`select_kv` to apply
    them.  A module may return cases that belong to another experiment (for example the
    unmodified design of ``context``) by setting ``Case.experiment`` accordingly: such a
    shared case is simulated once and stored under the experiment that owns it.

``parse(runs_dir: pathlib.Path, csv_dir: pathlib.Path) -> list[pathlib.Path]``
    Read the finished runs with :func:`load_results`, write the CSV files of the
    experiment into ``csv_dir`` with :func:`write_csv`, and return the written paths.
    Rows that cannot be computed yet (a run or one of its references is missing) are
    omitted, never filled with placeholders.  Aggregate rows (e.g. geometric means) are
    written only when all of their inputs are present.

``CSV_KEYS: dict[str, tuple[str, ...]]``
    For every CSV file name written by ``parse``, the columns that identify a row.
    ``parse_results.py --check`` uses them to align a CSV with ``expected/paper/<name>``.

``trace_statistics(trace: pathlib.Path, final_cycle: int) -> Any`` (optional)
    An experiment whose analysis needs the command trace defines this function.  It is
    called once for every case returned by the ``cases()`` of the module (without
    ``--models``/``--kv``), right after the simulation and before the trace is deleted,
    with the path of the command trace and the final cycle.  The returned (small, JSON
    serializable) value is stored in ``result.json`` under ``trace_statistics[NAME]``.  A
    case whose result lacks these statistics counts as not complete and is simulated again.

``run_case(case, *, runs_dir, keep_traces, timeout_s, revision, stop) -> dict`` (optional)
    Replaces :func:`run_case` for the cases owned by the experiment, for results that do
    not come from ``generate_trace.py``.  It must write ``result.json`` atomically with at
    least ``schema_version``, ``status``, the case definition, ``final_cycle`` and
    ``wall_time_s``, and return it.

A :class:`Case` is one simulation: a model configuration, a KV history, and configuration
overrides.  ``overrides`` has the schema of the ``--config`` file of ``generate_trace.py``:
the sections ``hardware`` and ``ramulator`` are merged over ``config/default.json`` and the
section ``pim_runtime`` overrides keys of ``config/pim/bitacc_runtime_default.json``.  The
shared configuration files are never modified; each run reads its own ``config.json``.

Run directory layout (``results/runs/<experiment>/<case_id>/``)
----------------------------------------------------------------
``result.json``         Written atomically after a successful run; its presence (with a
                        matching case definition) marks the case as complete.  Fields:
                        ``final_cycle``, ``latency_ns``, ``tokens_per_s``, ``model``,
                        ``kv_history``, ``overrides``, ``labels``, ``dram_org``,
                        ``command_counts``, ``cycle_breakdown``, ``trace``,
                        ``trace_statistics``, ``wall_time_s``, ``peak_rss_bytes``,
                        ``git_commit``, ...
``trace_summary.json``  Full statistics of the trace generator and the simulator (per
                        command cycles, runtime inputs, command counters); read it with
                        :func:`load_trace_summary`.
``config.json``         The overrides of the case, as passed to ``generate_trace.py``.
``stdout.log``, ``stderr.log``  Output of ``generate_trace.py``.
``trace_out.ch0``       The command trace; kept only with ``--keep-traces`` (as are the
                        simulator input traces ``trace_out.trace``, ``ramulator_trace.txt``).
``trace_out.yaml``, ``trace_out_run.out``, ``trace_out_cmd_counter.txt``
                        Simulator configuration, log, and command counters (small).

Peak memory is recorded only if ``psutil`` is installed.  On Windows it is the peak working
set reported by the OS; elsewhere it is the largest resident set size seen while sampling
every ``POLL_INTERVAL_S`` seconds.  ``peak_rss_bytes`` is that of the simulator
(``ramulator2``), ``peak_rss_generator_bytes`` that of the Python trace generator.
"""

from __future__ import annotations

import argparse
import csv
import functools
import importlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # Optional: only used to record the peak memory of a simulation.
    import psutil
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
EXPERIMENTS_DIR = SCRIPTS_DIR / "experiments"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
CSV_DIR = RESULTS_DIR / "csv"
BASELINES_DIR = ROOT / "baselines"
EXPECTED_DIR = ROOT / "expected" / "paper"
DEFAULT_CONFIG = ROOT / "config" / "default.json"

RESULT_SCHEMA_VERSION = 1
RESULT_NAME = "result.json"
SUMMARY_NAME = "trace_summary.json"
GENERATOR_SUMMARY_NAME = "latest_command_trace_summary.json"
COMMAND_TRACE_NAME = "trace_out.ch0"
TRACE_FILE_NAMES = (COMMAND_TRACE_NAME, "trace_out.trace", "ramulator_trace.txt")
DEFAULT_TIMEOUT_S = 24 * 3600
POLL_INTERVAL_S = 0.2

# The twelve model configurations of the paper, smallest first.
MODELS: tuple[str, ...] = (
    "125M", "350M", "700M", "1B", "1.5B", "2.5B",
    "3.8B", "7B", "13B", "30B", "70B", "100B",
)  # fmt: skip
# The KV histories (number of cached tokens) of the paper.
KV_HISTORIES: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048)
GEOMEAN_LABEL = "GeoMean"

# Signature of the optional ``trace_statistics`` function of an experiment module.
TraceAnalyzer = Callable[[Path, int], Any]


@dataclass(frozen=True)
class Case:
    """One simulation of one experiment."""

    experiment: str
    case_id: str
    model: str
    kv_history: int
    overrides: Mapping[str, Any] = field(default_factory=dict)
    labels: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.experiment, self.case_id)

    def run_dir(self, runs_dir: Path = RUNS_DIR) -> Path:
        return runs_dir / self.experiment / self.case_id

    def definition(self) -> dict[str, Any]:
        """The fields that identify the simulated configuration."""
        return {
            "experiment": self.experiment,
            "case_id": self.case_id,
            "model": self.model,
            "kv_history": int(self.kv_history),
            "overrides": _plain(self.overrides),
            "labels": _plain(self.labels),
        }


def base_case(model: str, kv_history: int) -> Case:
    """The unmodified BitAcc design for one model and KV history.

    These cases belong to the ``context`` experiment; other experiments return them from
    ``cases()`` to use the same simulation as their reference point.
    """
    return Case("context", f"{model_slug(model)}_kv{int(kv_history)}", model, int(kv_history))


def format_overrides(overrides: Mapping[str, Any]) -> str:
    """Compact, deterministic text of the overrides of a case."""
    return json.dumps(_plain(overrides), sort_keys=True, separators=(",", ":"))


class CaseFailed(RuntimeError):
    """A simulation did not produce a valid result."""


class SweepInterrupted(RuntimeError):
    """The sweep was asked to stop while a simulation was running."""


# --------------------------------------------------------------------------------------
# Models and case selection
# --------------------------------------------------------------------------------------


def model_name(label: str) -> str:
    """Name of a model in config/models.json, e.g. '7B' -> 'BITNET-7B'."""
    return f"BITNET-{label}"


def model_slug(label: str) -> str:
    """File-system friendly model label, e.g. '1.5B' -> '1p5b'."""
    return label.lower().replace(".", "p")


def model_label(value: str) -> str:
    """Normalize a user-supplied model ('7b', 'BITNET-7B', 'bitnet_1p5b') to its label."""
    text = value.strip().upper().replace("_", "-")
    if text.startswith("BITNET-"):
        text = text[len("BITNET-") :]
    for label in MODELS:
        if text in (label, model_slug(label).upper()):
            return label
    raise ValueError(f"unknown model {value!r}; available models: {', '.join(MODELS)}")


def select_models(args: argparse.Namespace, defaults: Sequence[str] = MODELS) -> list[str]:
    """Models of an experiment after applying ``--models`` (a filter on ``defaults``)."""
    requested = getattr(args, "models", None)
    if not requested:
        return list(defaults)
    wanted = {model_label(value) for value in requested}
    return [label for label in defaults if label in wanted]


def select_kv(
    args: argparse.Namespace,
    defaults: Sequence[int] = KV_HISTORIES,
    *,
    allow_other: bool = False,
) -> list[int]:
    """KV histories of an experiment after applying ``--kv``.

    ``--kv`` filters ``defaults``; with ``allow_other`` it may also name histories outside
    of ``defaults`` (for experiments that are defined for any history).
    """
    requested = getattr(args, "kv", None)
    if not requested:
        return list(defaults)
    wanted = sorted({int(value) for value in requested})
    if any(value <= 0 for value in wanted):
        raise ValueError("--kv values must be positive")
    return [value for value in wanted if allow_other or value in defaults]


# --------------------------------------------------------------------------------------
# Experiment discovery
# --------------------------------------------------------------------------------------


def discover_experiments() -> dict[str, ModuleType]:
    """Import every module in scripts/experiments/ and return them by name, sorted."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    modules: dict[str, ModuleType] = {}
    for path in sorted(EXPERIMENTS_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        module = importlib.import_module(f"experiments.{path.stem}")
        for attribute in ("NAME", "cases", "parse", "CSV_KEYS"):
            if not hasattr(module, attribute):
                raise AttributeError(f"{path} does not define {attribute}")
        if module.NAME != path.stem:
            raise ValueError(f"{path}: NAME is {module.NAME!r}, expected {path.stem!r}")
        modules[module.NAME] = module
    return modules


@functools.lru_cache(maxsize=None)
def trace_analyzers() -> dict[tuple[str, str], dict[str, TraceAnalyzer]]:
    """The ``trace_statistics`` functions to apply to a case: case key -> experiment -> function."""
    every_case = argparse.Namespace(models=None, kv=None)
    analyzers: dict[tuple[str, str], dict[str, TraceAnalyzer]] = {}
    for name, module in discover_experiments().items():
        analyze = getattr(module, "trace_statistics", None)
        if analyze is None:
            continue
        for case in module.cases(every_case):
            analyzers.setdefault(case.key, {})[name] = analyze
    return analyzers


def unique_cases(cases: Iterable[Case]) -> list[Case]:
    """Drop repeated cases (shared between experiments), keeping the first occurrence."""
    seen: dict[tuple[str, str], Case] = {}
    for case in cases:
        previous = seen.setdefault(case.key, case)
        if previous.definition() != case.definition():
            raise ValueError(
                f"case {case.experiment}/{case.case_id} is defined twice with different settings"
            )
    return list(seen.values())


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Normalize nested mappings/sequences to what a JSON round trip returns."""
    return json.loads(json.dumps(value, sort_keys=True))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write a JSON file so that readers never observe a partial file."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_result(run_dir: Path) -> dict[str, Any] | None:
    """Return the result of a finished run directory, or None if it has no valid result."""
    path = run_dir / RESULT_NAME
    if not path.is_file():
        return None
    try:
        result = _read_json(path)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    if result.get("schema_version") != RESULT_SCHEMA_VERSION or result.get("status") != "ok":
        return None
    if not isinstance(result.get("final_cycle"), int) or result["final_cycle"] <= 0:
        return None
    return result


def load_result(case: Case, runs_dir: Path = RUNS_DIR) -> dict[str, Any] | None:
    """Return the result of ``case`` if it is complete and was run with this definition."""
    result = read_result(case.run_dir(runs_dir))
    if result is None:
        return None
    definition = case.definition()
    if any(result.get(name) != value for name, value in definition.items()):
        return None
    statistics = result.get("trace_statistics", {})
    if any(name not in statistics for name in trace_analyzers().get(case.key, {})):
        return None
    return result


def load_results(runs_dir: Path, experiment: str) -> list[dict[str, Any]]:
    """All valid results stored under one experiment, sorted by case id."""
    experiment_dir = runs_dir / experiment
    if not experiment_dir.is_dir():
        return []
    results = []
    for run_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
        result = read_result(run_dir)
        if result is not None:
            results.append(result)
    return results


def load_trace_summary(runs_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Full generator/simulator statistics of a finished run (see the module docstring)."""
    return _read_json(runs_dir / result["experiment"] / result["case_id"] / SUMMARY_NAME)


# --------------------------------------------------------------------------------------
# Running one case
# --------------------------------------------------------------------------------------


def find_ramulator() -> Path:
    """Locate the simulator binary built in ramulator2/build/."""
    build = ROOT / "ramulator2" / "build"
    candidates = (
        build / "ramulator2",
        build / "ramulator2.exe",
        build / "Release" / "ramulator2.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path.relative_to(ROOT)) for path in candidates)
    raise FileNotFoundError(f"simulator binary not found (tried {tried}); build ramulator2 first")


def git_revision() -> str | None:
    """Commit of the artifact ('-dirty' if the work tree has changes), or None without git."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    if head.returncode != 0 or status.returncode != 0:
        return None
    return head.stdout.strip() + ("-dirty" if status.stdout.strip() else "")


def clock_period_ns(overrides: Mapping[str, Any]) -> float:
    """Period of the memory controller clock (ns) for a case."""
    period = _read_json(DEFAULT_CONFIG)["ramulator"]["tck_ns"]
    return float(overrides.get("ramulator", {}).get("tck_ns", period))


def timing(final_cycle: int, period_ns: float) -> tuple[float, float]:
    """Decoding latency (ns) and throughput (tokens/s) of a final cycle count."""
    latency_ns = final_cycle * period_ns
    return latency_ns, 1e9 / latency_ns


class _MemorySampler:
    """Track the peak resident memory of a process tree by periodic sampling."""

    def __init__(self, pid: int) -> None:
        self.peaks: dict[str, int] = {}
        self._root = None
        if psutil is not None:
            try:
                self._root = psutil.Process(pid)
            except psutil.Error:
                self._root = None

    def sample(self) -> None:
        if self._root is None:
            return
        try:
            processes = [self._root, *self._root.children(recursive=True)]
        except psutil.Error:
            return  # The tree exited between two samples.
        for process in processes:
            try:
                name = process.name().lower()
                info = process.memory_info()
            except psutil.Error:
                continue  # This process exited while it was being inspected.
            role = "simulator" if name.startswith("ramulator2") else "generator"
            # Windows reports the true peak; elsewhere the peak of the sampled values is used.
            rss = int(getattr(info, "peak_wset", info.rss))
            self.peaks[role] = max(self.peaks.get(role, 0), rss)


def _kill_tree(process: subprocess.Popen) -> None:
    """Terminate a child process together with the simulator it started."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _remove_traces(run_dir: Path) -> None:
    for name in TRACE_FILE_NAMES:
        path = run_dir / name
        if path.is_file():
            path.unlink()


def _strip_paths(value: Any) -> Any:
    """Drop the machine-specific file lists from the generator summary."""
    if isinstance(value, dict):
        return {
            key: _strip_paths(item)
            for key, item in value.items()
            if key not in ("files", "trace_files")
        }
    if isinstance(value, list):
        return [_strip_paths(item) for item in value]
    return value


def _tail(path: Path, lines: int = 15) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _simulate(
    case: Case, run_dir: Path, timeout_s: int, stop: threading.Event | None
) -> tuple[float, _MemorySampler]:
    """Run generate_trace.py for one case; return the wall time and the memory peaks."""
    config_path = run_dir / "config.json"
    write_json_atomic(config_path, _plain(case.overrides))
    command = [
        sys.executable,
        str(ROOT / "generate_trace.py"),
        "--model", model_name(case.model),
        "--kv-history", str(int(case.kv_history)),
        "--config", str(config_path),
        "--output-dir", str(run_dir),
        "--ramulator-bin", str(find_ramulator()),
        "--timeout", str(int(timeout_s)),
        "--keep-intermediates",
    ]  # fmt: skip
    group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    stderr_path = run_dir / "stderr.log"
    started = time.monotonic()
    with (run_dir / "stdout.log").open("wb") as out, stderr_path.open("wb") as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, **group)
        sampler = _MemorySampler(process.pid)
        try:
            while process.poll() is None:
                sampler.sample()
                if stop is not None and stop.is_set():
                    raise SweepInterrupted(f"{case.experiment}/{case.case_id} interrupted")
                if time.monotonic() - started > timeout_s:
                    raise CaseFailed(f"timed out after {timeout_s} s")
                time.sleep(POLL_INTERVAL_S)
        finally:
            _kill_tree(process)
    if process.returncode != 0:
        raise CaseFailed(
            f"generate_trace.py exited with code {process.returncode}:\n{_tail(stderr_path)}"
        )
    return time.monotonic() - started, sampler


def _collect(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and cross-check the outputs of a finished simulation."""
    try:
        report = json.loads((run_dir / "stdout.log").read_text(encoding="utf-8"))
        summary = _strip_paths(_read_json(run_dir / GENERATOR_SUMMARY_NAME))
        final_cycle = int(report["last_cycle"])
        simulator_cycles = int(summary["full_trace"]["memory_system_cycles"])
        summary["estimates"]["component_cycle_breakdown"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise CaseFailed(f"unreadable simulation outputs: {exc!r}") from exc
    if report.get("status") != "pass" or final_cycle <= 0 or final_cycle != simulator_cycles:
        raise CaseFailed(
            f"inconsistent outputs: status={report.get('status')!r}, last trace cycle "
            f"{final_cycle}, simulator cycles {simulator_cycles}"
        )
    return report, summary


def _trace_statistics(
    run_dir: Path, final_cycle: int, analyzers: Mapping[str, TraceAnalyzer]
) -> dict[str, Any]:
    """Apply the ``trace_statistics`` functions of the experiments to the command trace."""
    statistics: dict[str, Any] = {}
    for name, analyze in sorted(analyzers.items()):
        try:
            statistics[name] = analyze(run_dir / COMMAND_TRACE_NAME, final_cycle)
        except ValueError as exc:
            raise CaseFailed(f"trace statistics of {name!r}: {exc}") from exc
    return statistics


def run_case(
    case: Case,
    *,
    runs_dir: Path = RUNS_DIR,
    keep_traces: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    revision: str | None = None,
    stop: threading.Event | None = None,
    analyzers: Mapping[str, TraceAnalyzer] | None = None,
) -> dict[str, Any]:
    """Simulate one case in its own directory and return its result.

    ``result.json`` is written only after the simulation finished and its outputs were
    cross-checked; a failed or interrupted run leaves no ``result.json`` behind.
    ``analyzers`` are the ``trace_statistics`` functions to apply, by experiment name.
    """
    run_dir = case.run_dir(runs_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    try:
        wall_time_s, sampler = _simulate(case, run_dir, timeout_s, stop)
        report, summary = _collect(run_dir)
        statistics = _trace_statistics(run_dir, int(report["last_cycle"]), analyzers or {})
    finally:
        if not keep_traces:
            _remove_traces(run_dir)

    final_cycle = int(report["last_cycle"])
    tck_ns = clock_period_ns(case.overrides)
    latency_ns, tokens_per_s = timing(final_cycle, tck_ns)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "ok",
        **case.definition(),
        "model_name": model_name(case.model),
        "final_cycle": final_cycle,
        "clock_period_ns": tck_ns,
        "latency_ns": latency_ns,
        "tokens_per_s": tokens_per_s,
        "dram_org": report.get("dram_org"),
        "required_max_row": report.get("required_max_row"),
        "trace": {
            "line_count": report.get("line_count"),
            "sha256": report.get("sha256"),
            "normalized_sha256": report.get("normalized_sha256"),
        },
        "command_counts": {
            name: count
            for name, count in sorted(summary["full_trace"]["cmd_counter"].items())
            if count
        },
        "cycle_breakdown": summary["estimates"]["component_cycle_breakdown"],
        "trace_statistics": statistics,
        "wall_time_s": round(wall_time_s, 3),
        "peak_rss_bytes": sampler.peaks.get("simulator"),
        "peak_rss_generator_bytes": sampler.peaks.get("generator"),
        "git_commit": revision,
        "platform": platform.platform(),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / GENERATOR_SUMMARY_NAME).unlink()
    write_json_atomic(run_dir / SUMMARY_NAME, summary)
    write_json_atomic(run_dir / RESULT_NAME, result)
    return result


# --------------------------------------------------------------------------------------
# Running a sweep
# --------------------------------------------------------------------------------------


@dataclass
class SweepOutcome:
    completed: list[Case] = field(default_factory=list)
    skipped: list[Case] = field(default_factory=list)
    failed: list[tuple[Case, str]] = field(default_factory=list)
    interrupted: bool = False


def run_cases(
    cases: Sequence[Case],
    *,
    jobs: int = 1,
    keep_traces: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    runs_dir: Path = RUNS_DIR,
) -> SweepOutcome:
    """Simulate the cases that have no valid result yet, ``jobs`` at a time."""
    outcome = SweepOutcome()
    pending = []
    for case in unique_cases(cases):
        if load_result(case, runs_dir) is None:
            pending.append(case)
        else:
            outcome.skipped.append(case)
    print(f"{len(outcome.skipped)} case(s) already complete, {len(pending)} to run", flush=True)
    if not pending:
        return outcome
    find_ramulator()
    if psutil is None:
        print("note: psutil is not installed; peak memory will not be recorded", flush=True)

    revision = git_revision()
    stop = threading.Event()
    experiments = discover_experiments()
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        futures: dict[Future, Case] = {
            pool.submit(
                getattr(experiments[case.experiment], "run_case", run_case),
                case,
                runs_dir=runs_dir,
                keep_traces=keep_traces,
                timeout_s=timeout_s,
                revision=revision,
                stop=stop,
                analyzers=trace_analyzers().get(case.key),
            ): case
            for case in pending
        }
        remaining = set(futures)
        try:
            while remaining:
                # A finite timeout keeps the main thread responsive to Ctrl+C on Windows.
                finished, remaining = wait(
                    remaining, timeout=POLL_INTERVAL_S, return_when=FIRST_COMPLETED
                )
                # Report in submission order so that the log is deterministic for --jobs 1.
                for future in sorted(finished, key=lambda item: pending.index(futures[item])):
                    _record(future, futures[future], outcome, len(pending))
        except KeyboardInterrupt:
            outcome.interrupted = True
            stop.set()
            for future in remaining:
                future.cancel()
            print("interrupted; stopping the running simulations", flush=True)
    return outcome


def _record(future: Future, case: Case, outcome: SweepOutcome, total: int) -> None:
    name = f"{case.experiment}/{case.case_id}"
    progress = f"[{len(outcome.completed) + len(outcome.failed) + 1}/{total}]"
    try:
        result = future.result()
    except SweepInterrupted:
        return
    except (CaseFailed, OSError) as exc:  # OSError: e.g. the disk is full.
        outcome.failed.append((case, str(exc)))
        print(f"{progress} FAILED {name}: {exc}", flush=True)
        return
    outcome.completed.append(case)
    print(
        f"{progress} done {name}: final cycle {result['final_cycle']}, "
        f"{result['wall_time_s']:.1f} s",
        flush=True,
    )


# --------------------------------------------------------------------------------------
# CSV helpers for the experiment modules
# --------------------------------------------------------------------------------------


def geomean(values: Iterable[float]) -> float:
    """Geometric mean of a non-empty sequence of positive numbers."""
    items = [float(value) for value in values]
    if not items:
        raise ValueError("geometric mean of an empty sequence")
    return math.exp(sum(math.log(value) for value in items) / len(items))


def format_cell(value: Any) -> str:
    """CSV text of a value: floats keep full precision, None becomes an empty cell."""
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write rows (mappings from column to value) in the given order and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([format_cell(row.get(column)) for column in columns])
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of rows (mappings from column to text)."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

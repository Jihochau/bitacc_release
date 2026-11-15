#!/usr/bin/env python3
"""Generate the command trace of one BitAcc decoding iteration."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.config import load_json, load_models, load_runtime_bundle
from src.token_lifecycle_trace import (
    resolve_ramulator_for_workload,
    write_token_lifecycle_trace,
)
from src.types import ModelConfig, RunConfig
from verify_trace import inspect_trace, verify_trace

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "BITNET-125M"
DEFAULT_KV_HISTORY = 2
EXPECTED_DIR = ROOT / "expected"
PIM_RUNTIME_DEFAULTS = ROOT / "config" / "pim" / "bitacc_runtime_default.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the command trace of one BitAcc decoding iteration."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model name from config/models.json (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--kv-history",
        type=int,
        default=DEFAULT_KV_HISTORY,
        help=f"number of cached tokens visible before decode (default: {DEFAULT_KV_HISTORY})",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        help="optional JSON model registry; replaces config/models.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "optional runtime/hardware JSON merged over config/default.json; its "
            "'pim_runtime' section, if any, overrides keys of "
            "config/pim/bitacc_runtime_default.json"
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="list configured model names and exit without running Ramulator2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory; omitted means result/<model>_kv<history>",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        help="optional expected JSON manifest for exact trace verification",
    )
    parser.add_argument(
        "--ramulator-bin",
        type=Path,
        help="Ramulator2 executable; omitted means auto-detect ramulator2/build/ramulator2[.exe]",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="retain Ramulator input/config/summary files in addition to trace_out.ch0",
    )
    return parser


def _model_candidates(value: str) -> list[str]:
    raw = value.strip().upper()
    candidates = [raw]
    hyphenated = raw.replace("_", "-")
    if hyphenated not in candidates:
        candidates.append(hyphenated)
    for candidate in tuple(candidates):
        if not candidate.startswith("BITNET-"):
            candidates.append(f"BITNET-{candidate}")
    return candidates


def _select_model(value: str, models: dict[str, ModelConfig]) -> ModelConfig:
    for candidate in _model_candidates(value):
        if candidate in models:
            return models[candidate]
    available = ", ".join(sorted(models))
    raise ValueError(f"unknown model {value!r}; available models: {available}")


def _workload_slug(model_name: str, kv_history: int) -> str:
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return f"{model_slug}_kv{int(kv_history)}"


def _resolve_output_dir(explicit: Path | None, model_name: str, kv_history: int) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return (ROOT / "result" / _workload_slug(model_name, kv_history)).resolve()


def _resolve_expected_manifest(
    explicit: Path | None, model_name: str, kv_history: int
) -> Path | None:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
    else:
        candidate = EXPECTED_DIR / f"{_workload_slug(model_name, kv_history)}.json"
        if not candidate.is_file():
            return None
        candidate = candidate.resolve()

    metadata = json.loads(candidate.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    manifest_model = metadata.get("model")
    if manifest_model is not None and str(manifest_model).upper() != model_name.upper():
        mismatches.append(f"model={manifest_model!r}")
    manifest_kv = metadata.get("visible_kv_history")
    if manifest_kv is not None and int(manifest_kv) != int(kv_history):
        mismatches.append(f"visible_kv_history={manifest_kv!r}")
    manifest_lout = metadata.get("decode_tokens")
    if manifest_lout is not None and int(manifest_lout) != 1:
        mismatches.append(f"decode_tokens={manifest_lout!r}")
    if mismatches:
        raise ValueError(
            f"expected manifest {candidate} does not match {model_name}, "
            f"KV history {kv_history}: {', '.join(mismatches)}"
        )
    return candidate


def _find_ramulator(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    candidates = (
        ROOT / "ramulator2" / "build" / "ramulator2",
        ROOT / "ramulator2" / "build" / "ramulator2.exe",
        ROOT / "ramulator2" / "build" / "Release" / "ramulator2.exe",
        ROOT / "ramulator2" / "ramulator2",
        ROOT / "ramulator2" / "ramulator2.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Ramulator2 was not found. Build it first or pass --ramulator-bin. Tried:\n  - " + tried
    )


def _remove_intermediates(output_dir: Path) -> None:
    generated_names = {
        "latest_command_trace_summary.json",
        "ramulator_config.yaml",
        "ramulator_trace.txt",
        "trace_out.trace",
        "trace_out.yaml",
        "trace_out_run.out",
    }
    for name in generated_names:
        path = output_dir / name
        if path.is_file():
            path.unlink()
    for path in output_dir.glob("trace_out_cmd_counter*.txt"):
        if path.is_file():
            path.unlink()


def main() -> int:
    parser = _parser()
    args = parser.parse_args()

    if args.kv_history <= 0:
        parser.error("--kv-history must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    _default_run, hardware, ramulator, cpu, pim, default_models = load_runtime_bundle(
        config_path=args.config
    )
    models = load_models(args.models_file) if args.models_file is not None else default_models
    pim_runtime = load_json(args.config).get("pim_runtime", {}) if args.config else {}
    if not isinstance(pim_runtime, dict):
        parser.error("the 'pim_runtime' section of --config must be a JSON object")
    unknown = sorted(set(pim_runtime) - set(load_json(PIM_RUNTIME_DEFAULTS)))
    if unknown:
        parser.error(f"unknown 'pim_runtime' keys in --config: {', '.join(unknown)}")

    if args.list_models:
        for name in sorted(models):
            print(name)
        return 0

    try:
        model = _select_model(args.model, models)
        expected_path = _resolve_expected_manifest(
            args.expected, model.name, args.kv_history
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    ramulator_bin = _find_ramulator(args.ramulator_bin)
    output_dir = _resolve_output_dir(args.output_dir, model.name, args.kv_history)
    run = RunConfig(model=model.name, lin=args.kv_history, lout=1, batch=1)
    ramulator, hardware, requirements = resolve_ramulator_for_workload(
        model=model,
        run=run,
        hardware=hardware,
        cpu=cpu,
        pim=pim,
        ramulator=ramulator,
        pim_runtime=pim_runtime,
    )
    generated = write_token_lifecycle_trace(
        model=model,
        run=run,
        hardware=hardware,
        cpu=cpu,
        ramulator=ramulator,
        pim=pim,
        result_dir=output_dir,
        ramulator_bin=ramulator_bin,
        timeout_s=args.timeout,
        pim_runtime=pim_runtime,
    )

    trace_path = Path(generated["trace_out_ch0"]).resolve()
    actual = inspect_trace(trace_path)
    structural_errors = list(actual["parse_errors"])
    if actual["max_cycle"] != actual["last_cycle"]:
        structural_errors.append(
            f"max cycle {actual['max_cycle']} != last cycle {actual['last_cycle']}"
        )
    if structural_errors:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "trace": str(trace_path),
                    "errors": structural_errors,
                    "actual": actual,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        print("Validation failed; intermediate files were retained.", file=sys.stderr)
        return 1

    reference_status = "not_available"
    if expected_path is not None:
        verification = verify_trace(trace_path, expected_path)
        if not verification["ok"]:
            print(json.dumps(verification, indent=2), file=sys.stderr)
            print("Verification failed; intermediate files were retained.", file=sys.stderr)
            return 1
        reference_status = "pass"

    if not args.keep_intermediates:
        _remove_intermediates(output_dir)

    print(
        json.dumps(
            {
                "status": "pass",
                "model": model.name,
                "visible_kv_history": args.kv_history,
                "decode_tokens": 1,
                "dram_org": ramulator.org,
                "required_max_row": requirements["required_max_row"],
                "trace": str(trace_path),
                "line_count": actual["line_count"],
                "last_cycle": actual["last_cycle"],
                "sha256": actual["sha256"],
                "normalized_sha256": actual["normalized_sha256"],
                "reference_verification": reference_status,
                "expected": None if expected_path is None else str(expected_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
DeT ablation harness for Takenoko LoRA training.

Features:
1. Generate DeT-focused variant configs from a base TOML.
2. Run variants sequentially and capture logs.
3. Build an auto-ranked report from TensorBoard scalars with log-text fallback.

Typical usage:
    python src/enhancements/det_motion_transfer/det_ablation_runner.py generate \
        --base-config configs/examples/wan21_lora.toml

    python src/enhancements/det_motion_transfer/det_ablation_runner.py run \
        --base-config configs/examples/wan21_lora.toml \
        --max-train-steps 200

    python src/enhancements/det_motion_transfer/det_ablation_runner.py report \
        --manifest .artifacts/DeT/ablation_runs/<timestamp>/manifest.json
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import toml


LOG_NAN_RE = re.compile(r"\b(?:nan|inf|-inf)\b", re.IGNORECASE)
DIRECTION_SUFFIX_RE = re.compile(r"\s+\([^)]*better\)$")
DEFAULT_SCORE_WEIGHTS = {
    "stability": 0.35,
    "quality": 0.25,
    "motion": 0.20,
    "cost": 0.20,
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    overrides: Dict[str, Any]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_tag(tag: str) -> str:
    out = DIRECTION_SUFFIX_RE.sub("", tag.strip())
    return out


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = config
    for key in parts[:-1]:
        current = target.get(key)
        if not isinstance(current, dict):
            current = {}
            target[key] = current
        target = current
    target[parts[-1]] = value


def parse_override(override_text: str) -> Tuple[str, Any]:
    if "=" not in override_text:
        raise ValueError(f"Invalid override '{override_text}'. Expected KEY=VALUE.")
    key, raw_value = override_text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override '{override_text}'. Empty key.")
    parsed = toml.loads(f"value = {raw_value}")["value"]
    return key, parsed


def default_variant_specs() -> List[VariantSpec]:
    return [
        VariantSpec(
            name="baseline_off",
            description="Disable DeT motion losses and Wan adapter.",
            overrides={
                "enable_det_motion_transfer": False,
                "enable_det_adapter": False,
            },
        ),
        VariantSpec(
            name="det_local_losses",
            description="Enable DeT temporal+dense local losses only.",
            overrides={
                "enable_det_motion_transfer": True,
                "det_temporal_kernel_loss_weight": 0.1,
                "det_dense_tracking_loss_weight": 0.1,
                "det_external_tracking_enabled": False,
                "det_high_frequency_loss_enabled": False,
                "enable_det_adapter": False,
            },
        ),
        VariantSpec(
            name="det_local_probe_scale",
            description="Local DeT losses + attention-locality probe auto-scale.",
            overrides={
                "enable_det_motion_transfer": True,
                "det_temporal_kernel_loss_weight": 0.1,
                "det_dense_tracking_loss_weight": 0.1,
                "det_attention_locality_probe_enabled": True,
                "det_attention_locality_auto_policy": "scale",
                "det_attention_locality_auto_scale_min": 0.1,
                "enable_det_adapter": False,
            },
        ),
        VariantSpec(
            name="det_local_plus_adapter",
            description="Local DeT losses + Wan adapter with locality-follow and warmup.",
            overrides={
                "enable_det_motion_transfer": True,
                "det_temporal_kernel_loss_weight": 0.1,
                "det_dense_tracking_loss_weight": 0.1,
                "det_attention_locality_probe_enabled": True,
                "det_attention_locality_auto_policy": "scale",
                "enable_det_adapter": True,
                "det_adapter_follow_det_locality_scale": True,
                "det_adapter_locality_scale_source": "attention_probe",
                "det_adapter_gate_warmup_enabled": True,
                "det_adapter_gate_warmup_steps": 500,
                "det_adapter_gate_warmup_shape": "linear",
            },
        ),
        VariantSpec(
            name="adapter_only_warmup",
            description="Wan adapter-only baseline with gate warmup.",
            overrides={
                "enable_det_motion_transfer": False,
                "enable_det_adapter": True,
                "det_adapter_follow_det_locality_scale": False,
                "det_adapter_gate_warmup_enabled": True,
                "det_adapter_gate_warmup_steps": 500,
                "det_adapter_gate_warmup_shape": "linear",
            },
        ),
    ]


def choose_variants(selected: Optional[Sequence[str]]) -> List[VariantSpec]:
    variants = default_variant_specs()
    if not selected:
        return variants
    by_name = {v.name: v for v in variants}
    missing = [name for name in selected if name not in by_name]
    if missing:
        raise ValueError(
            f"Unknown variant(s): {missing}. Available: {sorted(by_name.keys())}"
        )
    return [by_name[name] for name in selected]


def render_command(template: Optional[str], config_path: Path) -> List[str]:
    if template:
        formatted = template.format(config=str(config_path))
        return shlex.split(formatted, posix=(os.name != "nt"))
    return [
        sys.executable,
        "src/takenoko.py",
        str(config_path),
        "--non-interactive",
        "--train",
    ]


def create_manifest(
    *,
    base_config_path: Path,
    work_dir: Path,
    variant_specs: Sequence[VariantSpec],
    command_template: Optional[str],
    max_train_steps: Optional[int],
    extra_overrides: Dict[str, Any],
    force: bool,
) -> Path:
    if work_dir.exists() and any(work_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Work directory '{work_dir}' is not empty. Use --force to overwrite."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    config_dir = work_dir / "configs"
    runs_dir = work_dir / "runs"
    config_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_config = toml.load(base_config_path)
    runs: List[Dict[str, Any]] = []

    for variant in variant_specs:
        run_dir = runs_dir / variant.name
        run_dir.mkdir(parents=True, exist_ok=True)

        run_config = copy.deepcopy(base_config)
        # Keep tag names stable for robust post-hoc metric extraction.
        run_config["tensorboard_append_direction_hints"] = False
        run_config["launch_tensorboard_server"] = False
        run_config["log_with"] = "tensorboard"
        run_config["output_dir"] = str((run_dir / "output").resolve())
        run_config["logging_dir"] = str((run_dir / "logs").resolve())
        run_config["output_name"] = f"det_ablation_{variant.name}"

        if max_train_steps is not None:
            run_config["max_train_steps"] = int(max_train_steps)

        for k, v in variant.overrides.items():
            set_nested(run_config, k, v)
        for k, v in extra_overrides.items():
            set_nested(run_config, k, v)

        config_path = config_dir / f"{variant.name}.toml"
        with config_path.open("w", encoding="utf-8") as f:
            toml.dump(run_config, f)

        log_path = run_dir / "train.log"
        cmd = render_command(command_template, config_path.resolve())
        runs.append(
            {
                "name": variant.name,
                "description": variant.description,
                "overrides": {**variant.overrides, **extra_overrides},
                "config_path": str(config_path.resolve()),
                "run_dir": str(run_dir.resolve()),
                "log_path": str(log_path.resolve()),
                "output_dir": str((run_dir / "output").resolve()),
                "logging_dir": str((run_dir / "logs").resolve()),
                "status": "pending",
                "return_code": None,
                "duration_sec": None,
                "command": cmd,
            }
        )

    manifest = {
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "base_config_path": str(base_config_path.resolve()),
        "command_template": command_template or "",
        "max_train_steps": max_train_steps,
        "work_dir": str(work_dir.resolve()),
        "runs": runs,
        "report_path": None,
        "report_json_path": None,
    }

    manifest_path = work_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def execute_runs(
    *,
    manifest_path: Path,
    continue_on_error: bool,
    dry_run: bool,
) -> None:
    manifest = load_manifest(manifest_path)
    repo_root = Path(__file__).resolve().parents[1]

    for run in manifest["runs"]:
        name = run["name"]
        if run.get("status") == "success":
            print(f"[skip] {name}: already marked success")
            continue

        cmd = run["command"]
        log_path = Path(run["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[run] {name}")
        print(f"      command: {' '.join(cmd)}")
        if dry_run:
            run["status"] = "dry_run"
            run["return_code"] = 0
            run["duration_sec"] = 0.0
            save_manifest(manifest_path, manifest)
            continue

        start = time.monotonic()
        with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
            return_code = int(proc.wait())

        duration = time.monotonic() - start
        run["return_code"] = return_code
        run["duration_sec"] = duration
        run["status"] = "success" if return_code == 0 else "failed"
        save_manifest(manifest_path, manifest)
        print(
            f"[done] {name}: status={run['status']} return_code={return_code} duration={duration:.1f}s"
        )

        if return_code != 0 and not continue_on_error:
            print("[stop] Halting after first failed run. Use --continue-on-error to continue.")
            break


def find_event_files(log_dir: Path) -> List[Path]:
    if not log_dir.exists():
        return []
    files = list(log_dir.rglob("events.out.tfevents.*"))
    return sorted(files)


def read_event_scalars(log_dir: Path) -> Tuple[Dict[str, List[Tuple[int, float]]], List[str]]:
    warnings: List[str] = []
    event_files = find_event_files(log_dir)
    if not event_files:
        return {}, warnings

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        warnings.append(f"TensorBoard event parser unavailable: {exc}")
        return {}, warnings

    series: Dict[str, List[Tuple[int, float]]] = {}
    for event_file in event_files:
        try:
            acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
            for tag in tags:
                canonical = sanitize_tag(tag)
                bucket = series.setdefault(canonical, [])
                for ev in acc.Scalars(tag):
                    bucket.append((int(ev.step), float(ev.value)))
        except Exception as exc:
            warnings.append(f"Failed reading event file '{event_file.name}': {exc}")

    for tag, points in series.items():
        points.sort(key=lambda x: (x[0], x[1]))
        dedup: Dict[int, float] = {}
        for step, value in points:
            dedup[step] = value
        series[tag] = sorted(dedup.items(), key=lambda x: x[0])

    return series, warnings


def pick_series(
    all_series: Dict[str, List[Tuple[int, float]]],
    candidates: Sequence[str],
) -> Optional[List[Tuple[int, float]]]:
    for tag in candidates:
        if tag in all_series and all_series[tag]:
            return all_series[tag]
    return None


def finite_values(series: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for step, value in series:
        if math.isfinite(value):
            out.append((step, value))
    return out


def scan_log_anomalies(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {"nan_token_hits": 0, "line_count": 0}
    nan_hits = 0
    line_count = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            if LOG_NAN_RE.search(line):
                nan_hits += 1
    return {"nan_token_hits": nan_hits, "line_count": line_count}


def compute_spike_ratio(values: Sequence[float], window: int = 20, factor: float = 2.5) -> float:
    if len(values) <= window:
        return 0.0
    spikes = 0
    checked = 0
    for idx in range(window, len(values)):
        history = values[idx - window : idx]
        baseline = abs(statistics.median(history))
        baseline = max(baseline, 1e-8)
        if abs(values[idx]) > baseline * factor:
            spikes += 1
        checked += 1
    if checked <= 0:
        return 0.0
    return float(spikes) / float(checked)


def compute_stability_score(
    *,
    nan_hits: int,
    loss_cv: Optional[float],
    spike_ratio: float,
) -> float:
    penalty = 0.0
    if nan_hits > 0:
        penalty += 0.85
    penalty += min(0.5, max(0.0, spike_ratio) * 1.2)
    if loss_cv is not None and math.isfinite(loss_cv):
        penalty += min(0.35, max(0.0, loss_cv - 0.2))
    return max(0.0, 1.0 - penalty)


def normalize_lower_is_better(
    values: Dict[str, Optional[float]],
    missing_default: float = 0.5,
) -> Dict[str, float]:
    finite = [v for v in values.values() if v is not None and math.isfinite(v)]
    if not finite:
        return {k: missing_default for k in values}
    lo = min(finite)
    hi = max(finite)
    out: Dict[str, float] = {}
    for key, value in values.items():
        if value is None or not math.isfinite(value):
            out[key] = missing_default
        elif hi == lo:
            out[key] = 1.0
        else:
            out[key] = (hi - value) / (hi - lo)
    return out


def normalize_higher_is_better(
    values: Dict[str, Optional[float]],
    missing_default: float = 0.5,
) -> Dict[str, float]:
    finite = [v for v in values.values() if v is not None and math.isfinite(v)]
    if not finite:
        return {k: missing_default for k in values}
    lo = min(finite)
    hi = max(finite)
    out: Dict[str, float] = {}
    for key, value in values.items():
        if value is None or not math.isfinite(value):
            out[key] = missing_default
        elif hi == lo:
            out[key] = 1.0
        else:
            out[key] = (value - lo) / (hi - lo)
    return out


def estimate_throughput(
    scalars: Dict[str, List[Tuple[int, float]]],
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    throughput_candidates = [
        "perf/it_s",
        "perf/iter_per_sec",
        "perf/steps_per_sec",
        "throughput/it_s",
        "throughput/steps_per_sec",
    ]
    throughput_series = pick_series(scalars, throughput_candidates)
    if throughput_series:
        finite = finite_values(throughput_series)
        if finite:
            values = [v for _, v in finite]
            return (
                float(statistics.fmean(values)),
                float(values[-1]),
                None,
                "direct",
            )

    iter_ms_candidates = [
        "perf/iter_ms",
        "perf/step_ms",
        "timing/iter_ms",
        "timing/step_ms",
    ]
    iter_ms_series = pick_series(scalars, iter_ms_candidates)
    if iter_ms_series:
        finite_ms = finite_values(iter_ms_series)
        positive = [v for _, v in finite_ms if v > 0.0]
        if positive:
            throughput_values = [1000.0 / v for v in positive]
            return (
                float(statistics.fmean(throughput_values)),
                float(throughput_values[-1]),
                float(statistics.fmean(positive)),
                "derived_from_iter_ms",
            )
    return None, None, None, "missing"


def summarize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    logging_dir = Path(run["logging_dir"])
    log_path = Path(run["log_path"])
    scalars, scalar_warnings = read_event_scalars(logging_dir)
    log_info = scan_log_anomalies(log_path)

    train_series = pick_series(
        scalars,
        [
            "loss/current",
            "loss/ema",
            "loss/average",
            "loss/mse",
            "loss",
        ],
    )
    val_series = pick_series(
        scalars,
        [
            "val/velocity_loss_avg",
            "val/direct_noise_loss_avg",
            "val/velocity_loss_avg_weighted",
            "val/direct_noise_loss_avg_weighted",
            "val/loss",
            "validation/loss",
        ],
    )

    motion_tags = [
        "loss/det_temporal_kernel",
        "loss/det_dense_tracking",
        "loss/det_external_tracking",
        "loss/det_high_frequency",
    ]
    motion_values: List[float] = []
    for tag in motion_tags:
        series = scalars.get(tag, [])
        finite = finite_values(series)
        if finite:
            motion_values.append(float(finite[-1][1]))

    train_finite = finite_values(train_series or [])
    val_finite = finite_values(val_series or [])
    train_values = [v for _, v in train_finite]

    train_final = float(train_values[-1]) if train_values else None
    train_mean = float(statistics.fmean(train_values)) if train_values else None
    train_std = (
        float(statistics.pstdev(train_values))
        if train_values and len(train_values) > 1
        else 0.0 if train_values else None
    )
    train_cv = (
        float(train_std / max(abs(train_mean), 1e-8))
        if train_values and train_mean is not None and train_std is not None
        else None
    )
    spike_ratio = compute_spike_ratio(train_values)
    val_final = float(val_finite[-1][1]) if val_finite else None
    motion_final = (
        float(statistics.fmean(motion_values)) if motion_values else None
    )
    throughput_mean, throughput_final, iter_ms_mean, throughput_source = (
        estimate_throughput(scalars)
    )
    observed_train_steps = (
        int(train_finite[-1][0] - train_finite[0][0] + 1) if len(train_finite) >= 2 else None
    )

    stability = compute_stability_score(
        nan_hits=int(log_info.get("nan_token_hits", 0)),
        loss_cv=train_cv,
        spike_ratio=spike_ratio,
    )

    return {
        "name": run["name"],
        "status": run.get("status", "unknown"),
        "return_code": run.get("return_code"),
        "duration_sec": run.get("duration_sec"),
        "train_final": train_final,
        "train_mean": train_mean,
        "train_cv": train_cv,
        "spike_ratio": spike_ratio,
        "val_final": val_final,
        "motion_final": motion_final,
        "throughput_mean": throughput_mean,
        "throughput_final": throughput_final,
        "iter_ms_mean": iter_ms_mean,
        "throughput_source": throughput_source,
        "observed_train_steps": observed_train_steps,
        "nan_token_hits": int(log_info.get("nan_token_hits", 0)),
        "line_count": int(log_info.get("line_count", 0)),
        "event_file_count": len(find_event_files(logging_dir)),
        "scalar_warning_count": len(scalar_warnings),
        "scalar_warnings": scalar_warnings,
        "stability_score": stability,
        "config_path": run["config_path"],
        "log_path": run["log_path"],
    }


def score_and_rank_summaries(
    summaries: Sequence[Dict[str, Any]],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    active_weights = dict(DEFAULT_SCORE_WEIGHTS)
    if weights:
        for key, value in weights.items():
            if key in active_weights and math.isfinite(float(value)) and float(value) >= 0.0:
                active_weights[key] = float(value)
    weight_sum = sum(active_weights.values())
    if weight_sum <= 0.0:
        active_weights = dict(DEFAULT_SCORE_WEIGHTS)
        weight_sum = sum(active_weights.values())
    for key in list(active_weights.keys()):
        active_weights[key] = active_weights[key] / weight_sum

    train_objectives = {row["name"]: row.get("train_final") for row in summaries}
    val_objectives = {row["name"]: row.get("val_final") for row in summaries}
    motion_objectives = {row["name"]: row.get("motion_final") for row in summaries}
    throughput_benefits = {row["name"]: row.get("throughput_mean") for row in summaries}
    duration_objectives = {row["name"]: row.get("duration_sec") for row in summaries}

    train_scores = normalize_lower_is_better(train_objectives)
    val_scores = normalize_lower_is_better(val_objectives)
    motion_scores = normalize_lower_is_better(motion_objectives)
    throughput_scores = normalize_higher_is_better(throughput_benefits)
    duration_scores = normalize_lower_is_better(duration_objectives)

    ranked_rows: List[Dict[str, Any]] = []
    for row in summaries:
        name = str(row["name"])
        status = str(row.get("status", "unknown"))
        stability = float(row.get("stability_score", 0.0))
        train_score = float(train_scores[name])
        val_score = float(val_scores[name])
        motion_score = float(motion_scores[name])
        throughput_score = float(throughput_scores[name])
        duration_score = float(duration_scores[name])

        train_available = row.get("train_final") is not None
        val_available = row.get("val_final") is not None
        quality_parts: List[float] = []
        if train_available:
            quality_parts.append(train_score)
        if val_available:
            quality_parts.append(val_score)
        quality_score = (
            float(statistics.fmean(quality_parts)) if quality_parts else 0.5
        )
        cost_score = float(statistics.fmean([throughput_score, duration_score]))

        total = 100.0 * (
            active_weights["stability"] * stability
            + active_weights["quality"] * quality_score
            + active_weights["motion"] * motion_score
            + active_weights["cost"] * cost_score
        )
        if status not in {"success", "dry_run"}:
            total *= 0.2

        enriched = dict(row)
        enriched["train_score"] = train_score
        enriched["val_score"] = val_score
        enriched["quality_score"] = quality_score
        enriched["motion_score"] = motion_score
        enriched["throughput_score"] = throughput_score
        enriched["duration_score"] = duration_score
        enriched["cost_score"] = cost_score
        enriched["score_weights"] = dict(active_weights)
        enriched["total_score"] = total
        ranked_rows.append(enriched)

    ranked_rows.sort(key=lambda x: x["total_score"], reverse=True)
    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx

    return ranked_rows


def build_report(manifest_path: Path) -> Tuple[Path, Path]:
    manifest = load_manifest(manifest_path)
    work_dir = Path(manifest["work_dir"])

    summaries = [summarize_run(run) for run in manifest["runs"]]
    ranked_rows = score_and_rank_summaries(summaries)

    top = ranked_rows[0] if ranked_rows else None
    generated_at = utc_now()
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    if top is not None and isinstance(top.get("score_weights"), dict):
        weights = dict(top["score_weights"])
    report_json = {
        "generated_at": generated_at,
        "manifest_path": str(manifest_path.resolve()),
        "base_config_path": manifest.get("base_config_path"),
        "score_weights": weights,
        "ranking": ranked_rows,
    }

    report_json_path = work_dir / "report.json"
    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    lines: List[str] = []
    lines.append("# DeT Ablation Report")
    lines.append("")
    lines.append(f"- Generated: {generated_at}")
    lines.append(f"- Base config: `{manifest.get('base_config_path', '')}`")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append("")
    if top is not None:
        lines.append(
            f"## Recommended Variant: `{top['name']}` (score={top['total_score']:.2f})"
        )
        lines.append(
            "Reason: best weighted combination of stability, quality, motion quality, and runtime cost."
        )
        lines.append("")

    lines.append("## Score Weights")
    lines.append("")
    lines.append(
        f"- stability={weights.get('stability', 0.0):.3f}, "
        f"quality={weights.get('quality', 0.0):.3f}, "
        f"motion={weights.get('motion', 0.0):.3f}, "
        f"cost={weights.get('cost', 0.0):.3f}"
    )
    lines.append("")

    lines.append("## Ranking")
    lines.append("")
    lines.append(
        "| Rank | Variant | Score | Stability | Quality | Motion | Cost | Throughput | Train Final | Val Final | Motion Final | NaN Hits | Spike Ratio | Status |"
    )
    lines.append(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for row in ranked_rows:
        lines.append(
            "| "
            f"{row['rank']} | "
            f"`{row['name']}` | "
            f"{row['total_score']:.2f} | "
            f"{row['stability_score']:.3f} | "
            f"{row['quality_score']:.3f} | "
            f"{row['motion_score']:.3f} | "
            f"{row['cost_score']:.3f} | "
            f"{_fmt_optional(row.get('throughput_mean'))} | "
            f"{_fmt_optional(row['train_final'])} | "
            f"{_fmt_optional(row['val_final'])} | "
            f"{_fmt_optional(row['motion_final'])} | "
            f"{row['nan_token_hits']} | "
            f"{row['spike_ratio']:.3f} | "
            f"{row['status']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Scores are heuristic and intended for quick run triage.")
    lines.append("- Lower objectives are better for train/val/motion losses; higher throughput is better.")
    lines.append("- `NaN Hits` are counted from raw training logs (case-insensitive token scan).")
    lines.append("- If TensorBoard scalars are missing, report quality is reduced.")
    lines.append("")
    lines.append("## Run Artifacts")
    lines.append("")
    for row in ranked_rows:
        lines.append(
            f"- `{row['name']}`: config=`{row['config_path']}`, log=`{row['log_path']}`"
        )

    report_md_path = work_dir / "report.md"
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    manifest["report_path"] = str(report_md_path.resolve())
    manifest["report_json_path"] = str(report_json_path.resolve())
    save_manifest(manifest_path, manifest)
    return report_md_path, report_json_path


def _fmt_optional(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.6g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeT ablation harness and auto-report.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--base-config",
        type=Path,
        help="Base TOML config to clone for ablation variants.",
    )
    common.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Ablation output directory. Default: .artifacts/DeT/ablation_runs/<timestamp>",
    )
    common.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Specific variant name to include (repeatable).",
    )
    common.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra TOML override KEY=VALUE applied to every variant (repeatable).",
    )
    common.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Override max_train_steps for each variant.",
    )
    common.add_argument(
        "--command-template",
        type=str,
        default=None,
        help='Optional command template with "{config}" placeholder.',
    )
    common.add_argument(
        "--force",
        action="store_true",
        help="Allow reusing a non-empty work directory.",
    )

    sub.add_parser("generate", parents=[common], help="Generate variant configs + manifest.")

    run_parser = sub.add_parser("run", parents=[common], help="Generate, execute, and report.")
    run_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs even if one run fails.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute training commands; mark as dry_run.",
    )

    report_parser = sub.add_parser("report", help="Build report from existing manifest.")
    report_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a previously generated manifest.json",
    )

    return parser.parse_args()


def default_work_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(".artifacts") / "DeT" / "ablation_runs" / stamp


def require_base_config(path: Optional[Path]) -> Path:
    if path is None:
        raise ValueError("--base-config is required for this command.")
    if not path.exists():
        raise FileNotFoundError(f"Base config not found: {path}")
    return path


def main() -> int:
    args = parse_args()
    if args.command == "report":
        manifest_path = args.manifest
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        report_md, report_json = build_report(manifest_path)
        print(f"Report written: {report_md}")
        print(f"JSON written:   {report_json}")
        return 0

    base_config = require_base_config(args.base_config)
    work_dir = args.work_dir or default_work_dir()
    variants = choose_variants(args.variant or None)
    extra_overrides: Dict[str, Any] = {}
    for item in args.override:
        key, value = parse_override(item)
        extra_overrides[key] = value

    manifest_path = create_manifest(
        base_config_path=base_config,
        work_dir=work_dir,
        variant_specs=variants,
        command_template=args.command_template,
        max_train_steps=args.max_train_steps,
        extra_overrides=extra_overrides,
        force=bool(args.force),
    )
    print(f"Manifest written: {manifest_path}")

    if args.command == "generate":
        print("Configs generated. Use `run` or `report` next.")
        return 0

    execute_runs(
        manifest_path=manifest_path,
        continue_on_error=bool(args.continue_on_error),
        dry_run=bool(args.dry_run),
    )
    report_md, report_json = build_report(manifest_path)
    print(f"Report written: {report_md}")
    print(f"JSON written:   {report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Motion-preservation ablation harness for Takenoko training.

Features:
1. Generate motion-preservation variant configs from a base TOML.
2. Run variants sequentially and capture logs.
3. Build an auto-ranked report from TensorBoard scalars with log-text fallback.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import toml


LOG_NAN_RE = re.compile(r"\b(?:nan|inf|-inf)\b", re.IGNORECASE)
DIRECTION_SUFFIX_RE = re.compile(r"\s+\([^)]*better\)$")
DEFAULT_SCORE_WEIGHTS = {
    "stability": 0.30,
    "quality": 0.25,
    "preservation": 0.30,
    "cost": 0.15,
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    overrides: Dict[str, Any]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_tag(tag: str) -> str:
    return DIRECTION_SUFFIX_RE.sub("", tag.strip())


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
            description="Disable motion preservation and attention replay.",
            overrides={
                "motion_preservation": False,
                "motion_attention_preservation": False,
                "ewc_lambda": 0.0,
            },
        ),
        VariantSpec(
            name="replay_dataset",
            description="Replay only with dataset anchors.",
            overrides={
                "motion_preservation": True,
                "motion_preservation_anchor_source": "dataset",
                "motion_attention_preservation": False,
                "ewc_lambda": 0.0,
            },
        ),
        VariantSpec(
            name="replay_hybrid",
            description="Replay only with hybrid anchors.",
            overrides={
                "motion_preservation": True,
                "motion_preservation_anchor_source": "hybrid",
                "motion_attention_preservation": False,
                "ewc_lambda": 0.0,
            },
        ),
        VariantSpec(
            name="replay_hybrid_attention",
            description="Replay with hybrid anchors plus attention preservation.",
            overrides={
                "motion_preservation": True,
                "motion_preservation_anchor_source": "hybrid",
                "motion_attention_preservation": True,
                "ewc_lambda": 0.0,
            },
        ),
        VariantSpec(
            name="replay_hybrid_attention_ewc",
            description="Replay + attention preservation + EWC. EWC is effective on full FT and ignored on LoRA.",
            overrides={
                "motion_preservation": True,
                "motion_preservation_anchor_source": "hybrid",
                "motion_attention_preservation": True,
                "ewc_lambda": 0.01,
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
        run_config["tensorboard_append_direction_hints"] = False
        run_config["launch_tensorboard_server"] = False
        run_config["log_with"] = "tensorboard"
        run_config["output_dir"] = str((run_dir / "output").resolve())
        run_config["logging_dir"] = str((run_dir / "logs").resolve())
        run_config["output_name"] = f"motion_ablation_{variant.name}"

        if max_train_steps is not None:
            run_config["max_train_steps"] = int(max_train_steps)

        for key, value in variant.overrides.items():
            set_nested(run_config, key, value)
        for key, value in extra_overrides.items():
            set_nested(run_config, key, value)

        config_path = config_dir / f"{variant.name}.toml"
        with config_path.open("w", encoding="utf-8") as handle:
            toml.dump(run_config, handle)

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
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def execute_runs(
    *,
    manifest_path: Path,
    continue_on_error: bool,
    dry_run: bool,
) -> None:
    manifest = load_manifest(manifest_path)
    repo_root = Path(__file__).resolve().parents[3]

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
        with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
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
                log_handle.write(line)
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
    return sorted(log_dir.rglob("events.out.tfevents.*"))


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
            for tag in acc.Tags().get("scalars", []):
                canonical = sanitize_tag(tag)
                bucket = series.setdefault(canonical, [])
                for event in acc.Scalars(tag):
                    bucket.append((int(event.step), float(event.value)))
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
    return [(step, value) for step, value in series if math.isfinite(value)]


def scan_log_anomalies(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {"nan_token_hits": 0, "line_count": 0}
    nan_hits = 0
    line_count = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
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
        baseline = max(abs(statistics.median(history)), 1e-8)
        if abs(values[idx]) > baseline * factor:
            spikes += 1
        checked += 1
    return float(spikes) / float(max(checked, 1))


def compute_stability_score(
    *,
    nan_hits: int,
    loss_cv: Optional[float],
    spike_ratio: float,
    invalid_skip_rate: Optional[float],
) -> float:
    penalty = 0.0
    if nan_hits > 0:
        penalty += 0.85
    penalty += min(0.5, max(0.0, spike_ratio) * 1.2)
    if loss_cv is not None and math.isfinite(loss_cv):
        penalty += min(0.35, max(0.0, loss_cv - 0.2))
    if invalid_skip_rate is not None and math.isfinite(invalid_skip_rate):
        penalty += min(0.4, max(0.0, invalid_skip_rate) * 2.0)
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
            values = [value for _, value in finite]
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
        positive = [value for _, value in finite_ms if value > 0.0]
        if positive:
            throughput_values = [1000.0 / value for value in positive]
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
        ["loss/current", "loss/ema", "loss/average", "loss/mse", "loss"],
    )
    val_series = pick_series(
        scalars,
        [
            "val/velocity_loss_avg",
            "val/direct_noise_loss_avg",
            "val/loss",
            "validation/loss",
        ],
    )
    replay_series = pick_series(
        scalars,
        ["loss/motion_preservation", "motion_preservation/loss"],
    )
    attention_series = pick_series(
        scalars,
        ["loss/motion_attention_preservation"],
    )
    apply_rate_series = pick_series(scalars, ["motion_preservation/apply_rate"])
    fallback_rate_series = pick_series(
        scalars, ["motion_preservation/temporal_fallback_rate"]
    )
    invalid_skip_series = pick_series(
        scalars, ["motion_preservation/invalid_anchor_skip_rate"]
    )
    replay_ratio_series = pick_series(
        scalars, ["motion_preservation/total_to_task"]
    )

    train_finite = finite_values(train_series or [])
    val_finite = finite_values(val_series or [])
    replay_finite = finite_values(replay_series or [])
    attention_finite = finite_values(attention_series or [])
    apply_rate_finite = finite_values(apply_rate_series or [])
    fallback_rate_finite = finite_values(fallback_rate_series or [])
    invalid_skip_finite = finite_values(invalid_skip_series or [])
    replay_ratio_finite = finite_values(replay_ratio_series or [])

    train_values = [value for _, value in train_finite]
    replay_values = [value for _, value in replay_finite]
    attention_values = [value for _, value in attention_finite]

    train_final = float(train_values[-1]) if train_values else None
    train_mean = float(statistics.fmean(train_values)) if train_values else None
    train_std = (
        float(statistics.pstdev(train_values))
        if len(train_values) > 1
        else 0.0 if train_values else None
    )
    train_cv = (
        float(train_std / max(abs(train_mean), 1e-8))
        if train_mean is not None and train_std is not None and train_values
        else None
    )
    spike_ratio = compute_spike_ratio(train_values)
    val_final = float(val_finite[-1][1]) if val_finite else None
    replay_final = float(replay_values[-1]) if replay_values else None
    replay_mean = float(statistics.fmean(replay_values)) if replay_values else None
    attention_final = float(attention_values[-1]) if attention_values else None
    apply_rate_final = float(apply_rate_finite[-1][1]) if apply_rate_finite else None
    fallback_rate_final = (
        float(fallback_rate_finite[-1][1]) if fallback_rate_finite else None
    )
    invalid_skip_rate_final = (
        float(invalid_skip_finite[-1][1]) if invalid_skip_finite else None
    )
    replay_ratio_final = (
        float(replay_ratio_finite[-1][1]) if replay_ratio_finite else None
    )
    throughput_mean, throughput_final, iter_ms_mean, throughput_source = (
        estimate_throughput(scalars)
    )
    observed_train_steps = (
        int(train_finite[-1][0] - train_finite[0][0] + 1) if len(train_finite) >= 2 else None
    )

    stability_score = compute_stability_score(
        nan_hits=int(log_info.get("nan_token_hits", 0)),
        loss_cv=train_cv,
        spike_ratio=spike_ratio,
        invalid_skip_rate=invalid_skip_rate_final,
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
        "replay_final": replay_final,
        "replay_mean": replay_mean,
        "attention_final": attention_final,
        "apply_rate_final": apply_rate_final,
        "fallback_rate_final": fallback_rate_final,
        "invalid_skip_rate_final": invalid_skip_rate_final,
        "replay_ratio_final": replay_ratio_final,
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
        "stability_score": stability_score,
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

    train_scores = normalize_lower_is_better(
        {row["name"]: row.get("train_final") for row in summaries}
    )
    val_scores = normalize_lower_is_better(
        {row["name"]: row.get("val_final") for row in summaries}
    )
    replay_scores = normalize_lower_is_better(
        {row["name"]: row.get("replay_final") for row in summaries}
    )
    attention_scores = normalize_lower_is_better(
        {row["name"]: row.get("attention_final") for row in summaries}
    )
    apply_rate_scores = normalize_higher_is_better(
        {row["name"]: row.get("apply_rate_final") for row in summaries}
    )
    fallback_scores = normalize_lower_is_better(
        {row["name"]: row.get("fallback_rate_final") for row in summaries}
    )
    replay_ratio_scores = normalize_lower_is_better(
        {row["name"]: row.get("replay_ratio_final") for row in summaries}
    )
    throughput_scores = normalize_higher_is_better(
        {row["name"]: row.get("throughput_mean") for row in summaries}
    )
    duration_scores = normalize_lower_is_better(
        {row["name"]: row.get("duration_sec") for row in summaries}
    )

    ranked_rows: List[Dict[str, Any]] = []
    for row in summaries:
        name = str(row["name"])
        status = str(row.get("status", "unknown"))
        stability = float(row.get("stability_score", 0.0))

        quality_parts = [train_scores[name]]
        if row.get("val_final") is not None:
            quality_parts.append(val_scores[name])
        quality_score = float(statistics.fmean(quality_parts))

        preservation_score = float(
            statistics.fmean(
                [
                    replay_scores[name],
                    attention_scores[name],
                    apply_rate_scores[name],
                    fallback_scores[name],
                    replay_ratio_scores[name],
                ]
            )
        )
        cost_score = float(statistics.fmean([throughput_scores[name], duration_scores[name]]))

        total = 100.0 * (
            active_weights["stability"] * stability
            + active_weights["quality"] * quality_score
            + active_weights["preservation"] * preservation_score
            + active_weights["cost"] * cost_score
        )
        if status not in {"success", "dry_run"}:
            total *= 0.2

        enriched = dict(row)
        enriched["quality_score"] = quality_score
        enriched["preservation_score"] = preservation_score
        enriched["cost_score"] = cost_score
        enriched["score_weights"] = dict(active_weights)
        enriched["total_score"] = total
        ranked_rows.append(enriched)

    ranked_rows.sort(key=lambda row: row["total_score"], reverse=True)
    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx
    return ranked_rows


def _fmt_optional(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.6g}"


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
    with report_json_path.open("w", encoding="utf-8") as handle:
        json.dump(report_json, handle, indent=2)

    lines: List[str] = []
    lines.append("# Motion Preservation Ablation Report")
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
            "Reason: best weighted combination of stability, core quality, preservation behavior, and runtime cost."
        )
        lines.append("")

    lines.append("## Score Weights")
    lines.append("")
    lines.append(
        f"- stability={weights.get('stability', 0.0):.3f}, "
        f"quality={weights.get('quality', 0.0):.3f}, "
        f"preservation={weights.get('preservation', 0.0):.3f}, "
        f"cost={weights.get('cost', 0.0):.3f}"
    )
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append(
        "| Rank | Variant | Score | Stability | Quality | Preservation | Cost | Replay Final | Apply Rate | Fallback Rate | Throughput | Train Final | NaN Hits | Status |"
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
            f"{row['preservation_score']:.3f} | "
            f"{row['cost_score']:.3f} | "
            f"{_fmt_optional(row.get('replay_final'))} | "
            f"{_fmt_optional(row.get('apply_rate_final'))} | "
            f"{_fmt_optional(row.get('fallback_rate_final'))} | "
            f"{_fmt_optional(row.get('throughput_mean'))} | "
            f"{_fmt_optional(row.get('train_final'))} | "
            f"{row['nan_token_hits']} | "
            f"{row['status']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Scores are heuristic and meant for fast triage, not publication.")
    lines.append("- Preservation score favors low replay loss, high replay apply rate, low fallback rate, and low replay-to-task ratio.")
    lines.append("- The `replay_hybrid_attention_ewc` variant is most meaningful on full fine-tuning; on LoRA, EWC is expected to be ignored.")
    lines.append("- If TensorBoard scalars are missing, report quality is reduced.")
    lines.append("")
    lines.append("## Run Artifacts")
    lines.append("")
    for row in ranked_rows:
        lines.append(
            f"- `{row['name']}`: config=`{row['config_path']}`, log=`{row['log_path']}`"
        )

    report_md_path = work_dir / "report.md"
    with report_md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")

    manifest["report_path"] = str(report_md_path.resolve())
    manifest["report_json_path"] = str(report_json_path.resolve())
    save_manifest(manifest_path, manifest)
    return report_md_path, report_json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motion-preservation ablation harness and auto-report."
    )
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
        help="Ablation output directory. Default: .artifacts/MotionPreservation/ablation_runs/<timestamp>",
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
    return Path(".artifacts") / "MotionPreservation" / "ablation_runs" / stamp


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

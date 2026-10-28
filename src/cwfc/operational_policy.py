"""Step 6: historical workload alerts with forward-only residual calibration.

Run ``python -m cwfc.operational_policy`` after Step 5. No models are fitted,
no holdout targets are opened, and no final forecast file is produced.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from cwfc.error_analysis import CPU, HORIZON, TRAIN_ENDS, load_training_targets, validate_training_targets

PRIMARY = "Ridge AR (6)"
FALLBACK = "Moving Average (12)"
MODELS = (PRIMARY, FALLBACK)
PEAK_QUANTILE = 0.95
CONSECUTIVE_STEPS = 3  # Illustrative 15-minute persistence rule; not tuned.
HOURS = {"First hour": (1, 12), "Second hour": (13, 24)}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    quantile: float | None


# Fixed illustrative choices, declared before examining policy performance.
POLICIES = (PolicySpec("point_only", None), PolicySpec("historical_q90", 0.90),
            PolicySpec("historical_q95", 0.95))
PROPOSED_POLICY = "historical_q90"


def residual_margin(residuals, quantile: float) -> float:
    """Nonnegative empirical quantile of signed errors, not positive-only errors.

    Empty historical error samples are uncalibrated (NaN). This is an
    approximate historical upper error margin, not a coverage guarantee.
    """
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("residuals must be a finite one-dimensional sequence")
    if values.size == 0:
        return float("nan")
    return max(0.0, float(np.quantile(values, quantile, method="linear")))


def conservative_forecast(predicted, margin) -> np.ndarray:
    """Apply a nonnegative margin in CPU percentage points and clip to [0,100]."""
    point = np.asarray(predicted, dtype=float)
    margins = np.asarray(margin, dtype=float)
    if (point.ndim != 1 or not np.isfinite(point).all()
            or not np.isfinite(margins).all() or (margins < 0).any()
            or (point < 0).any() or (point > 100).any()):
        raise ValueError("Predictions must be in [0,100] and margins finite and nonnegative")
    if margins.ndim > 1 or (margins.ndim == 1 and margins.shape != point.shape):
        raise ValueError("Margin must be scalar or aligned with predictions")
    return np.clip(point + margins, 0, 100)


def classify_alerts(predicted, upper, threshold: float,
                    consecutive_steps: int = CONSECUTIVE_STEPS) -> tuple[np.ndarray, np.ndarray]:
    """Three workload-alert levels; call once per contiguous forecast origin.

    High risk starts at the third consecutive POINT exceedance (inclusive),
    not merely a sustained upper-margin exceedance. Runs are not backfilled.
    No actual validation demand or infrastructure-capacity signal is accepted.
    """
    point = np.asarray(predicted, dtype=float)
    bound = np.asarray(upper, dtype=float)
    if (point.ndim != 1 or bound.shape != point.shape
            or not np.isfinite(point).all() or not np.isfinite(bound).all()
            or (point < 0).any() or (bound > 100).any() or (bound < point).any()
            or not np.isfinite(threshold) or not 0 <= threshold <= 100):
        raise ValueError("Provide aligned bounded predictions, upper values and a valid threshold")
    if isinstance(consecutive_steps, bool) or not isinstance(consecutive_steps, int) or consecutive_steps < 1:
        raise ValueError("consecutive_steps must be a positive integer")
    levels, runs = [], []
    run = 0
    for estimate, upper_value in zip(point, bound):
        run = run + 1 if estimate >= threshold else 0
        level = "Normal"
        if upper_value >= threshold:
            level = "Elevated demand"
        if run >= consecutive_steps:
            level = "High-risk condition"
        levels.append(level)
        runs.append(run)
    return np.asarray(levels, dtype=str), np.asarray(runs, dtype=int)


def validate_oof_predictions(predictions: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """Verify saved Step 5 CPU forecasts against the training-only timeline.

    Cached residual/threshold columns are ignored and recomputed. Forecast
    provenance comes from the fixed Step 5 implementation and its tests.
    """
    training = validate_training_targets(training)
    columns = ["target", "model", "fold", "train_size", "origin_timestamp",
               "horizon_step", "timestamp", "actual", "predicted"]
    if not set(columns).issubset(predictions.columns):
        raise ValueError(f"OOF predictions require columns {columns}")
    selected = predictions.loc[(predictions["target"] == CPU) & predictions["model"].isin(MODELS), columns].copy()
    if len(selected) != 2 * len(TRAIN_ENDS) * HORIZON:
        raise ValueError("Expected 72 aligned OOF observations for each selected CPU method")
    numeric = [column for column in columns if column not in ("target", "model")]
    selected[numeric] = selected[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(selected[numeric].to_numpy()).all():
        raise ValueError("OOF predictions must be finite")
    for model in MODELS:
        for fold, end in enumerate(TRAIN_ENDS, start=1):
            part = selected[(selected["model"] == model) & (selected["fold"] == fold)]
            expected = training.iloc[end:end + HORIZON]
            if (len(part) != HORIZON or not np.array_equal(part["horizon_step"], np.arange(1, HORIZON + 1))
                    or not np.array_equal(part["timestamp"], expected["timestamp"])
                    or not (part["train_size"] == end).all()
                    or not (part["origin_timestamp"] == training.iloc[end - 1]["timestamp"]).all()
                    or not np.allclose(part["actual"], expected[CPU], rtol=0, atol=1e-12)
                    or not part["predicted"].between(0, 100).all()):
                raise ValueError(f"Invalid chronological OOF alignment for {model}, fold {fold}")
    selected["residual"] = selected["actual"] - selected["predicted"]
    selected["horizon_group"] = np.where(selected["horizon_step"] <= 12, "First hour", "Second hour")
    return selected.reset_index(drop=True)


def available_errors(predictions: pd.DataFrame, model: str, origin: int, hour: str) -> pd.DataFrame:
    """Only forecasts already resolved at this origin may calibrate its margin."""
    return predictions[(predictions["model"] == model) & (predictions["timestamp"] <= origin)
                       & (predictions["origin_timestamp"] < origin)
                       & (predictions["horizon_group"] == hour)]


def residual_summary(part: pd.DataFrame) -> dict:
    errors = part["residual"].to_numpy(dtype=float)
    return {
        "sample_count": len(errors),
        "source_folds": ",".join(str(int(fold)) for fold in sorted(part["fold"].unique())),
        "latest_error_timestamp": int(part["timestamp"].max()) if len(errors) else np.nan,
        "mean_residual": float(errors.mean()) if len(errors) else np.nan,
        "residual_std": float(errors.std(ddof=0)) if len(errors) else np.nan,
        "underprediction_count": int((errors > 0).sum()),
        "overprediction_count": int((errors < 0).sum()),
        "signed_q90": float(np.quantile(errors, 0.90, method="linear")) if len(errors) else np.nan,
        "signed_q95": float(np.quantile(errors, 0.95, method="linear")) if len(errors) else np.nan,
        "margin_q90": residual_margin(errors, 0.90),
        "margin_q95": residual_margin(errors, 0.95),
    }


def uncertainty_analysis(training: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Separate retrospective diagnostics from origin-available parameters.

    Origin 4 uses all 110 training points and resolved OOF errors to document
    possible next-origin parameters only. It produces no future predictions.
    """
    rows = []
    for model in MODELS:
        model_oof = oof[oof["model"] == model]
        for fold in (0, 1, 2, 3):
            part = model_oof if fold == 0 else model_oof[model_oof["fold"] == fold]
            for hour in ("All steps", *HOURS):
                subset = part if hour == "All steps" else part[part["horizon_group"] == hour]
                rows.append({"analysis_type": "descriptive_oof", "model": model, "fold": fold,
                             "train_size": np.nan, "origin_timestamp": np.nan,
                             "horizon_group": hour, "threshold": np.nan, **residual_summary(subset)})
        for fold, end in enumerate([*TRAIN_ENDS, len(training)], start=1):
            origin = int(training.iloc[end - 1]["timestamp"])
            threshold = float(np.quantile(training[CPU].iloc[:end], PEAK_QUANTILE, method="linear"))
            for hour in HOURS:
                rows.append({
                    "analysis_type": "origin_calibration", "model": model, "fold": fold,
                    "train_size": end, "origin_timestamp": origin, "horizon_group": hour,
                    "threshold": threshold, **residual_summary(available_errors(oof, model, origin, hour)),
                })
    return pd.DataFrame(rows)


def replay_policy(training: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Replay fixed policies; observed demand is used solely for scoring labels."""
    rows = []
    for fold, end in enumerate(TRAIN_ENDS, start=1):
        part = oof[(oof["model"] == PRIMARY) & (oof["fold"] == fold)].copy()
        origin = int(training.iloc[end - 1]["timestamp"])
        threshold = float(np.quantile(training[CPU].iloc[:end], PEAK_QUANTILE, method="linear"))
        historical = {hour: available_errors(oof, PRIMARY, origin, hour) for hour in HOURS}
        common_eligible = all(len(part) > 0 for part in historical.values())
        for policy in POLICIES:
            margins = np.array([
                0.0 if policy.quantile is None else residual_margin(historical[hour]["residual"], policy.quantile)
                for hour in part["horizon_group"]
            ])
            available = bool(np.isfinite(margins).all())
            if available:
                upper = conservative_forecast(part["predicted"], margins)
                levels, runs = classify_alerts(part["predicted"], upper, threshold)
            else:
                upper = np.full(len(part), np.nan)
                levels = np.full(len(part), "Uncalibrated", dtype=object)
                runs = np.zeros(len(part), dtype=int)
            for i, record in enumerate(part.to_dict("records")):
                errors = historical[record["horizon_group"]]
                rows.append({
                    **record, "policy": policy.name, "margin_quantile": policy.quantile,
                    "threshold": threshold, "calibration_count": len(errors),
                    "calibration_latest_timestamp": int(errors["timestamp"].max()) if len(errors) else np.nan,
                    "common_eligible": common_eligible, "policy_available": available,
                    "nominal_margin": margins[i], "upper_forecast": upper[i],
                    "applied_margin": upper[i] - record["predicted"],
                    "alert_level": levels[i], "point_exceedance_run": runs[i],
                    "alert": bool(levels[i] != "Normal") if available else pd.NA,
                    "observed_elevated": bool(record["actual"] >= threshold),
                })
    result = pd.DataFrame(rows)
    result["alert"] = result["alert"].astype("boolean")
    return result


def policy_metrics(rows: pd.DataFrame) -> dict:
    """Score only available policies; undefined rates and empty errors are NaN.

    Alerts are five-minute forecast periods, not notifications or scaling
    actions. An episode is a contiguous alert run within a single origin.
    """
    part = rows[rows["policy_available"]]
    metric_names = (
        "observed_elevated_count", "alert_count", "alert_episodes", "high_risk_count",
        "true_positive", "false_positive", "false_negative", "true_negative", "alert_rate",
        "precision", "recall", "false_positive_rate", "underprediction_rate", "overprediction_rate",
        "average_positive_forecast_error", "mean_error_when_underpredicted",
        "average_excess", "mean_residual", "average_conservative_margin",
    )
    result = {"candidate_count": len(rows), "evaluated_count": len(part),
              "unavailable_count": len(rows) - len(part),
              "candidate_observed_elevated_count": int(rows["observed_elevated"].sum())}
    if part.empty:
        return {**result, **dict.fromkeys(metric_names, np.nan)}
    alert = part["alert"].to_numpy(dtype=bool)
    actual_event = part["observed_elevated"].to_numpy(dtype=bool)
    tp, fp = int((alert & actual_event).sum()), int((alert & ~actual_event).sum())
    fn, tn = int((~alert & actual_event).sum()), int((~alert & ~actual_event).sum())
    error = (part["actual"] - part["upper_forecast"]).to_numpy(dtype=float)
    episodes = 0
    for _, block in part.groupby("fold"):
        flags = block["alert"].to_numpy(dtype=bool)
        episodes += int((flags & ~np.r_[False, flags[:-1]]).sum())
    return {
        **result, "observed_elevated_count": int(actual_event.sum()), "alert_count": int(alert.sum()),
        "alert_episodes": episodes, "high_risk_count": int((part["alert_level"] == "High-risk condition").sum()),
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "alert_rate": float(alert.mean()), "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "underprediction_rate": float((error > 0).mean()), "overprediction_rate": float((error < 0).mean()),
        "average_positive_forecast_error": float(np.maximum(error, 0).mean()),
        "mean_error_when_underpredicted": float(error[error > 0].mean()) if (error > 0).any() else np.nan,
        "average_excess": float(np.maximum(-error, 0).mean()), "mean_residual": float(error.mean()),
        "average_conservative_margin": float(part["applied_margin"].mean()),
    }


def summarize_policy(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in POLICIES:
        part = validation[validation["policy"] == policy.name]
        for fold in (0, 1, 2, 3):
            subset = part if fold == 0 else part[part["fold"] == fold]
            rows.append({"policy": policy.name, "cohort": "all_origins", "fold": fold,
                         **policy_metrics(subset)})
        rows.append({"policy": policy.name, "cohort": "common_calibrated_origins", "fold": 0,
                     **policy_metrics(part[part["common_eligible"]])})
    return pd.DataFrame(rows)


def run_analysis(training_path: Path, predictions_path: Path, evaluation_manifest_path: Path,
                 category_path: Path, output_dir: Path, report_dir: Path) -> dict[str, pd.DataFrame]:
    from cwfc.operational_reporting import write_report, write_figures

    training = load_training_targets(training_path)
    manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(training.to_csv(index=False).encode()).hexdigest()
    if (manifest.get("validated_training_sha256") != digest or manifest.get("train_ends") != TRAIN_ENDS
            or manifest.get("horizon") != HORIZON or manifest.get("ridge_alpha") != 1.0
            or manifest.get("holdout_targets_accessed") is not False):
        raise ValueError("Step 5 manifest does not match the training data and fixed validation protocol")
    oof = validate_oof_predictions(pd.read_csv(predictions_path), training)
    categories = pd.read_csv(category_path)
    uncertainty = uncertainty_analysis(training, oof)
    validation = replay_policy(training, oof)
    summary = summarize_policy(validation)
    tables = {"uncertainty_analysis": uncertainty, "operational_policy_validation": validation,
              "policy_summary": summary}
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.17g")
    figures = write_figures(validation, summary, report_dir / "figures")
    write_report(training, oof, tables, categories, report_dir / "operational_recommendations.md", figures)
    import matplotlib

    policy_manifest = {
        "training_sha256": digest, "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (predictions_path, evaluation_manifest_path, category_path)},
        "primary_model": PRIMARY, "fallback_model": FALLBACK, "vm_count_model": "Persistence",
        "peak_quantile": PEAK_QUANTILE, "quantile_method": "linear", "threshold_comparison": ">=",
        "policies": {policy.name: policy.quantile for policy in POLICIES}, "proposed_policy": PROPOSED_POLICY,
        "policy_status": "illustrative; fixed choices, not selected by policy validation results",
        "consecutive_point_exceedances": CONSECUTIVE_STEPS, "horizon_groups": HOURS,
        "calibration_rule": "same-model OOF errors with timestamp <= origin; separate first/second hour",
        "warmup_rule": "No previous OOF errors: empirical policies unavailable; point-only comparator retained",
        "holdout_targets_accessed": False, "new_models_fitted": False, "formal_interval_coverage": False,
    }
    (output_dir / "operational_policy_manifest.json").write_text(
        json.dumps(policy_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("outputs/train_targets.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/evaluation_validation_predictions.csv"))
    parser.add_argument("--evaluation-manifest", type=Path, default=Path("outputs/evaluation_manifest.json"))
    parser.add_argument("--categories", type=Path, default=Path("outputs/category_statistics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    tables = run_analysis(args.training, args.predictions, args.evaluation_manifest, args.categories,
                          args.output_dir, args.report_dir)
    print(tables["policy_summary"].query("cohort == 'common_calibrated_origins'").round(5).to_string(index=False))
    print(f"\nStep 6 report: {args.report_dir / 'operational_recommendations.md'}")
    print("No holdout targets accessed, no models changed, no final predictions.csv generated.")


if __name__ == "__main__":
    main()

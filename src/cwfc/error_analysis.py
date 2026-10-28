"""Leakage-safe Step 5 evaluation of the fixed Step 3/4 forecasting methods.

Run from the project root with ``python -m cwfc.error_analysis``. No raw VM
readings or holdout targets are loaded; all selection uses the first 110 rows
of the training-target artifact. CPU errors are in percentage points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.advanced_evaluation import MODELS as ADVANCED_MODELS
from cwfc.baseline_evaluation import HORIZON, TRAIN_ENDS, MODELS as BASELINE_MODELS
from cwfc.baselines import persistence_forecast
from cwfc.metrics import mae, rmse

CPU = "aggregate_cpu_utilization"
VM = "active_vm_count"
TRAIN_SIZE = 110
LEADING_MODELS = ("Ridge AR (6)", "Ridge AR (3)", "Moving Average (12)", "Persistence")
HORIZON_GROUPS = {
    "Steps 1-6": (1, 6), "Steps 7-12": (7, 12),
    "Steps 13-18": (13, 18), "Steps 19-24": (19, 24),
    "First hour": (1, 12), "Second hour": (13, 24),
}


def validate_training_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Reject missing, unordered, irregular or invalid training-only targets.

    No sorting, interpolation, target imputation or calendar-date conversion
    is performed. Timestamps are source-relative seconds.
    """
    columns = ["timestamp", VM, CPU]
    if not set(columns).issubset(data.columns):
        raise ValueError(f"Required columns: {columns}")
    if len(data) != TRAIN_SIZE:
        raise ValueError("Expected exactly 110 training observations; exclude holdout")
    clean = data[columns].apply(pd.to_numeric, errors="raise").reset_index(drop=True)
    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("Training targets and timestamps must be finite")
    ts = clean["timestamp"].to_numpy(dtype=float)
    if not np.equal(ts, np.floor(ts)).all() or not np.equal(np.diff(ts), 300).all():
        raise ValueError("Timestamps must be ordered, unique, integer seconds at 300-second cadence")
    if not clean[CPU].between(0, 100).all():
        raise ValueError("CPU utilization must be in [0, 100]")
    counts = clean[VM].to_numpy(dtype=float)
    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Active VM counts must be nonnegative integers")
    return clean


def load_training_targets(path: Path) -> pd.DataFrame:
    """Parse at most 110 rows, even if a caller accidentally passes a full series.

    The normal input is outputs/train_targets.csv, already produced by EDA.
    The cap prevents parsing target observations 111-134. Short files fail.
    """
    return validate_training_targets(pd.read_csv(path, nrows=TRAIN_SIZE))


def forecast_cpu(model: str, train: np.ndarray) -> np.ndarray:
    """Reuse unchanged forecasts, with fixed windows, alpha=0.3/1.0 and clipping.

    A private copy of history is the only data supplied to a forecasting method.
    Recursive ridge forecasts append predictions, never validation observations.
    """
    history = np.array(train, dtype=float, copy=True)
    if model in BASELINE_MODELS:
        prediction = np.full(HORIZON, BASELINE_MODELS[model](history), dtype=float)
    else:
        prediction = np.asarray(ADVANCED_MODELS[model](history), dtype=float)
    if prediction.shape != (HORIZON,) or not np.isfinite(prediction).all():
        raise ValueError(f"{model} must produce 24 finite predictions")
    return prediction


def validation_predictions(data: pd.DataFrame) -> pd.DataFrame:
    """Forecast 24 steps at each of the three fixed, expanding origins."""
    data = validate_training_targets(data)
    models = list(dict.fromkeys([*BASELINE_MODELS, *ADVANCED_MODELS]))
    records = []
    for fold, end in enumerate(TRAIN_ENDS, start=1):
        train = data.iloc[:end]
        validation = data.iloc[end:end + HORIZON]
        if len(validation) != HORIZON:
            raise ValueError("Every validation fold needs 24 future observations")
        threshold = float(np.quantile(train[CPU], 0.95, method="linear"))
        # The VM forecast receives timestamp-only future information.
        vm_forecast = persistence_forecast(train, validation[["timestamp"]], VM).to_numpy()
        forecasts = [(CPU, name, forecast_cpu(name, train[CPU].to_numpy())) for name in models]
        forecasts.append((VM, "Persistence", vm_forecast))
        for target, model, predicted in forecasts:
            for step, (timestamp, actual, estimate) in enumerate(
                zip(validation["timestamp"], validation[target], predicted), start=1
            ):
                records.append({
                    "target": target, "model": model, "fold": fold, "train_size": end,
                    "origin_timestamp": int(train.iloc[-1]["timestamp"]),
                    "horizon_step": step, "timestamp": int(timestamp),
                    "elapsed_minutes": float((timestamp - data.iloc[0]["timestamp"]) / 60),
                    "actual": float(actual), "predicted": float(estimate),
                    "peak_threshold": threshold if target == CPU else np.nan,
                    # Inclusive comparison preserves the Step 3/4 definition.
                    "is_peak": bool(actual >= threshold) if target == CPU else False,
                })
    result = pd.DataFrame(records)
    result["residual"] = result["actual"] - result["predicted"]
    result["absolute_error"] = result["residual"].abs()
    return result


def relative_improvement(error: float, baseline_error: float) -> float:
    """Percent reduction; undefined (NaN) for a zero-error baseline, even 0/0."""
    if baseline_error == 0:
        return float("nan")
    return float(100 * (baseline_error - error) / baseline_error)


def error_statistics(group: pd.DataFrame) -> dict:
    """Empty subsets retain count=0 and undefined errors, never zero errors."""
    if group.empty:
        return {"count": 0, "mae": np.nan, "rmse": np.nan, "mean_residual": np.nan}
    return {
        "count": len(group), "mae": mae(group["actual"], group["predicted"]),
        "rmse": rmse(group["actual"], group["predicted"]),
        "mean_residual": float(group["residual"].mean()),
    }


def analyze_predictions(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pool unrounded point errors; retain per-fold, horizon and subset detail.

    ``fold=0`` denotes pooled validation in peak/residual output tables.
    Residual standard deviations use ddof=0 (descriptive population SD).
    """
    pooled, folds, horizons, peaks, residuals, robustness = [], [], [], [], [], []
    for (target, model), group in predictions.groupby(["target", "model"], sort=True):
        key = {"target": target, "model": model}
        pooled.append({**key, **error_statistics(group)})
        for fold, part in group.groupby("fold"):
            folds.append({
                **key, "fold": int(fold), "train_size": int(part["train_size"].iloc[0]),
                "origin_timestamp": int(part["origin_timestamp"].iloc[0]),
                "validation_start": int(part["timestamp"].min()),
                "validation_end": int(part["timestamp"].max()),
                **error_statistics(part),
            })
        if target != CPU:
            continue
        for label, (start, stop) in HORIZON_GROUPS.items():
            part = group[group["horizon_step"].between(start, stop)]
            horizons.append({**key, "horizon_group": label, "start_step": start,
                             "end_step": stop, **error_statistics(part)})
        for fold, part in [(0, group), *list(group.groupby("fold"))]:
            for subset, mask in [("peak", part["is_peak"]), ("nonpeak", ~part["is_peak"])]:
                peaks.append({
                    **key, "fold": int(fold), "subset": subset,
                    "peak_threshold": float(part["peak_threshold"].iloc[0]) if fold else np.nan,
                    **error_statistics(part[mask]),
                })
            absolute = part["absolute_error"]
            residuals.append({
                **key, "fold": int(fold), **error_statistics(part),
                "residual_std": float(part["residual"].std(ddof=0)),
                "largest_positive_residual": float(part.loc[part["residual"] > 0, "residual"].max()),
                "largest_negative_residual": float(part.loc[part["residual"] < 0, "residual"].min()),
                "residual_min": float(part["residual"].min()),
                "residual_max": float(part["residual"].max()),
                "underprediction_count": int((part["residual"] > 0).sum()),
                "overprediction_count": int((part["residual"] < 0).sum()),
                "absolute_error_median": float(absolute.median()),
                "absolute_error_p75": float(absolute.quantile(0.75)),
                "absolute_error_p90": float(absolute.quantile(0.90)),
                "absolute_error_p95": float(absolute.quantile(0.95)),
                "absolute_error_max": float(absolute.max()),
            })
    comparison = pd.DataFrame(pooled)
    for target, group in comparison.groupby("target"):
        baseline = group[group["model"] == "Persistence"].iloc[0]
        for metric in ("mae", "rmse"):
            comparison.loc[group.index, f"{metric}_improvement_pct"] = [
                relative_improvement(value, baseline[metric]) for value in group[metric]
            ]
    fold_table = pd.DataFrame(folds)
    cpu_folds = fold_table[fold_table["target"] == CPU].copy()
    cpu_folds["mae_rank"] = cpu_folds.groupby("fold")["mae"].rank(method="min")
    for model, group in cpu_folds.groupby("model"):
        robustness.append({
            "model": model, "fold_count": len(group),
            "fold_mae_mean": group["mae"].mean(),
            "fold_mae_std": group["mae"].std(ddof=0),
            "best_fold_mae": group["mae"].min(), "worst_fold_mae": group["mae"].max(),
            "worst_fold_rmse": group["rmse"].max(),
            "fold_mae_wins": int((group["mae_rank"] == 1).sum()),
        })
    return {
        "model_comparison": comparison.sort_values(["target", "mae", "rmse", "model"]).reset_index(drop=True),
        "fold_performance": fold_table,
        "horizon_error_analysis": pd.DataFrame(horizons),
        "peak_error_analysis": pd.DataFrame(peaks),
        "residual_analysis": pd.DataFrame(residuals),
        "model_robustness": pd.DataFrame(robustness),
    }


def run_analysis(input_path: Path, output_dir: Path, report_dir: Path) -> dict[str, pd.DataFrame]:
    """Write separate Step 5 artifacts; preserve all Step 1-4 evaluation files."""
    from cwfc.evaluation_reporting import write_figures, write_report

    data = load_training_targets(input_path)
    predictions = validation_predictions(data)
    tables = analyze_predictions(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {**tables, "evaluation_validation_predictions": predictions}.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.17g")
    figure_paths = write_figures(predictions, tables, report_dir / "figures")
    write_report(data, predictions, tables, report_dir / "model_evaluation.md", figure_paths,
                 input_path=input_path)
    import matplotlib

    manifest = {
        "input": str(input_path), "input_rows_used": len(data),
        "validated_training_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "train_ends": TRAIN_ENDS, "horizon": HORIZON, "peak_quantile": 0.95,
        "peak_quantile_method": "linear", "peak_comparison": "actual >= training P95",
        "residual_convention": "actual - predicted", "residual_std_ddof": 0,
        "models": list(dict.fromkeys([*BASELINE_MODELS, *ADVANCED_MODELS])),
        "smoothing_alpha": 0.3, "ridge_alpha": 1.0,
        "holdout_targets_accessed": False, "selection_status": "provisional; validation only",
        "relative_improvement_with_zero_baseline": "undefined (NaN / blank CSV cell)",
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/train_targets.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    tables = run_analysis(args.input, args.output_dir, args.report_dir)
    print(tables["model_comparison"].round(5).to_string(index=False))
    print(f"\nStep 5 report: {args.report_dir / 'model_evaluation.md'}")
    print("Holdout targets were not accessed. No final predictions.csv was generated.")


if __name__ == "__main__":
    main()


from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.metrics import mae, rmse


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = Path("outputs/train_targets.csv")
OUTPUT_DIR = Path("outputs")

HORIZON = 24
TRAIN_ENDS = [38, 62, 86]

# --------------------------------------------------
# 3. Forecasting methods
# --------------------------------------------------

def persistence(values):
    return float(values[-1])


def moving_average(values, window):
    return float(np.mean(values[-window:]))


def exponential_smoothing(values, alpha=0.3):
    return float(
        pd.Series(values)
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .iloc[-1]
    )


MODELS = {
    "Persistence": persistence,
    "Moving Average (6)": lambda x: moving_average(x, 6),
    "Moving Average (12)": lambda x: moving_average(x, 12),
    "Exponential Smoothing": exponential_smoothing,
}

# --------------------------------------------------
# 4. Expanding-window validation
# --------------------------------------------------

def main():
    """Run the original evaluation without side effects on import."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
    required = {"timestamp", "active_vm_count", "aggregate_cpu_utilization"}
    assert required.issubset(data.columns)
    assert len(data) == 110, "Expected 110 training timestamps"
    assert data["timestamp"].is_unique
    print("Training observations:", len(data))

    records = []
    fold_records = []

    cpu_values = data["aggregate_cpu_utilization"].to_numpy()

    for fold, train_end in enumerate(TRAIN_ENDS, start=1):

        train = cpu_values[:train_end]

        actual = cpu_values[
            train_end:train_end + HORIZON
        ]

        assert len(actual) == HORIZON

        # Peak threshold calculated from training data only
        peak_threshold = np.quantile(train, 0.95)

        for model_name, forecast_function in MODELS.items():

            # Direct 24-step forecast:
            # same forecast origin, no validation values used.
            estimate = forecast_function(train)

            predictions = np.full(
                HORIZON, estimate, dtype=float
            )

            fold_mae = mae(actual, predictions)
            fold_rmse = rmse(actual, predictions)

            peak_mask = actual >= peak_threshold

            peak_mae = (
                mae(actual[peak_mask], predictions[peak_mask])
                if peak_mask.any()
                else np.nan
            )

            fold_records.append({
                "fold": fold,
                "model": model_name,
                "train_size": train_end,
                "mae": fold_mae,
                "rmse": fold_rmse,
                "peak_mae": peak_mae,
                "peak_count": int(peak_mask.sum()),
            })

            for i in range(HORIZON):
                records.append({
                    "fold": fold,
                    "model": model_name,
                    "timestamp": int(
                        data.iloc[train_end + i]["timestamp"]
                    ),
                    "actual": float(actual[i]),
                    "predicted": float(predictions[i]),
                })

    # --------------------------------------------------
    # 5. Summarize validation performance
    # --------------------------------------------------

    predictions_df = pd.DataFrame(records)
    fold_results = pd.DataFrame(fold_records)

    summary_rows = []

    for model_name, group in predictions_df.groupby("model"):

        summary_rows.append({
            "model": model_name,
            "mae": mae(group["actual"], group["predicted"]),
            "rmse": rmse(group["actual"], group["predicted"]),
        })

    summary = pd.DataFrame(summary_rows).sort_values("mae")

    baseline_mae = summary.loc[
        summary["model"] == "Persistence", "mae"
    ].iloc[0]

    summary["improvement_pct"] = (
        100 * (baseline_mae - summary["mae"]) / baseline_mae
    )

    print("\n========== BASELINE RESULTS ==========")
    print(summary.round(5).to_string(index=False))

    print("\n========== FOLD RESULTS ==========")
    print(fold_results.round(5).to_string(index=False))

    # --------------------------------------------------
    # 6. Active VM count baseline
    # --------------------------------------------------

    vm_counts = data["active_vm_count"].to_numpy()

    print("\n========== ACTIVE VM COUNT ==========")

    for train_end in TRAIN_ENDS:

        actual = vm_counts[train_end:train_end + HORIZON]
        predicted = np.full(HORIZON, vm_counts[train_end - 1])

        print(
            f"Train size {train_end}: "
            f"MAE={mae(actual, predicted):.4f}, "
            f"RMSE={rmse(actual, predicted):.4f}"
        )

    # --------------------------------------------------
    # 7. Save reproducible results
    # --------------------------------------------------

    summary.to_csv(
        OUTPUT_DIR / "baseline_summary.csv", index=False
    )

    fold_results.to_csv(
        OUTPUT_DIR / "baseline_fold_results.csv", index=False
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "baseline_validation_predictions.csv",
        index=False
    )

    print("\nBaseline evaluation completed successfully!")


if __name__ == "__main__":
    main()

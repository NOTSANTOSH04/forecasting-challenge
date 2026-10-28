
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
# 2. Existing baseline
# --------------------------------------------------

def moving_average_12(train, horizon=24):
    estimate = np.mean(train[-12:])
    return np.full(horizon, estimate)


# --------------------------------------------------
# 3. Linear trend forecasting
# --------------------------------------------------

def linear_trend(train, horizon=24, window=12):

    recent = np.asarray(train[-window:], dtype=float)

    x = np.arange(window, dtype=float)

    slope, intercept = np.polyfit(x, recent, 1)

    future_x = np.arange(
        window, window + horizon, dtype=float
    )

    forecasts = slope * future_x + intercept

    return np.clip(forecasts, 0, 100)


# --------------------------------------------------
# 4. Ridge autoregressive model
# --------------------------------------------------

def ridge_autoregression(
    train,
    horizon=24,
    lags=3,
    alpha=1.0,
):

    train = np.asarray(train, dtype=float)

    if len(train) <= lags:
        raise ValueError("Not enough training observations")

    # Create supervised training examples.
    # Each row uses only previous observations.
    X = np.array([
        train[i-lags:i][::-1]
        for i in range(lags, len(train))
    ])

    y = train[lags:]

    # Center inputs and target so the intercept
    # is not penalized by regularization.
    x_mean = X.mean(axis=0)
    y_mean = y.mean()

    X_centered = X - x_mean
    y_centered = y - y_mean

    coefficients = np.linalg.solve(
        X_centered.T @ X_centered
        + alpha * np.eye(lags),
        X_centered.T @ y_centered,
    )

    intercept = y_mean - x_mean @ coefficients

    # Recursive forecasting:
    # Future predictions become inputs for
    # subsequent prediction steps.
    history = list(train)
    forecasts = []

    for _ in range(horizon):

        lag_values = np.array(
            history[-lags:][::-1],
            dtype=float,
        )

        prediction = float(
            intercept + lag_values @ coefficients
        )

        prediction = float(np.clip(prediction, 0, 100))

        forecasts.append(prediction)
        history.append(prediction)

    return np.array(forecasts)


# --------------------------------------------------
# 5. Models to evaluate
# --------------------------------------------------

MODELS = {
    "Moving Average (12)": moving_average_12,

    "Linear Trend (12)": lambda x: linear_trend(
        x, window=12
    ),

    "Linear Trend (24)": lambda x: linear_trend(
        x, window=24
    ),

    "Ridge AR (3)": lambda x: ridge_autoregression(
        x, lags=3
    ),

    "Ridge AR (6)": lambda x: ridge_autoregression(
        x, lags=6
    ),
}


# --------------------------------------------------
# 6. Expanding-window validation
# --------------------------------------------------

def main():
    """Run the original evaluation without side effects on import."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
    assert len(data) == 110
    assert data["timestamp"].is_unique
    values = data["aggregate_cpu_utilization"].to_numpy(dtype=float)

    records = []
    fold_records = []

    for fold, train_end in enumerate(TRAIN_ENDS, start=1):

        train = values[:train_end]

        actual = values[
            train_end:train_end + HORIZON
        ]

        assert len(actual) == HORIZON

        # Calculate peak threshold using training only.
        peak_threshold = np.quantile(train, 0.95)

        print(f"\nEvaluating fold {fold}")

        for name, forecast_function in MODELS.items():

            predictions = forecast_function(train)

            assert len(predictions) == HORIZON
            assert np.isfinite(predictions).all()

            peak_mask = actual >= peak_threshold

            peak_error = (
                mae(actual[peak_mask], predictions[peak_mask])
                if peak_mask.any()
                else np.nan
            )

            fold_records.append({
                "fold": fold,
                "model": name,
                "mae": mae(actual, predictions),
                "rmse": rmse(actual, predictions),
                "peak_mae": peak_error,
                "peak_count": int(peak_mask.sum()),
            })

            for step in range(HORIZON):

                records.append({
                    "fold": fold,
                    "model": name,
                    "horizon_step": step + 1,
                    "timestamp": int(
                        data.iloc[train_end + step]["timestamp"]
                    ),
                    "actual": actual[step],
                    "predicted": predictions[step],
                })


    # --------------------------------------------------
    # 7. Overall model comparison
    # --------------------------------------------------

    results = pd.DataFrame(records)
    fold_results = pd.DataFrame(fold_records)

    summary = []

    for name, group in results.groupby("model"):

        summary.append({
            "model": name,
            "mae": mae(group["actual"], group["predicted"]),
            "rmse": rmse(group["actual"], group["predicted"]),
        })

    summary = pd.DataFrame(summary).sort_values("mae")

    baseline_mae = summary.loc[
        summary["model"] == "Moving Average (12)", "mae"
    ].iloc[0]

    summary["improvement_pct"] = (
        100 * (baseline_mae - summary["mae"]) / baseline_mae
    )

    print("\n========== ADVANCED MODEL RESULTS ==========")
    print(summary.round(5).to_string(index=False))

    print("\n========== FOLD RESULTS ==========")
    print(fold_results.round(5).to_string(index=False))


    # --------------------------------------------------
    # 8. Short-horizon vs long-horizon errors
    # --------------------------------------------------

    print("\n========== FORECAST HORIZON ANALYSIS ==========")

    results["horizon_group"] = np.where(
        results["horizon_step"] <= 12,
        "First hour",
        "Second hour",
    )

    for (name, period), group in results.groupby(
        ["model", "horizon_group"]
    ):

        print(
            name,
            "|", period,
            "| MAE:", round(
                mae(group["actual"], group["predicted"]), 5
            )
        )


    # --------------------------------------------------
    # 9. Save results
    # --------------------------------------------------

    summary.to_csv(
        OUTPUT_DIR / "advanced_model_summary.csv",
        index=False,
    )

    fold_results.to_csv(
        OUTPUT_DIR / "advanced_fold_results.csv",
        index=False,
    )

    results.to_csv(
        OUTPUT_DIR / "advanced_validation_predictions.csv",
        index=False,
    )

    print("\nAdvanced model evaluation completed!")


if __name__ == "__main__":
    main()

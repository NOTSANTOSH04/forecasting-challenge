"""Generate the submission directly from the compressed challenge release.

Official data use the fixed 110/24 split. Smaller smoke inputs reserve the
requested horizon (default 24) and require at least seven training timestamps.
Holdout target values are discarded after aggregation; only their grid is kept.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.advanced_evaluation import ridge_autoregression
from cwfc.baselines import persistence_forecast

CPU = "aggregate_cpu_utilization"
VM = "active_vm_count"
OUTPUT_COLUMNS = ["timestamp", VM, CPU]
CPU_COLUMNS = ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]
CADENCE = 300
OFFICIAL_SIZE = 134
OFFICIAL_HORIZON = 24
LAGS = 6
RIDGE_ALPHA = 1.0
LOGGER = logging.getLogger(__name__)


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _timestamps(values: pd.Series, name: str) -> np.ndarray:
    try:
        numbers = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be numeric relative seconds") from error
    if (not np.isfinite(numbers).all() or (numbers < 0).any()
            or (numbers > 2**53 - 1).any() or not np.equal(numbers, np.floor(numbers)).all()):
        raise ValueError(f"{name} must be finite nonnegative integer relative seconds")
    return numbers.astype(np.int64)


def _vm_ids(frame: pd.DataFrame, name: str) -> pd.Series:
    if "vm_id" not in frame or frame["vm_id"].isna().any():
        raise ValueError(f"{name} requires nonmissing vm_id values")
    identifiers = frame["vm_id"].astype("string")
    if identifiers.str.strip().eq("").any():
        raise ValueError(f"{name} requires nonblank vm_id values")
    return identifiers


def aggregate_targets(cpu: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Reject invalid records, then compute targets independently per timestamp.

    Rows may arrive in VM-major or arbitrary order. Duplicate VM/timestamp keys,
    missing/nonnumeric CPU values, invalid ranges and unmatched metadata fail;
    nothing is imputed or silently dropped. Changing VM membership is allowed.
    """
    if cpu.empty or metadata.empty:
        raise ValueError("CPU readings and metadata must not be empty")
    if not set(CPU_COLUMNS).issubset(cpu.columns):
        raise ValueError(f"CPU readings require columns: {', '.join(CPU_COLUMNS)}")
    readings = cpu[CPU_COLUMNS].copy()
    readings["vm_id"] = _vm_ids(readings, "CPU readings")
    metadata_ids = _vm_ids(metadata, "Metadata")
    if metadata_ids.duplicated().any():
        raise ValueError("Metadata must contain one unique row per vm_id")
    if set(readings["vm_id"].unique()) != set(metadata_ids):
        raise ValueError("CPU and metadata VM populations must match")
    readings["timestamp"] = _timestamps(readings["timestamp"], "CPU timestamps")
    if readings.duplicated(["vm_id", "timestamp"]).any():
        raise ValueError("Duplicate VM-timestamp CPU records are not allowed")
    columns = ["min_cpu", "avg_cpu", "max_cpu"]
    try:
        utilization = readings[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("CPU fields must be numeric") from error
    if (not np.isfinite(utilization).all() or (utilization < 0).any() or (utilization > 100).any()
            or (utilization[:, 0] > utilization[:, 1]).any()
            or (utilization[:, 1] > utilization[:, 2]).any()):
        raise ValueError("CPU fields must be finite and satisfy 0 <= min_cpu <= avg_cpu <= max_cpu <= 100")
    readings[columns] = utilization
    targets = readings.groupby("timestamp", as_index=False, sort=True).agg(
        active_vm_count=("vm_id", "nunique"), aggregate_cpu_utilization=("avg_cpu", "mean")
    )
    if not np.equal(np.diff(targets["timestamp"]), CADENCE).all():
        raise ValueError("Aggregate timestamps must form a complete 300-second grid; gaps are not imputed")
    return targets[OUTPUT_COLUMNS]


def load_targets(data_dir: Path) -> pd.DataFrame:
    """Read both original compressed CSVs; metadata CPU summaries are not features."""
    try:
        cpu = pd.read_csv(data_dir / "vm_cpu_readings.csv.gz", dtype={"vm_id": "string"})
        metadata = pd.read_csv(data_dir / "vm_metadata.csv.gz", usecols=["vm_id"], dtype={"vm_id": "string"})
    except (OSError, EOFError, ValueError) as error:
        raise ValueError("Cannot read required compressed CPU/metadata CSVs; check paths, headers and gzip integrity") from error
    return aggregate_targets(cpu, metadata)


def split_targets(targets: pd.DataFrame, horizon: int = OFFICIAL_HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return training targets and a timestamp-only future template.

    The official-size series always uses 110/24. A shorter synthetic series
    uses its prefix for training and its last `horizon` timestamps for output.
    Larger unexpected series are rejected rather than changing the contract.
    """
    _positive_integer(horizon, "horizon")
    if len(targets) > OFFICIAL_SIZE:
        raise ValueError("Expected the 134-timestamp official release or a smaller smoke dataset")
    if len(targets) == OFFICIAL_SIZE and horizon != OFFICIAL_HORIZON:
        raise ValueError("The official 134-timestamp release requires the fixed 110/24 split")
    if len(targets) - horizon <= LAGS:
        raise ValueError("Ridge AR (6) requires at least seven training timestamps before the requested horizon")
    train = targets.iloc[:-horizon][OUTPUT_COLUMNS].copy().reset_index(drop=True)
    future = targets.iloc[-horizon:][["timestamp"]].copy().reset_index(drop=True)
    return train, future


def forecast_targets(train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Reuse the unchanged six-lag recursive ridge and VM persistence functions.

    The future argument must contain timestamps only: observed future targets
    cannot be passed across this forecasting boundary.
    """
    if list(future.columns) != ["timestamp"]:
        raise ValueError("Future information must contain only the timestamp column")
    if not set(OUTPUT_COLUMNS).issubset(train.columns) or len(train) <= LAGS or future.empty:
        raise ValueError("Provide at least seven historical targets and nonempty future timestamps")
    training_times = _timestamps(train["timestamp"], "Training timestamps")
    future_times = _timestamps(future["timestamp"], "Future timestamps")
    if not np.equal(np.diff(np.r_[training_times, future_times]), CADENCE).all():
        raise ValueError("Training and future timestamps must be chronological and contiguous at 300-second cadence")
    history = train[[VM, CPU]].to_numpy(dtype=float)
    if (not np.isfinite(history).all() or (history[:, 0] < 0).any()
            or not np.equal(history[:, 0], np.floor(history[:, 0])).all()
            or (history[:, 1] < 0).any() or (history[:, 1] > 100).any()):
        raise ValueError("Training targets must have finite nonnegative integer VM counts and CPU in [0,100]")
    cpu = ridge_autoregression(history[:, 1].copy(), horizon=len(future_times), lags=LAGS, alpha=RIDGE_ALPHA)
    counts = persistence_forecast(train, future, VM).to_numpy(dtype=float)
    predictions = pd.DataFrame({"timestamp": future_times, VM: counts.astype(np.int64), CPU: cpu})
    validate_predictions(predictions, future_times)
    return predictions


def validate_predictions(predictions: pd.DataFrame, expected_timestamps) -> None:
    """Enforce the exact submission schema, grid, row count and numeric bounds."""
    expected = np.asarray(expected_timestamps)
    if (list(predictions.columns) != OUTPUT_COLUMNS or predictions.empty or expected.ndim != 1
            or len(predictions) != len(expected)):
        raise ValueError("Prediction columns/order and row count must match the submission contract")
    if not all(pd.api.types.is_numeric_dtype(predictions[column]) for column in OUTPUT_COLUMNS):
        raise ValueError("All prediction columns must be numeric")
    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("Predictions must not contain missing or nonfinite values")
    timestamps = _timestamps(predictions["timestamp"], "Prediction timestamps")
    if (not np.array_equal(timestamps, expected) or not np.equal(np.diff(timestamps), CADENCE).all()
            or len(np.unique(timestamps)) != len(timestamps)):
        raise ValueError("Prediction timestamps must match the unique ordered future grid exactly")
    if not predictions[CPU].between(0, 100).all() or not predictions[VM].ge(0).all():
        raise ValueError("CPU forecasts must be in [0,100] and VM counts nonnegative")
    if not np.equal(predictions[VM], np.floor(predictions[VM])).all():
        raise ValueError("Persistence VM forecasts must be integer counts")


def generate_predictions(data_dir: Path, horizon: int = OFFICIAL_HORIZON) -> pd.DataFrame:
    _positive_integer(horizon, "horizon")
    targets = load_targets(data_dir)
    train, future = split_targets(targets, horizon=horizon)
    # Keep no observed holdout targets in the forecasting workflow.
    del targets
    LOGGER.info("Training timestamps: %d; forecast horizon: %d; CPU model: Ridge AR (6), alpha=1.0; VM model: persistence",
                len(train), len(future))
    if len(train) + len(future) < OFFICIAL_SIZE:
        LOGGER.info("Smaller smoke dataset; prefix/%d split with %d training timestamps", len(future), len(train))
    return forecast_targets(train, future)


def write_predictions(predictions: pd.DataFrame, output: Path) -> None:
    validate_predictions(predictions, predictions["timestamp"].to_numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False, float_format="%.17g", lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("challenge_data"), help="Directory containing both compressed release CSVs")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--horizon", type=int, default=OFFICIAL_HORIZON, help="Final timestamps to forecast; official data require 24")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.output.resolve() in {(args.data / name).resolve() for name in ("vm_cpu_readings.csv.gz", "vm_metadata.csv.gz", "DATA_CARD.md", "SHA256SUMS")}:
            raise ValueError("Output must not overwrite an input release file")
        predictions = generate_predictions(args.data, horizon=args.horizon)
        write_predictions(predictions, args.output)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.exit(2, f"error: {error}\n")
    LOGGER.info("Wrote %d predictions to %s; holdout targets were not used for fitting or evaluation", len(predictions), args.output)


if __name__ == "__main__":
    main()

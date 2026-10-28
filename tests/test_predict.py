"""Submission contract, raw input validation and end-to-end holdout isolation."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cwfc import predict
from cwfc.advanced_evaluation import ridge_autoregression


def release_frames(size=134, vm_count=3, start=900):
    times = np.arange(size) * 300 + start
    cpu = pd.DataFrame([
        {"timestamp": int(timestamp), "vm_id": f"vm-{vm}", "min_cpu": 0.0,
         "max_cpu": 100.0, "avg_cpu": 10 + vm + 0.2 * np.sin(step / 6)}
        for vm in range(vm_count) for step, timestamp in enumerate(times)
    ])
    metadata = pd.DataFrame({"vm_id": [f"vm-{vm}" for vm in range(vm_count)]})
    return cpu, metadata


def write_release(directory, cpu, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    cpu.to_csv(directory / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    metadata.to_csv(directory / "vm_metadata.csv.gz", index=False, compression="gzip")


def test_target_aggregation_is_unweighted_and_ordered():
    cpu, metadata = release_frames(size=10, vm_count=2)
    # Deliberately unsorted raw records; target ordering remains chronological.
    targets = predict.aggregate_targets(cpu.iloc[::-1], metadata)
    assert list(targets.columns) == predict.OUTPUT_COLUMNS
    np.testing.assert_array_equal(targets.timestamp, np.arange(10) * 300 + 900)
    np.testing.assert_array_equal(targets.active_vm_count, 2)
    np.testing.assert_allclose(targets.aggregate_cpu_utilization, 10.5 + 0.2 * np.sin(np.arange(10) / 6))


def test_official_size_has_exact_110_24_split(tmp_path):
    cpu, metadata = release_frames()
    write_release(tmp_path, cpu, metadata)
    targets = predict.load_targets(tmp_path)
    train, future = predict.split_targets(targets)
    assert len(train) == 110 and len(future) == 24
    assert list(future.columns) == ["timestamp"]
    result = predict.generate_predictions(tmp_path)
    assert result.shape == (24, 3)
    assert list(result.columns) == predict.OUTPUT_COLUMNS
    np.testing.assert_array_equal(result.timestamp, targets.timestamp.iloc[-24:])
    assert result.active_vm_count.eq(3).all()
    assert result.aggregate_cpu_utilization.between(0, 100).all()
    assert np.isfinite(result.to_numpy()).all()
    with pytest.raises(ValueError, match="110/24"):
        predict.generate_predictions(tmp_path, horizon=12)


def test_holdout_cpu_targets_cannot_change_final_predictions(tmp_path):
    cpu, metadata = release_frames()
    write_release(tmp_path, cpu, metadata)
    before = predict.generate_predictions(tmp_path)
    forecast_start = before.timestamp.min()
    cpu.loc[cpu.timestamp >= forecast_start, "avg_cpu"] = 97.0
    write_release(tmp_path, cpu, metadata)
    after = predict.generate_predictions(tmp_path)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_holdout_vm_counts_cannot_change_vm_forecasts(tmp_path):
    cpu, metadata = release_frames()
    write_release(tmp_path, cpu, metadata)
    before = predict.generate_predictions(tmp_path)
    cpu = cpu[~((cpu.timestamp >= before.timestamp.min()) & (cpu.vm_id != "vm-0"))]
    write_release(tmp_path, cpu, metadata)
    after = predict.generate_predictions(tmp_path)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_fitting_boundary_receives_only_chronological_training(tmp_path, monkeypatch):
    cpu, metadata = release_frames()
    write_release(tmp_path, cpu, metadata)
    expected = predict.aggregate_targets(cpu, metadata).iloc[:110].aggregate_cpu_utilization.to_numpy()
    captured = []

    def record_fit(history, horizon, lags, alpha):
        np.testing.assert_array_equal(history, expected)
        assert horizon == 24 and lags == 6 and alpha == 1.0
        captured.append(history.copy())
        return ridge_autoregression(history, horizon=horizon, lags=lags, alpha=alpha)

    monkeypatch.setattr(predict, "ridge_autoregression", record_fit)
    predict.generate_predictions(tmp_path)
    assert len(captured) == 1


@pytest.mark.parametrize("end", [38, 62, 86])
def test_production_model_equals_original_on_validation_origins(end):
    cpu, metadata = release_frames()
    targets = predict.aggregate_targets(cpu, metadata)
    train = targets.iloc[:end]
    future = targets.iloc[end:end + 24][["timestamp"]]
    actual = predict.forecast_targets(train, future)
    expected = ridge_autoregression(train.aggregate_cpu_utilization.to_numpy(), horizon=24, lags=6, alpha=1.0)
    np.testing.assert_array_equal(actual.aggregate_cpu_utilization, expected)
    assert actual.active_vm_count.eq(train.active_vm_count.iloc[-1]).all()


def test_recursive_forecasting_uses_generated_lags(monkeypatch):
    cpu, metadata = release_frames(size=14)
    targets = predict.aggregate_targets(cpu, metadata)
    train, future = predict.split_targets(targets, horizon=4)
    history = train.aggregate_cpu_utilization.to_numpy()
    coefficients = np.array([0.4, 0.2, 0.1, 0.05, 0.03, 0.02])
    x = np.array([history[i-6:i][::-1] for i in range(6, len(history))])
    intercept = history[6:].mean() - x.mean(axis=0) @ coefficients
    monkeypatch.setattr(np.linalg, "solve", lambda *_: coefficients)
    expected, recursive = [], list(history)
    for _ in range(4):
        value = np.clip(intercept + np.asarray(recursive[-6:][::-1]) @ coefficients, 0, 100)
        expected.append(value)
        recursive.append(value)
    np.testing.assert_allclose(predict.forecast_targets(train, future).aggregate_cpu_utilization, expected)


def test_smoke_cli_is_deterministic_and_needs_no_intermediate_files(tmp_path):
    cpu, metadata = release_frames(size=18, vm_count=2)
    data = tmp_path / "challenge_data"
    write_release(data, cpu, metadata)
    command = [sys.executable, "-m", "cwfc.predict", "--data", str(data), "--horizon", "4", "--output", "predictions.csv"]
    for _ in range(2):
        completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
        assert "Ridge AR (6)" in completed.stderr
        assert "Smaller smoke dataset" in completed.stderr
        current = (tmp_path / "predictions.csv").read_bytes()
        if _ == 0:
            first = current
        else:
            assert current == first
    output = pd.read_csv(tmp_path / "predictions.csv")
    assert output.shape == (4, 3)
    assert not (tmp_path / "outputs").exists()


def test_vm_persistence_uses_last_training_population(tmp_path):
    cpu, metadata = release_frames(size=18, vm_count=3)
    last_training_time = 900 + 13 * 300
    cpu = cpu[~((cpu.timestamp == last_training_time) & (cpu.vm_id != "vm-0"))]
    write_release(tmp_path, cpu, metadata)
    assert predict.generate_predictions(tmp_path, horizon=4).active_vm_count.eq(1).all()


@pytest.mark.parametrize("constant", [0.0, 100.0])
def test_constant_cpu_and_bounds(tmp_path, constant):
    cpu, metadata = release_frames(size=18)
    cpu["avg_cpu"] = constant
    write_release(tmp_path, cpu, metadata)
    predictions = predict.generate_predictions(tmp_path, horizon=4)
    np.testing.assert_allclose(predictions.aggregate_cpu_utilization, constant)


@pytest.mark.parametrize("problem", ["empty", "columns", "missing_cpu", "non_numeric", "nonfinite", "bounds", "order", "duplicate", "missing_vm", "blank_vm", "missing_meta", "duplicate_meta", "gap", "fractional_time"])
def test_malformed_raw_inputs_are_rejected(problem):
    cpu, metadata = release_frames(size=18)
    if problem == "empty":
        cpu = cpu.iloc[:0]
    elif problem == "columns":
        cpu = cpu.drop(columns=["avg_cpu"])
    elif problem == "missing_cpu":
        cpu.loc[0, "avg_cpu"] = np.nan
    elif problem == "non_numeric":
        cpu["avg_cpu"] = cpu["avg_cpu"].astype(object)
        cpu.loc[0, "avg_cpu"] = "bad"
    elif problem == "nonfinite":
        cpu.loc[0, "avg_cpu"] = np.inf
    elif problem == "bounds":
        cpu.loc[0, "avg_cpu"] = 101
    elif problem == "order":
        cpu.loc[0, "min_cpu"] = 90
    elif problem == "duplicate":
        cpu = pd.concat([cpu, cpu.iloc[:1]], ignore_index=True)
    elif problem == "missing_vm":
        cpu.loc[0, "vm_id"] = None
    elif problem == "blank_vm":
        cpu.loc[0, "vm_id"] = " "
    elif problem == "missing_meta":
        metadata = metadata.iloc[:1]
    elif problem == "duplicate_meta":
        metadata = pd.concat([metadata, metadata.iloc[:1]], ignore_index=True)
    elif problem == "gap":
        cpu = cpu[cpu.timestamp != 900 + 5 * 300]
    else:
        cpu["timestamp"] = cpu.timestamp.astype(float)
        cpu.loc[0, "timestamp"] = 901.5
    with pytest.raises(ValueError):
        predict.aggregate_targets(cpu, metadata)


@pytest.mark.parametrize("horizon", [0, -1, 134, True])
def test_invalid_horizons(horizon):
    cpu, metadata = release_frames()
    targets = predict.aggregate_targets(cpu, metadata)
    with pytest.raises(ValueError):
        predict.split_targets(targets, horizon)


def test_insufficient_and_unexpected_history_rejected():
    for size in [10, 135]:
        cpu, metadata = release_frames(size=size)
        with pytest.raises(ValueError):
            predict.split_targets(predict.aggregate_targets(cpu, metadata), horizon=4)


def test_forecasting_boundary_rejects_future_targets():
    cpu, metadata = release_frames(size=18)
    targets = predict.aggregate_targets(cpu, metadata)
    with pytest.raises(ValueError, match="only the timestamp"):
        predict.forecast_targets(targets.iloc[:14], targets.iloc[14:])
    train, future = predict.split_targets(targets, horizon=4)
    with pytest.raises(ValueError, match="chronological"):
        predict.forecast_targets(train.iloc[::-1], future)


@pytest.mark.parametrize("problem", ["columns", "count", "timestamp", "duplicate", "nan", "negative_vm", "cpu_bounds"])
def test_submission_validator(problem):
    cpu, metadata = release_frames(size=18)
    train, future = predict.split_targets(predict.aggregate_targets(cpu, metadata), horizon=4)
    result = predict.forecast_targets(train, future)
    if problem == "columns":
        result = result[predict.OUTPUT_COLUMNS[::-1]]
    elif problem == "count":
        result = result.iloc[:3]
    elif problem == "timestamp":
        result.loc[0, "timestamp"] += 1
    elif problem == "duplicate":
        result.loc[1, "timestamp"] = result.loc[0, "timestamp"]
    elif problem == "nan":
        result.loc[0, "aggregate_cpu_utilization"] = np.nan
    elif problem == "negative_vm":
        result.loc[0, "active_vm_count"] = -1
    else:
        result.loc[0, "aggregate_cpu_utilization"] = 101
    with pytest.raises(ValueError):
        predict.validate_predictions(result, future.timestamp.to_numpy())


def test_cli_errors_are_clear(tmp_path):
    completed = subprocess.run([sys.executable, "-m", "cwfc.predict", "--data", str(tmp_path)],
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "error:" in completed.stderr and "Traceback" not in completed.stderr

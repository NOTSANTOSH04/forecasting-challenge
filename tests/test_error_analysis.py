"""Tests for the evaluation protocol, not fitted rankings on the public dataset."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cwfc import error_analysis as analysis
from cwfc.advanced_evaluation import ridge_autoregression


@pytest.fixture
def training():
    step = np.arange(110)
    return pd.DataFrame({
        "timestamp": 900 + 300 * step,
        analysis.VM: np.full(110, 10000),
        analysis.CPU: 9.0 + 0.3 * np.sin(step / 7) + 0.002 * step,
    })


def test_pooled_metrics_and_residual_convention(training):
    predictions = analysis.validation_predictions(training)
    # Each fold has a different known error magnitude, making pooled RMSE
    # different from a simple mean of fold RMSEs.
    residual = predictions["fold"].map({1: 3.0, 2: -1.0, 3: 0.0})
    predictions["predicted"] = predictions["actual"] - residual
    predictions["residual"] = residual
    predictions["absolute_error"] = residual.abs()
    tables = analysis.analyze_predictions(predictions)
    row = tables["model_comparison"].query("target == @analysis.CPU and model == 'Persistence'").iloc[0]
    assert row["mae"] == pytest.approx(4 / 3)
    assert row["rmse"] == pytest.approx(np.sqrt(10 / 3))
    assert row["rmse"] != pytest.approx(4 / 3)
    assert row["mean_residual"] == pytest.approx(2 / 3)
    residuals = tables["residual_analysis"]
    row = residuals[(residuals["model"] == "Persistence") & (residuals["fold"] == 0)].iloc[0]
    assert row["largest_positive_residual"] == 3
    assert row["largest_negative_residual"] == -1
    assert row["residual_std"] == pytest.approx(np.std([3, -1, 0], ddof=0))
    assert row["absolute_error_p95"] == 3


def test_fixed_origins_and_24_step_alignment(training):
    predictions = analysis.validation_predictions(training)
    assert len(predictions) == 3 * 24 * 9  # Eight CPU methods, one VM baseline.
    assert not predictions.duplicated(["target", "model", "fold", "timestamp"]).any()
    for (target, model, fold), group in predictions.groupby(["target", "model", "fold"]):
        end = [38, 62, 86][fold - 1]
        np.testing.assert_array_equal(group["horizon_step"], np.arange(1, 25))
        np.testing.assert_array_equal(group["timestamp"], training["timestamp"].iloc[end:end + 24])
        np.testing.assert_array_equal(group["actual"], training[target].iloc[end:end + 24])
        assert (group["timestamp"] > group["origin_timestamp"]).all()
        np.testing.assert_array_equal(
            group["timestamp"] - group["origin_timestamp"], 300 * group["horizon_step"]
        )
        assert (group["train_size"] == end).all()


@pytest.mark.parametrize("fold,end", [(1, 38), (2, 62), (3, 86)])
def test_future_target_changes_cannot_change_forecasts_or_thresholds(training, fold, end):
    before = analysis.validation_predictions(training)
    poisoned = training.copy()
    poisoned.loc[end:, analysis.CPU] = 99.0
    poisoned.loc[end:, analysis.VM] = 1
    after = analysis.validation_predictions(poisoned)
    columns = ["target", "model", "fold", "horizon_step", "timestamp", "predicted", "peak_threshold"]
    pd.testing.assert_frame_equal(
        before.loc[before["fold"] == fold, columns].reset_index(drop=True),
        after.loc[after["fold"] == fold, columns].reset_index(drop=True),
    )
    cpu = before[(before["fold"] == fold) & (before["target"] == analysis.CPU)]
    expected_threshold = np.quantile(training[analysis.CPU].iloc[:end], 0.95)
    np.testing.assert_allclose(cpu["peak_threshold"], expected_threshold)


def test_ridge_recursive_inputs_are_previous_predictions(monkeypatch):
    # Force known fitted coefficients; subsequent predictions must use their
    # own generated lag values while keeping the fitted coefficients fixed.
    train = np.array([10.0, 11.0, 9.0, 12.0, 8.0, 13.0])
    coefficients = np.array([0.5, 0.2])
    monkeypatch.setattr(np.linalg, "solve", lambda *_: coefficients)
    x = np.array([train[i - 2:i][::-1] for i in range(2, len(train))])
    intercept = train[2:].mean() - x.mean(axis=0) @ coefficients
    history = list(train)
    expected = []
    for _ in range(24):
        value = np.clip(intercept + 0.5 * history[-1] + 0.2 * history[-2], 0, 100)
        expected.append(value)
        history.append(value)
    np.testing.assert_allclose(ridge_autoregression(train, lags=2), expected)


def test_empty_peak_subsets_are_undefined(training):
    # Later decreasing observations never exceed their training-derived P95.
    training[analysis.CPU] = np.linspace(20, 10, 110)
    tables = analysis.analyze_predictions(analysis.validation_predictions(training))
    peaks = tables["peak_error_analysis"]
    empty = peaks[peaks["subset"] == "peak"]
    assert (empty["count"] == 0).all()
    assert empty[["mae", "rmse", "mean_residual"]].isna().all().all()
    nonpeak = peaks[(peaks["subset"] == "nonpeak") & (peaks["fold"] > 0)]
    assert (nonpeak["count"] == 24).all()


def test_constant_targets_and_zero_baseline(training):
    training[analysis.CPU] = 7.5
    predictions = analysis.validation_predictions(training)
    np.testing.assert_allclose(predictions.loc[predictions["target"] == analysis.CPU, "predicted"], 7.5)
    tables = analysis.analyze_predictions(predictions)
    vm_folds = tables["fold_performance"].query("target == @analysis.VM")
    assert len(vm_folds) == 3
    assert (vm_folds[["mae", "rmse"]] == 0).all().all()
    assert tables["model_comparison"][["mae_improvement_pct", "rmse_improvement_pct"]].isna().all().all()
    residuals = tables["residual_analysis"]
    persistence = residuals[residuals["model"] == "Persistence"]
    assert persistence[["largest_positive_residual", "largest_negative_residual"]].isna().all().all()
    # Inclusive training-P95 convention: all identical observations are peaks.
    peaks = tables["peak_error_analysis"]
    assert (peaks[(peaks["subset"] == "peak") & (peaks["fold"] > 0)]["count"] == 24).all()


@pytest.mark.parametrize("error,baseline,expected", [(2, 4, 50), (6, 4, -50), (4, 4, 0)])
def test_relative_improvements(error, baseline, expected):
    assert analysis.relative_improvement(error, baseline) == expected


@pytest.mark.parametrize("error", [0, 3])
def test_zero_baseline_relative_improvement_is_undefined(error):
    assert np.isnan(analysis.relative_improvement(error, 0))


def test_horizon_groups_pool_correct_steps(training):
    predictions = analysis.validation_predictions(training)
    tables = analysis.analyze_predictions(predictions)
    for row in tables["horizon_error_analysis"].itertuples():
        selected = predictions[(predictions["target"] == analysis.CPU) &
                               (predictions["model"] == row.model) &
                               predictions["horizon_step"].between(row.start_step, row.end_step)]
        assert row.count == 3 * (row.end_step - row.start_step + 1)
        assert row.mae == pytest.approx(np.abs(selected["actual"] - selected["predicted"]).mean())
        assert row.rmse == pytest.approx(np.sqrt(np.square(selected["actual"] - selected["predicted"]).mean()))


@pytest.mark.parametrize("problem", ["short", "full", "reverse", "duplicate", "gap", "nan", "cpu", "vm", "columns"])
def test_invalid_training_data_rejected(training, problem):
    data = training.copy()
    if problem == "short":
        data = data.iloc[:-1]
    elif problem == "full":
        data = pd.concat([data, data.iloc[:24]], ignore_index=True)
    elif problem == "reverse":
        data = data.iloc[::-1]
    elif problem == "duplicate":
        data.loc[1, "timestamp"] = data.loc[0, "timestamp"]
    elif problem == "gap":
        data.loc[50:, "timestamp"] += 300
    elif problem == "nan":
        data.loc[50, analysis.CPU] = np.nan
    elif problem == "cpu":
        data.loc[50, analysis.CPU] = 101
    elif problem == "vm":
        data.loc[50, analysis.VM] = -1
    else:
        data = data.drop(columns=[analysis.VM])
    with pytest.raises(ValueError):
        analysis.validation_predictions(data)


def test_loader_caps_target_parsing_at_110_rows(training, tmp_path):
    path = tmp_path / "training.csv"
    training.to_csv(path, index=False)
    expected = analysis.load_training_targets(path)
    # Sentinel rows are deliberately not numeric: parsing them would fail.
    with path.open("a") as stream:
        stream.write("33900,DO_NOT_PARSE,DO_NOT_PARSE\n" * 24)
    pd.testing.assert_frame_equal(analysis.load_training_targets(path), expected)


def test_model_modules_are_import_safe(monkeypatch):
    def reject_io(*args, **kwargs):
        raise AssertionError("Import must not read or write evaluation files")
    monkeypatch.setattr(pd, "read_csv", reject_io)
    monkeypatch.setattr(pd.DataFrame, "to_csv", reject_io)
    monkeypatch.setattr(Path, "mkdir", reject_io)
    for name in ("cwfc.baseline_evaluation", "cwfc.advanced_evaluation"):
        importlib.reload(importlib.import_module(name))


def test_forecast_validation(training, monkeypatch):
    monkeypatch.setitem(analysis.BASELINE_MODELS, "Persistence", lambda _: np.nan)
    with pytest.raises(ValueError, match="24 finite predictions"):
        analysis.forecast_cpu("Persistence", training[analysis.CPU].to_numpy())


@pytest.mark.parametrize("scenario", ["no_peaks", "constant"])
def test_report_handles_degenerate_subsets(training, tmp_path, scenario):
    from cwfc.evaluation_reporting import write_report
    training[analysis.CPU] = np.linspace(20, 10, 110) if scenario == "no_peaks" else 7.5
    predictions = analysis.validation_predictions(training)
    tables = analysis.analyze_predictions(predictions)
    report = tmp_path / "model_evaluation.md"
    write_report(training, predictions, tables, report, [], input_path=Path("training.csv"))
    text = report.read_text(encoding="utf-8")
    assert "training.csv" in text
    assert "N/A" in text
    if scenario == "no_peaks":
        assert "There are no peak observations" in text


def test_reproducible_analysis_and_artifacts(training, tmp_path, monkeypatch):
    # Covers the complete command pipeline, including figures and the report.
    path = tmp_path / "training.csv"
    training.to_csv(path, index=False)
    output = tmp_path / "outputs"
    report = tmp_path / "reports"
    read_csv = pd.read_csv
    reads = []

    def read_training_only(source, **kwargs):
        assert source == path
        assert kwargs["nrows"] == 110
        reads.append(source)
        return read_csv(source, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_training_only)
    # Existing unrelated artifacts must survive evaluation.
    output.mkdir()
    previous = output / "baseline_summary.csv"
    previous.write_text("preserve earlier work\n")
    analysis.run_analysis(path, output, report)
    first = {str(file.relative_to(tmp_path)): file.read_bytes()
             for directory in (output, report) for file in directory.rglob("*") if file.is_file()}
    analysis.run_analysis(path, output, report)
    second = {str(file.relative_to(tmp_path)): file.read_bytes()
              for directory in (output, report) for file in directory.rglob("*") if file.is_file()}
    assert first == second
    assert reads == [path, path]
    assert len(list((report / "figures").glob("evaluation_*.png"))) == 7
    assert previous.read_text() == "preserve earlier work\n"
    assert not (tmp_path / "predictions.csv").exists()

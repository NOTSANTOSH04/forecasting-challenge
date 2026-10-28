"""Operational-policy math, timing, missing-history and artifact tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cwfc.error_analysis import CPU, VM, validation_predictions
from cwfc import operational_policy as policy


@pytest.fixture
def training():
    steps = np.arange(110)
    return pd.DataFrame({"timestamp": steps * 300, VM: np.full(110, 10000),
                         CPU: 9 + 0.3 * np.sin(steps / 7) + 0.002 * steps})


@pytest.fixture
def oof(training):
    return policy.validate_oof_predictions(validation_predictions(training), training)


@pytest.mark.parametrize("errors,q,expected", [
    ([-2, -1, 0, 1, 2], 0.9, 1.6), ([-3, -2, -1], 0.95, 0),
    ([0, 0, 0], 0.9, 0), ([-1, 3], 0.5, 1),
])
def test_signed_residual_margin(errors, q, expected):
    assert policy.residual_margin(errors, q) == pytest.approx(expected)


def test_empty_residual_sample_is_uncalibrated():
    assert np.isnan(policy.residual_margin([], 0.9))


@pytest.mark.parametrize("errors,q", [([np.nan], 0.9), ([np.inf], 0.9), ([[1]], 0.9), ([1], 1.1)])
def test_invalid_margin_inputs(errors, q):
    with pytest.raises(ValueError):
        policy.residual_margin(errors, q)


def test_conservative_forecast_clipping_and_alignment():
    np.testing.assert_array_equal(policy.conservative_forecast([0, 50, 100], [0, 10, 10]), [0, 60, 100])
    np.testing.assert_array_equal(policy.conservative_forecast([99, 100], 2), [100, 100])
    for predicted, margin in [([10], -1), ([101], 1), ([1, 2], [1]), ([1], np.nan)]:
        with pytest.raises(ValueError):
            policy.conservative_forecast(predicted, margin)


def test_alert_levels_and_sustained_point_exceedance():
    point = [8, 9, 10, 10, 10, 9, 10, 10, 10]
    upper = [9, 10, 11, 11, 11, 10, 11, 11, 11]
    levels, runs = policy.classify_alerts(point, upper, threshold=10)
    assert list(levels) == ["Normal", "Elevated demand", "Elevated demand", "Elevated demand",
                            "High-risk condition", "Elevated demand", "Elevated demand",
                            "Elevated demand", "High-risk condition"]
    np.testing.assert_array_equal(runs, [0, 0, 1, 2, 3, 0, 1, 2, 3])
    # Upper-bound-only exceedances do not become high risk.
    levels, _ = policy.classify_alerts([9] * 24, [11] * 24, threshold=10)
    assert set(levels) == {"Elevated demand"}
    # A fresh origin always resets persistence state.
    assert policy.classify_alerts([10, 10], [10, 10], 10)[1].tolist() == [1, 2]


def test_alert_runs_can_cross_hour_boundary():
    point = np.r_[np.full(10, 8.0), np.full(4, 10.0)]
    levels, runs = policy.classify_alerts(point, point, 10)
    assert levels[11] == "Elevated demand"
    assert levels[12] == "High-risk condition"
    assert runs[12] == 3


def test_constant_forecasts_and_no_alert_subsets():
    levels, runs = policy.classify_alerts([7.5] * 24, [7.5] * 24, threshold=8)
    assert set(levels) == {"Normal"}
    assert not runs.any()
    # Inclusive equality follows the existing event convention.
    levels, runs = policy.classify_alerts([7.5] * 24, [7.5] * 24, threshold=7.5)
    assert (levels == "High-risk condition").sum() == 22


def test_calibration_chronology_and_next_origin_parameters(training, oof):
    uncertainty = policy.uncertainty_analysis(training, oof)
    parameters = uncertainty[uncertainty["analysis_type"] == "origin_calibration"]
    for row in parameters.itertuples():
        expected_counts = {1: 0, 2: 12, 3: 24, 4: 36}
        assert row.sample_count == expected_counts[row.fold]
        if row.sample_count:
            assert row.latest_error_timestamp <= row.origin_timestamp
        else:
            assert np.isnan(row.margin_q90)
            assert np.isnan(row.margin_q95)
        expected_threshold = np.quantile(training[CPU].iloc[:int(row.train_size)], 0.95)
        assert row.threshold == pytest.approx(expected_threshold)
    # Fallback uses its own same-model residual sample.
    origin = int(training.iloc[61]["timestamp"])
    errors = policy.available_errors(oof, policy.FALLBACK, origin, "First hour")
    row = parameters[(parameters["model"] == policy.FALLBACK) &
                     (parameters["fold"] == 2) & (parameters["horizon_group"] == "First hour")].iloc[0]
    assert row.margin_q90 == pytest.approx(policy.residual_margin(errors.residual, 0.9))


def test_first_fold_is_not_silently_given_zero_uncertainty(training, oof):
    validation = policy.replay_policy(training, oof)
    warmup = validation[(validation["fold"] == 1) & (validation["policy"] != "point_only")]
    assert not warmup["policy_available"].any()
    assert set(warmup["alert_level"]) == {"Uncalibrated"}
    assert warmup[["upper_forecast", "nominal_margin", "alert"]].isna().all().all()
    summary = policy.summarize_policy(validation)
    unavailable = summary[(summary["fold"] == 1) & (summary["policy"] != "point_only")]
    assert (unavailable["evaluated_count"] == 0).all()
    assert unavailable[["true_positive", "false_positive", "false_negative", "underprediction_rate"]].isna().all().all()


@pytest.mark.parametrize("fold,end", [(1, 38), (2, 62), (3, 86)])
def test_future_targets_cannot_change_origin_policy(training, oof, fold, end):
    before = policy.replay_policy(training, oof)
    changed = training.copy()
    changed.loc[end:, CPU] = 70
    changed_oof = policy.validate_oof_predictions(validation_predictions(changed), changed)
    after = policy.replay_policy(changed, changed_oof)
    columns = ["policy", "horizon_step", "timestamp", "predicted", "threshold", "calibration_count",
               "calibration_latest_timestamp", "upper_forecast", "alert_level", "point_exceedance_run"]
    pd.testing.assert_frame_equal(before.loc[before["fold"] == fold, columns].reset_index(drop=True),
                                  after.loc[after["fold"] == fold, columns].reset_index(drop=True))


@pytest.mark.parametrize("problem", ["timestamp", "horizon", "actual", "origin", "missing", "duplicate", "order", "nonfinite"])
def test_invalid_oof_alignment_rejected(training, oof, problem):
    changed = oof.copy()
    if problem in ("timestamp", "horizon", "actual", "origin"):
        column = {"timestamp": "timestamp", "horizon": "horizon_step", "actual": "actual", "origin": "origin_timestamp"}[problem]
        changed.loc[0, column] += 1
    elif problem == "missing":
        changed = changed.iloc[1:]
    elif problem == "duplicate":
        changed.iloc[1] = changed.iloc[0]
    elif problem == "order":
        changed = changed.iloc[::-1]
    else:
        changed.loc[0, "predicted"] = np.inf
    with pytest.raises(ValueError):
        policy.validate_oof_predictions(changed, training)


def test_policy_confusion_metrics_and_signed_errors():
    rows = pd.DataFrame({"policy_available": [True] * 4, "observed_elevated": [True, False, True, False],
                         "alert": [True, True, False, False], "fold": [1] * 4,
                         "alert_level": ["Elevated demand", "High-risk condition", "Normal", "Normal"],
                         "actual": [12, 8, 11, 9], "upper_forecast": [11, 12, 9, 9],
                         "applied_margin": [1, 2, 1, 1]})
    result = policy.policy_metrics(rows)
    for key in ("true_positive", "false_positive", "false_negative", "true_negative", "alert_episodes", "high_risk_count"):
        assert result[key] == 1
    assert result["precision"] == result["recall"] == result["false_positive_rate"] == 0.5
    assert result["underprediction_rate"] == 0.5
    assert result["overprediction_rate"] == 0.25
    assert result["average_positive_forecast_error"] == 0.75
    assert result["mean_error_when_underpredicted"] == 1.5
    assert result["average_excess"] == 1
    assert result["mean_residual"] == -0.25
    assert result["average_conservative_margin"] == 1.25
    rows["alert"] = False
    rows["alert_level"] = "Normal"
    rows["observed_elevated"] = False
    result = policy.policy_metrics(rows)
    assert result["alert_count"] == result["false_positive"] == result["false_negative"] == 0
    assert np.isnan(result["precision"]) and np.isnan(result["recall"])
    assert result["false_positive_rate"] == 0
    assert np.isnan(policy.policy_metrics(rows.iloc[:0])["underprediction_rate"])


def test_replay_alignment_and_common_comparison(training, oof):
    validation = policy.replay_policy(training, oof)
    assert len(validation) == 216
    for (_, fold), part in validation.groupby(["policy", "fold"]):
        end = [38, 62, 86][fold - 1]
        np.testing.assert_array_equal(part.timestamp, training.timestamp.iloc[end:end + 24])
        np.testing.assert_array_equal(part.horizon_step, np.arange(1, 25))
    summary = policy.summarize_policy(validation)
    common = summary[summary["cohort"] == "common_calibrated_origins"]
    assert (common["evaluated_count"] == 48).all()
    assert (common["candidate_count"] == 48).all()
    assert common["observed_elevated_count"].nunique() == 1


def test_reproducible_outputs_and_input_access(training, tmp_path, monkeypatch):
    training_path = tmp_path / "train_targets.csv"
    predictions_path = tmp_path / "oof.csv"
    manifest_path = tmp_path / "evaluation_manifest.json"
    category_path = tmp_path / "categories.csv"
    training.to_csv(training_path, index=False)
    # Match the loader's exact parsed digest, as Step 5 does.
    loaded = policy.load_training_targets(training_path)
    digest = hashlib.sha256(loaded.to_csv(index=False).encode()).hexdigest()
    validation_predictions(loaded).to_csv(predictions_path, index=False)
    manifest_path.write_text(json.dumps({"validated_training_sha256": digest, "train_ends": [38, 62, 86],
                                        "horizon": 24, "ridge_alpha": 1.0, "holdout_targets_accessed": False}))
    pd.DataFrame({"vm_category": ["Interactive", "Unknown", "Delay-insensitive"],
                  "mean": [11.5, 10.3, 8.2]}).to_csv(category_path, index=False)
    output, report = tmp_path / "outputs", tmp_path / "reports"
    output.mkdir()
    preserved = output / "model_comparison.csv"
    preserved.write_text("existing results\n")
    read_csv = pd.read_csv
    reads = []

    def check_sources(path, **kwargs):
        assert path in (training_path, predictions_path, category_path)
        if path == training_path:
            assert kwargs["nrows"] == 110
        reads.append(path)
        return read_csv(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", check_sources)
    arguments = (training_path, predictions_path, manifest_path, category_path, output, report)
    policy.run_analysis(*arguments)
    first = {str(p.relative_to(tmp_path)): p.read_bytes() for folder in (output, report)
             for p in folder.rglob("*") if p.is_file()}
    policy.run_analysis(*arguments)
    second = {str(p.relative_to(tmp_path)): p.read_bytes() for folder in (output, report)
              for p in folder.rglob("*") if p.is_file()}
    assert first == second
    assert reads == [training_path, predictions_path, category_path] * 2
    assert len(list((report / "figures").glob("operational_*.png"))) == 3
    assert preserved.read_text() == "existing results\n"
    assert not (tmp_path / "predictions.csv").exists()
    manifest_path.write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        policy.run_analysis(*arguments)

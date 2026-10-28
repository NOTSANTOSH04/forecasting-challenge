"""Figures and a data-derived supporting report for Step 5 evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.error_analysis import CPU, VM, LEADING_MODELS


COLORS = {
    "Ridge AR (6)": "#0072B2", "Ridge AR (3)": "#CC79A7",
    "Moving Average (12)": "#009E73", "Persistence": "#D55E00",
}


def write_figures(predictions: pd.DataFrame, tables: dict, directory: Path) -> list[Path]:
    """Use an offline backend, source-relative time and explicitly labeled units."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        path = directory / f"evaluation_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    cpu = predictions[predictions["target"] == CPU]
    comparison = tables["model_comparison"].query("target == @CPU")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        for metric in ("mae", "rmse"):
            ranked = comparison.sort_values(metric, ascending=False)
            fig, ax = plt.subplots(figsize=(9, 4.8))
            bars = ax.barh(ranked["model"], ranked[metric],
                           color=[COLORS.get(name, "#858585") for name in ranked["model"]])
            ax.bar_label(bars, fmt="%.4f", padding=4)
            ax.set(xlabel=f"{metric.upper()} (CPU percentage points)",
                   title=f"Aggregate CPU: pooled validation {metric.upper()}")
            ax.set_xlim(0, ranked[metric].max() * 1.2)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            save(fig, f"model_{metric}")

        horizon = tables["horizon_error_analysis"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, metric in zip(axes, ("mae", "rmse")):
            for model in LEADING_MODELS:
                part = horizon[(horizon["model"] == model) & (horizon["end_step"] - horizon["start_step"] == 5)]
                ax.plot(part["end_step"], part[metric], marker="o", label=model, color=COLORS[model])
            ax.set(xticks=[6, 12, 18, 24], xticklabels=["1-6", "7-12", "13-18", "19-24"],
                   xlabel="Forecast steps (5 minutes per step)",
                   ylabel=f"{metric.upper()} (CPU percentage points)")
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle("Horizon errors pooled across three validation windows")
        save(fig, "horizon_errors")

        # Keep near-identical AR(3) out of these time-series panels for readability.
        displayed = (comparison.iloc[0]["model"], "Moving Average (12)", "Persistence")
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharey=True)
        for fold, ax in enumerate(axes, start=1):
            actual = cpu[(cpu["fold"] == fold) & (cpu["model"] == "Persistence")]
            ax.plot(actual["elapsed_minutes"], actual["actual"], color="black", linewidth=2,
                    marker=".", label="Observed CPU")
            ax.plot(actual["elapsed_minutes"], actual["peak_threshold"], color="#777777",
                    linestyle=":", label="Training P95")
            for i, model in enumerate(dict.fromkeys(displayed)):
                part = cpu[(cpu["fold"] == fold) & (cpu["model"] == model)]
                ax.plot(part["elapsed_minutes"], part["predicted"],
                        linestyle=["--", "-.", ":"][i], color=COLORS.get(model, "#CC79A7"), label=model)
            ax.set(title=f"Fold {fold}: single origin at {actual['origin_timestamp'].iloc[0] / 60:g} minutes",
                   ylabel="Mean CPU utilization (%)", xlabel="Minutes since first training timestamp")
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8, ncol=3)
        save(fig, "actual_vs_predicted")

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        leading = cpu[cpu["model"].isin(LEADING_MODELS)]
        bins = np.linspace(leading["residual"].min() - 0.01, leading["residual"].max() + 0.01, 19)
        for model, ax in zip(LEADING_MODELS, axes.flat):
            residual = cpu.loc[cpu["model"] == model, "residual"]
            ax.hist(residual, bins=bins, color=COLORS[model], edgecolor="white")
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(residual.mean(), color="black", linestyle=":", label="Mean residual")
            ax.set(title=model, xlabel="Actual - predicted (CPU percentage points)", ylabel="Observation count")
            ax.legend(fontsize=8)
        fig.suptitle("Residual distributions: positive values mean underprediction")
        save(fig, "residual_distribution")

        folds = tables["fold_performance"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, metric in zip(axes, ("mae", "rmse")):
            for model in LEADING_MODELS:
                part = folds[(folds["target"] == CPU) & (folds["model"] == model)]
                ax.plot(part["fold"], part[metric], marker="o", label=model, color=COLORS[model])
            ax.set(xticks=[1, 2, 3], xlabel="Validation fold",
                   ylabel=f"{metric.upper()} (CPU percentage points)")
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle("Leading methods: variation across forecast origins")
        save(fig, "fold_performance")

        peaks = tables["peak_error_analysis"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
        x = np.arange(len(LEADING_MODELS))
        for fold, ax in enumerate(axes, start=1):
            part = peaks[peaks["fold"] == fold]
            for offset, subset, color in [(-0.18, "peak", "#D55E00"), (0.18, "nonpeak", "#0072B2")]:
                values = part[part["subset"] == subset].set_index("model").reindex(LEADING_MODELS)
                ax.bar(x + offset, values["mae"], width=0.36, color=color, label=subset.capitalize())
            count = int(part[(part["model"] == "Persistence") & (part["subset"] == "peak")]["count"].iloc[0])
            ax.set(title=f"Fold {fold}: {count} peak observations", xticks=x,
                   xticklabels=["AR(6)", "AR(3)", "MA(12)", "Persist."])
            ax.tick_params(axis="x", labelrotation=30)
            ax.grid(axis="y", alpha=0.2)
            if count == 0:
                ax.text(0.5, 0.9, "Peak errors: unavailable", ha="center", transform=ax.transAxes)
        axes[0].set_ylabel("MAE (CPU percentage points)")
        axes[0].legend(fontsize=8)
        fig.suptitle("Training-relative P95 subsets: descriptive only; peak sample is small")
        save(fig, "peak_errors")
    return paths


def markdown_table(frame: pd.DataFrame, labels: dict[str, str] | None = None) -> str:
    """Avoid introducing tabulate solely to format the supporting report."""
    labels = labels or {}
    columns = [labels.get(str(column), str(column)) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("N/A")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.5f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_report(data: pd.DataFrame, predictions: pd.DataFrame, tables: dict,
                 path: Path, figure_paths: list[Path], *, input_path: Path) -> None:
    """All numerical findings below come from full-precision generated tables."""
    comparison = tables["model_comparison"]
    cpu = comparison[comparison["target"] == CPU].set_index("model")
    folds = tables["fold_performance"].query("target == @CPU")
    peaks = tables["peak_error_analysis"]
    residuals = tables["residual_analysis"]
    horizons = tables["horizon_error_analysis"]
    best = cpu.index[0]
    weakest = cpu["mae"].idxmax()
    fallback = "Moving Average (12)" if best != "Moving Average (12)" else "Persistence"
    best_row = cpu.loc[best]
    fallback_row = cpu.loc[fallback]
    advantage = 100 * (fallback_row["mae"] - best_row["mae"]) / fallback_row["mae"] if fallback_row["mae"] else np.nan
    peak_pooled = peaks[(peaks["fold"] == 0) & (peaks["subset"] == "peak")].set_index("model")
    nonpeak = peaks[(peaks["fold"] == 0) & (peaks["subset"] == "nonpeak")].set_index("model")
    best_peak = peak_pooled["mae"].idxmin() if peak_pooled["mae"].notna().any() else None
    leading = list(LEADING_MODELS)
    fold_mae = folds.pivot(index="model", columns="fold", values="mae").reindex(cpu.index)
    fold_rmse = folds.pivot(index="model", columns="fold", values="rmse").reindex(cpu.index)
    fold_summary = pd.DataFrame({"model": cpu.index})
    for fold in (1, 2, 3):
        fold_summary[f"Fold {fold} MAE"] = fold_mae[fold].to_numpy()
        fold_summary[f"Fold {fold} RMSE"] = fold_rmse[fold].to_numpy()
    peak_detail = peaks[peaks["fold"] > 0]
    thresholds = peak_detail[(peak_detail["model"] == "Persistence") & (peak_detail["subset"] == "peak")]
    hourly = horizons[horizons["horizon_group"].isin(["First hour", "Second hour"])][
        ["model", "horizon_group", "count", "mae", "rmse"]
    ]
    block_table = horizons[(horizons["model"].isin(leading)) &
                           (horizons["end_step"] - horizons["start_step"] == 5)][
        ["model", "horizon_group", "count", "mae", "rmse"]
    ]
    leading_residuals = residuals[(residuals["fold"] == 0) & residuals["model"].isin(leading)]
    vm = comparison[comparison["target"] == VM]
    vm_folds = tables["fold_performance"].query("target == @VM")
    cpu_predictions = predictions[predictions["target"] == CPU]
    counts = data[VM]
    lines = [
        "# Comprehensive model evaluation and error analysis (Step 5)",
        "Supporting section for the final 4-6 page assignment report. Model choice is provisional; Step 6 and final holdout forecasting are outside this analysis.",
        "## Evaluation methodology",
        f"Input: `{input_path.as_posix()}` (first {len(data)} training observations only). Training mean CPU utilization is {data[CPU].mean():.5f}%. Source-relative seconds are preserved; plots use minutes since the first training timestamp. No calendar dates are inferred.",
        "| Fold | Training observations (1-based) | Validation observations (1-based) | Forecast length |\n| --- | --- | --- | --- |\n| 1 | 1-38 | 39-62 | 24 steps |\n| 2 | 1-62 | 63-86 | 24 steps |\n| 3 | 1-86 | 87-110 | 24 steps |",
        "Every method predicts the same 24 future five-minute timestamps from one origin per fold. Validation windows do not overlap. Earlier validation observations become historical training data at later origins, as in deployment. No validation target is used as a recursive input within its own forecast. There is no shuffling, retuning or holdout evaluation.",
        "Forecast implementations are reused from Steps 3-4: persistence; flat moving averages with windows 6/12; flat exponential smoothing with alpha=0.3 and adjust=False; linear trends fitted to the last 12/24 points; and centered ridge autoregressions with lags 3/6, alpha=1.0 and an unpenalized intercept. Trends and recursive ridge predictions retain their original [0,100] clipping. Ridge is refit using only available historical lag/target pairs at each origin. No future-derived features or full-trace metadata statistics are used.",
        "MAE and RMSE are calculated from all 72 unrounded validation point predictions per CPU method. In particular, pooled RMSE is the square root of pooled squared errors, not the mean of fold RMSEs. CPU errors, bias and residual spread are in **percentage points**, while target values are percentages. Relative improvement = 100 x (persistence error - model error) / persistence error; negative values indicate deterioration. Zero baseline error produces N/A, including 0/0. Residual = actual - predicted: positive means underprediction; negative means overprediction.",
        "## Model comparison and relative improvements",
        markdown_table(cpu.reset_index()[["model", "mae", "rmse", "mae_improvement_pct", "rmse_improvement_pct"]]),
        f"**{best}** has the lowest pooled MAE ({best_row['mae']:.5f}) and pooled RMSE rank {int(cpu['rmse'].rank(method='min').loc[best])} ({best_row['rmse']:.5f}). Its MAE/RMSE improvements over persistence are {best_row['mae_improvement_pct']:.2f}%/{best_row['rmse_improvement_pct']:.2f}%. Its MAE advantage over {fallback} is {advantage:.2f}%.",
        f"The weakest pooled method is **{weakest}**, with MAE {cpu.loc[weakest, 'mae']:.5f} and RMSE {cpu.loc[weakest, 'rmse']:.5f}. "
        + ("Extrapolating a recent local slope over two hours is vulnerable to reversals; its large fold errors below are consistent with that failure mode. This is an interpretation of observed errors, not a causal test. " if weakest.startswith("Linear Trend") else "")
        + "The original method is retained and reported without corrective tuning.",
        "## Fold-level performance and robustness",
        markdown_table(fold_summary),
        markdown_table(tables["model_robustness"].set_index("model").loc[leading].reset_index()),
        "Fold standard deviation is descriptive (ddof=0). Fold wins count the best MAE among all eight CPU methods, with ties shared; they are not significance tests.",
    ]
    best_folds = fold_mae.loc[best]
    fallback_folds = fold_mae.loc[fallback]
    wins = int((best_folds < fallback_folds).sum())
    lines.append(
        f"{best} beats {fallback} on MAE in {wins} of the three windows. Their worst-fold MAEs are {best_folds.max():.5f} and {fallback_folds.max():.5f}, respectively. "
        f"The difference in pooled MAE between Ridge AR (6) and Ridge AR (3) is {abs(cpu.loc['Ridge AR (6)', 'mae'] - cpu.loc['Ridge AR (3)', 'mae']):.5f}. "
        "The small separation and changing fold rankings make the precise lag ranking uncertain. A lower fold-error spread alone can also reflect consistently poor predictions, so inspect it alongside absolute error levels."
    )
    for fold in (1, 2, 3):
        winner = fold_mae[fold].idxmin()
        lines.append(f"Fold {fold}: best MAE is {fold_mae.loc[winner, fold]:.5f} ({winner}); {best} = {fold_mae.loc[best, fold]:.5f}, {fallback} = {fold_mae.loc[fallback, fold]:.5f}.")
    lines.extend([
        "## Peak-demand error analysis",
        "Each origin uses the linear-interpolated 95th percentile of its **training CPU window only**. A validation observation is a peak when actual >= that threshold, preserving the earlier evaluation convention. This threshold identifies unusually high aggregate utilization relative to historical training data; it is **not an infrastructure capacity limit**. Actual validation values classify errors after forecasting and never determine forecast inputs or thresholds.",
        markdown_table(thresholds[["fold", "peak_threshold", "count"]], {"count": "Peak count (of 24)"}),
        "Pooled peak and nonpeak errors (every observation uses its own origin's threshold):",
    ])
    peak_comparison = pd.DataFrame({"model": cpu.index})
    for subset, frame in [("peak", peak_pooled), ("nonpeak", nonpeak)]:
        for metric in ("count", "mae", "rmse", "mean_residual"):
            peak_comparison[f"{subset}_{metric}"] = frame.reindex(cpu.index)[metric].to_numpy()
    lines.append(markdown_table(peak_comparison))
    lines.extend([
        "Fold-specific peak/nonpeak errors for the leading methods (N/A means no qualifying observations):",
        markdown_table(peak_detail[peak_detail["model"].isin(leading)][
            ["model", "fold", "subset", "count", "mae", "rmse", "mean_residual"]
        ]),
    ])
    peak_count = int(peak_pooled.loc[best, "count"])
    if peak_count:
        lines.append(
            f"There are only {peak_count} distinct peak validation observations. {best} has peak MAE {peak_pooled.loc[best, 'mae']:.5f}, "
            f"peak RMSE {peak_pooled.loc[best, 'rmse']:.5f}, and peak mean residual {peak_pooled.loc[best, 'mean_residual']:.5f}. "
            f"{fallback} has peak MAE {peak_pooled.loc[fallback, 'mae']:.5f}. The lowest pooled peak MAE is {peak_pooled.loc[best_peak, 'mae']:.5f} ({best_peak}); "
            "peak performance alone does not offset poor overall performance. Positive peak bias indicates underprediction and suggests that a point forecast alone may provide insufficient headroom. No headroom policy is finalized in Step 5."
        )
        lines.append(
            f"The primary model's peak MAE is {peak_pooled.loc[best, 'mae'] - peak_pooled.loc['Persistence', 'mae']:+.5f} "
            f"relative to persistence ({peak_pooled.loc['Persistence', 'mae']:.5f}), while its peak RMSE is "
            f"{peak_pooled.loc[best, 'rmse'] - peak_pooled.loc['Persistence', 'rmse']:+.5f} relative to persistence "
            f"({peak_pooled.loc['Persistence', 'rmse']:.5f}). This peak comparison is a material limitation of the pooled ranking."
        )
    else:
        lines.append("There are no peak observations. Peak errors are undefined; this dataset cannot support a peak-performance ranking.")
    lines.extend([
        "Peak plots are descriptive illustrations only: the subset is small, fold-dependent and temporally correlated. The figure omits missing peak bars rather than showing them as zero. Empty subsets remain count=0 and blank/NaN in CSVs. No peak uncertainty or reliable capacity guarantee can be estimated from these samples.",
        "## Forecast horizon analysis",
        "First versus second hour, pooled across folds (36 observations per hour per model):",
        markdown_table(hourly),
        "Six-step blocks for the leading models (18 observations per block):",
        markdown_table(block_table),
    ])
    for model in leading:
        part = horizons[horizons["model"] == model].set_index("horizon_group")
        first, second = part.loc["First hour"], part.loc["Second hour"]
        blocks = part.loc[["Steps 1-6", "Steps 7-12", "Steps 13-18", "Steps 19-24"]]
        monotonic = bool((np.diff(blocks["mae"]) >= 0).all())
        lines.append(
            f"{model}: second-hour minus first-hour MAE = {second['mae'] - first['mae']:+.5f}; RMSE = {second['rmse'] - first['rmse']:+.5f}. "
            f"MAE across consecutive six-step blocks {'increases monotonically' if monotonic else 'does not increase monotonically'}."
        )
    lines.extend([
        "Horizon groups describe realized errors on these particular workload trajectories; a longer lead time does not necessarily produce a monotonically larger error. These three windows cannot establish a general error-growth law.",
        "## Residual and absolute-error analysis",
        "Population residual SD uses ddof=0. Largest positive/negative residuals identify the most severe under/overprediction; a missing sign is N/A. Absolute-error percentiles are empirical descriptive summaries, not prediction intervals.",
        markdown_table(leading_residuals[["model", "mean_residual", "residual_std",
                                        "largest_positive_residual", "largest_negative_residual",
                                        "absolute_error_median", "absolute_error_p90", "absolute_error_p95", "absolute_error_max"]]),
        "Mean residual by fold (positive = average underprediction, negative = average overprediction):",
        markdown_table(residuals[(residuals["fold"] > 0) & residuals["model"].isin(leading)][
            ["model", "fold", "mean_residual", "residual_std", "underprediction_count", "overprediction_count"]
        ]),
    ])
    for model in leading:
        part = residuals[(residuals["model"] == model) & (residuals["fold"] > 0)]
        pooled = leading_residuals[leading_residuals["model"] == model].iloc[0]
        lines.append(
            f"{model}: pooled bias {pooled['mean_residual']:+.5f}; average underprediction in {int((part['mean_residual'] > 0).sum())} folds "
            f"and average overprediction in {int((part['mean_residual'] < 0).sum())} folds. "
            f"There are {int(pooled['underprediction_count'])} point underpredictions and {int(pooled['overprediction_count'])} point overpredictions out of {int(pooled['count'])}."
        )
    # Locate individual extremes using the same residual convention as the CSV.
    best_points = cpu_predictions[cpu_predictions["model"] == best]
    for label, index in [("Largest residual", best_points["residual"].idxmax()),
                         ("Smallest residual", best_points["residual"].idxmin())]:
        point = best_points.loc[index]
        lines.append(
            f"{best}, {label.lower()}: {point['residual']:+.5f} at source-relative timestamp {int(point['timestamp'])} seconds "
            f"(fold {int(point['fold'])}, forecast step {int(point['horizon_step'])}); actual {point['actual']:.5f}%, prediction {point['predicted']:.5f}%."
        )
    vm_constant = counts.nunique() == 1
    vm_zero = vm.iloc[0]["mae"] == 0
    fallback_third = fold_mae.loc[fallback, 3] < fold_mae.loc[best, 3]
    lines.extend([
        "Fold biases and the extreme cases show why pooled bias alone can hide failures in opposite directions. Residual serial dependence is plausible for multistep forecasts; the histogram does not establish independent or normally distributed errors.",
        "## Active VM count (evaluated separately)",
        markdown_table(vm_folds[["model", "fold", "count", "mae", "rmse"]]),
        f"Training VM count ranges from {int(counts.min())} to {int(counts.max())} and has {counts.nunique()} distinct value(s). "
        f"Persistence has pooled MAE {vm.iloc[0]['mae']:.5f} VMs and RMSE {vm.iloc[0]['rmse']:.5f} VMs. "
        + ("Its relative improvements are undefined because baseline error is zero. " if vm_zero else "Persistence relative improvement is zero against itself. ")
        + ("A constant forecast is justified for this curated, fixed challenge population and its observed training behavior. " if vm_constant else "The observed population varies, so a permanent constant-population assumption would not be justified on this input. ")
        + "This is not evidence that real cloud deployments have no VM churn, nor a claim about unseen holdout performance.",
        "## Provisional model selection recommendation",
        f"Use **{best} for aggregate CPU utilization**, with **{fallback} as the fallback baseline**, and **persistence for active VM count**. "
        f"The CPU recommendation prioritizes pooled MAE ({best_row['mae']:.5f}), considers pooled RMSE ({best_row['rmse']:.5f}), "
        f"and recognizes the {wins}/3 fold MAE wins over the fallback and the worst-fold comparisons above. "
        "The ridge models fit small regularized linear systems and need only the available CPU history; moving averages have even lower complexity and no fitted lag coefficients. Fixed ridge regularization is useful given short training windows, but it does not eliminate overfitting or recursive error accumulation.",
        f"{fallback} is reasonable when fitted autoregressive behavior is unstable or a simpler operational baseline is desired. "
        + ("Its lower third-window MAE is a material counterexample to a universal advantage for the primary model. " if fallback_third else "The fold table documents when either method is more dependable. ")
        + "Peak errors and horizon behavior remain risks for the primary model. The observed difference between Ridge AR (3) and Ridge AR (6), documented above, is insufficient to establish a reliable lag preference on other data. "
        "The preferred model is not a reliable peak-capacity predictor, and the trend model's peak results do not rescue its large overall errors. This is a validation-based engineering judgment, not a statistical-significance claim.",
        "## Limitations, risks and next-stage boundary",
        "Only three correlated forecast origins and 110 five-minute training observations are available. The validation set has 72 distinct timestamps, not hundreds of independent observations merely because multiple models predict them. Training windows overlap, ranking may change with new windows, and there is insufficient history to justify daily/weekly seasonality or strong generalization claims. Selection uses the same validation windows as the earlier model exploration, so validation selection optimism remains possible. No significance test or resampling-based ranking confidence is claimed.",
        "The unweighted VM mean is not a core-weighted measure of infrastructure capacity; aggregate smoothness can conceal individual VM or segment spikes. P95 peaks reflect this short, curated population and can shift as the training window expands. Short-window trend extrapolation and nearly stationary recursive ridge forecasts can both miss workload shifts. A constant active population limits what can be learned about churn.",
        "**Holdout audit:** this Step 5 execution reads only the capped training-target CSV, not raw traces, holdout timestamps or observations 111-134. The final 24 observed target values were neither accessed nor used for fitting, features, tuning, thresholds, selection or conclusions. No final `predictions.csv` is generated. Earlier EDA code constructs the complete target series before splitting as required by the assignment; this step does not rerun that code or re-open its raw inputs.",
        "Step 5 is complete when the documented command and automated tests pass. The project is ready for review before Step 6; operational policy, final model fitting, final prediction export and any authorized final holdout assessment remain for the next stage.",
        "## Reproduction and artifacts",
        "From the repository root, after `python -m pip install -e \".[dev]\"`:\n\n```powershell\npython -m cwfc.error_analysis\npython -m pytest -q -p no:cacheprovider\n```",
        "The analysis accepts `--input`, `--output-dir` and `--report-dir`. Input must contain the first 110 chronological training observations at five-minute cadence; parsing is capped at 110 rows and shorter inputs fail validation. No sorting, resampling or imputation is performed.",
        "Outputs: `model_comparison.csv`, `fold_performance.csv`, `horizon_error_analysis.csv`, `peak_error_analysis.csv`, `residual_analysis.csv`, `model_robustness.csv`, `evaluation_validation_predictions.csv` and `evaluation_manifest.json` under `outputs/`. The manifest records configuration, dependency versions and a SHA256 digest of the validated training data. `fold=0` denotes pooled rows in peak/residual CSVs. CSVs preserve full precision; this report rounds displayed values. Previous baseline/advanced evaluation files and EDA figures are preserved.",
        "## Figures",
    ])
    for figure in figure_paths:
        lines.append(f"![{figure.stem.replace('evaluation_', '').replace('_', ' ').capitalize()}](figures/{figure.name})")
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")

"""Measured results and monitoring guidance for the Step 6 policy demonstration."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.evaluation_reporting import markdown_table


def write_figures(validation: pd.DataFrame, summary: pd.DataFrame, directory: Path) -> list[Path]:
    """Combine uncertainty and threshold plots; reuse Step 5 residual figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name, layout_top=None):
        path = directory / f"operational_{name}.png"
        if layout_top is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=(0, 0, 1, layout_top))
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    q90 = validation[validation["policy"] == "historical_q90"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharey=True)
        for fold, ax in enumerate(axes, start=1):
            part = q90[q90["fold"] == fold]
            # Horizon minutes make clear that each panel is a fresh origin.
            minutes = part["horizon_step"] * 5
            ax.plot(minutes, part["actual"], color="black", marker=".", label="Observed demand")
            ax.plot(minutes, part["predicted"], color="#0072B2", linestyle="--", label="Ridge AR (6)")
            ax.axhline(part["threshold"].iloc[0], color="#D55E00", linestyle=":", label="Training P95")
            if part["policy_available"].all():
                ax.fill_between(minutes, part["predicted"], part["upper_forecast"],
                                color="#56B4E9", alpha=0.25, label="Historical Q90 upper margin")
                ax.plot(minutes, part["upper_forecast"], color="#0072B2", linestyle=":")
                alerts = part["alert"].to_numpy(dtype=bool)
                ax.scatter(minutes[alerts], part.loc[alerts, "upper_forecast"], marker="^",
                           color="#D55E00", s=28, zorder=4, label="Relative-demand alert")
            else:
                ax.text(0.03, 0.9, "No earlier OOF errors: upper margin unavailable",
                        transform=ax.transAxes, fontsize=9)
            ax.axvline(60, color="#999999", linewidth=0.8, alpha=0.6)
            origin = int(part["origin_timestamp"].iloc[0])
            ax.set(title=f"Fold {fold}: source-relative origin {origin} seconds",
                   xlabel="Minutes ahead of forecast origin", ylabel="Mean VM CPU utilization (%)")
            ax.grid(alpha=0.2)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955))
        fig.suptitle("Historical error margins and workload alerts (no formal interval coverage)")
        save(fig, "uncertainty_and_alerts", layout_top=0.91)

        common = summary[summary["cohort"] == "common_calibrated_origins"].set_index("policy")
        policies = ["point_only", "historical_q90", "historical_q95"]
        labels = ["Point only", "Q90 margin", "Q95 margin"]
        common = common.reindex(policies)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        x = np.arange(3)
        axes[0].bar(x, common["underprediction_rate"] * 100, color="#0072B2")
        axes[0].set_ylabel("Underprediction frequency (%)")
        for offset, column, label, color in [(-0.18, "average_positive_forecast_error", "Mean shortfall", "#D55E00"),
                                             (0.18, "average_excess", "Mean excess", "#009E73")]:
            axes[1].bar(x + offset, common[column], width=0.36, label=label, color=color)
        axes[1].set_ylabel("CPU percentage points (all-period mean)")
        axes[1].legend(fontsize=8)
        for offset, column, label, color in [(-0.25, "true_positive", "Correct alerts", "#009E73"),
                                             (0, "false_positive", "False positives", "#E69F00"),
                                             (0.25, "false_negative", "Missed events", "#D55E00")]:
            axes[2].bar(x + offset, common[column], width=0.25, label=label, color=color)
        axes[2].set_ylabel("Forecast-period count")
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.set(xticks=x, xticklabels=labels)
            ax.tick_params(axis="x", labelrotation=25)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        count = int(common["evaluated_count"].iloc[0])
        fig.suptitle(f"Fixed margin trade-offs on the same {count} periods (folds 2-3 only)")
        save(fig, "margin_tradeoffs")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
        columns = [("true_positive", "Correct alerts", "#009E73"),
                   ("false_positive", "False positives", "#E69F00"),
                   ("false_negative", "Missed events", "#D55E00")]
        for fold, ax in enumerate(axes, start=1):
            row = summary[(summary["policy"] == "historical_q90") &
                          (summary["cohort"] == "all_origins") & (summary["fold"] == fold)].iloc[0]
            if row["evaluated_count"]:
                bars = ax.bar(np.arange(3), [row[c[0]] for c in columns], color=[c[2] for c in columns])
                ax.bar_label(bars, fmt="%.0f", padding=3)
                ax.set_xticks(np.arange(3), [c[1] for c in columns], rotation=25, ha="right")
            else:
                ax.text(0.5, 0.55, "Uncalibrated\nNo policy score", ha="center", transform=ax.transAxes)
                ax.set_xticks([])
            ax.set_title(f"Fold {fold}: {int(row['candidate_observed_elevated_count'])} observed events")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Five-minute period count")
        fig.suptitle("Illustrative Q90 alert policy: historical fold outcomes")
        save(fig, "fold_alert_performance")
    return paths


def write_report(training: pd.DataFrame, oof: pd.DataFrame, tables: dict,
                 categories: pd.DataFrame, path: Path, figures: list[Path]) -> None:
    """Generate evidence-based guidance; policy choices are not optimized here."""
    # Avoid importing the CLI module when it is executing as __main__.
    primary = "Ridge AR (6)"
    uncertainty = tables["uncertainty_analysis"]
    validation = tables["operational_policy_validation"]
    summary = tables["policy_summary"]
    params = uncertainty[(uncertainty["analysis_type"] == "origin_calibration") &
                         (uncertainty["model"] == primary)]
    diagnostics = uncertainty[(uncertainty["analysis_type"] == "descriptive_oof") &
                              (uncertainty["model"] == primary)]
    common = summary[summary["cohort"] == "common_calibrated_origins"].set_index("policy")
    baseline = summary[(summary["cohort"] == "all_origins") & (summary["fold"] == 0) &
                       (summary["policy"] == "point_only")].iloc[0]
    q90, point, q95 = (common.loc[name] for name in ("historical_q90", "point_only", "historical_q95"))
    observed_total = int(baseline["observed_elevated_count"])
    primary_oof = oof[oof["model"] == primary]
    residual = primary_oof["residual"]
    next_params = params[params["train_size"] == len(training)]
    fold_scores = summary[(summary["policy"] == "historical_q90") & (summary["cohort"] == "all_origins") &
                          (summary["fold"] > 0)]
    next_threshold = next_params["threshold"].iloc[0]
    evidence_note = (
        "The uncertainty margins reduced numerical shortfalls, but the alert replay did not identify any of the evaluable high-demand events. "
        if q90["true_positive"] == 0 and q90["false_negative"] > 0 else
        "The replay below describes workload alerts on the limited evaluable sample. "
    )
    lines = [
        "# Forecast-based operational recommendations (Step 6)",
        "Supporting material for the final assignment report. **Recommendation: use this policy in shadow monitoring only; do not connect it to automatic scaling.** "
        + evidence_note + "No new model, model tuning, final forecast or Step 7 work is included.",
        "## Purpose and what the targets mean",
        "Forecasting gives a monitoring team advance notice to investigate demand changes, check headroom and prepare a response. The selected CPU method remains Ridge AR (6), Moving Average (12) remains its fallback, and VM-count persistence remains unchanged. Step 5 found good average CPU accuracy but weaker peak behavior and longer-horizon errors; an accurate average forecast is insufficient evidence for an operational scaling controller.",
        "Aggregate CPU utilization is the **unweighted mean of VM-level average CPU percentages**. It is not core-weighted CPU demand, physical server utilization, total cluster usage, available capacity or a server-count requirement. Different VM sizes and memory configurations make a direct conversion invalid. `active_vm_count` counts distinct VMs with valid readings, not hosts, scaling demand or spare capacity. Its constant training value reflects this curated population; it does not show that real clouds lack churn.",
        "## Fixed illustrative monitoring policy",
        "The following rules are engineering assumptions chosen for illustration before the policy replay, not thresholds optimized against its results. The proposed illustration uses Q90 residual margins; Q95 is a fixed sensitivity comparison and point-only is a comparator. Neither empirical quantile denotes guaranteed prediction-interval coverage.",
        "At each origin, derive T = linear-interpolated training CPU P95. Separately for steps 1-12 and 13-24, take the empirical Q90 of **signed** same-model residuals already observed at that origin: m = max(0, Q90(actual - predicted)). Construct U = clip(predicted + m, 0, 100). Quantiles include negative and zero errors; they are not calculated from positive errors alone. Errors and margins are CPU percentage points; T and forecasts are CPU percentages.",
        "| Level | Exact condition at a forecast period | Monitoring response |\n"
        "| --- | --- | --- |\n"
        "| Normal | U < T | Continue routine monitoring; this does not certify infrastructure health. |\n"
        "| Elevated demand | U >= T, before the sustained point-exceedance rule applies | Inspect high-load VMs and categories; increase monitoring frequency as an illustrative action. |\n"
        "| High-risk condition | Point forecast >= T for at least 3 consecutive five-minute steps at this origin | Investigate actual headroom, service latency and contention; consider scaling or redistribution only if real capacity metrics support it. |",
        "The 3-step rule represents a prespecified 15-minute sustained forecast condition, not an empirically optimal duration. The first two exceeding steps remain Elevated; High-risk starts on the third and continues while point exceedances persist. Runs reset when the point forecast falls below T and at each new origin, including across folds. They may span the first/second-hour boundary within one origin. Repeated upper-bound exceedances alone do not trigger High-risk. A High-risk label still describes relative workload risk, not confirmed physical saturation. No infrastructure-capacity verification signal is present or synthesized in this implementation.",
        "If no earlier OOF errors exist for either hour, the empirical policy is **Uncalibrated**, rather than Normal with a zero margin. A point-only comparison remains available, but it offers no uncertainty protection. In deployment, first collect resolved forecast errors and retain ordinary telemetry alerts during this warm-up. Recalculate forecasts when new measurements arrive and deduplicate notifications in a real monitoring service; these notification semantics are not simulated here.",
        "## Exact thresholds and origin-available margins",
        "The P95 threshold is a **relative historical high-demand threshold, not a physical infrastructure capacity limit**. The inclusive >= comparison preserves Step 5's event definition. Every fold uses its own training window; no validation targets determine its threshold. Values below are rounded to eight decimals; CSVs retain full precision.",
    ]
    parameter_table = params[["fold", "train_size", "origin_timestamp", "horizon_group", "threshold",
                              "sample_count", "margin_q90", "margin_q95"]].copy()
    parameter_table[["train_size", "origin_timestamp"]] = parameter_table[["train_size", "origin_timestamp"]].astype("int64")
    # Show exact useful precision separately from the shared report formatter.
    for column in ("threshold", "margin_q90", "margin_q95"):
        parameter_table[column] = parameter_table[column].map(lambda x: f"{x:.8f}" if pd.notna(x) else "N/A")
    lines.extend([
        markdown_table(parameter_table),
        f"The row marked fold 4/train size {len(training)} documents parameters available after the complete training window only: T = {next_threshold:.8f}%. "
        "No next-window forecasts are generated and no holdout values are opened. Fallback calibration parameters are also recorded in `uncertainty_analysis.csv`; a fallback must use its own residual margins rather than inheriting Ridge's errors.",
        "## Calibration chronology and uncertainty limitations",
        "Reuse the existing saved expanding-window OOF predictions; verify their timestamps, 24-step alignment, target values and origins against the training-only data and verify the Step 5 manifest. No forecasting method is refit. For an origin at t, calibration admits only same-model errors whose observed timestamp <= t and whose original forecast origin < t. Thus fold 2 uses resolved fold-1 errors, fold 3 uses resolved folds 1-2, and fold 1 has no calibration data. Short- and long-horizon samples are kept separate. Current-fold residuals are never used for its margin, and calibration labels record the latest error timestamp for audit.",
        "The replay assumes readings through the latest training timestamp have arrived when the forecast is issued. A live implementation must check measurement arrival times and exclude delayed observations even when their source timestamp is earlier than the origin.",
        "The uncertainty CSV separates `origin_calibration` from `descriptive_oof`. Pooled/per-fold retrospective residual summaries below are diagnostics, not margins applied back to those same predictions. Only origin-available samples drive historical alerts. Margin samples are small and temporally dependent; adjacent forecast errors share a fitted model and often a common workload shift. Training windows also overlap. The primary model was selected using these same validation windows, so calibration and replay remain subject to model-selection optimism despite forward-only margin construction. No conformal, normality, independence or formal 90%/95% coverage claim is made.",
        markdown_table(diagnostics[["fold", "horizon_group", "sample_count", "mean_residual", "residual_std",
                                    "signed_q90", "signed_q95"]]),
        "Q95 uses the extreme tail of very few observations, so it can be especially unstable. Hour-specific quantiles describe historical errors, not a law requiring every later-origin margin to grow with horizon. Regime changes can make past upper margins unrepresentative.",
        "## Historical policy evaluation",
        f"The point-only comparator covers all {int(baseline['evaluated_count'])} validation periods with {observed_total} observed P95 events. "
        f"The empirical policies cannot be evaluated on fold 1. The fair, common comparison uses only {int(point['evaluated_count'])} periods in folds 2-3, "
        f"containing {int(point['observed_elevated_count'])} observed elevated-demand periods. Do not compare rates from the 72-period point-only sample directly against the 48-period calibrated sample.",
        "Metric definitions: an alert period is a five-minute forecast timestamp classified Elevated or High-risk. An observed event is actual >= origin-specific T. TP is an alerted event; FP an alert without an event; FN an event without an alert; TN neither. Alert rate = alerts/N, precision = TP/(TP+FP), recall = TP/(TP+FN), and false-positive rate = FP/(FP+TN). Zero denominators are N/A. An episode is a contiguous alert run within one origin; forecast-period counts are not numbers of notifications or servers. Uncalibrated policies have no score, not zero-error performance.",
        "Common-origin event results:",
        markdown_table(common.reset_index()[["policy", "evaluated_count", "observed_elevated_count", "alert_count",
                                            "alert_episodes", "high_risk_count", "true_positive", "false_positive",
                                            "false_negative", "true_negative", "precision", "recall", "false_positive_rate"]]),
        "Q90 results by fold (candidate observed events are shown even when calibration is unavailable):",
        markdown_table(fold_scores[["fold", "candidate_count", "evaluated_count", "candidate_observed_elevated_count",
                                   "observed_elevated_count", "alert_count", "true_positive", "false_positive", "false_negative"]]),
        f"On the common sample, Q90 produced {int(q90['alert_count'])} alert periods in {int(q90['alert_episodes'])} episodes: "
        f"{int(q90['true_positive'])} correct alerts, {int(q90['false_positive'])} false positives and {int(q90['false_negative'])} missed events. "
        f"It produced {int(q90['high_risk_count'])} sustained point-exceedance High-risk periods. Q95 produced {int(q95['alert_count'])} alerts, "
        f"{int(q95['true_positive'])} correct alerts and {int(q95['false_negative'])} misses. These are workload-event labels, not verified service incidents or capacity shortfalls.",
        evidence_note + "Broad long-horizon margins can warn about a workload rise at the wrong time, while a small short-horizon margin can miss an early spike. Review the timestamp-level CSV and forecast/threshold plot for these cases. The thin event sample makes precision, recall and false-positive rates unreliable estimates for future operations. Keep the disappointing results visible; no extra threshold or persistence search is performed to improve them.",
        "## Safety-margin and signed-error trade-offs",
        "Underprediction rate = fraction with actual > conservative forecast; overprediction rate = fraction with actual < conservative forecast. Average positive forecast error = mean(max(actual - conservative forecast, 0)) across **all** evaluated periods. The conditional mean when underpredicted is separately available in CSVs. Average excess = mean(max(conservative forecast - actual, 0)); average conservative margin = mean(conservative forecast - point forecast) after clipping. These quantities are in percentage points and do not measure physical under/over-provisioning.",
        markdown_table(common.reset_index()[["policy", "evaluated_count", "underprediction_rate", "overprediction_rate",
                                            "average_positive_forecast_error", "average_excess", "mean_residual",
                                            "average_conservative_margin", "alert_rate"]]),
        f"Across all {len(residual)} OOF point forecasts, Ridge underpredicts {int((residual > 0).sum())} times "
        f"({100 * (residual > 0).mean():.2f}%) and overpredicts {int((residual < 0).sum())} times; pooled signed bias = {residual.mean():+.5f}. "
        "An overall negative bias can coexist with underprediction of peaks; Step 5 documents that distinction.",
        f"On the common sample, adding Q90 changes underprediction frequency from {100 * point['underprediction_rate']:.2f}% "
        f"to {100 * q90['underprediction_rate']:.2f}%, mean shortfall from {point['average_positive_forecast_error']:.5f} "
        f"to {q90['average_positive_forecast_error']:.5f}, and mean excess from {point['average_excess']:.5f} "
        f"to {q90['average_excess']:.5f}. Q95 changes the underprediction frequency to {100 * q95['underprediction_rate']:.2f}% "
        f"with mean excess {q95['average_excess']:.5f}. Larger historical margins reduce numerical underprediction here, "
        "but do not necessarily detect more events or improve alert precision. The illustrated quantile choices remain unchanged by these results.",
        "## Under-provisioning, over-provisioning and category considerations",
        "In a real system, insufficient resources may cause contention, latency, performance degradation and reduced reliability during spikes. Excess resources may leave idle capacity, raise expenditure and create unnecessary scaling activity. A larger uncertainty margin places more weight on avoiding shortfalls but can increase investigation effort and false alerts. The preferred operating point depends on service-level objectives, workload criticality, response lead time and infrastructure economics. This dataset supplies neither those costs nor measured latency, capacity headroom or scaling outcomes; no financial saving, pricing assumption or physical capacity benefit is estimated.",
        "The existing category EDA uses training readings only. Its means are descriptive:",
        markdown_table(categories[["vm_category", "mean"]], {"mean": "Training mean VM CPU (%)"}),
        "Interactive workloads may warrant closer latency monitoring and more conservative protection **if their actual SLOs justify it**. Delay-insensitive workloads may permit deferral or scheduling flexibility after checking deadlines and dependencies. Unknown workloads need classification before confident policy assignment. These are operational interpretations, not causal evidence that category membership determines latency sensitivity or spare capacity. Category means do not validate segment-specific alert thresholds, and no new segment forecasting models are introduced.",
        "## Applying this to a real cloud environment",
        "Start with shadow alerts and log point forecasts, the origin-available margin, threshold, observed demand and operator outcomes. Collect CPU-core allocations and usage, exact VM sizes, placement, host/cluster headroom, throttling, runnable queues, memory/IO pressure, application latency/error rates, service dependencies and scaling lead times. Use these signals to establish genuine capacity limits and verify service risk before scaling or redistributing work. The aggregate mean can hide a saturated VM or bottleneck even while relative-demand alerts stay Normal.",
        "Collect more independent forecast windows and high-load incidents before choosing alert quantiles or run lengths. Reserve a later prospective period to evaluate a frozen policy and notification deduplication. Calibrate each deployed model and monitor drift, missed events and alert fatigue; if the CPU model fails or required lag history is unavailable, use the validated Moving Average (12) fallback with its own calibration and retain ordinary monitoring. The present replay does not validate automated failover or notification handling.",
        "## Reproduction, audit and stage boundary",
        "From the repository root, with the existing environment and Step 5 artifacts:\n\n```powershell\npython -m cwfc.operational_policy\npython -m pytest -q -p no:cacheprovider\n```",
        "The command accepts `--training`, `--predictions`, `--evaluation-manifest`, `--categories`, `--output-dir` and `--report-dir`. "
        "It writes `outputs/uncertainty_analysis.csv`, `outputs/operational_policy_validation.csv`, `outputs/policy_summary.csv`, "
        "`outputs/operational_policy_manifest.json`, this report and three `reports/figures/operational_*.png` figures. "
        "CSV rates are fractions, not percentages; report values are rounded. The manifest records fixed policy settings, dependency versions and input digests. "
        "Step 5's residual-distribution plot is reused when present. Earlier artifacts are preserved.",
        "**Holdout audit:** only the capped 110-row training-target artifact, saved validation predictions, Step 5 manifest and training-category statistics are read. "
        "No raw VM traces, holdout timestamps or final 24 observed holdout targets are opened or used. No model or hyperparameter changes, formal coverage claims, "
        "final `predictions.csv`, live infrastructure actions or cost estimates are produced. Step 6 is ready for review after checks pass; Step 7 and final submission remain unstarted.",
        "## Figures",
    ])
    for figure in figures:
        lines.append(f"![{figure.stem.replace('operational_', '').replace('_', ' ').capitalize()}](figures/{figure.name})")
    residual_figure = path.parent / "figures" / "evaluation_residual_distribution.png"
    if residual_figure.exists():
        lines.append("The existing residual-distribution plot uses actual - predicted; positive errors are underprediction. "
                     "It includes the primary method and provides the signed-error context without generating a duplicate figure.")
        lines.append("![Step 5 residual distributions](figures/evaluation_residual_distribution.png)")
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")

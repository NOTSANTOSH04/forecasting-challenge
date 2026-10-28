
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Setup
# --------------------------------------------------

DATA_DIR = Path("challenge_data")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("reports/figures")

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

cpu = pd.read_csv(DATA_DIR / "vm_cpu_readings.csv.gz")
metadata = pd.read_csv(DATA_DIR / "vm_metadata.csv.gz")

print("CPU dataset shape:", cpu.shape)
print("Metadata shape:", metadata.shape)

# --------------------------------------------------
# 2. Construct forecasting targets
# --------------------------------------------------

targets = (
    cpu.groupby("timestamp", as_index=False)
    .agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
    )
    .sort_values("timestamp")
    .reset_index(drop=True)
)

assert len(targets) == 134
assert targets["timestamp"].is_unique
assert targets["timestamp"].diff().dropna().eq(300).all()

# Construct the complete target series FIRST.
# Then separate training and holdout timestamps.

train_targets = targets.iloc[:110].copy()

# Keep ONLY timestamps for the holdout.
# Do not inspect or use the observed holdout targets.
holdout_timestamps = targets.iloc[110:][["timestamp"]].copy()

train_targets.to_csv(
    OUTPUT_DIR / "train_targets.csv", index=False
)

holdout_timestamps.to_csv(
    OUTPUT_DIR / "holdout_timestamps.csv", index=False
)

print("\nTraining timestamps:", len(train_targets))
print("Holdout timestamps:", len(holdout_timestamps))

print("\nTraining target statistics:")
print(
    train_targets[
        ["active_vm_count", "aggregate_cpu_utilization"]
    ].describe()
)

# --------------------------------------------------
# 3. Restrict workload EDA to training timestamps
# --------------------------------------------------

training_times = set(train_targets["timestamp"])

train_cpu = cpu[
    cpu["timestamp"].isin(training_times)
].copy()

train_cpu["elapsed_minutes"] = (
    train_cpu["timestamp"] - train_cpu["timestamp"].min()
) / 60

train_targets["elapsed_minutes"] = (
    train_targets["timestamp"]
    - train_targets["timestamp"].min()
) / 60

print("\nTraining CPU observations:", len(train_cpu))

# --------------------------------------------------
# 4. CPU utilization distribution
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    train_cpu["avg_cpu"],
    bins=50,
    color="steelblue",
    edgecolor="white",
)

plt.title("Distribution of VM CPU Utilization")
plt.xlabel("Average CPU Utilization (%)")
plt.ylabel("Number of Observations")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "cpu_distribution.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 5. Aggregate CPU utilization over time
# --------------------------------------------------

plt.figure(figsize=(11, 5))

plt.plot(
    train_targets["elapsed_minutes"],
    train_targets["aggregate_cpu_utilization"],
    color="royalblue",
    linewidth=2,
)

plt.title("Aggregate CPU Utilization Over Time")
plt.xlabel("Elapsed Time (minutes)")
plt.ylabel("Aggregate CPU Utilization (%)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "cpu_time_series.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 6. Active VM count over time
# --------------------------------------------------

plt.figure(figsize=(11, 5))

plt.plot(
    train_targets["elapsed_minutes"],
    train_targets["active_vm_count"],
    color="darkgreen",
    linewidth=2,
)

plt.title("Active VM Count Over Time")
plt.xlabel("Elapsed Time (minutes)")
plt.ylabel("Active VM Count")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "active_vm_count.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 7. CPU utilization by VM category
# --------------------------------------------------

# Merge stable VM categories with training CPU data.
# Do not use full-trace summary statistics as features.

category_data = train_cpu.merge(
    metadata[["vm_id", "vm_category"]],
    on="vm_id",
    how="left",
    validate="many_to_one",
)

category_stats = (
    category_data.groupby("vm_category")["avg_cpu"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
)

print("\nCPU utilization by VM category:")
print(category_stats)

category_stats.to_csv(
    OUTPUT_DIR / "category_statistics.csv"
)

plt.figure(figsize=(10, 5))

category_stats["mean"].plot.bar(
    color="mediumpurple"
)

plt.title("Mean CPU Utilization by VM Category")
plt.xlabel("VM Category")
plt.ylabel("Mean CPU Utilization (%)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "cpu_by_category.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 8. Peak workload analysis
# --------------------------------------------------

peak_threshold = train_targets[
    "aggregate_cpu_utilization"
].quantile(0.95)

peak_periods = train_targets[
    train_targets["aggregate_cpu_utilization"]
    >= peak_threshold
]

print("\n========== PEAK WORKLOAD ==========")
print("Training P95 threshold:", round(peak_threshold, 4))
print("Number of peak timestamps:", len(peak_periods))

print(
    peak_periods[
        ["timestamp", "aggregate_cpu_utilization"]
    ].to_string(index=False)
)

plt.figure(figsize=(11, 5))

plt.plot(
    train_targets["elapsed_minutes"],
    train_targets["aggregate_cpu_utilization"],
    label="Aggregate CPU",
)

plt.axhline(
    peak_threshold,
    color="red",
    linestyle="--",
    label="Training P95 threshold",
)

plt.scatter(
    peak_periods["elapsed_minutes"],
    peak_periods["aggregate_cpu_utilization"],
    color="red",
    zorder=3,
)

plt.title("High-Demand Workload Periods")
plt.xlabel("Elapsed Time (minutes)")
plt.ylabel("Aggregate CPU Utilization (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "peak_workload.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 9. Summary
# --------------------------------------------------

print("\n========== EDA SUMMARY ==========")

print(
    "Mean training aggregate CPU:",
    round(
        train_targets[
            "aggregate_cpu_utilization"
        ].mean(), 4
    ),
)

print(
    "Minimum training aggregate CPU:",
    round(
        train_targets[
            "aggregate_cpu_utilization"
        ].min(), 4
    ),
)

print(
    "Maximum training aggregate CPU:",
    round(
        train_targets[
            "aggregate_cpu_utilization"
        ].max(), 4
    ),
)

print(
    "Active VM count constant:",
    train_targets["active_vm_count"].nunique() == 1,
)

print("\nAnalysis completed successfully!")
print("Figures saved in:", FIGURE_DIR)
print("Analysis data saved in:", OUTPUT_DIR)


# --------------------------------------------------
# 10. VM lifetime and churn analysis
# --------------------------------------------------

print("\n========== VM LIFETIME ANALYSIS ==========")

created = pd.to_numeric(
    metadata["timestamp_vm_created"],
    errors="coerce"
)

deleted = pd.to_numeric(
    metadata["timestamp_vm_deleted"],
    errors="coerce"
)

print("\nCreation timestamp statistics:")
print(created.describe())

print("\nDeletion timestamp statistics:")
print(deleted.describe())

# Investigate recorded creation/deletion events
# relative to the observed training window.

start = train_targets["timestamp"].min()
end = train_targets["timestamp"].max()

created_during_training = created.between(start, end)
deleted_during_training = deleted.between(start, end)

print(
    "\nVMs created during training window:",
    created_during_training.sum()
)

print(
    "VMs deleted during training window:",
    deleted_during_training.sum()
)

print(
    "Missing/unparseable creation timestamps:",
    created.isna().sum()
)

print(
    "Missing/unparseable deletion timestamps:",
    deleted.isna().sum()
)
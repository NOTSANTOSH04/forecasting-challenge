from pathlib import Path
import pandas as pd

DATA_DIR = Path("challenge_data")

# Load datasets
cpu = pd.read_csv(DATA_DIR / "vm_cpu_readings.csv.gz")
metadata = pd.read_csv(DATA_DIR / "vm_metadata.csv.gz")

print("\n========== CPU DATA ==========")

print("\nShape:")
print(cpu.shape)

print("\nColumns:")
print(cpu.columns.tolist())

print("\nFirst 5 rows:")
print(cpu.head())

print("\nData types:")
print(cpu.dtypes)

print("\nMissing values:")
print(cpu.isnull().sum())

print("\n========== VM METADATA ==========")

print("\nShape:")
print(metadata.shape)

print("\nColumns:")
print(metadata.columns.tolist())

print("\nFirst 5 rows:")
print(metadata.head())

print("\nMissing values:")
print(metadata.isnull().sum())

print("\n========== BASIC CHECKS ==========")

print("Unique VMs:", cpu["vm_id"].nunique())
print("Unique timestamps:", cpu["timestamp"].nunique())

print(
    "Duplicate VM-timestamp rows:",
    cpu.duplicated(subset=["vm_id", "timestamp"]).sum()
)

print("\nCPU statistics:")
print(cpu[["min_cpu", "max_cpu", "avg_cpu"]].describe())

print("\nReadings per VM:")
print(cpu.groupby("vm_id").size().describe())

print("\n========== FINAL DATA CHECKS ==========")

# 1. Timestamp intervals
timestamps = sorted(cpu["timestamp"].unique())

intervals = pd.Series(timestamps).diff().dropna()

print("\nTimestamp interval counts:")
print(intervals.value_counts())

# 2. CPU range checks
invalid_cpu = (
    (cpu["min_cpu"] < 0) |
    (cpu["max_cpu"] > 100) |
    (cpu["min_cpu"] > cpu["avg_cpu"]) |
    (cpu["avg_cpu"] > cpu["max_cpu"])
)

print("\nInvalid CPU readings:", invalid_cpu.sum())

# 3. Metadata matching
missing_metadata = (
    set(cpu["vm_id"]) - set(metadata["vm_id"])
)

print("VMs missing metadata:", len(missing_metadata))

# 4. Active VM count
active_counts = cpu.groupby("timestamp")["vm_id"].nunique()

print("\nActive VM count:")
print(active_counts.describe())
# Cloud Workload Forecasting & Analysis

Atomity AI/ML internship assessment using the fixed Microsoft Azure Public
Dataset V2 challenge subset. CPU forecasting uses the validated Ridge AR (6);
active VM count uses persistence. The operational recommendation is shadow
monitoring and decision support only.

## Requirements and installation

Python **3.11+**. Runtime dependencies are NumPy, pandas and Matplotlib; pytest
is provided by the development extra. Dependencies are declared in
`pyproject.toml`.

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation, invoke `.venv\Scripts\python.exe` directly
in place of `python`; changing execution policy is unnecessary.

## Dataset

Download `data-v1.tar.gz` from the fixed
[data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)
and extract it into `challenge_data/`:

```text
challenge_data/
  vm_cpu_readings.csv.gz
  vm_metadata.csv.gz
  DATA_CARD.md
  SHA256SUMS
```

Use the exact release, not a different Azure subset. See
[DATASET.md](DATASET.md) for fields and target definitions.

Verify checksums on Linux/macOS from `challenge_data/`:

```bash
sha256sum -c SHA256SUMS
```

Or from the project root in PowerShell:

```powershell
Get-Content challenge_data/SHA256SUMS | ForEach-Object {
    $expectedHash, $filename = $_ -split '\s+', 2
    $releasePath = Join-Path 'challenge_data' $filename.Trim()
    $actualHash = (Get-FileHash -LiteralPath $releasePath -Algorithm SHA256).Hash
    if ($actualHash.ToLowerInvariant() -ne $expectedHash.ToLowerInvariant()) {
        throw "Checksum mismatch: $filename"
    }
}
```

## Generate the required predictions

From the project root:

```powershell
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This command works directly from the two compressed input CSVs; EDA and
intermediate files under `outputs/` are not prerequisites. It validates rows,
aggregates the complete chronological target series and separates the fixed
110/24 official split. Observed holdout targets are discarded; only their
future timestamps enter the forecasting function. No holdout accuracy is
calculated.

CPU forecasts reuse the original six-lag Ridge algorithm with `alpha=1.0`,
centered inputs/targets, an unpenalized intercept and recursive predictions
clipped to [0,100]. VM persistence repeats the latest training count; 10,000 is
not hard-coded. Output has exactly 24 rows and these columns in this order:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

The official output grid is 33,000-39,900 relative seconds at 300-second
intervals. All predictions are finite; CPU is in [0,100] and VM counts are
nonnegative. Historical validation metrics in the reports are not holdout
results.

Smaller synthetic smoke inputs may use a configurable horizon:

```powershell
python -m cwfc.predict --data path/to/smoke_data --horizon 4 --output smoke_predictions.csv
```

Inputs smaller than 134 timestamps use the final `--horizon` timestamps as the
output grid and require at least seven training timestamps. An official-size
134-timestamp input always requires 110/24; larger unexpected series fail.
The predictor never silently changes models. Moving Average (12) remains the
validated fallback for a separately justified operational exception, but no
automatic fallback is enabled: insufficient Ridge history fails clearly.

Raw CPU input requires `timestamp,vm_id,min_cpu,max_cpu,avg_cpu`; metadata
requires unique `vm_id` values matching the CPU population. Other metadata
columns are not forecasting features. Missing/nonfinite CPU values, invalid
CPU ranges, duplicate VM-timestamp keys, missing/blank VM IDs, population
mismatches and aggregate timestamp gaps fail clearly. Raw rows may be unsorted;
aggregation sorts timestamps. Changing represented VM membership is allowed.
No target values are imputed and no records are silently dropped.

## Tests and analysis reproduction

```powershell
python -m pytest -q -p no:cacheprovider
```

Tests cover aggregation, chronological alignment, recursive forecasting,
contract validation, smaller smoke data, deterministic CSVs and holdout
isolation by changing only future observed targets. Existing evaluation and
policy tests are preserved.

To regenerate the supporting analysis from the downloaded release:

```powershell
python notebooks/eda_analysis.py
python -m cwfc.error_analysis
python -m cwfc.operational_policy
```

Evaluation uses three expanding origins (38/62/86 historical observations),
each forecasting 24 steps. Policy margins use only errors resolved by each
origin; fold 1 is uncalibrated. Neither evaluation command uses final holdout
targets. Full-trace metadata CPU summaries are not model features.

## Reports and local packaging

- [Final technical report](reports/final_report.md) and [PDF](reports/final_report.pdf).
- [Model evaluation](reports/model_evaluation.md).
- [Operational recommendations](reports/operational_recommendations.md).
- [Submission audit](reports/submission_audit.md).

The PDF contains six main-report pages and one appendix page. To rebuild it
with an installed Chrome/Chromium/Edge browser, without extra Python packages:

```powershell
python scripts/render_report.py
```

The small renderer supports the final report's Markdown constructs and keeps
local document links as plain text in the PDF. The Markdown source retains
clickable relative links. PDF regeneration requires a compatible local browser;
prediction generation does not.

Prepare and inspect the local submission archive:

```powershell
python scripts/package_submission.py
```

The archive includes source, tests, documentation, reports/figures, selected
small analysis artifacts and `predictions.csv`. It excludes raw compressed
datasets, virtual environments, caches, credentials and local audit scratch
files. Download the official release separately to reproduce it. Packaging
never uploads, pushes or submits anything. See [SUBMISSION.md](SUBMISSION.md)
and [ASSIGNMENT.md](ASSIGNMENT.md) for the original contracts.

## AI-assisted development disclosure

This submission was developed with assistance from OpenAI ChatGPT and Codex for implementation guidance, code generation and debugging, testing workflows, analysis, and technical documentation. The forecasting results, validation approach, source code, and limitations are provided for independent review. The candidate is responsible for understanding and explaining the submitted work.

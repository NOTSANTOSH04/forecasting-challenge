# Local submission readiness audit

The local candidate is ready for review. Nothing has been submitted, uploaded,
pushed or published externally.

## Prediction contract

Generated directly from the official compressed `data-v1` release with:

```powershell
python -m cwfc.predict --data challenge_data --output predictions.csv
```

| Check | Observed result |
| --- | --- |
| Rows | 24 |
| Columns, in order | `timestamp,active_vm_count,aggregate_cpu_utilization` |
| First / last timestamp | 33,000 / 39,900 relative seconds |
| Timestamp cadence | 300 seconds; unique and increasing |
| Minimum / maximum forecast CPU | 9.202333016521417% / 9.369857021241707% |
| Minimum / maximum predicted VM count | 10,000 / 10,000 |
| Missing or nonfinite predictions | None |
| CPU bounds / nonnegative VM counts | Passed |
| CSV SHA256 | `debf6da561aedaee9fefd5120afcc073eccac0785af67f6c091a4bfd669c0a87` |

The output retains full numeric precision. Displayed extrema above are rounded
float representations; they are forecasts, not observed holdout values.

## Methodology and holdout protection

The predictor calls the unchanged `ridge_autoregression` implementation with
six lags, `alpha=1.0`, centered inputs/targets, an unpenalized intercept,
recursive generated inputs and [0,100] clipping. Active VM count uses the
existing persistence function and the latest training value, with no hard-coded
population. There is no automatic model fallback or new model selection.

The official series uses exactly 110 training observations and 24 future
timestamps. Raw holdout readings are processed only for structural validation
and full target construction as permitted by the assignment. The split retains
their timestamps and discards their observed target values before forecasting.
They were never used for fitting, features, tuning, model choice, policy
calibration or accuracy evaluation. No final holdout accuracy was calculated.

## Verification performed

- All three published SHA256 entries (CPU readings, metadata and release card)
  match the supplied release.
- Raw input checks reject missing/nonfinite values, invalid CPU ranges,
  duplicate VM-timestamp keys, unmatched metadata populations and irregular
  aggregate timestamps. No missing targets are invented.
- Regenerated training targets agree with the existing training artifact to
  absolute tolerance `1e-12`.
- Production forecasts on the three historical origins agree with the saved
  Ridge AR (6) validation predictions to absolute tolerance `1e-12`; synthetic
  tests also verify equality with the original forecasting function.
- All **101 automated tests passed**, including 41 new pipeline tests. They
  cover aggregation, schema, output bounds, chronological alignment,
  recursive inputs, smaller datasets, malformed records and deterministic
  command execution. Holdout-isolation tests change only future observed CPU
  values or VM membership and confirm unchanged forecasts.
- Two commands executed from an isolated source copy, without `outputs/` or
  EDA artifacts, produced byte-identical CSVs matching the root output hash.
  Module provenance was checked to ensure the isolated copy was imported.
- `pip check` found no broken installed runtime requirements; `git diff --check`
  passed. Prediction CSV remains available and is not excluded by `.gitignore`.
- Existing evaluation outputs, model implementations and tests were preserved.

Tests were run with:

```powershell
python -m pytest -q -p no:cacheprovider
python -m pip check
git diff --check
```

Runtime used for this audit: Python 3.13.17, NumPy 2.5.4, pandas 3.1.0 and
Matplotlib 3.11.2. Declared compatibility remains Python 3.11 or newer.

## Report verification

The [Markdown report](final_report.md) retains all nine required sections,
four existing figures and the historical model/policy metrics. Model names,
hyperparameters, comparison tables and relative links were checked against
saved artifacts. Its operational recommendation remains **shadow monitoring
and decision support only**, with no automatic scaling or financial benefits
claimed.

The [PDF](final_report.pdf) was rendered from that source using an installed
browser, then rasterized with Windows' local PDF renderer. All pages were
inspected, including full-size checks of the results table, operational figure
and appendix. It contains **six main-report pages plus one appendix page**;
fonts, tables, figure captions and page breaks are readable without clipping.
The renderer adds no Python dependencies:

```powershell
python scripts/render_report.py
```

## Submission candidate

`atomity_forecasting_submission.zip` is a local candidate, built with:

```powershell
python scripts/package_submission.py
```

It contains an enclosing `forecasting-challenge/` directory with source code,
tests, project configuration, README and original contracts, the final
`predictions.csv`, Markdown/PDF reports, supporting figures, EDA scripts and
selected small analysis CSVs/manifests. The release card, checksum list and
dataset-download instructions are included; compressed raw datasets are not.

The packaging allowlist excludes Git internals, virtual environments, raw
compressed data, credentials, Python/notebook caches, local browser profiles,
audit scratch directories and redundant early-stage prediction intermediates.
Archive entries and CRC integrity are inspected. No upload or Git push is part
of packaging.

The inspected candidate contains **69 files**. After extraction, every relative
link in the README and final report resolves, the archived `predictions.csv`
matches the root file byte for byte, and the full **101-test suite also passes
from the extracted candidate**. The final archive rebuild changes this audit
note only; the source, tests, figures and predictions remain identical to those
tested.

## Remaining limits

A fresh dependency installation could not be verified in this environment:
the declared setuptools build backend is absent and the configured package
source supplied no installable version. The clean source-copy execution used
the available runtime dependencies; it is not a claim of a fresh environment
bootstrap. `pyproject.toml` declares `setuptools>=68`; reviewers need access to
that build backend and the declared runtime dependencies when installing.

Forecast ranking still rests on three windows. Only two elevated events were
available for the calibrated policy comparison, and the illustrative alerts
missed both. The unweighted CPU mean is not physical capacity telemetry.
These limitations remain in the final report and do not justify automated
infrastructure actions.

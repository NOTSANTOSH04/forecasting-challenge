"""Build an inspected local ZIP candidate from an explicit project allowlist.

Run after generating predictions and the final reports. No network operations,
Git mutations or raw dataset copies are performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import zipfile


DOCUMENTS = (
    "README.md", "pyproject.toml", ".gitignore", "ASSIGNMENT.md", "DATASET.md",
    "SUBMISSION.md", "SOURCES.md", "LICENSE", "CODE_OF_CONDUCT.md", "DATA_RELEASE.md",
    "DATA_CARD_TEMPLATE.md", "inspect_data.py", "predictions.csv",
    "challenge_data/README.md", "challenge_data/DATA_CARD.md", "challenge_data/SHA256SUMS",
)
ANALYSIS = (
    "train_targets.csv", "category_statistics.csv", "model_comparison.csv", "fold_performance.csv",
    "horizon_error_analysis.csv", "peak_error_analysis.csv", "residual_analysis.csv", "model_robustness.csv",
    "evaluation_validation_predictions.csv", "evaluation_manifest.json", "uncertainty_analysis.csv",
    "operational_policy_validation.csv", "policy_summary.csv", "operational_policy_manifest.json",
)
PATTERNS = ("src/cwfc/*.py", "tests/test_*.py", "scripts/*.py", "notebooks/*.py",
            "reports/*.md", "reports/final_report.pdf", "reports/figures/*.png")
REQUIRED = ("src/cwfc/predict.py", "tests/test_predict.py", "reports/final_report.md",
            "reports/final_report.pdf", "reports/submission_audit.md")
SECRET_MARKERS = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}")


def submission_files(root: Path) -> list[Path]:
    root = root.resolve()
    selected = {root / name for name in DOCUMENTS}
    selected.update(root / "outputs" / name for name in ANALYSIS)
    for pattern in PATTERNS:
        selected.update(root.glob(pattern))
    for name in (*DOCUMENTS, *REQUIRED, *("outputs/" + name for name in ANALYSIS)):
        if not (root / name).is_file():
            raise ValueError(f"Missing submission file: {name}")
    for path in selected:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Submission paths must be ordinary files inside the project")
        if path.stat().st_size > 2_000_000:
            raise ValueError(f"Unexpectedly large submission artifact: {path.relative_to(root)}")
        if path.suffix in (".py", ".md", ".toml", ".json") and SECRET_MARKERS.search(path.read_bytes()):
            raise ValueError(f"Possible secret in {path.relative_to(root)}; inspect before packaging")
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def build_archive(root: Path, output: Path) -> tuple[int, int]:
    root = root.resolve()
    files = submission_files(root)
    if output.resolve() in {path.resolve() for path in files}:
        raise ValueError("Archive output must not overwrite a selected source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = "forecasting-challenge/" + path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive failed CRC integrity verification")
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != len(files):
            raise ValueError("Archive contains duplicate or missing entries")
        forbidden = {".git", ".venv", ".aws", ".codex", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".submission-audit"}
        for name in names:
            path = Path(name)
            if (forbidden.intersection(path.parts) or name.endswith(".csv.gz")
                    or any(part.startswith(".env") for part in path.parts)):
                raise ValueError(f"Forbidden archive entry: {name}")
        if "forecasting-challenge/predictions.csv" not in names:
            raise ValueError("Required predictions.csv missing from archive")
    return len(files), output.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("atomity_forecasting_submission.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    count, size = build_archive(root, args.output)
    print(f"Inspected local archive: {args.output} ({count} files, {size:,} bytes)")
    print("Includes predictions.csv; excludes raw datasets, credentials, environments and caches. No external submission performed.")


if __name__ == "__main__":
    main()

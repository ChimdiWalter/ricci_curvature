from __future__ import annotations

from pathlib import Path


def append_deviation(root: Path, baseline: str, reason: str) -> None:
    path = root / "docs" / "deviations.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {baseline}\n\n{reason.strip()}\n")


def record_external_baseline_deviations(root: Path) -> None:
    append_deviation(
        root,
        "Monocle3 wrapper",
        "The Python pipeline records a wrapper contract. Exact execution requires an R environment "
        "with Monocle3 installed and an AnnData-to-R conversion bridge. No Monocle3 metric is "
        "reported unless `results/metrics/monocle3_metrics.csv` exists.",
    )
    append_deviation(
        root,
        "Slingshot wrapper",
        "The Python pipeline records a wrapper contract. Exact execution requires an R/Bioconductor "
        "environment with Slingshot installed. No Slingshot metric is reported unless "
        "`results/metrics/slingshot_metrics.csv` exists.",
    )


from __future__ import annotations

import argparse
from pathlib import Path

from ricci_cell_fate.reporting.figures import write_figures
from ricci_cell_fate.reporting.tables import write_report_tables
from ricci_cell_fate.utils.paths import find_project_root
from ricci_cell_fate.utils.provenance import artifact_record, read_json, write_json


def generate_report(root: Path | None = None) -> dict:
    root = root or find_project_root()
    table_paths = write_report_tables(root)
    figure_manifest = write_figures(root)
    run_manifest = {
        "tables": [artifact_record(path, task="reporting", root=root) for path in table_paths],
        "figures": figure_manifest.get("figures", []),
        "metric_summary": read_json(root / "results" / "metrics" / "summary_metrics.json", default={}),
    }
    write_json(run_manifest, root / "results" / "manifests" / "run_manifest.json")
    return run_manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    generate_report()


if __name__ == "__main__":  # pragma: no cover
    main()


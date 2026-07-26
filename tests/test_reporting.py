from __future__ import annotations

import json

import pandas as pd

from ricci_cell_fate.manuscript.build import build_manuscript
from ricci_cell_fate.reporting.figures import write_figures
from ricci_cell_fate.reporting.pipeline import generate_report
from ricci_cell_fate.reporting.tables import write_report_tables


def test_reporting_and_manuscript_placeholders(tmp_path):
    (tmp_path / "results" / "metrics").mkdir(parents=True)
    (tmp_path / "results" / "tables").mkdir(parents=True)
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "manuscript" / "auto").mkdir(parents=True)
    pd.DataFrame([{"dataset": "tiny", "task": "branch", "macro_f1": 0.5}]).to_csv(
        tmp_path / "results" / "metrics" / "curvature_metrics.csv",
        index=False,
    )
    (tmp_path / "results" / "metrics" / "summary_metrics.json").write_text(
        json.dumps({"metric_summaries": [{"file": "curvature_metrics.csv", "n_rows": 1, "status_proxy_labels_mean": 1.0}]}),
        encoding="utf-8",
    )
    table_paths = write_report_tables(tmp_path)
    assert table_paths
    figure_manifest = write_figures(tmp_path, dpi=72)
    assert figure_manifest["figures"]
    run_manifest = generate_report(tmp_path)
    assert "figures" in run_manifest
    manuscript = build_manuscript(tmp_path)
    assert (tmp_path / "manuscript" / "paper.md").exists()
    assert "paper" in manuscript
    assert "status_proxy_labels" in (tmp_path / "manuscript" / "auto" / "results_summary.md").read_text(encoding="utf-8")

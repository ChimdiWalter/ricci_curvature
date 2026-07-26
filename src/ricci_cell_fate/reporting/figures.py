from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ricci_cell_fate.reporting.placeholders import placeholder_figure
from ricci_cell_fate.utils.paths import find_project_root
from ricci_cell_fate.utils.provenance import artifact_record, write_json


def _save(fig, base: Path, dpi: int = 300) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in [".png", ".pdf"]:
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def workflow_schematic(root: Path, dpi: int = 300) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    steps = ["Download", "Preprocess", "Rebuild graph", "Curvature", "Tasks", "Manuscript"]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, (x, label) in enumerate(zip(xs, steps, strict=False)):
        ax.add_patch(plt.Rectangle((x - 0.065, 0.38), 0.13, 0.2, fill=False, lw=1.4))
        ax.text(x, 0.48, label, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.07, 0.48), xytext=(x + 0.07, 0.48), arrowprops={"arrowstyle": "->"})
    ax.text(0.5, 0.78, "Artifact-gated Ricci cell fate pipeline", ha="center", fontsize=12, weight="bold")
    return _save(fig, root / "results" / "figures" / "workflow_schematic", dpi)


def embedding_overlay(root: Path, dpi: int = 300) -> list[Path]:
    feature_files = sorted((root / "results" / "tables").glob("*/*_curvature_node_features.csv"))
    if not feature_files:
        return [Path(p) for p in placeholder_figure(root / "results" / "figures" / "embedding_curvature_overlay", "Missing Result", "Run curvature experiments first.", dpi)]
    df = pd.read_csv(feature_files[0], index_col=0)
    x_col = "PC1" if "PC1" in df else df.select_dtypes("number").columns[0]
    y_col = "PC2" if "PC2" in df else df.select_dtypes("number").columns[min(1, len(df.select_dtypes("number").columns) - 1)]
    color_col = "curvature_extremum_score" if "curvature_extremum_score" in df else df.select_dtypes("number").columns[-1]
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(df[x_col], df[y_col], c=df[color_col], s=8, cmap="viridis", linewidths=0)
    fig.colorbar(sc, ax=ax, label=color_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Curvature overlay")
    return _save(fig, root / "results" / "figures" / "embedding_curvature_overlay", dpi)


def branch_visualization(root: Path, dpi: int = 300) -> list[Path]:
    feature_files = sorted((root / "results" / "tables").glob("*/*_curvature_node_features.csv"))
    if not feature_files:
        return [Path(p) for p in placeholder_figure(root / "results" / "figures" / "branch_localization", "Missing Result", "Run curvature experiments first.", dpi)]
    df = pd.read_csv(feature_files[0], index_col=0)
    y = df["branch_score"].sort_values(ascending=False).head(100) if "branch_score" in df else pd.Series(dtype=float)
    if y.empty:
        return [Path(p) for p in placeholder_figure(root / "results" / "figures" / "branch_localization", "Missing Result", "No branch scores were found.", dpi)]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(len(y)), y.to_numpy(), lw=1.5)
    ax.set_xlabel("Ranked cell")
    ax.set_ylabel("Branch score")
    ax.set_title("Branch localization ranking")
    return _save(fig, root / "results" / "figures" / "branch_localization", dpi)


def metrics_panel(root: Path, name: str, metric_file: str, dpi: int = 300) -> list[Path]:
    path = root / "results" / "metrics" / metric_file
    if not path.exists():
        return [Path(p) for p in placeholder_figure(root / "results" / "figures" / name, "Missing Result", f"{metric_file} is absent.", dpi)]
    df = pd.read_csv(path)
    numeric = [c for c in df.select_dtypes("number").columns if not c.startswith("status_")]
    if df.empty or not numeric:
        return [Path(p) for p in placeholder_figure(root / "results" / "figures" / name, "Missing Result", f"{metric_file} has no numeric metrics.", dpi)]
    metric = "macro_f1" if "macro_f1" in numeric else numeric[0]
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return [Path(p) for p in placeholder_figure(root / "results" / "figures" / name, "Missing Result", f"{metric} is unavailable.", dpi)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = plot_df.get("dataset", pd.Series(range(len(plot_df)))).astype(str)
    ax.bar(np.arange(len(plot_df)), plot_df[metric].to_numpy())
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(name.replace("_", " ").title())
    return _save(fig, root / "results" / "figures" / name, dpi)


def write_figures(root: Path | None = None, dpi: int = 300) -> dict[str, Any]:
    root = root or find_project_root()
    figure_paths: list[Path] = []
    figure_paths.extend(workflow_schematic(root, dpi))
    figure_paths.extend(embedding_overlay(root, dpi))
    figure_paths.extend(branch_visualization(root, dpi))
    figure_paths.extend(metrics_panel(root, "early_fate_metrics", "curvature_metrics.csv", dpi))
    figure_paths.extend(metrics_panel(root, "ablation_plot", "ablation_metrics.csv", dpi))
    figure_paths.extend(metrics_panel(root, "transfer_generalization", "transfer_metrics.csv", dpi))
    manifest = {
        "figures": [
            artifact_record(path, task="reporting", root=root)
            for path in figure_paths
            if Path(path).exists()
        ]
    }
    write_json(manifest, root / "manuscript" / "auto" / "figure_manifest.json")
    return manifest


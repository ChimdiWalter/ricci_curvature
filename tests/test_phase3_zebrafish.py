from __future__ import annotations

from pathlib import Path

import pandas as pd

from ricci_cell_fate.tasks.phase3_zebrafish import (
    build_branch_region_target,
    build_early_lineage_task,
    build_family_map,
)
from ricci_cell_fate.utils.config import read_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_zebrafish_task_design_construction() -> None:
    cfg = read_yaml(ROOT / "configs" / "tasks" / "zebrafish_phase3.yaml")
    obs = pd.read_csv(ROOT / "data" / "processed" / "zebrafish" / "zebrafish_node_metadata.csv", index_col=0)
    edge = pd.read_csv(ROOT / "data" / "processed" / "zebrafish" / "zebrafish_edge_weights.csv")

    branch = build_branch_region_target(
        obs,
        edge,
        cluster_key=str(cfg["branch_region_task"]["cluster_key"]),
        family_map=build_family_map(cfg["branch_region_task"]["stage_family_map"]),
        anchor_clusters=[str(item) for item in cfg["branch_region_task"]["anchor_stages"]],
        tolerance_hops=int(cfg["branch_region_task"]["tolerance_hops"]),
    )
    early = build_early_lineage_task(
        obs,
        stage_key=str(cfg["early_fate_task"]["stage_key"]),
        lineage_key=str(cfg["early_fate_task"]["lineage_key"]),
        early_stages=[str(item) for item in cfg["early_fate_task"]["early_stages"]],
        supported_labels=[str(item) for item in cfg["early_fate_task"]["supported_labels"]],
    )

    assert int(branch.node_labels["branch_region_exact"].sum()) == 473
    assert int(branch.node_labels["branch_region_tolerant"].sum()) == 563
    assert set(branch.anchor_clusters) == {"04.8-30%", "05.3-50%"}

    support = early.class_support.set_index("lineage_label")["n_cells"].to_dict()
    assert support == {"Notochord": 305, "Prechordal Plate": 197}
    assert set(early.supported_labels) == {"Notochord", "Prechordal Plate"}
    assert not early.cell_labels["lineage_label"].isin(["Early Blastomeres"]).any()
    assert set(early.cell_labels["stage"].unique()) == {"04.3-DOME", "04.8-30%", "05.3-50%"}

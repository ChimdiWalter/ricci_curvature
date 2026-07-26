# Discrete Ricci Curvature for Transition-Region Ranking in Single-Cell Lineage Graphs

Code and configurations for:

**"Dataset-Dependent Utility of Discrete Ricci Curvature for Transition-Region Ranking in Single-Cell Lineage Graphs"**

*Chimdi Walter Ndubuisi, University of Missouri*

## Overview

Controlled evaluation of when Forman-Ricci and Ollivier-Ricci curvature features add information beyond strong graph-topology baselines for transition-region ranking in single-cell lineage graphs. Three datasets (Paul15, pancreas, zebrafish), five classifiers, out-of-sample graph attachment, strictly inductive evaluation, and synthetic ground-truth benchmarks.

## Key Findings

- Curvature provides dataset- and classifier-dependent incremental signal for branch-region ranking
- Largest canonical Ollivier gain on pancreas (AUPRC +0.092, 95% CI [0.076, 0.114])
- Gains diminish with non-linear classifiers on two of three datasets
- Signal persists under out-of-sample graph attachment
- Curvature-only models remain weak; graph topology is essential

## Installation

```bash
pip install -e .
```

## Data

Datasets load via standard APIs (no manual download):
- Paul15: `scanpy.datasets.paul15()`
- Pancreas: `cellrank.datasets.pancreas(kind='raw')`
- Zebrafish: `cellrank.datasets.zebrafish()`

## Structure

```
src/ricci_cell_fate/
    curvature/     # Forman-Ricci and Ollivier-Ricci
    graphs/        # Graph construction and features
    tasks/         # Task definitions per dataset
    evaluation/    # Metrics
configs/           # Dataset and task configurations
scripts/           # Experiment runners
tests/             # Test suite
```

## License

MIT

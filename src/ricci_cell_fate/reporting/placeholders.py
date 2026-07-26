from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def placeholder_figure(path_base: str | Path, title: str, message: str, dpi: int = 300) -> list[str]:
    base = Path(path_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in [".png", ".pdf"]:
        path = base.with_suffix(suffix)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14, weight="bold")
        ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(str(path))
    return paths


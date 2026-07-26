from __future__ import annotations

from pathlib import Path
from typing import Any


def read_h5ad(path: str | Path) -> Any:
    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("anndata is required to read .h5ad files") from exc
    return ad.read_h5ad(path)


def write_h5ad(adata: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(p)


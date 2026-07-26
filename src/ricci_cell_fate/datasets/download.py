from __future__ import annotations

import argparse
import importlib
import shutil
from pathlib import Path
from typing import Any

from ricci_cell_fate.datasets.registry import DatasetSpec, load_registry
from ricci_cell_fate.datasets.validate import validate_h5ad, write_metadata_table
from ricci_cell_fate.io.anndata_io import write_h5ad
from ricci_cell_fate.utils.paths import ensure_dir, find_project_root
from ricci_cell_fate.utils.provenance import artifact_record, write_json, write_placeholder


class DatasetDownloadError(RuntimeError):
    """Raised when an official dataset route cannot be executed."""


def _import_loader(spec: DatasetSpec):
    try:
        module = importlib.import_module(spec.module_name)
    except Exception as exc:
        raise DatasetDownloadError(
            f"Could not import official loader module {spec.module_name!r} for {spec.name}: {exc}"
        ) from exc
    try:
        return getattr(module, spec.function_name)
    except AttributeError as exc:
        raise DatasetDownloadError(f"Missing loader function {spec.loader!r}") from exc


def _official_path_argument(spec: DatasetSpec, raw_dir: Path) -> Path | None:
    if spec.loader.startswith("scanpy.datasets."):
        return None
    if spec.name == "pancreas" and spec.loader_kwargs.get("kind") == "raw":
        return raw_dir / "pancreas.h5ad"
    return raw_dir / spec.raw_filename


def download_dataset(
    spec: DatasetSpec,
    *,
    root: Path | None = None,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    root = root or find_project_root()
    raw_dir = ensure_dir(root / "data" / "raw" / spec.name)
    manifest_dir = ensure_dir(root / "data" / "manifests")
    raw_path = raw_dir / spec.raw_filename
    placeholder = raw_dir / "MISSING_DATA_DEPENDENCY.md"

    try:
        loader = _import_loader(spec)
        kwargs = dict(spec.loader_kwargs)
        path_arg = _official_path_argument(spec, raw_dir)
        if path_arg is not None:
            kwargs["path"] = path_arg
        adata = loader(**kwargs)
        if not raw_path.exists():
            write_h5ad(adata, raw_path)
        if placeholder.exists():
            placeholder.unlink()
        summary = validate_h5ad(raw_path, spec, manifest_dir / f"{spec.name}_schema.json")
        write_metadata_table(summary, manifest_dir / f"{spec.name}_metadata_summary.csv")
        record = artifact_record(
            raw_path,
            dataset=spec.name,
            config_path="configs/datasets/registry.yaml",
            source_url=spec.official_api_url,
            notes=f"Official loader {spec.loader}; backing route {spec.backing_url}",
            root=root,
        )
        record["retrieval_route"] = {
            "loader": spec.loader,
            "loader_kwargs": spec.loader_kwargs,
            "official_api_url": spec.official_api_url,
            "backing_url": spec.backing_url,
        }
        return record
    except Exception as exc:
        msg = (
            f"Dataset {spec.name} could not be downloaded through official route {spec.loader}.\n\n"
            f"Official API URL: {spec.official_api_url}\n"
            f"Backing URL or package route: {spec.backing_url}\n\n"
            f"Error: {type(exc).__name__}: {exc}\n"
        )
        write_placeholder(placeholder, "Missing Data Dependency", msg)
        if allow_placeholders:
            return {
                "path": str(placeholder),
                "checksum": None,
                "dataset": spec.name,
                "source_url": spec.official_api_url,
                "status": "placeholder",
                "error": msg,
            }
        raise DatasetDownloadError(msg) from exc


def download_many(
    datasets: list[str],
    *,
    root: Path | None = None,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    root = root or find_project_root()
    registry = load_registry(root / "configs" / "datasets" / "registry.yaml")
    records = []
    for name in datasets:
        spec = registry[name]
        if not spec.enabled:
            continue
        records.append(download_dataset(spec, root=root, allow_placeholders=allow_placeholders))
    manifest = {"artifacts": records}
    write_json(manifest, root / "data" / "manifests" / "download_manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pancreas", "lung", "zebrafish", "paul15"])
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args(argv)
    download_many(args.datasets, allow_placeholders=args.allow_placeholders)


if __name__ == "__main__":  # pragma: no cover
    main()

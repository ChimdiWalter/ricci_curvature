from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ricci_cell_fate.utils.config import read_yaml
from ricci_cell_fate.utils.paths import find_project_root


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    enabled: bool
    loader: str
    loader_kwargs: dict[str, Any]
    official_api_url: str
    backing_url: str
    expected_shape: tuple[int, int] | None
    organism: str
    biological_system: str
    time_key_candidates: tuple[str, ...] = field(default_factory=tuple)
    lineage_key_candidates: tuple[str, ...] = field(default_factory=tuple)
    batch_key_candidates: tuple[str, ...] = field(default_factory=tuple)

    @property
    def raw_filename(self) -> str:
        return f"{self.name}_raw.h5ad"

    @property
    def processed_filename(self) -> str:
        return f"{self.name}_processed.h5ad"

    @property
    def module_name(self) -> str:
        return ".".join(self.loader.split(".")[:-1])

    @property
    def function_name(self) -> str:
        return self.loader.split(".")[-1]


def _coerce_spec(name: str, data: dict[str, Any]) -> DatasetSpec:
    expected = data.get("expected_shape")
    return DatasetSpec(
        name=name,
        enabled=bool(data.get("enabled", True)),
        loader=str(data["loader"]),
        loader_kwargs=dict(data.get("loader_kwargs", {})),
        official_api_url=str(data["official_api_url"]),
        backing_url=str(data["backing_url"]),
        expected_shape=tuple(expected) if expected else None,
        organism=str(data.get("organism", "unknown")),
        biological_system=str(data.get("biological_system", "unknown")),
        time_key_candidates=tuple(data.get("time_key_candidates", [])),
        lineage_key_candidates=tuple(data.get("lineage_key_candidates", [])),
        batch_key_candidates=tuple(data.get("batch_key_candidates", [])),
    )


def load_registry(path: str | Path | None = None) -> dict[str, DatasetSpec]:
    root = find_project_root()
    registry_path = Path(path) if path else root / "configs" / "datasets" / "registry.yaml"
    payload = read_yaml(registry_path)
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"Malformed dataset registry: {registry_path}")
    return {name: _coerce_spec(name, spec) for name, spec in datasets.items()}


def get_dataset(name: str, path: str | Path | None = None) -> DatasetSpec:
    registry = load_registry(path)
    try:
        return registry[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(registry)}") from exc


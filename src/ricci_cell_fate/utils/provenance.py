from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .hashing import sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_commit(root: str | Path | None = None) -> str | None:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(cmd, cwd=root, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


@dataclass
class ArtifactRecord:
    path: str
    checksum: str | None
    created_at: str
    dataset: str | None = None
    task: str | None = None
    split_id: str | None = None
    config_path: str | None = None
    seed: int | None = None
    software: dict[str, str | None] | None = None
    source_url: str | None = None
    git_commit: str | None = None
    notes: str | None = None


def artifact_record(
    path: str | Path,
    *,
    dataset: str | None = None,
    task: str | None = None,
    split_id: str | None = None,
    config_path: str | None = None,
    seed: int | None = None,
    source_url: str | None = None,
    notes: str | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    p = Path(path)
    checksum = sha256_file(p) if p.exists() and p.is_file() else None
    rec = ArtifactRecord(
        path=str(p),
        checksum=checksum,
        created_at=utc_now(),
        dataset=dataset,
        task=task,
        split_id=split_id,
        config_path=config_path,
        seed=seed,
        software={
            "python": platform.python_version(),
            "ricci_cell_fate": package_version("ricci-cell-fate"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
            "networkx": package_version("networkx"),
            "anndata": package_version("anndata"),
            "scanpy": package_version("scanpy"),
            "cellrank": package_version("cellrank"),
        },
        source_url=source_url,
        git_commit=git_commit(root),
        notes=notes,
    )
    return asdict(rec)


def read_json(path: str | Path, default: Any | None = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_manifest(record: dict[str, Any], path: str | Path, key: str = "artifacts") -> None:
    manifest = read_json(path, default={key: []})
    manifest.setdefault(key, [])
    manifest[key].append(record)
    manifest["updated_at"] = utc_now()
    write_json(manifest, path)


def write_placeholder(path: str | Path, title: str, message: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n{message.strip()}\n")
    return p


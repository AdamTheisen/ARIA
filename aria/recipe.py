from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

from .region import Region, region_from_dict

@dataclass
class ARIARecipe:
    name: str
    region: Region
    sources: dict[str, Any] = field(default_factory=dict)
    time: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=lambda: {"backend": "icechunk"})
    compute: dict[str, Any] = field(default_factory=lambda: {"engine": "dask", "scheduler": "threads"})

    @classmethod
    def from_dict(cls, payload):
        if "name" not in payload:
            raise ValueError("Recipe requires a name.")
        if "region" not in payload:
            raise ValueError("Recipe requires a region definition.")
        return cls(
            name=str(payload["name"]),
            region=region_from_dict(payload["region"]),
            sources=dict(payload.get("sources", {})),
            time=dict(payload.get("time", {})),
            storage=dict(payload.get("storage", {"backend": "icechunk"})),
            compute=dict(payload.get("compute", {"engine": "dask", "scheduler": "threads"})),
        )

    @classmethod
    def load(cls, path):
        payload = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(payload)

    def validate(self):
        if not self.sources:
            raise ValueError("Recipe must define at least one source.")
        backend = self.storage.get("backend", "icechunk")
        if backend not in {"icechunk", "zarr"}:
            raise ValueError("storage.backend must be 'icechunk' or 'zarr'.")
        return True

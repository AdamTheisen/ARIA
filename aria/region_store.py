from __future__ import annotations

import json
from pathlib import Path

from .region import Region, region_from_dict


def region_store_path() -> Path:
    return Path.home() / ".config" / "aria" / "regions.json"


def load_saved_regions(path: Path | None = None) -> dict[str, Region]:
    path = Path(path or region_store_path())
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    out = {}
    for name, item in (payload or {}).items():
        try:
            data = dict(item)
            data["name"] = name
            out[name] = region_from_dict(data)
        except Exception:
            continue
    return out


def save_region(region: Region, path: Path | None = None) -> None:
    path = Path(path or region_store_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text()) or {}
        except Exception:
            payload = {}
    payload[region.name] = {
        "west": region.west,
        "south": region.south,
        "east": region.east,
        "north": region.north,
        "query_states": list(region.query_states_override),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def delete_region(name: str, path: Path | None = None) -> bool:
    path = Path(path or region_store_path())
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text()) or {}
    except Exception:
        return False
    if name not in payload:
        return False
    del payload[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return True

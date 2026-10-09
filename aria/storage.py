from __future__ import annotations

"""Persistent ARIA regional storage manager.

The Streamlit dashboard only edits configuration and reads state. A scheduler
(cron, launchd, systemd, Kubernetes, etc.) should periodically run
``aria storage update`` so collection continues when the dashboard is closed.

v0.21 stores core Xarray scientific datasets in appendable Icechunk-backed
Zarr v3 repositories and supports multiple named regional storage profiles.
Live collection and historical backfill write into the same profile stores.
Naturally tabular observations and derived catalogs remain Parquet.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
import os
import re
import shutil
import socket
import time
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .region import GPGL_REGION, region_from_dict

CONFIG_PATH = Path(os.environ.get("ARIA_STORAGE_CONFIG", Path.home()/".config"/"aria"/"storage.json")).expanduser()
STATE_PATH = Path(os.environ.get("ARIA_STORAGE_STATE", Path.home()/".cache"/"aria"/"storage"/"state.json")).expanduser()
DEFAULT_ROOT = Path(os.environ.get("ARIA_STORAGE_ROOT", Path.home()/".local"/"share"/"aria"/"storage")).expanduser()

DEFAULT_SOURCES = {
    "surface": {"enabled": True, "interval_minutes": 5},
    "mrms": {"enabled": True, "interval_minutes": 5},
    "hrrr": {
        "enabled": True,
        "interval_minutes": 60,
        # Basic: legacy F00 surface/radar fields only.
        # Standard: selected forecast leads + curated pressure-level atmosphere.
        # Full-Campaign: hourly leads through F18 + expanded pressure-level fields.
        "storage_tier": "standard",
        "forecast_hours": [0, 1, 3, 6, 12],
        "pressure_levels_hpa": [1000, 925, 850, 700, 500, 300, 250],
    },
    "air_quality": {"enabled": False, "interval_minutes": 30},
    "radiosonde": {"enabled": False, "interval_minutes": 60},
    "sst": {"enabled": False, "interval_minutes": 1440},
    "marine": {"enabled": False, "interval_minutes": 10},
    # Native individual-radar persistence is opt-in because Level-II volume
    # storage can grow quickly. v0.24 establishes the profile configuration
    # and canonical storage path; collection is activated only for explicitly
    # selected sites.
    "nexrad": {
        "enabled": False,
        "interval_minutes": 5,
        "storage_mode": "selected",
        "selected_radars": [],
        "native_polar": True,
    },
    # Standard integrated differences are generated from already-persisted
    # ARIA sources. No extra upstream download is required.
    "model_obs": {"enabled": True, "interval_minutes": 60},
}
SOURCE_LABELS = {
    "surface": "Surface observations",
    "mrms": "MRMS reflectivity",
    "hrrr": "HRRR surface",
    "air_quality": "Air Quality",
    "radiosonde": "Radiosondes",
    "sst": "Sea/Lake Surface Temperature",
    "marine": "Great Lakes / marine observations",
    "nexrad": "Individual NEXRAD Level II",
    "model_obs": "Model–observation differences",
}

SOURCE_PATHS = {
    "surface": ("observations", "surface"),
    "air_quality": ("observations", "air_quality"),
    "radiosonde": ("observations", "radiosonde"),
    "mrms": ("analyses", "mrms"),
    "sst": ("analyses", "sst"),
    "marine": ("observations", "marine"),
    "hrrr": ("models", "hrrr"),
    "nexrad": ("radar", "nexrad"),
    "model_obs": ("derived", "model_obs"),
}
SOURCE_STORAGE_FORMAT = {
    "surface": "Icechunk / Zarr v3 + partitioned Parquet",
    "air_quality": "Icechunk / Zarr v3 + partitioned Parquet",
    "radiosonde": "Icechunk / Zarr v3 + partitioned Parquet",
    "mrms": "Icechunk / Zarr v3",
    "sst": "Icechunk / Zarr v3",
    "marine": "Partitioned Parquet",
    "hrrr": "Icechunk / Zarr v3",
    "nexrad": "Radar DataTree / Icechunk Zarr v3 (opt-in)",
    "model_obs": "Icechunk / Zarr v3",
}

SOURCE_NATIVE_CADENCE = {
    "surface": "~5 min observations",
    "mrms": "~2–5 min product",
    "hrrr": "Hourly model cycles",
    "air_quality": "Typically hourly",
    "radiosonde": "Typically 12-hour launches",
    "sst": "Daily gridded analysis",
    "marine": "~10 min latest buoy/platform observations",
    "nexrad": "~4–10 min native Level-II volumes",
    "model_obs": "Hourly derived analyses",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _utcnow()).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def _copy_sources(sources=None) -> dict:
    base = json.loads(json.dumps(DEFAULT_SOURCES))
    if sources:
        for key, spec in sources.items():
            base.setdefault(key, {})
            if isinstance(spec, dict):
                base[key].update(spec)
    return base


def _profile_payload(region=GPGL_REGION, *, enabled=False, sources=None) -> dict:
    return {
        "enabled": bool(enabled),
        "region": region.as_dict() if hasattr(region, "as_dict") else dict(region),
        "sources": _copy_sources(sources),
        "created_utc": _iso(),
        "updated_utc": _iso(),
    }


def _canonicalize_config(data: dict | None = None) -> dict:
    data = json.loads(json.dumps(data or {}, default=str))
    # Upgrade the single-region v0.20 schema in place. The default profile slug
    # matches the old region directory, so existing data remain where they are.
    if "profiles" not in data:
        region_payload = data.get("region", GPGL_REGION.as_dict())
        try:
            region = region_from_dict(region_payload)
        except Exception:
            region = GPGL_REGION
        profile_name = _slug(region.name)
        profile = _profile_payload(
            region,
            enabled=bool(data.get("enabled", False)),
            sources=data.get("sources", {}),
        )
        if data.get("created_utc"):
            profile["created_utc"] = data["created_utc"]
        data = {
            "version": 3,
            "root": str(data.get("root", DEFAULT_ROOT)),
            "active_profile": profile_name,
            "profiles": {profile_name: profile},
            "created_utc": data.get("created_utc", _iso()),
            "updated_utc": _iso(),
        }
    data["version"] = 3
    data.setdefault("root", str(DEFAULT_ROOT))
    data.setdefault("profiles", {})
    if not data["profiles"]:
        name = _slug(GPGL_REGION.name)
        data["profiles"][name] = _profile_payload(GPGL_REGION)
    data.setdefault("active_profile", next(iter(data["profiles"])))
    if data["active_profile"] not in data["profiles"]:
        data["active_profile"] = next(iter(data["profiles"]))
    for name, profile in data["profiles"].items():
        profile.setdefault("enabled", False)
        profile.setdefault("region", GPGL_REGION.as_dict())
        profile["sources"] = _copy_sources(profile.get("sources", {}))
        profile.setdefault("created_utc", _iso())
        profile.setdefault("updated_utc", _iso())
    data.setdefault("created_utc", _iso())
    data.setdefault("updated_utc", _iso())
    return data


def _load_config_canonical() -> dict:
    if not CONFIG_PATH.exists():
        return _canonicalize_config({})
    try:
        raw = json.loads(CONFIG_PATH.read_text())
        canonical = _canonicalize_config(raw)
        if raw.get("version") != 3 or "profiles" not in raw:
            _atomic_json(CONFIG_PATH, canonical)
        return canonical
    except Exception:
        return _canonicalize_config({})


def _project_profile(config: dict, profile_name: str | None = None) -> dict:
    canonical = _canonicalize_config(config)
    name = profile_name or canonical.get("active_profile")
    if name not in canonical["profiles"]:
        raise KeyError(f"Unknown storage profile: {name}")
    out = json.loads(json.dumps(canonical))
    profile = canonical["profiles"][name]
    out["profile_name"] = name
    out["enabled"] = bool(profile.get("enabled", False))
    out["region"] = json.loads(json.dumps(profile.get("region", {})))
    out["sources"] = json.loads(json.dumps(profile.get("sources", {})))
    return out


def default_config(region=GPGL_REGION) -> dict:
    name = _slug(region.name)
    canonical = {
        "version": 3,
        "root": str(DEFAULT_ROOT),
        "active_profile": name,
        "profiles": {name: _profile_payload(region)},
        "created_utc": _iso(),
        "updated_utc": _iso(),
    }
    return _project_profile(canonical, name)


def load_config(profile_name: str | None = None) -> dict:
    return _project_profile(_load_config_canonical(), profile_name)


def save_config(config: dict) -> dict:
    previous=_load_config_canonical()
    canonical=_canonicalize_config(config)
    name=config.get("profile_name") or config.get("active_profile") or canonical["active_profile"]
    previous_region_key=None
    try:
        if name in previous.get("profiles",{}):
            previous_region_key=region_from_dict(previous["profiles"][name]["region"]).cache_key
    except Exception:
        previous_region_key=None
    if name not in canonical["profiles"]:
        canonical["profiles"][name] = _profile_payload(config.get("region", GPGL_REGION.as_dict()))
    profile = canonical["profiles"][name]
    # Compatibility projection: existing dashboard code edits these top-level
    # fields. Persist them back into the selected named profile.
    for key in ("enabled", "region", "sources"):
        if key in config:
            profile[key] = json.loads(json.dumps(config[key], default=str))
    profile["sources"] = _copy_sources(profile.get("sources", {}))
    profile["updated_utc"] = _iso()
    canonical["active_profile"] = name
    canonical["updated_utc"] = _iso()
    persist = {k:v for k,v in canonical.items() if k not in {"profile_name","enabled","region","sources"}}
    _atomic_json(CONFIG_PATH,persist)

    # Repointing a profile to another region/grid must make every enabled source
    # due immediately. The writer will preserve any incompatible previous grid
    # repository under _grid_archive when the next collection runs.
    try:
        new_region_key=region_from_dict(profile["region"]).cache_key
    except Exception:
        new_region_key=None
    if previous_region_key and new_region_key and previous_region_key!=new_region_key:
        try:
            state=_load_state_canonical()
            pstate=state["profiles"].setdefault(name,{"sources":{}})
            pstate["sources"]={}
            pstate["last_collector_result"]="region changed; sources reset"
            _save_state_canonical(state)
        except Exception:
            pass
    return _project_profile(persist,name)


def list_storage_profiles(config: dict | None = None) -> dict[str, dict]:
    canonical = _canonicalize_config(config or _load_config_canonical())
    return json.loads(json.dumps(canonical["profiles"]))


def create_storage_profile(name: str, region=GPGL_REGION, *, copy_from: str | None = None, enabled=False) -> dict:
    canonical = _load_config_canonical()
    slug = _slug(name)
    if slug in canonical["profiles"]:
        raise ValueError(f"Storage profile already exists: {slug}")
    if copy_from:
        if copy_from not in canonical["profiles"]:
            raise KeyError(f"Unknown storage profile: {copy_from}")
        src = canonical["profiles"][copy_from]
        profile = _profile_payload(region, enabled=enabled, sources=src.get("sources", {}))
    else:
        profile = _profile_payload(region, enabled=enabled)
    profile["display_name"] = str(name)
    canonical["profiles"][slug] = profile
    canonical["active_profile"] = slug
    canonical["updated_utc"] = _iso()
    _atomic_json(CONFIG_PATH, canonical)
    return _project_profile(canonical, slug)


def set_active_profile(name: str) -> dict:
    canonical = _load_config_canonical()
    if name not in canonical["profiles"]:
        raise KeyError(f"Unknown storage profile: {name}")
    canonical["active_profile"] = name
    canonical["updated_utc"] = _iso()
    _atomic_json(CONFIG_PATH, canonical)
    return _project_profile(canonical, name)


def profile_for_region(region, config: dict | None = None, *, require_enabled=False) -> dict | None:
    canonical = _canonicalize_config(config or _load_config_canonical())
    for name, profile in canonical["profiles"].items():
        try:
            stored = region_from_dict(profile["region"])
            if stored.cache_key == region.cache_key and (not require_enabled or profile.get("enabled", False)):
                return _project_profile(canonical, name)
        except Exception:
            continue
    return None


def _canonicalize_state(data: dict | None = None, config: dict | None = None) -> dict:
    data = json.loads(json.dumps(data or {}, default=str))
    cfg = _canonicalize_config(config or _load_config_canonical())
    if "profiles" not in data:
        name = cfg.get("active_profile")
        pstate = {
            "sources": data.get("sources", {}),
            "last_collector_start_utc": data.get("last_collector_start_utc"),
            "last_collector_finish_utc": data.get("last_collector_finish_utc"),
            "last_collector_result": data.get("last_collector_result"),
        }
        data = {"version": 3, "profiles": {name:pstate}, "history": data.get("history", [])}
    data["version"] = 3
    data.setdefault("profiles", {})
    for name in cfg["profiles"]:
        data["profiles"].setdefault(name, {"sources": {}})
        data["profiles"][name].setdefault("sources", {})
    data.setdefault("history", [])
    return data


def _load_state_canonical() -> dict:
    if not STATE_PATH.exists():
        return _canonicalize_state({})
    try:
        return _canonicalize_state(json.loads(STATE_PATH.read_text()))
    except Exception:
        return _canonicalize_state({})


def _project_state(state: dict, profile_name: str | None = None) -> dict:
    cfg = _load_config_canonical()
    canonical = _canonicalize_state(state, cfg)
    name = profile_name or cfg.get("active_profile")
    pstate = canonical["profiles"].setdefault(name, {"sources": {}})
    out = json.loads(json.dumps(canonical))
    out["profile_name"] = name
    out["sources"] = json.loads(json.dumps(pstate.get("sources", {})))
    for key in ("last_collector_start_utc","last_collector_finish_utc","last_collector_result","collector_host","collector_pid"):
        if key in pstate:
            out[key] = pstate[key]
    return out


def load_state(profile_name: str | None = None) -> dict:
    return _project_state(_load_state_canonical(), profile_name)


def _save_state_canonical(state: dict) -> dict:
    state = _canonicalize_state(state)
    state["updated_utc"] = _iso()
    state["history"] = state.get("history", [])[-500:]
    _atomic_json(STATE_PATH, state)
    return state


def save_state(state: dict, profile_name: str | None = None) -> dict:
    canonical = _canonicalize_state(state)
    cfg = _load_config_canonical()
    name = profile_name or state.get("profile_name") or cfg.get("active_profile")
    pstate = canonical["profiles"].setdefault(name, {"sources": {}})
    if "sources" in state:
        pstate["sources"] = json.loads(json.dumps(state["sources"], default=str))
    for key in ("last_collector_start_utc","last_collector_finish_utc","last_collector_result","collector_host","collector_pid"):
        if key in state:
            pstate[key] = state[key]
    canonical = _save_state_canonical(canonical)
    return _project_state(canonical, name)

def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "region"


def region_store_root(config: dict, profile_name: str | None = None) -> Path:
    if profile_name is not None and "profiles" in config:
        cfg = _project_profile(config, profile_name)
    elif config.get("profile_name"):
        cfg = config
    elif "profiles" in config:
        cfg = _project_profile(config)
    else:
        cfg = config
    region = region_from_dict(cfg["region"])
    root = Path(cfg.get("root", DEFAULT_ROOT)).expanduser()
    # Existing upgraded stores retain their old region slug because the
    # migration profile is created with that same slug. New named profiles get
    # their own independent directory even if two profiles share one region.
    store_name = cfg.get("profile_name") or _slug(region.name)
    return root / _slug(store_name)


def _source_dir(config: dict, source: str) -> Path:
    parts = SOURCE_PATHS.get(source, ("observations", source))
    path = region_store_root(config).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _legacy_source_dir(config: dict, source: str) -> Path:
    """v0.19 source directory retained for migration/read fallback."""
    return region_store_root(config) / source


def _ensure_store_manifest(config: dict) -> Path:
    root = region_store_root(config)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "store_manifest.json"
    payload = {
        "aria_storage_schema": 4,
        "core_scientific_format": "Icechunk / Zarr v3",
        "tabular_format": "Partitioned Parquet",
        "profile": config.get("profile_name"),
        "region": config.get("region", {}),
        "updated_utc": _iso(),
    }
    _atomic_json(path, payload)
    return path


def _json_safe(value: Any) -> Any:
    """Convert common scientific Python values to JSON/NetCDF-safe scalars.

    pandas/NumPy booleans and timestamps are common in adapter metadata.
    NetCDF4 does not accept boolean attributes, and Parquet serializes
    DataFrame.attrs through JSON, so normalize them once at the storage layer.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bytes)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _sanitize_attrs(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()

    def clean(attrs):
        out = {}
        for key, value in attrs.items():
            safe = _json_safe(value)
            if safe is None:
                continue
            # netCDF attributes accept scalars and homogeneous numeric/string
            # sequences, but nested mappings do not. Preserve nested metadata
            # as a JSON string instead of losing it.
            if isinstance(safe, dict):
                out[key] = json.dumps(safe, sort_keys=True)
            elif isinstance(safe, list):
                if all(isinstance(v, (str, int, float, bytes)) for v in safe):
                    out[key] = safe
                else:
                    out[key] = json.dumps(safe, sort_keys=True)
            else:
                out[key] = safe
        return out

    ds.attrs = clean(ds.attrs)
    for name in ds.variables:
        ds[name].attrs = clean(ds[name].attrs)
    return ds


def _timestamp_from_dataset(ds: xr.Dataset) -> pd.Timestamp:
    for attr in ("analysis_time", "valid_time", "time_coverage_start"):
        if attr in ds.attrs:
            try: return pd.Timestamp(ds.attrs[attr])
            except Exception: pass
    if "time" in ds.coords and ds["time"].size:
        try: return pd.Timestamp(ds["time"].values[-1])
        except Exception: pass
    return pd.Timestamp.now(tz="UTC")


def _stamp(value: Any = None) -> str:
    try:
        t = pd.Timestamp(value) if value is not None else pd.Timestamp.now(tz="UTC")
        if t.tzinfo is None: t = t.tz_localize("UTC")
        return t.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
    except Exception:
        return pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")


def _icechunk_repo_path(directory: Path, stem: str) -> Path:
    return directory / f"{stem}.icechunk"


def _prepare_dataset_for_icechunk(ds: xr.Dataset, when=None) -> xr.Dataset:
    ds = _sanitize_attrs(ds)
    stamp = pd.Timestamp(when if when is not None else _timestamp_from_dataset(ds))
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)

    # Every persistent scientific store has an explicit appendable time axis.
    if "time" in ds.dims:
        try:
            times = pd.to_datetime(ds["time"].values)
            ds = ds.assign_coords(time=("time", times.to_numpy(dtype="datetime64[ns]")))
        except Exception:
            pass
    elif "time" in ds.coords and ds["time"].ndim == 0:
        ds = ds.expand_dims(time=[stamp.to_datetime64()])
    else:
        ds = ds.expand_dims(time=[stamp.to_datetime64()])

    # Use one explicit, sufficiently fine CF time encoding for new stores.
    # This avoids xarray/Icechunk repeatedly renegotiating inherited
    # "days since ..." encodings when sub-hourly timestamps are appended.
    if "time" in ds.coords:
        try:
            ds["time"].encoding.clear()
            ds["time"].encoding.update({
                "dtype": "int64",
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
            })
        except Exception:
            pass

    ds.attrs["aria_storage_format"] = "icechunk_zarr_v3"
    ds.attrs["aria_storage_schema"] = 4
    return ds


def _icechunk_encoding(ds: xr.Dataset) -> dict:
    """Reasonable regional chunks with a stable explicit time encoding."""
    encoding = {}
    if "time" in ds.coords:
        encoding["time"] = {
            "dtype": "int64",
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }
    for name, var in ds.data_vars.items():
        chunks = []
        for dim, size in var.sizes.items():
            if dim == "time":
                chunks.append(1)
            elif dim in ("latitude", "longitude", "x", "y", "x_km", "y_km"):
                chunks.append(min(int(size), 256))
            elif dim in ("altitude_km", "pressure_hpa", "level"):
                chunks.append(min(int(size), 8))
            else:
                chunks.append(min(int(size), 64))
        if chunks:
            encoding[name] = {"chunks": tuple(chunks)}
    return encoding


def _open_icechunk_repository(path: Path):
    try:
        import icechunk as ic
    except ImportError as exc:
        raise RuntimeError(
            "Persistent ARCO storage requires icechunk. Install ARIA dependencies "
            "with `python -m pip install -e .`."
        ) from exc
    path.mkdir(parents=True, exist_ok=True)
    storage = ic.local_filesystem_storage(str(path))
    return ic.Repository.open_or_create(storage)



def _dataset_layout_signature(ds: xr.Dataset) -> str:
    """Stable signature for the append-compatible Xarray layout.

    Time length is deliberately excluded. Spatial/vertical dimensions,
    non-time coordinates, and data-variable dimension layouts are included.
    This catches region/grid changes and schema changes before xarray/to_zarr
    attempts an incompatible append.
    """
    h=hashlib.sha256()
    dims={str(k):int(v) for k,v in ds.sizes.items() if str(k)!="time"}
    h.update(json.dumps(sorted(dims.items()),separators=(",",":")).encode())

    for name in sorted(ds.coords):
        coord=ds.coords[name]
        if name=="time" or "time" in coord.dims:
            continue
        h.update(str(name).encode())
        h.update(json.dumps(tuple(str(d) for d in coord.dims)).encode())
        arr=np.asarray(coord.values)
        h.update(str(arr.dtype).encode())
        h.update(json.dumps(arr.shape).encode())
        try:
            if np.issubdtype(arr.dtype,np.floating):
                payload=np.round(arr.astype("float64"),8).tobytes()
            elif np.issubdtype(arr.dtype,np.number) or np.issubdtype(arr.dtype,np.datetime64):
                payload=np.ascontiguousarray(arr).tobytes()
            else:
                payload=json.dumps(arr.astype(str).ravel().tolist(),separators=(",",":")).encode()
        except Exception:
            payload=json.dumps(np.asarray(arr).astype(str).ravel().tolist(),separators=(",",":")).encode()
        h.update(payload)

    variables=[]
    for name,var in sorted(ds.data_vars.items()):
        variables.append((str(name),tuple(str(d) for d in var.dims if str(d)!="time")))
    h.update(json.dumps(variables,separators=(",",":")).encode())
    return h.hexdigest()[:20]


def _archive_incompatible_icechunk(repo_path: Path, directory: Path, stem: str, old_signature: str | None) -> Path:
    """Move an incompatible active repository aside without deleting data."""
    archive_root=directory/"_grid_archive"
    archive_root.mkdir(parents=True,exist_ok=True)
    stamp=pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    sig=(old_signature or "unknown")[:12]
    target=archive_root/f"{stem}-{stamp}-{sig}.icechunk"
    counter=1
    while target.exists():
        target=archive_root/f"{stem}-{stamp}-{sig}-{counter}.icechunk"
        counter+=1
    shutil.move(str(repo_path),str(target))
    return target


def _write_dataset(ds: xr.Dataset, directory: Path, stem: str, when=None) -> Path:
    """Append an Xarray dataset to Icechunk/Zarr v3, rotating incompatible grids.

    A storage profile can be repointed to a different region/grid and ARIA
    schemas can gain variables over time. Those changes are not append-compatible
    with an existing Zarr array layout. Rather than fail the collector, preserve
    the previous repository under ``_grid_archive`` and initialize a fresh
    active repository for the new layout.
    """
    from icechunk.xarray import to_icechunk

    repo_path=_icechunk_repo_path(directory,stem)
    prepared=_prepare_dataset_for_icechunk(ds,when)
    prepared_signature=_dataset_layout_signature(prepared)
    prepared.attrs["aria_layout_signature"]=prepared_signature

    repo=_open_icechunk_repository(repo_path)
    initialized=False
    existing_times=set()
    existing_signature=None
    incompatible=False

    try:
        read_session=repo.readonly_session("main")
        existing=xr.open_zarr(
            read_session.store,
            consolidated=False,
            zarr_format=3,
            chunks=None,
        )
        initialized=bool(existing.data_vars)
        if initialized:
            existing_signature=existing.attrs.get("aria_layout_signature") or _dataset_layout_signature(existing)
            incompatible=(existing_signature!=prepared_signature)
        if initialized and not incompatible and "time" in existing.coords:
            existing_times={
                pd.Timestamp(v).to_datetime64()
                for v in np.asarray(existing["time"].values).ravel()
            }
        existing.close()
    except Exception:
        # A repository that cannot be read should not be overwritten here; let
        # Icechunk surface the underlying problem unless it is simply empty.
        initialized=False

    if initialized and incompatible:
        _archive_incompatible_icechunk(
            repo_path,directory,stem,existing_signature,
        )
        repo=_open_icechunk_repository(repo_path)
        initialized=False
        existing_times=set()

    if initialized and "time" in prepared.coords:
        new_times=np.asarray(prepared["time"].values).ravel()
        keep=np.array(
            [pd.Timestamp(v).to_datetime64() not in existing_times for v in new_times],
            dtype=bool,
        )
        if not keep.any():
            return repo_path
        prepared=prepared.isel(time=np.where(keep)[0])

    session=repo.writable_session("main")
    encoding=_icechunk_encoding(prepared)
    if initialized:
        to_icechunk(
            prepared,
            session,
            append_dim="time",
        )
        message=f"ARIA append {stem}: {prepared.sizes.get('time',1)} time step(s)"
    else:
        to_icechunk(
            prepared,
            session,
            mode="w",
            encoding=encoding or None,
        )
        message=f"ARIA initialize {stem} layout={prepared_signature}"
    session.commit(message)
    return repo_path


def _load_icechunk_latest(path: Path) -> xr.Dataset | None:
    if not path.exists():
        return None
    try:
        import icechunk as ic
        storage = ic.local_filesystem_storage(str(path))
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        opened = xr.open_zarr(
            session.store,
            consolidated=False,
            zarr_format=3,
            chunks=None,
        )
        if "time" in opened.dims and opened.sizes.get("time", 0):
            loaded = opened.isel(time=[-1]).load()
        else:
            loaded = opened.load()
        opened.close()
        loaded.attrs["aria_data_origin"] = "persistent_store"
        loaded.attrs["aria_storage_path"] = str(path)
        loaded.attrs["aria_storage_format"] = "icechunk_zarr_v3"
        return loaded
    except Exception:
        return None



def load_stored_time_range(
    region,
    source: str,
    stem: str,
    start,
    end,
    *,
    max_frames: int | None = None,
) -> xr.Dataset | None:
    """Read an arbitrary UTC time range from a matching profile Icechunk store."""
    config = profile_for_region(region)
    if config is None:
        return None
    path = _icechunk_repo_path(_source_dir(config, source), stem)
    if not path.exists():
        return None
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end < start:
        raise ValueError("end must be after start")
    try:
        import icechunk as ic
        storage = ic.local_filesystem_storage(str(path))
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        opened = xr.open_zarr(session.store, consolidated=False, zarr_format=3, chunks=None)
        if "time" not in opened.coords:
            loaded = opened.load()
        else:
            times = pd.to_datetime(opened["time"].values, utc=True, errors="coerce")
            keep = np.where((times >= start) & (times <= end))[0]
            if not len(keep):
                opened.close()
                return None
            if max_frames is not None and len(keep) > int(max_frames):
                pick = np.linspace(0, len(keep)-1, int(max_frames)).round().astype(int)
                keep = keep[pick]
            loaded = opened.isel(time=keep).load()
        opened.close()
        loaded.attrs["aria_data_origin"] = "persistent_store_history"
        loaded.attrs["aria_storage_profile"] = config.get("profile_name")
        loaded.attrs["aria_storage_path"] = str(path)
        return loaded
    except Exception:
        return None



def _parquet_dataset_root(directory: Path, stem: str) -> Path:
    """Root of one partitioned Parquet dataset."""
    return directory / stem


def _parquet_snapshot_path(directory: Path, stem: str, when=None) -> Path:
    stamp = pd.Timestamp(when if when is not None else pd.Timestamp.now(tz="UTC"))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    stamp = stamp.tz_convert("UTC")
    root = _parquet_dataset_root(directory, stem)
    partition = root / f"year={stamp.year:04d}" / f"month={stamp.month:02d}" / f"day={stamp.day:02d}"
    partition.mkdir(parents=True, exist_ok=True)
    return partition / f"part-{stamp.strftime('%Y%m%dT%H%M%SZ')}.parquet"


def _parquet_files(directory: Path, stem: str) -> list[Path]:
    """Return partitioned Parquet parts plus old snapshot-style files."""
    files=[]
    root=_parquet_dataset_root(directory, stem)
    if root.exists():
        files.extend(root.rglob("*.parquet"))
    files.extend(directory.glob(f"{stem}_*.parquet"))
    return sorted(set(files))


def _snapshot_time_from_parquet_path(path: Path) -> pd.Timestamp | None:
    name=path.stem
    token=None
    if name.startswith("part-"):
        token=name[5:]
    elif "_" in name:
        token=name.rsplit("_",1)[-1]
    if not token:
        return None
    try:
        return pd.to_datetime(token,format="%Y%m%dT%H%M%SZ",utc=True)
    except Exception:
        return None


def load_stored_frame_range(region, source: str, stem: str, start, end) -> pd.DataFrame | None:
    """Read Parquet snapshots spanning an arbitrary UTC time range."""
    config = profile_for_region(region)
    if config is None:
        return None
    directory = _source_dir(config, source)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    frames=[]
    for path in _parquet_files(directory,stem):
        stamp=_snapshot_time_from_parquet_path(path)
        if stamp is None:
            continue
        if start <= stamp <= end:
            frame=pd.read_parquet(path)
            if "aria_snapshot_time" not in frame:
                frame["aria_snapshot_time"]=stamp
            frames.append(frame)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out.attrs["aria_storage_profile"] = config.get("profile_name")
    return out


def load_stored_dataset_history(
    region,
    source: str,
    stem: str,
    *,
    hours: float = 3.0,
    max_frames: int | None = None,
) -> xr.Dataset | None:
    """Load a recent time window from an Icechunk store.

    Unlike :func:`load_stored_dataset`, this is intended for local playback and
    therefore returns multiple time steps. It never triggers a network update.
    """
    config = profile_for_region(region)
    if config is None:
        return None
    path = _icechunk_repo_path(_source_dir(config, source), stem)
    if not path.exists():
        return None
    try:
        import icechunk as ic
        storage = ic.local_filesystem_storage(str(path))
        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        opened = xr.open_zarr(
            session.store,
            consolidated=False,
            zarr_format=3,
            chunks=None,
        )
        if "time" not in opened.coords or not opened["time"].size:
            loaded = opened.load()
        else:
            times = pd.to_datetime(opened["time"].values, utc=True, errors="coerce")
            latest = times.max()
            cutoff = latest - pd.Timedelta(hours=float(hours))
            keep = np.where(times >= cutoff)[0]
            if max_frames is not None and len(keep) > int(max_frames):
                # Evenly sample the stored frames so long windows remain smooth
                # without loading hundreds of regional grids into memory.
                pick = np.linspace(0, len(keep)-1, int(max_frames)).round().astype(int)
                keep = keep[pick]
            loaded = opened.isel(time=keep).load()
        opened.close()
        loaded.attrs["aria_data_origin"] = "persistent_store_history"
        loaded.attrs["aria_storage_path"] = str(path)
        return loaded
    except Exception:
        return None


def _write_frame(df: pd.DataFrame, directory: Path, stem: str, when=None) -> Path | None:
    """Write tabular observations as a date-partitioned Parquet dataset."""
    if df is None or df.empty:
        return None
    if when is None and "time" in df:
        when=pd.to_datetime(df["time"],utc=True,errors="coerce").max()
    stamp=pd.Timestamp(when if when is not None else pd.Timestamp.now(tz="UTC"))
    if stamp.tzinfo is None:
        stamp=stamp.tz_localize("UTC")
    stamp=stamp.tz_convert("UTC")
    path=_parquet_snapshot_path(directory,stem,stamp)
    if not path.exists():
        tmp=path.with_suffix(".parquet.tmp")
        safe_df=df.copy()
        if "aria_snapshot_time" not in safe_df.columns:
            safe_df["aria_snapshot_time"]=stamp

        # Stabilize Arrow schemas. The shared observation `value` column is
        # always numeric. Other mixed object columns containing text are
        # explicitly stored as nullable strings rather than relying on
        # PyArrow's type inference.
        for col in safe_df.columns:
            if safe_df[col].dtype != object:
                continue
            if col=="value":
                safe_df[col]=pd.to_numeric(safe_df[col],errors="coerce")
                continue
            nonnull=safe_df[col].dropna()
            if nonnull.empty:
                continue
            numeric=pd.to_numeric(nonnull,errors="coerce")
            if numeric.notna().all():
                safe_df[col]=pd.to_numeric(safe_df[col],errors="coerce")
            elif any(isinstance(v,str) for v in nonnull):
                safe_df[col]=safe_df[col].astype("string")

        safe_attrs=_json_safe(dict(getattr(df,"attrs",{})))
        safe_df.attrs=safe_attrs if isinstance(safe_attrs,dict) else {}
        safe_df.to_parquet(tmp,index=False)
        os.replace(tmp,path)
    return path



HRRR_TIER_DEFAULTS = {
    "basic": {
        "forecast_hours": [0],
        "pressure_levels_hpa": [],
        "surface_variables": [
            "air_temperature_2m", "dew_point_temperature_2m",
            "u_wind_10m", "v_wind_10m",
        ],
        "radar_variables": ["composite_reflectivity", "reflectivity_1km"],
        "pressure_variables": [],
    },
    "standard": {
        "forecast_hours": [0, 1, 3, 6, 12],
        "pressure_levels_hpa": [1000, 925, 850, 700, 500, 300, 250],
        "surface_variables": [
            "air_temperature_2m", "dew_point_temperature_2m",
            "u_wind_10m", "v_wind_10m",
        ],
        "radar_variables": ["composite_reflectivity", "reflectivity_1km"],
        "pressure_variables": [
            "air_temperature", "relative_humidity",
            "u_wind", "v_wind", "geopotential_height",
        ],
    },
    "full_campaign": {
        "forecast_hours": list(range(0, 19)),
        "pressure_levels_hpa": [
            1000, 950, 925, 900, 850, 800, 750, 700, 650, 600,
            550, 500, 450, 400, 350, 300, 250, 200, 150, 100,
        ],
        "surface_variables": [
            "air_temperature_2m", "dew_point_temperature_2m",
            "u_wind_10m", "v_wind_10m", "surface_pressure",
            "precipitation_rate",
        ],
        "radar_variables": ["composite_reflectivity", "reflectivity_1km"],
        "pressure_variables": [
            "air_temperature", "dew_point_temperature", "relative_humidity",
            "u_wind", "v_wind", "geopotential_height", "vertical_velocity",
        ],
    },
}


def _hrrr_storage_settings(region) -> dict:
    """Resolve one profile's HRRR storage tier with backward-compatible defaults."""
    cfg=profile_for_region(region,require_enabled=False)
    spec=(cfg or {}).get("sources",{}).get("hrrr",{})
    tier=str(spec.get("storage_tier","standard")).strip().lower().replace("-","_")
    if tier not in HRRR_TIER_DEFAULTS:
        tier="standard"
    settings=json.loads(json.dumps(HRRR_TIER_DEFAULTS[tier]))
    if tier!="basic":
        if spec.get("forecast_hours"):
            settings["forecast_hours"]=sorted({
                int(x) for x in spec["forecast_hours"] if 0 <= int(x) <= 48
            })
        if spec.get("pressure_levels_hpa"):
            settings["pressure_levels_hpa"]=sorted({
                int(x) for x in spec["pressure_levels_hpa"] if 50 <= int(x) <= 1100
            }, reverse=True)
    settings["tier"]=tier
    return settings


def _hrrr_member_for_cube(ds: xr.Dataset, run, forecast_hour: int) -> xr.Dataset:
    """Normalize one HRRR member before concatenating it into a forecast cube."""
    ds=ds.copy()
    # Remove source scalar clock coordinates; ARIA stores initialization on the
    # appendable time axis and valid time on the forecast-hour coordinate.
    for name in ("time","step","valid_time"):
        if name in ds.coords and name not in ds.dims:
            try:
                ds=ds.drop_vars(name)
            except Exception:
                pass
    ds=ds.expand_dims(forecast_hour=[int(forecast_hour)])
    ds=ds.assign_coords(
        valid_time=(
            "forecast_hour",
            np.asarray([pd.Timestamp(run.valid_time).to_datetime64()],dtype="datetime64[ns]"),
        )
    )
    ds["forecast_hour"].attrs.update({
        "long_name":"Forecast lead time",
        "units":"hours",
    })
    ds["valid_time"].attrs["long_name"]="Model valid time"
    return ds


def _concat_hrrr_members(members: list[xr.Dataset], *, tier: str, product: str) -> xr.Dataset:
    if not members:
        raise RuntimeError(f"No HRRR {product} forecast members were retrieved.")
    ds=xr.concat(
        members,
        dim="forecast_hour",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="outer",
    ).sortby("forecast_hour")
    ds.attrs.update({
        "aria_hrrr_storage_tier":tier,
        "aria_hrrr_product":product,
        "aria_forecast_cube":"initialization_time × forecast_hour × spatial/vertical dimensions",
    })
    return ds


def _subset_pressure_levels(ds: xr.Dataset, levels: list[int]) -> xr.Dataset:
    if not levels or "pressure_hpa" not in ds.coords:
        return ds
    available=np.asarray(ds["pressure_hpa"].values,dtype=float)
    wanted=[float(x) for x in levels]
    keep=[]
    for level in wanted:
        if available.size:
            idx=int(np.nanargmin(np.abs(available-level)))
            if abs(float(available[idx])-level) <= 1.0:
                keep.append(float(available[idx]))
    keep=list(dict.fromkeys(keep))
    return ds.sel(pressure_hpa=keep) if keep else ds.isel(pressure_hpa=slice(0,0))


def _open_forecast_cube_latest(region, stem: str) -> xr.Dataset | None:
    """Open the latest initialization from an HRRR forecast cube."""
    return load_stored_dataset(region,"hrrr",stem,max_age_minutes=24*60)


def _select_hrrr_valid_member(region, stem: str, target_time, *, max_offset_minutes=30):
    """Select the persisted initialization/lead nearest a requested valid time."""
    cfg=profile_for_region(region,require_enabled=False)
    if cfg is None:
        return None, None
    path=_icechunk_repo_path(_source_dir(cfg,"hrrr"),stem)
    if not path.exists():
        return None, None
    target=pd.Timestamp(target_time)
    target=target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    try:
        import icechunk as ic
        repo=ic.Repository.open(ic.local_filesystem_storage(str(path)))
        session=repo.readonly_session("main")
        opened=xr.open_zarr(session.store,consolidated=False,zarr_format=3,chunks=None)
        if "time" not in opened.coords or "forecast_hour" not in opened.coords:
            opened.close()
            return None, None
        inits=pd.to_datetime(opened["time"].values,utc=True,errors="coerce")
        leads=np.asarray(opened["forecast_hour"].values,dtype=int)
        best=None
        for ti,init in enumerate(inits):
            if pd.isna(init):
                continue
            for fi,lead in enumerate(leads):
                valid=pd.Timestamp(init)+pd.Timedelta(hours=int(lead))
                offset=abs((valid-target).total_seconds())/60.0
                if best is None or offset < best[0]:
                    best=(offset,ti,fi,pd.Timestamp(init),int(lead),valid)
        if best is None or best[0] > float(max_offset_minutes):
            opened.close()
            return None, None
        offset,ti,fi,init,lead,valid=best
        member=opened.isel(time=ti,forecast_hour=fi).load()
        opened.close()
        meta={
            "initialization_time":init,
            "forecast_hour":lead,
            "valid_time":valid,
            "offset_minutes":float((valid-target).total_seconds()/60.0),
        }
        return member,meta
    except Exception:
        return None,None



def load_stored_hrrr_valid_member(
    region,
    product: str,
    target_time,
    *,
    max_offset_minutes: float = 30,
):
    """Public local-first selector for persisted HRRR forecast cubes.

    ``product`` is one of ``surface``, ``radar``, or ``pressure``.
    Returns ``(dataset, match_metadata)`` or ``(None, None)``.
    """
    stems={
        "surface":"surface_forecast",
        "radar":"radar_forecast",
        "pressure":"pressure_forecast",
    }
    key=str(product).lower()
    if key not in stems:
        raise ValueError(f"Unknown HRRR forecast-cube product: {product}")
    return _select_hrrr_valid_member(
        region,stems[key],target_time,
        max_offset_minutes=max_offset_minutes,
    )


def _collect(source: str, region, directory: Path) -> dict:
    """Run one source adapter/workflow and persist a snapshot."""
    from . import workflows

    files: list[str] = []
    data_time = None
    if source == "surface":
        ds, latest, start, end = workflows.build_latest_surface(region=region, include_history=False)
        data_time = _timestamp_from_dataset(ds)
        files.append(str(_write_dataset(ds, directory, "analysis", data_time)))
        p = _write_frame(latest, directory, "observations", data_time)
        if p: files.append(str(p))
    elif source == "mrms":
        ds, scan_time = workflows.build_mrms_at_time(region=region)
        data_time = pd.Timestamp(scan_time)
        files.append(str(_write_dataset(ds, directory, "reflectivity", data_time)))
    elif source == "hrrr":
        settings=_hrrr_storage_settings(region)
        tier=settings["tier"]
        leads=settings["forecast_hours"]

        # Use one initialization cycle for every lead in this collector run.
        # build_hrrr_surface(None, F00) resolves the newest conservatively
        # available initialization; subsequent members explicitly reuse it.
        base_vars=list(dict.fromkeys(
            settings["surface_variables"]+settings["radar_variables"]
        ))
        f00,base_run=workflows.build_hrrr_surface(
            region=region,forecast_hour=0,variables=base_vars
        )
        cycle=pd.Timestamp(base_run.initialization_time)
        data_time=cycle

        surface_members=[]
        radar_members=[]
        pressure_members=[]

        for fxx in leads:
            if int(fxx)==0:
                surface_raw=f00[[
                    v for v in settings["surface_variables"] if v in f00
                ]]
                radar_raw=f00[[
                    v for v in settings["radar_variables"] if v in f00
                ]]
                run=base_run
            else:
                surface_raw,run=workflows.build_hrrr_surface(
                    region=region,cycle=cycle,forecast_hour=int(fxx),
                    variables=settings["surface_variables"],
                )
                radar_raw,_=workflows.build_hrrr_surface(
                    region=region,cycle=cycle,forecast_hour=int(fxx),
                    variables=settings["radar_variables"],
                )

            surface_members.append(_hrrr_member_for_cube(surface_raw,run,int(fxx)))
            radar_members.append(_hrrr_member_for_cube(radar_raw,run,int(fxx)))

            if settings["pressure_variables"]:
                pressure_raw,pressure_run=workflows.build_hrrr_pressure(
                    region=region,cycle=cycle,forecast_hour=int(fxx),
                    variables=settings["pressure_variables"],
                )
                pressure_raw=_subset_pressure_levels(
                    pressure_raw,settings["pressure_levels_hpa"]
                )
                pressure_members.append(
                    _hrrr_member_for_cube(pressure_raw,pressure_run,int(fxx))
                )

        surface_cube=_concat_hrrr_members(
            surface_members,tier=tier,product="surface"
        )
        radar_cube=_concat_hrrr_members(
            radar_members,tier=tier,product="radar"
        )
        files.append(str(_write_dataset(
            surface_cube,directory,"surface_forecast",cycle
        )))
        files.append(str(_write_dataset(
            radar_cube,directory,"radar_forecast",cycle
        )))

        if pressure_members:
            pressure_cube=_concat_hrrr_members(
                pressure_members,tier=tier,product="pressure"
            )
            files.append(str(_write_dataset(
                pressure_cube,directory,"pressure_forecast",cycle
            )))

        # Transitional compatibility store used by existing dashboard readers.
        # It contains only F00 and can be retired once all readers select from
        # surface_forecast/radar_forecast.
        files.append(str(_write_dataset(f00,directory,"f00",base_run.valid_time)))
    elif source == "air_quality":
        ds, latest, observations, latest_time = workflows.build_latest_air_quality(region=region, max_distance_km=300.0)
        data_time = pd.Timestamp(latest_time)
        files.append(str(_write_dataset(ds, directory, "analysis", data_time)))
        p = _write_frame(latest, directory, "observations", data_time)
        if p: files.append(str(p))
    elif source == "radiosonde":
        ds, matched, trajectory, points = workflows.build_latest_atmosphere(region=region)
        data_time = pd.Timestamp(ds.attrs.get("raob_cycle", pd.Timestamp.now(tz="UTC")))
        files.append(str(_write_dataset(ds, directory, "analysis", data_time)))
        for stem, frame in (("profiles", matched), ("trajectory", trajectory), ("points", points)):
            if isinstance(frame, pd.DataFrame):
                p = _write_frame(frame, directory, stem, data_time)
                if p: files.append(str(p))
    elif source == "marine":
        from .adapters.marine import load_latest_marine_observations
        latest = load_latest_marine_observations(region)
        if latest.empty:
            raise RuntimeError("No marine observations were available in the selected region.")
        data_time = pd.to_datetime(latest["time"], utc=True, errors="coerce").max()
        p = _write_frame(latest, directory, "observations", data_time)
        if p: files.append(str(p))
    elif source == "sst":
        from .adapters.sst import open_sst
        ds, product = open_sst(region)
        try:
            loaded = ds.load()
        finally:
            try: ds.close()
            except Exception: pass
        data_time = _timestamp_from_dataset(loaded)
        files.append(str(_write_dataset(loaded, directory, product.key, data_time)))
    elif source == "model_obs":
        # Build standard comparison products exclusively from the local
        # persistent stores. This keeps derived generation reproducible and
        # prevents the collector from silently making another network request.
        from .integrated import build_surface_difference, build_radar_difference

        # Model members are selected by observation valid time from the
        # persisted forecast cubes rather than comparing observations with
        # whichever F00 happens to be newest.
        def _ds_time(ds, attr_names=()):
            if ds is None:
                return None
            if "time" in ds.coords and ds["time"].size:
                vals=pd.to_datetime(ds["time"].values,utc=True,errors="coerce")
                vals=[pd.Timestamp(x) for x in vals.ravel() if not pd.isna(x)]
                if vals:
                    return max(vals)
            for name in attr_names:
                value=ds.attrs.get(name)
                if value:
                    stamp=pd.Timestamp(value)
                    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            return None

        built=0
        failures=[]

        surface=load_stored_dataset(region,"surface","analysis",max_age_minutes=120)
        surface_time=_ds_time(surface)
        if surface is not None and surface_time is not None:
            model,match=_select_hrrr_valid_member(
                region,"surface_forecast",surface_time,max_offset_minutes=30
            )
            if model is not None and match is not None:
                for model_variable,stem in (
                    ("air_temperature_2m","hrrr_surface_temperature"),
                    ("dew_point_temperature_2m","hrrr_surface_dew_point"),
                ):
                    if model_variable not in model:
                        continue
                    try:
                        comp=build_surface_difference(model,surface,model_variable)
                        comp.attrs.update({
                            "aria_derived_product":"model_observation_difference",
                            "model_source":"HRRR",
                            "observation_source":"ARIA surface analysis",
                            "model_initialization_time":match["initialization_time"].isoformat(),
                            "forecast_hour":int(match["forecast_hour"]),
                            "model_valid_time":match["valid_time"].isoformat(),
                            "observation_time":surface_time.isoformat(),
                            "observation_offset_minutes":float(match["offset_minutes"]),
                            "matching_rule":"nearest persisted HRRR valid time within 30 minutes",
                        })
                        files.append(str(_write_dataset(
                            comp,directory,stem,match["valid_time"]
                        )))
                        built+=1
                    except Exception as exc:
                        failures.append(f"{stem}: {exc}")

        mrms=load_stored_dataset(region,"mrms","reflectivity",max_age_minutes=120)
        mrms_time=_ds_time(mrms,("analysis_time",))
        if mrms is not None and mrms_time is not None:
            model,match=_select_hrrr_valid_member(
                region,"radar_forecast",mrms_time,max_offset_minutes=30
            )
            if model is not None and match is not None and "composite_reflectivity" in model:
                try:
                    comp=build_radar_difference(
                        model,mrms,model_variable="composite_reflectivity"
                    )
                    comp.attrs.update({
                        "aria_derived_product":"model_observation_difference",
                        "model_source":"HRRR",
                        "observation_source":"MRMS QC Composite",
                        "model_initialization_time":match["initialization_time"].isoformat(),
                        "forecast_hour":int(match["forecast_hour"]),
                        "model_valid_time":match["valid_time"].isoformat(),
                        "observation_time":mrms_time.isoformat(),
                        "observation_offset_minutes":float(match["offset_minutes"]),
                        "matching_rule":"nearest persisted HRRR valid time within 30 minutes",
                    })
                    files.append(str(_write_dataset(
                        comp,directory,"hrrr_mrms_reflectivity",match["valid_time"]
                    )))
                    built+=1
                except Exception as exc:
                    failures.append(f"hrrr_mrms_reflectivity: {exc}")

        data_time=max(
            [x for x in (surface_time,mrms_time) if x is not None],
            default=pd.Timestamp.now(tz="UTC"),
        )
        if not built:
            detail="; ".join(failures) if failures else "no locally persisted source pairs met the 30-minute matching criterion"
            raise RuntimeError(f"No model–observation products were generated: {detail}")
    elif source == "nexrad":
        from .adapters.nexrad import nexrad_sites_for_region
        spec = load_config().get("sources", {}).get("nexrad", {})
        selected = [str(x).upper() for x in spec.get("selected_radars", [])]
        mode = str(spec.get("storage_mode", "selected"))
        if mode == "all_region":
            selected = sorted(nexrad_sites_for_region(region))
        if not selected:
            raise RuntimeError(
                "Individual NEXRAD storage is enabled but no radar sites are selected. "
                "Choose Selected radars or All radars in region in Data Storage."
            )
        # Native Radar DataTree/Icechunk persistence is intentionally guarded
        # until the xradar/radar-datatree writer is available in this environment.
        # The configuration is persisted now without silently falling back to
        # raw Level-II duplication.
        raise RuntimeError(
            "Individual NEXRAD persistence is configured but the native Radar DataTree "
            "writer is not enabled in this ARIA build. The profile selection is preserved; "
            "MRMS collection and interactive NEXRAD access are unaffected."
        )
    else:
        raise ValueError(f"Unknown storage source: {source}")
    return {"data_time": str(data_time), "files": files}


def storage_matches_region(region, config: dict | None = None) -> bool:
    return profile_for_region(region, config) is not None


def latest_stored_file(source: str, stem: str, *, suffix: str, config: dict | None = None) -> Path | None:
    config = config or load_config()
    directory = _source_dir(config, source)
    if not directory.exists():
        return None
    files = sorted(directory.glob(f"{stem}_*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def stored_source_is_current(region, source: str, max_age_minutes: float, *, config: dict | None = None, state: dict | None = None) -> bool:
    matched = profile_for_region(region, config)
    if matched is None or not matched.get("enabled", False):
        return False
    if not matched.get("sources", {}).get(source, {}).get("enabled", False):
        return False
    matched_state = load_state(matched.get("profile_name")) if state is None or state.get("profile_name") != matched.get("profile_name") else state
    source_state = matched_state.get("sources", {}).get(source, {})
    if source_state.get("status") not in ("current", "success"):
        return False
    last = source_state.get("last_success_utc")
    if not last:
        return False
    try:
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last)).total_seconds() / 60.0
        return age <= float(max_age_minutes)
    except Exception:
        return False


def load_stored_dataset(region, source: str, stem: str, *, max_age_minutes: float) -> xr.Dataset | None:
    config = profile_for_region(region)
    if config is None:
        return None
    state = load_state(config.get("profile_name"))
    if not stored_source_is_current(region, source, max_age_minutes, config=config, state=state):
        return None

    path = _icechunk_repo_path(_source_dir(config, source), stem)
    ds = _load_icechunk_latest(path)
    if ds is not None:
        ds.attrs["aria_storage_profile"] = config.get("profile_name")
        return ds

    # Active ARIA readers intentionally do not fall back to NetCDF. Legacy
    # snapshots must be migrated into Icechunk/Zarr v3 first.
    return None


def load_stored_frame(region, source: str, stem: str, *, max_age_minutes: float) -> pd.DataFrame | None:
    config = profile_for_region(region)
    if config is None:
        return None
    state = load_state(config.get("profile_name"))
    if not stored_source_is_current(region, source, max_age_minutes, config=config, state=state):
        return None
    directory=_source_dir(config,source)
    files=sorted(_parquet_files(directory,stem),key=lambda p:p.stat().st_mtime,reverse=True)
    if not files:
        legacy=_legacy_source_dir(config,source)
        files=sorted(_parquet_files(legacy,stem),key=lambda p:p.stat().st_mtime,reverse=True) if legacy.exists() else []
    if not files:
        return None
    frame=pd.read_parquet(files[0])
    frame.attrs["aria_storage_profile"] = config.get("profile_name")
    return frame


def _legacy_archive_root(config: dict) -> Path:
    active_root=Path(config.get("root",DEFAULT_ROOT)).expanduser()
    return active_root.parent / f"{active_root.name}_legacy_archive" / _slug(config.get("profile_name") or region_from_dict(config["region"]).name)


def _archive_legacy_file(path: Path, config: dict, source: str) -> Path:
    target_root=_legacy_archive_root(config) / source
    target_root.mkdir(parents=True,exist_ok=True)
    target=target_root/path.name
    if target.exists():
        target=target_root/f"{path.stem}-{int(time.time())}{path.suffix}"
    shutil.move(str(path),str(target))
    return target


def migrate_legacy_snapshots(
    *,
    source: str | None = None,
    config: dict | None = None,
    archive_legacy: bool = True,
) -> dict:
    """Convert legacy ARIA storage into active cloud-optimized stores.

    Gridded NetCDF snapshots are appended to Icechunk-backed Zarr v3.
    Snapshot-style Parquet files are rewritten into date-partitioned Parquet
    datasets. After a successful verified conversion, legacy files are moved
    outside the active storage root by default.
    """
    config=config or load_config()
    sources=[source] if source else list(DEFAULT_SOURCES)
    summary={}
    for key in sources:
        active=_source_dir(config,key)
        legacy=_legacy_source_dir(config,key)
        nc_files=[]
        parquet_files=[]
        if legacy.exists():
            nc_files.extend(legacy.glob("*.nc"))
            parquet_files.extend(legacy.glob("*.parquet"))
        # v0.20/v0.21 snapshot-style Parquet can also exist in the active dir.
        parquet_files.extend(active.glob("*_*.parquet"))
        # Defensive: capture any NetCDF snapshots left in active source dirs.
        nc_files.extend(active.glob("*.nc"))
        nc_files=sorted(set(nc_files))
        parquet_files=sorted(set(parquet_files))

        migrated_nc=0
        migrated_parquet=0
        archived=[]
        failures=[]

        for path in nc_files:
            stem=re.sub(r"_\\d{8}T\\d{6}Z$","",path.stem)
            try:
                with xr.open_dataset(path) as opened:
                    ds=opened.load()
                target=_write_dataset(ds,active,stem,_timestamp_from_dataset(ds))
                # Verify the repository is readable before removing the legacy source.
                if _load_icechunk_latest(Path(target)) is None:
                    raise RuntimeError("Icechunk verification failed after NetCDF migration")
                migrated_nc+=1
                if archive_legacy:
                    archived.append(str(_archive_legacy_file(path,config,key)))
            except Exception as exc:
                failures.append({"file":str(path),"error":str(exc)})

        for path in parquet_files:
            try:
                stamp=_snapshot_time_from_parquet_path(path)
                if stamp is None:
                    # Old naming convention is stem_YYYYMMDDTHHMMSSZ.parquet.
                    match=re.search(r"(\\d{8}T\\d{6}Z)$",path.stem)
                    stamp=pd.to_datetime(match.group(1),format="%Y%m%dT%H%M%SZ",utc=True) if match else None
                stem=re.sub(r"_\\d{8}T\\d{6}Z$","",path.stem)
                frame=pd.read_parquet(path)
                target=_write_frame(frame,active,stem,stamp)
                if target is None or not target.exists():
                    raise RuntimeError("Partitioned Parquet verification failed")
                pd.read_parquet(target,columns=[] if len(frame.columns)==0 else None)
                migrated_parquet+=1
                if archive_legacy and path.resolve()!=target.resolve():
                    archived.append(str(_archive_legacy_file(path,config,key)))
            except Exception as exc:
                failures.append({"file":str(path),"error":str(exc)})

        summary[key]={
            "netcdf_found":len(nc_files),
            "netcdf_to_icechunk":migrated_nc,
            "parquet_snapshots_found":len(parquet_files),
            "parquet_to_partitioned":migrated_parquet,
            "archived":archived,
            "failures":failures,
        }
    _ensure_store_manifest(config)
    return summary



class SourceLock:
    def __init__(self, config: dict, source: str, stale_seconds: int = 7200):
        self.path = region_store_root(config) / ".locks" / f"{source}.lock"
        self.stale_seconds = stale_seconds
        self.acquired = False
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                age = time.time() - self.path.stat().st_mtime
                if age > self.stale_seconds:
                    shutil.rmtree(self.path)
            except Exception:
                pass
        try:
            self.path.mkdir()
            (self.path / "owner.json").write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started_utc": _iso()}))
            self.acquired = True
        except FileExistsError:
            self.acquired = False
        return self.acquired
    def __exit__(self, *exc):
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)


def source_due(config: dict, state: dict, source: str, *, force=False, now=None) -> bool:
    if force: return True
    spec = config.get("sources", {}).get(source, {})
    if not spec.get("enabled", False): return False
    last = state.get("sources", {}).get(source, {}).get("last_success_utc")
    if not last: return True
    try:
        elapsed = (pd.Timestamp(now or _utcnow()) - pd.Timestamp(last)).total_seconds()/60.0
        return elapsed >= float(spec.get("interval_minutes", 60))
    except Exception:
        return True


def update_storage(*, source: str | None = None, force: bool = False, allow_disabled: bool = False, profile: str | None = None) -> dict:
    """Run collection for one profile or all enabled storage profiles.

    With ``profile=None`` (the scheduler/default path), every enabled profile is
    evaluated independently. This decouples background collection from the
    region currently open in the dashboard.
    """
    config = _load_config_canonical()
    state = _load_state_canonical()
    state["last_collector_start_utc"] = _iso()
    state["collector_host"] = socket.gethostname()
    state["collector_pid"] = os.getpid()
    _save_state_canonical(state)

    if profile is not None:
        if profile not in config["profiles"]:
            raise KeyError(f"Unknown storage profile: {profile}")
        names = [profile]
    else:
        names = list(config["profiles"])

    ran_any = False
    for name in names:
        pcfg = _project_profile(config, name)
        if not pcfg.get("enabled", False) and not allow_disabled:
            continue
        pstate = state["profiles"].setdefault(name, {"sources": {}})
        pstate["last_collector_start_utc"] = _iso()
        pstate["collector_host"] = socket.gethostname()
        pstate["collector_pid"] = os.getpid()
        region=region_from_dict(pcfg["region"])
        # Ensure old NetCDF/snapshot-Parquet data are converted before this
        # profile resumes live collection. The operation is idempotent; after
        # successful conversion the legacy files are archived outside the
        # active storage root.
        migrate_legacy_snapshots(config=pcfg,archive_legacy=True)
        _ensure_store_manifest(pcfg)
        requested=[source] if source else list(DEFAULT_SOURCES)
        profile_ran = False
        for key in requested:
            spec = pcfg.get("sources", {}).get(key, {})
            if not spec.get("enabled", False) and not (force and allow_disabled):
                continue
            projected_state = _project_state(state, name)
            if not source_due(pcfg, projected_state, key, force=force):
                continue
            with SourceLock(pcfg, key) as locked:
                if not locked:
                    pstate.setdefault("sources", {}).setdefault(key, {})["status"] = "already running"
                    continue
                src_state = pstate.setdefault("sources", {}).setdefault(key, {})
                src_state.update({"status": "running", "last_attempt_utc": _iso(), "error": None})
                _save_state_canonical(state)
                started = time.time()
                try:
                    result = _collect(key, region, _source_dir(pcfg, key))
                    src_state.update({
                        "status": "current",
                        "last_success_utc": _iso(),
                        "data_time": result.get("data_time"),
                        "files": result.get("files", []),
                        "duration_seconds": round(time.time()-started, 2),
                        "error": None,
                    })
                    state.setdefault("history", []).append({
                        "time": _iso(), "profile": name, "source": key,
                        "mode": "live", "result": "success",
                        "data_time": result.get("data_time"),
                    })
                except Exception as exc:
                    src_state.update({
                        "status": "error",
                        "duration_seconds": round(time.time()-started, 2),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    state.setdefault("history", []).append({
                        "time": _iso(), "profile": name, "source": key,
                        "mode": "live", "result": "error", "error": str(exc),
                    })
                _save_state_canonical(state)
                profile_ran = True
                ran_any = True
        pstate["last_collector_finish_utc"] = _iso()
        pstate["last_collector_result"] = "complete" if profile_ran else "nothing due"
        _save_state_canonical(state)

    state["last_collector_finish_utc"] = _iso()
    state["last_collector_result"] = "complete" if ran_any else "nothing due or all profiles paused"
    state = _save_state_canonical(state)
    active = config.get("active_profile")
    projected = _project_state(state, active)
    projected["last_collector_finish_utc"] = state.get("last_collector_finish_utc")
    projected["last_collector_result"] = state.get("last_collector_result")
    return projected


def storage_size_bytes(config: dict | None = None, profile_name: str | None = None) -> int:
    cfg = config or load_config(profile_name)
    root = region_store_root(cfg, profile_name)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def total_storage_size_bytes(config: dict | None = None) -> int:
    canonical = _canonicalize_config(config or _load_config_canonical())
    root = Path(canonical.get("root", DEFAULT_ROOT)).expanduser()
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())



def source_time_coverage(
    config: dict | None = None,
    *,
    profile_name: str | None = None,
) -> list[dict]:
    """Summarize trustworthy timestamps present in each source's persistent store.

    Rules:
      * Prefer timestamped Parquet snapshots for tabular observation sources.
      * Use Icechunk time coordinates for gridded/model sources.
      * Reject impossible future timestamps from scientific availability.
      * Represent discrete sources as points instead of continuous spans.
    """
    config = config or load_config(profile_name)
    if "profiles" in config and profile_name:
        config = _project_profile(config, profile_name)
    pname = config.get("profile_name") or profile_name
    state = load_state(pname)

    rows=[]
    discrete_sources={"radiosonde","sst"}
    parquet_preferred={"surface","air_quality","marine","radiosonde"}
    now=pd.Timestamp.now(tz="UTC")
    # Allow a small clock/product tolerance, but never accept months-ahead dates.
    future_cutoff=now+pd.Timedelta(hours=6)

    def _clean_times(values):
        cleaned=[]
        rejected_future=0
        for value in values:
            try:
                stamp=pd.Timestamp(value)
                stamp=stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            except Exception:
                continue
            if pd.isna(stamp):
                continue
            if stamp > future_cutoff:
                rejected_future += 1
                continue
            cleaned.append(stamp)
        return sorted(set(cleaned)), rejected_future

    for source, spec in config.get("sources", {}).items():
        directory=_source_dir(config,source)
        parquet_times=[]
        icechunk_times=[]
        rejected_future=0

        # Timestamped snapshot files are the authoritative timeline source for
        # tabular observations. These filenames are written from collection
        # timestamps and avoid stale/bad historical coordinate encodings.
        try:
            for stem in (
                "observations",
                "profiles",
                "trajectory",
                "points",
            ):
                for path in _parquet_files(directory,stem):
                    stamp=_snapshot_time_from_parquet_path(path)
                    if stamp is not None:
                        parquet_times.append(stamp)
        except Exception:
            pass

        # Icechunk is authoritative for gridded/model products and remains a
        # fallback when a tabular source has no readable snapshots.
        for path in sorted(directory.glob("*.icechunk")):
            try:
                import icechunk as ic
                repo=ic.Repository.open(ic.local_filesystem_storage(str(path)))
                session=repo.readonly_session("main")
                opened=xr.open_zarr(
                    session.store,
                    consolidated=False,
                    zarr_format=3,
                    chunks=None,
                )
                try:
                    if "time" in opened.coords:
                        vals=pd.to_datetime(
                            opened["time"].values,
                            utc=True,
                            errors="coerce",
                        )
                        icechunk_times.extend(x for x in vals if not pd.isna(x))
                finally:
                    opened.close()
            except Exception:
                pass

        parquet_times, rejected_pq=_clean_times(parquet_times)
        icechunk_times, rejected_ic=_clean_times(icechunk_times)
        rejected_future += rejected_pq + rejected_ic

        if source in parquet_preferred and parquet_times:
            times=parquet_times
            timeline_source="parquet_snapshots"
        elif icechunk_times:
            times=icechunk_times
            timeline_source="icechunk_time_coordinate"
        elif parquet_times:
            times=parquet_times
            timeline_source="parquet_snapshots"
        else:
            # Last-resort point from state; do not infer a historical span.
            latest=state.get("sources",{}).get(source,{}).get("data_time")
            fallback, rejected_state=_clean_times([latest] if latest else [])
            rejected_future += rejected_state
            times=fallback
            timeline_source="collector_state"

        interval=max(1,int(spec.get("interval_minutes",60)))
        gap_threshold=pd.Timedelta(minutes=max(interval*3,15))
        segments=[]
        isolated=[]

        if times and source not in discrete_sources:
            seg_start=times[0]
            previous=times[0]
            for current in times[1:]:
                if current-previous > gap_threshold:
                    if previous > seg_start:
                        segments.append((seg_start,previous))
                    else:
                        isolated.append(seg_start)
                    seg_start=current
                previous=current
            if previous > seg_start:
                segments.append((seg_start,previous))
            else:
                isolated.append(seg_start)

        ss=state.get("sources",{}).get(source,{})
        rows.append({
            "source":source,
            "label":SOURCE_LABELS.get(source,source),
            "enabled":bool(spec.get("enabled",False)),
            "discrete":source in discrete_sources,
            "count":len(times),
            "first_time":times[0].isoformat() if times else None,
            "last_time":times[-1].isoformat() if times else None,
            "times":[x.isoformat() for x in times],
            "segments":[(a.isoformat(),b.isoformat()) for a,b in segments],
            "isolated_times":[x.isoformat() for x in isolated],
            "timeline_source":timeline_source,
            "rejected_future_times":int(rejected_future),
            "last_success_utc":ss.get("last_success_utc"),
            "status":ss.get("status","not run"),
            "interval_minutes":interval,
            "product_cadence":SOURCE_NATIVE_CADENCE.get(source,""),
        })
    return rows

def status_rows(config: dict | None = None, state: dict | None = None, *, profile_name: str | None = None) -> list[dict]:
    config = config or load_config(profile_name)
    if "profiles" in config and profile_name:
        config = _project_profile(config, profile_name)
    pname = config.get("profile_name") or profile_name
    state = state or load_state(pname)
    if state.get("profile_name") != pname and pname:
        state = load_state(pname)
    rows=[]
    now=pd.Timestamp.now(tz="UTC")
    for key, spec in config.get("sources", {}).items():
        ss=state.get("sources",{}).get(key,{})
        last=ss.get("last_success_utc")
        lag=None
        if last:
            try: lag=round((now-pd.Timestamp(last)).total_seconds()/60.0,1)
            except Exception: pass
        interval = int(spec.get("interval_minutes", 60))
        next_check = None
        if last:
            try: next_check = (pd.Timestamp(last) + pd.Timedelta(minutes=interval)).isoformat()
            except Exception: pass
        rows.append({
            "profile": pname,
            "source": key,
            "label": SOURCE_LABELS.get(key,key),
            "enabled": bool(spec.get("enabled",False)),
            "interval_minutes": interval,
            "product_cadence": SOURCE_NATIVE_CADENCE.get(key, ""),
            "storage_format": SOURCE_STORAGE_FORMAT.get(key, "Icechunk"),
            "status": ss.get("status","not run"),
            "data_time": ss.get("data_time"),
            "last_attempt_utc": ss.get("last_attempt_utc"),
            "last_success_utc": last,
            "next_check_utc": next_check,
            "lag_minutes": lag,
            "duration_seconds": ss.get("duration_seconds"),
            "error": ss.get("error"),
        })
    return rows


def all_status_rows(config: dict | None = None, state: dict | None = None) -> list[dict]:
    canonical = _canonicalize_config(config or _load_config_canonical())
    canonical_state = _canonicalize_state(state or _load_state_canonical(), canonical)
    rows=[]
    for name in canonical["profiles"]:
        rows.extend(status_rows(_project_profile(canonical,name), _project_state(canonical_state,name), profile_name=name))
    return rows

from __future__ import annotations
import os

def configure_dask(scheduler=None, workers=None):
    """Configure ARIA's local Dask execution.

    Defaults are intentionally laptop/workstation friendly.  A distributed
    client can be supplied by applications later without changing workflows.
    """
    import dask
    scheduler = scheduler or os.environ.get("ARIA_DASK_SCHEDULER", "threads")
    config = {"scheduler": scheduler}
    if workers is not None:
        config["num_workers"] = int(workers)
    dask.config.set(config)
    return config

def recommended_chunks(dataset_kind: str):
    kind = str(dataset_kind).lower()
    if kind in {"mrms", "surface", "air_quality"}:
        return {"time": 1, "latitude": 400, "longitude": 400}
    if kind in {"hrrr", "hrrr_surface"}:
        return {"init_time": 1, "forecast_hour": 1, "latitude": 400, "longitude": 400}
    if kind in {"atmosphere", "radar3d", "radar"}:
        return {"time": 1, "altitude_km": 8, "y_km": 200, "x_km": 200}
    return "auto"

def maybe_chunk(ds, dataset_kind="generic", enabled=True):
    if not enabled:
        return ds
    chunks = recommended_chunks(dataset_kind)
    usable = {k: v for k, v in chunks.items() if k in ds.dims} if isinstance(chunks, dict) else chunks
    return ds.chunk(usable)

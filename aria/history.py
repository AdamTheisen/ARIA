from __future__ import annotations

"""Historical retrieval/backfill into the same ARIA stores used by live collection.

v0.21 establishes the storage contract and initial backfill engines for MRMS and
HRRR. Additional observation sources can adopt the same interface as their
historical upstream APIs are added.
"""

from dataclasses import dataclass
import time

import pandas as pd

from .region import region_from_dict
from .storage import (
    _load_config_canonical,
    _load_state_canonical,
    _save_state_canonical,
    _project_profile,
    _source_dir,
    _write_dataset,
    _iso,
)

HISTORICAL_SOURCE_CAPABILITIES = {
    "mrms": {"step_minutes": 5, "label": "MRMS reflectivity"},
    "hrrr": {"step_minutes": 60, "label": "HRRR F00 surface/model fields"},
}

HRRR_HISTORY_VARIABLES = [
    "air_temperature_2m",
    "dew_point_temperature_2m",
    "u_wind_10m",
    "v_wind_10m",
    "composite_reflectivity",
    "reflectivity_1km",
]


def _utc(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def historical_plan(profile: str, start, end, sources=("mrms", "hrrr"), *, step_minutes=None) -> dict:
    config = _load_config_canonical()
    if profile not in config["profiles"]:
        raise KeyError(f"Unknown storage profile: {profile}")
    start = _utc(start)
    end = _utc(end)
    if end < start:
        raise ValueError("Historical end time must be after start time.")
    rows=[]
    total=0
    for source in sources:
        if source not in HISTORICAL_SOURCE_CAPABILITIES:
            rows.append({"source":source,"supported":False,"steps":0,"step_minutes":None})
            continue
        step = int(step_minutes or HISTORICAL_SOURCE_CAPABILITIES[source]["step_minutes"])
        count = max(1, int((end-start).total_seconds()//(step*60))+1)
        total += count
        rows.append({"source":source,"supported":True,"steps":count,"step_minutes":step})
    return {
        "profile": profile,
        "region": config["profiles"][profile]["region"],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sources": rows,
        "total_steps": total,
    }


def _historical_collect(source: str, when: pd.Timestamp, profile_config: dict) -> dict:
    from . import workflows

    region = region_from_dict(profile_config["region"])
    directory = _source_dir(profile_config, source)
    files=[]
    if source == "mrms":
        ds, scan_time = workflows.build_mrms_at_time(when, region=region)
        ds = ds.copy()
        ds.attrs.update(
            aria_ingest_mode="historical_backfill",
            aria_requested_time=when.isoformat(),
        )
        files.append(str(_write_dataset(ds, directory, "reflectivity", scan_time)))
        return {"requested_time":when.isoformat(),"data_time":str(scan_time),"files":files}
    if source == "hrrr":
        cycle = when.floor("1h")
        ds, run = workflows.build_hrrr_surface(
            region=region,
            cycle=cycle,
            forecast_hour=0,
            variables=HRRR_HISTORY_VARIABLES,
        )
        ds = ds.copy()
        ds.attrs.update(
            aria_ingest_mode="historical_backfill",
            aria_requested_time=when.isoformat(),
        )
        files.append(str(_write_dataset(ds, directory, "f00", run.valid_time)))
        return {"requested_time":when.isoformat(),"data_time":str(run.valid_time),"files":files}
    raise ValueError(f"Historical backfill is not implemented for {source!r} in v0.21.")


def backfill_storage(
    profile: str,
    start,
    end,
    *,
    sources=("mrms", "hrrr"),
    step_minutes=None,
    max_steps: int | None = None,
    continue_on_error=True,
) -> dict:
    """Backfill supported historical sources into an existing named store.

    Icechunk append de-duplication makes the operation resumable: re-running a
    completed time range does not duplicate already-stored timestamps.
    """
    config = _load_config_canonical()
    if profile not in config["profiles"]:
        raise KeyError(f"Unknown storage profile: {profile}")
    pcfg = _project_profile(config, profile)
    state = _load_state_canonical()
    pstate = state["profiles"].setdefault(profile, {"sources": {}})
    start=_utc(start); end=_utc(end)
    if end < start:
        raise ValueError("Historical end time must be after start time.")

    results=[]
    attempted=0
    succeeded=0
    failed=0
    for source in sources:
        if source not in HISTORICAL_SOURCE_CAPABILITIES:
            results.append({"source":source,"status":"unsupported","error":"Historical adapter not yet implemented"})
            continue
        step=int(step_minutes or HISTORICAL_SOURCE_CAPABILITIES[source]["step_minutes"])
        times=pd.date_range(start,end,freq=f"{step}min",inclusive="both")
        for when in times:
            if max_steps is not None and attempted >= int(max_steps):
                return {
                    "profile":profile,"start":start.isoformat(),"end":end.isoformat(),
                    "attempted":attempted,"succeeded":succeeded,"failed":failed,
                    "stopped_at_max_steps":True,"results":results,
                }
            attempted += 1
            started=time.time()
            src_state=pstate.setdefault("sources",{}).setdefault(source,{})
            src_state.update({"status":"historical running","last_attempt_utc":_iso(),"error":None})
            _save_state_canonical(state)
            try:
                result=_historical_collect(source,pd.Timestamp(when),pcfg)
                succeeded += 1
                src_state.update({
                    "status":"current",
                    "last_success_utc":_iso(),
                    "data_time":result.get("data_time"),
                    "duration_seconds":round(time.time()-started,2),
                    "error":None,
                })
                state.setdefault("history",[]).append({
                    "time":_iso(),"profile":profile,"source":source,
                    "mode":"historical_backfill","result":"success",
                    "requested_time":pd.Timestamp(when).isoformat(),
                    "data_time":result.get("data_time"),
                })
                results.append({"source":source,"status":"success",**result})
            except Exception as exc:
                failed += 1
                src_state.update({
                    "status":"error",
                    "duration_seconds":round(time.time()-started,2),
                    "error":f"{type(exc).__name__}: {exc}",
                })
                state.setdefault("history",[]).append({
                    "time":_iso(),"profile":profile,"source":source,
                    "mode":"historical_backfill","result":"error",
                    "requested_time":pd.Timestamp(when).isoformat(),
                    "error":str(exc),
                })
                results.append({"source":source,"status":"error","requested_time":pd.Timestamp(when).isoformat(),"error":str(exc)})
                if not continue_on_error:
                    _save_state_canonical(state)
                    raise
            _save_state_canonical(state)
    return {
        "profile":profile,"start":start.isoformat(),"end":end.isoformat(),
        "attempted":attempted,"succeeded":succeeded,"failed":failed,
        "stopped_at_max_steps":False,"results":results,
    }

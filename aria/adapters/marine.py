from __future__ import annotations

"""Marine surface observations for ARIA.

The initial v0.20.2 adapter uses NOAA/NDBC's real-time latest-observation table.
It is capability-based: a platform may report air temperature, water
temperature, wind, pressure, dew point, and/or waves. Missing variables remain
NaN rather than excluding the platform.
"""

from io import StringIO
import numpy as np
import pandas as pd
import requests

NDBC_LATEST_OBS = "https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt"

# Canonical NDBC columns used by the real-time latest observation table.
_COLUMNS = [
    "station_id","latitude","longitude","year","month","day","hour","minute",
    "wind_direction_deg","wind_speed_ms","wind_gust_ms","wave_height_m",
    "dominant_period_s","average_period_s","mean_wave_direction_deg",
    "pressure_hpa","pressure_tendency_hpa","air_temperature_c",
    "water_temperature_c","dew_point_c","visibility_nm","tide_ft",
]

_NUMERIC = [c for c in _COLUMNS if c != "station_id"]


def _read_ndbc_latest(text: str) -> pd.DataFrame:
    # NDBC provides two header/comment rows beginning with '#'.  Reading the
    # body with known column names is more stable than depending on exact header
    # punctuation across server revisions.
    body = "\n".join(line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    if not body.strip():
        return pd.DataFrame(columns=_COLUMNS + ["time","source","platform_type"])
    frame = pd.read_csv(
        StringIO(body), sep=r"\s+", names=_COLUMNS, engine="python",
        na_values=["MM","999","9999","999.0","99.0"],
    )
    # If NDBC adds trailing fields, pandas may create malformed rows. Keep only
    # rows with plausible coordinates and coerce all known numeric fields.
    for col in _NUMERIC:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[
        frame["latitude"].between(-90,90, inclusive="both")
        & frame["longitude"].between(-180,180, inclusive="both")
    ].copy()
    time_parts = dict(
        year=frame["year"], month=frame["month"], day=frame["day"],
        hour=frame["hour"], minute=frame["minute"],
    )
    frame["time"] = pd.to_datetime(time_parts, utc=True, errors="coerce")
    frame["source"] = "NOAA/NDBC"
    frame["platform_type"] = "marine_station"
    return frame


def load_latest_marine_observations(region, *, timeout: int = 45) -> pd.DataFrame:
    """Return current NDBC marine stations inside *region*.

    No station is required to carry a particular sensor.  This lets ARIA use
    air temperature wherever present while retaining water temperature/wind/
    pressure from platforms with different payloads.
    """
    response = requests.get(NDBC_LATEST_OBS, timeout=timeout)
    response.raise_for_status()
    frame = _read_ndbc_latest(response.text)
    if frame.empty:
        return frame
    keep = (
        frame["latitude"].between(float(region.south), float(region.north), inclusive="both")
        & frame["longitude"].between(float(region.west), float(region.east), inclusive="both")
    )
    return frame.loc[keep].reset_index(drop=True)

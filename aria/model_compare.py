from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def compare_surface_to_asos(model_ds, observations, variable="air_temperature_2m"):
    """
    Match HRRR surface fields to nearest ASOS sites near model valid time.
    """
    mapping = {
        "air_temperature_2m": ("air_temperature_f", lambda x: (x - 32.0) * 5.0 / 9.0, "degC"),
        "dew_point_temperature_2m": ("dew_point_temperature_f", lambda x: (x - 32.0) * 5.0 / 9.0, "degC"),
        "u_wind_10m": ("u_wind_kt", lambda x: x * 0.514444, "m s-1"),
        "v_wind_10m": ("v_wind_kt", lambda x: x * 0.514444, "m s-1"),
    }
    if variable not in mapping:
        raise ValueError(
            f"ASOS comparison is not configured for {variable}. "
            f"Available: {list(mapping)}"
        )

    obs_var, converter, units = mapping[variable]
    valid = pd.Timestamp(model_ds.attrs["valid_time"])
    if valid.tzinfo is None:
        valid = valid.tz_localize("UTC")

    obs = observations[observations.variable == obs_var].copy()
    obs["time"] = pd.to_datetime(obs.time, utc=True)
    obs["dt_min"] = (obs.time - valid).abs().dt.total_seconds() / 60.0
    obs = obs[obs.dt_min <= 20].sort_values("dt_min").drop_duplicates("station_id")
    if obs.empty:
        return pd.DataFrame()

    lat = model_ds["latitude"]
    lon = model_ds["longitude"]
    da = model_ds[variable].squeeze(drop=True)

    tree = cKDTree(np.column_stack([lat.values.ravel(), lon.values.ravel()]))
    _, flat = tree.query(obs[["latitude", "longitude"]].to_numpy())
    model_values = da.values.ravel()[flat]

    out = obs[
        ["station_id","time","latitude","longitude","value","dt_min"]
    ].copy()
    out["observation"] = converter(out["value"].astype(float).to_numpy())
    out["model"] = model_values
    out["model_minus_observation"] = out["model"] - out["observation"]
    out["units"] = units
    return out.drop(columns=["value"])


def comparison_metrics(df):
    if df is None or df.empty:
        return {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan}
    d = df["model_minus_observation"].to_numpy(float)
    return {
        "n": int(np.isfinite(d).sum()),
        "bias": float(np.nanmean(d)),
        "mae": float(np.nanmean(np.abs(d))),
        "rmse": float(np.sqrt(np.nanmean(d ** 2))),
    }

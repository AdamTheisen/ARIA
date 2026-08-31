from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


SURFACE_COMPARISON = {
    "air_temperature_2m": ("air_temperature_f", "Air Temperature", "degC"),
    "dew_point_temperature_2m": ("dew_point_temperature_f", "Dew Point", "degC"),
}


def _model_latlon(ds):
    return ds["latitude"], ds["longitude"]


def regrid_model_to_regular(model_da, model_ds, target_lat, target_lon, method="linear"):
    """Interpolate a curvilinear HRRR field onto a regular lat/lon target."""
    lat, lon = _model_latlon(model_ds)
    values = np.asarray(model_da.squeeze().values, float)
    points = np.column_stack([lon.values.ravel(), lat.values.ravel()])
    vals = values.ravel()
    good = np.isfinite(points).all(axis=1) & np.isfinite(vals)

    tlon, tlat = np.meshgrid(np.asarray(target_lon), np.asarray(target_lat))
    out = griddata(points[good], vals[good], (tlon, tlat), method=method)

    # Fill convex-hull edge gaps using nearest neighbor.
    if np.isnan(out).any():
        nearest = griddata(points[good], vals[good], (tlon, tlat), method="nearest")
        out = np.where(np.isfinite(out), out, nearest)

    return out


def regrid_curvilinear_to_curvilinear(
    source_da, source_lat, source_lon, target_lat, target_lon, method="linear"
):
    values = np.asarray(source_da.squeeze().values, float)
    points = np.column_stack([source_lon.values.ravel(), source_lat.values.ravel()])
    vals = values.ravel()
    good = np.isfinite(points).all(axis=1) & np.isfinite(vals)
    target = (np.asarray(target_lon), np.asarray(target_lat))
    out = griddata(points[good], vals[good], target, method=method)
    if np.isnan(out).any():
        nearest = griddata(points[good], vals[good], target, method="nearest")
        out = np.where(np.isfinite(out), out, nearest)
    return out


def build_surface_difference(model_ds, surface_ds, model_variable):
    """
    Put HRRR and GPGL surface analysis on the surface-analysis grid and return
    model, observation, and model-minus-observation fields.
    """
    if model_variable not in SURFACE_COMPARISON:
        raise ValueError(f"No gridded surface comparison for {model_variable}")

    obs_variable, label, units = SURFACE_COMPARISON[model_variable]
    obs = surface_ds[obs_variable].squeeze(drop=True).astype(float)

    if obs_variable.endswith("_f"):
        obs_c = (obs - 32.0) * 5.0 / 9.0
    else:
        obs_c = obs

    model = regrid_model_to_regular(
        model_ds[model_variable],
        model_ds,
        surface_ds.latitude.values,
        surface_ds.longitude.values,
    )

    result = xr.Dataset(
        {
            "model": (("latitude","longitude"), model),
            "observation": (("latitude","longitude"), obs_c.values),
            "difference": (("latitude","longitude"), model - obs_c.values),
        },
        coords={
            "latitude": surface_ds.latitude.values,
            "longitude": surface_ds.longitude.values,
        },
        attrs={
            "comparison": f"HRRR {label} vs GPGL ASOS analysis",
            "variable": model_variable,
            "units": units,
            "valid_time": model_ds.attrs.get("valid_time",""),
            "difference_definition": "model_minus_observation",
        },
    )
    return result


def build_radar_difference(model_ds, radar_ds, model_variable="composite_reflectivity"):
    """
    Regrid HRRR simulated reflectivity onto an observed radar grid and calculate
    HRRR minus observed reflectivity. Supports both regular MRMS lat/lon grids
    and the curvilinear GPGL Level-II mosaic.
    """
    radar = radar_ds["reflectivity"].squeeze(drop=True).astype(float)

    if radar_ds["latitude"].ndim == 1 and radar_ds["longitude"].ndim == 1:
        model_on_radar = regrid_model_to_regular(
            model_ds[model_variable],
            model_ds,
            radar_ds.latitude.values,
            radar_ds.longitude.values,
        )
        obs = radar.values
        difference = model_on_radar - obs
        echo = (model_on_radar >= 5.0) | (obs >= 5.0)
        difference = np.where(echo, difference, np.nan)
        result = xr.Dataset(
            {
                "model": (("latitude","longitude"), model_on_radar),
                "observation": (("latitude","longitude"), obs),
                "difference": (("latitude","longitude"), difference),
            },
            coords={
                "latitude": radar_ds.latitude.values,
                "longitude": radar_ds.longitude.values,
            },
        )
    else:
        model_on_radar = regrid_curvilinear_to_curvilinear(
            model_ds[model_variable],
            model_ds.latitude,
            model_ds.longitude,
            radar_ds.latitude,
            radar_ds.longitude,
        )
        obs = radar.values
        difference = model_on_radar - obs
        echo = (model_on_radar >= 5.0) | (obs >= 5.0)
        difference = np.where(echo, difference, np.nan)
        result = xr.Dataset(
            {
                "model": (("y_km","x_km"), model_on_radar),
                "observation": (("y_km","x_km"), obs),
                "difference": (("y_km","x_km"), difference),
            },
            coords={
                "y_km": radar_ds.y_km,
                "x_km": radar_ds.x_km,
                "latitude": (("y_km","x_km"), radar_ds.latitude.values),
                "longitude": (("y_km","x_km"), radar_ds.longitude.values),
            },
        )

    source = radar_ds.attrs.get("source", "regional NEXRAD")
    result.attrs.update({
        "comparison": f"HRRR simulated reflectivity vs {source}",
        "variable": model_variable,
        "units": "dBZ",
        "valid_time": model_ds.attrs.get("valid_time",""),
        "radar_time": radar_ds.attrs.get("analysis_time",""),
        "radar_source": source,
        "difference_definition": "model_minus_observation",
    })
    return result


def radar_verification_metrics(comparison, thresholds=(10,20,30,40)):
    """Simple grid-cell categorical verification plus reflectivity errors."""
    m = comparison.model.values
    o = comparison.observation.values
    valid = np.isfinite(m) & np.isfinite(o)
    rows = []
    for threshold in thresholds:
        hit = valid & (m >= threshold) & (o >= threshold)
        miss = valid & (m < threshold) & (o >= threshold)
        false = valid & (m >= threshold) & (o < threshold)
        hits, misses, false_alarms = hit.sum(), miss.sum(), false.sum()
        pod = hits / (hits + misses) if hits + misses else np.nan
        far = false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan
        csi = hits / (hits + misses + false_alarms) if hits + misses + false_alarms else np.nan
        rows.append({
            "threshold_dbz": threshold,
            "hits": int(hits),
            "misses": int(misses),
            "false_alarms": int(false_alarms),
            "POD": pod,
            "FAR": far,
            "CSI": csi,
            "model_echo_fraction": float(np.mean(m[valid] >= threshold)) if valid.any() else np.nan,
            "observed_echo_fraction": float(np.mean(o[valid] >= threshold)) if valid.any() else np.nan,
        })
    return pd.DataFrame(rows)


def radar_fractions_skill_score(comparison, thresholds=(20,40), neighborhoods_km=(10,25,50,100)):
    """Fractions Skill Score for spatially tolerant reflectivity verification."""
    from scipy.ndimage import uniform_filter

    m=np.asarray(comparison.model.values,float)
    o=np.asarray(comparison.observation.values,float)
    valid=np.isfinite(m)&np.isfinite(o)

    # Estimate native grid spacing. MRMS is regular lat/lon; fallback to 4 km.
    spacing_km=4.0
    if "latitude" in comparison.coords and comparison.latitude.ndim==1 and comparison.latitude.size>1:
        spacing_km=max(abs(float(np.nanmedian(np.diff(comparison.latitude.values))))*111.0,0.1)
    elif "x_km" in comparison.coords and comparison.x_km.size>1:
        spacing_km=max(abs(float(np.nanmedian(np.diff(comparison.x_km.values)))),0.1)

    rows=[]
    for threshold in thresholds:
        mb=np.where(valid,m>=threshold,0.0).astype(float)
        ob=np.where(valid,o>=threshold,0.0).astype(float)
        for km in neighborhoods_km:
            size=max(1,int(round(float(km)/spacing_km)))
            mf=uniform_filter(mb,size=size,mode="nearest")
            of=uniform_filter(ob,size=size,mode="nearest")
            num=np.nansum((mf-of)**2)
            den=np.nansum(mf**2+of**2)
            fss=1.0-num/den if den>0 else np.nan
            rows.append({"threshold_dbz":int(threshold),"neighborhood_km":float(km),"FSS":float(fss) if np.isfinite(fss) else np.nan})
    return pd.DataFrame(rows)


def match_raob_hrrr_profile(model_ds, raob_df, station_id, variable):
    """
    Compare a radiosonde profile with nearest HRRR pressure-grid column.
    """
    source_map = {
        "air_temperature": ("air_temperature_c", 1.0),
        "dew_point_temperature": ("dew_point_temperature_c", 1.0),
        "u_wind": ("u_wind_kt", 0.514444),
        "v_wind": ("v_wind_kt", 0.514444),
        "wind_speed": ("wind_speed_kt", 0.514444),
    }
    if variable not in source_map:
        raise ValueError(f"Unsupported profile comparison variable: {variable}")

    source, factor = source_map[variable]
    obs = raob_df[raob_df.station_id == station_id].dropna(
        subset=["pressure_hpa", source]
    ).copy()
    if obs.empty:
        return pd.DataFrame()

    lat0 = float(obs.launch_latitude.dropna().iloc[0])
    lon0 = float(obs.launch_longitude.dropna().iloc[0])
    lat, lon = model_ds.latitude.values, model_ds.longitude.values
    tree = cKDTree(np.column_stack([lat.ravel(), lon.ravel()]))
    _, flat = tree.query([[lat0, lon0]])
    yi, xi = np.unravel_index(flat[0], lat.shape)
    ydim, xdim = model_ds.latitude.dims

    da = model_ds[variable].isel({ydim:int(yi), xdim:int(xi)})
    mp = np.asarray(da.pressure_hpa.values, float)
    mv = np.asarray(da.values, float)

    op = obs.pressure_hpa.to_numpy(float)
    ov = obs[source].to_numpy(float) * factor
    order = np.argsort(mp)
    interp = np.interp(op, mp[order], mv[order], left=np.nan, right=np.nan)

    return pd.DataFrame({
        "pressure_hpa": op,
        "observation": ov,
        "model": interp,
        "model_minus_observation": interp - ov,
    }).sort_values("pressure_hpa", ascending=False)

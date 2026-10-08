from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from .trajectory_match import lonlat_to_xy_km

UPPER_AIR_VARIABLES = {
    "air_temperature": {
        "source": "air_temperature_c", "units": "degC",
        "long_name": "Air temperature",
    },
    "dew_point_temperature": {
        "source": "dew_point_temperature_c", "units": "degC",
        "long_name": "Dew point temperature",
    },
    "wind_speed": {
        "source": "wind_speed_kt", "units": "kt",
        "long_name": "Wind speed",
    },
    "u_wind": {
        "source": "u_wind_kt", "units": "kt",
        "long_name": "Eastward wind component",
    },
    "v_wind": {
        "source": "v_wind_kt", "units": "kt",
        "long_name": "Northward wind component",
    },
}

def _smooth_nan(field, sigma=0.8):
    valid = np.isfinite(field)
    if not valid.any() or sigma <= 0:
        return field.copy()
    data = np.where(valid, field, 0.0)
    weight = valid.astype(float)
    sd = gaussian_filter(data, sigma=sigma, mode="nearest")
    sw = gaussian_filter(weight, sigma=sigma, mode="nearest")
    out = np.full_like(field, np.nan, dtype=float)
    good = sw > 1e-8
    out[good] = sd[good] / sw[good]
    return out

def profile_points_at_levels(raob_df, altitude_levels_km, source_variable):
    rows = []
    if raob_df is None or raob_df.empty or source_variable not in raob_df:
        return pd.DataFrame(rows)
    for station, sdf in raob_df.groupby("station_id", dropna=True):
        sdf = sdf.dropna(subset=["height_m", source_variable]).sort_values("height_m")
        if len(sdf) < 2:
            continue
        z = pd.to_numeric(sdf["height_m"], errors="coerce").to_numpy(float)
        val = pd.to_numeric(sdf[source_variable], errors="coerce").to_numpy(float)
        tlat = pd.to_numeric(sdf.get("trajectory_latitude"), errors="coerce") if "trajectory_latitude" in sdf else pd.Series(np.nan, index=sdf.index)
        tlon = pd.to_numeric(sdf.get("trajectory_longitude"), errors="coerce") if "trajectory_longitude" in sdf else pd.Series(np.nan, index=sdf.index)
        lat0 = pd.to_numeric(sdf["latitude"], errors="coerce")
        lon0 = pd.to_numeric(sdf["longitude"], errors="coerce")
        lat = tlat.where(tlat.notna(), lat0).to_numpy(float)
        lon = tlon.where(tlon.notna(), lon0).to_numpy(float)
        good = np.isfinite(z) & np.isfinite(val) & np.isfinite(lat) & np.isfinite(lon)
        z, val, lat, lon = z[good], val[good], lat[good], lon[good]
        if len(z) < 2:
            continue
        order = np.argsort(z)
        z, val, lat, lon = z[order], val[order], lat[order], lon[order]
        zu, idx = np.unique(z, return_index=True)
        val, lat, lon = val[idx], lat[idx], lon[idx]
        if len(zu) < 2:
            continue
        targets = np.asarray(altitude_levels_km, float) * 1000.0
        fv = interp1d(zu, val, bounds_error=False, fill_value=np.nan)(targets)
        fla = interp1d(zu, lat, bounds_error=False, fill_value=np.nan)(targets)
        flo = interp1d(zu, lon, bounds_error=False, fill_value=np.nan)(targets)
        for alt, vv, la, lo in zip(altitude_levels_km, fv, fla, flo):
            if np.isfinite(vv) and np.isfinite(la) and np.isfinite(lo):
                rows.append({"station_id": station, "altitude_km": float(alt),
                             "latitude": float(la), "longitude": float(lo),
                             "value": float(vv)})
    return pd.DataFrame(rows)

def _grid_points(points, lon, lat, altitudes, west, east, south, north,
                 local_radius_km=600, background_radius_km=1000,
                 smoothing_sigma=0.8):
    lon2d, lat2d = np.meshgrid(lon, lat)
    shape = (len(altitudes), len(lat), len(lon))
    field = np.full(shape, np.nan)
    nearest = np.full(shape, np.nan)
    count = np.zeros(shape, np.int16)
    confidence = np.zeros(shape, float)
    if points.empty:
        return field, nearest, count, confidence
    lon0=(west+east)/2; lat0=(south+north)/2
    tx,ty=lonlat_to_xy_km(lon2d.ravel(),lat2d.ravel(),lon0,lat0)
    targets=np.column_stack([tx,ty])
    for zi, alt in enumerate(altitudes):
        lev=points[np.isclose(points.altitude_km,alt)]
        if lev.empty: continue
        px,py=lonlat_to_xy_km(lev.longitude,lev.latitude,lon0,lat0)
        tree=cKDTree(np.column_stack([px,py])); vals=lev.value.to_numpy(float)
        k=min(6,len(lev)); d,idx=tree.query(targets,k=k)
        if k==1: d=d[:,None]; idx=idx[:,None]
        wl=np.where(d<=local_radius_km,1/np.maximum(d,1e-6)**2,0)
        den=wl.sum(1); n=(wl>0).sum(1); local=np.full(len(targets),np.nan)
        g=den>0; local[g]=(wl*vals[idx]).sum(1)[g]/den[g]
        kb=min(12,len(lev)); db,ib=tree.query(targets,k=kb)
        if kb==1: db=db[:,None]; ib=ib[:,None]
        wb=np.where(db<=background_radius_km,1/np.maximum(db,1e-6),0)
        denb=wb.sum(1); bg=np.full(len(targets),np.nan); gb=denb>0
        bg[gb]=(wb*vals[ib]).sum(1)[gb]/denb[gb]
        nd=np.min(d,axis=1)
        frac=np.clip(1-(nd-150)/350,0,1)
        out=np.full(len(targets),np.nan)
        both=np.isfinite(local)&np.isfinite(bg); out[both]=frac[both]*local[both]+(1-frac[both])*bg[both]
        out[np.isfinite(local)&~np.isfinite(bg)]=local[np.isfinite(local)&~np.isfinite(bg)]
        out[~np.isfinite(local)&np.isfinite(bg)]=bg[~np.isfinite(local)&np.isfinite(bg)]
        field[zi]=_smooth_nan(out.reshape(lat2d.shape),smoothing_sigma)
        nearest[zi]=nd.reshape(lat2d.shape); count[zi]=n.reshape(lat2d.shape)
        confidence[zi]=(0.65*np.clip(1-nd/background_radius_km,0,1)+0.35*np.clip(n/4,0,1)).reshape(lat2d.shape)
    return field, nearest, count, confidence

def build_upper_air_dataset(raob_df, west,east,south,north,
                            altitude_levels_km=None,horizontal_resolution_deg=0.5):
    if altitude_levels_km is None:
        altitude_levels_km=np.array([0.5,1,2,3,4,5,6,8,10,12,14,16],float)
    else: altitude_levels_km=np.asarray(altitude_levels_km,float)
    lon=np.arange(west,east+horizontal_resolution_deg/2,horizontal_resolution_deg)
    lat=np.arange(south,north+horizontal_resolution_deg/2,horizontal_resolution_deg)
    ds=xr.Dataset(coords={"altitude_km":altitude_levels_km,"latitude":lat,"longitude":lon})
    all_points={}
    common_diag=None
    for name,meta in UPPER_AIR_VARIABLES.items():
        pts=profile_points_at_levels(raob_df,altitude_levels_km,meta["source"])
        all_points[name]=pts
        f,nearest,count,conf=_grid_points(pts,lon,lat,altitude_levels_km,west,east,south,north)
        ds[name]=(("altitude_km","latitude","longitude"),f)
        ds[name].attrs.update({"units":meta["units"],"long_name":meta["long_name"]})
        if common_diag is None and name=="air_temperature":
            ds["nearest_profile_distance_km"]=(("altitude_km","latitude","longitude"),nearest)
            ds["n_contributing_profiles"]=(("altitude_km","latitude","longitude"),count)
            ds["analysis_confidence"]=(("altitude_km","latitude","longitude"),conf)
            common_diag=True
    # Guarantee a usable scalar wind-speed analysis whenever vector winds are
    # available. This also repairs older/source-specific profiles where the
    # reported speed column is missing but u/v were successfully derived.
    if "u_wind" in ds and "v_wind" in ds:
        derived=np.hypot(ds["u_wind"],ds["v_wind"])
        ds["wind_speed"]=derived
        ds["wind_speed"].attrs.update({
            "units":"kt","long_name":"Wind speed","derived_from":"u_wind,v_wind"
        })

    ds.attrs.update({"analysis_type":"trajectory-aware radiosonde multi-variable analysis",
                     "variables":",".join(UPPER_AIR_VARIABLES)})
    return ds, all_points

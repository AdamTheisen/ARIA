from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


EARTH_KM_PER_DEG_LAT = 110.57


def lonlat_to_xy_km(longitude, latitude, lon0, lat0):
    """
    Convert lon/lat to a local equirectangular Cartesian coordinate system.

    x: km east of (lon0, lat0)
    y: km north of (lon0, lat0)
    """
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)

    x = (
        111.32
        * np.cos(np.deg2rad(lat0))
        * (longitude - lon0)
    )
    y = EARTH_KM_PER_DEG_LAT * (latitude - lat0)
    return x, y


def xy_km_to_lonlat(x_km, y_km, lon0, lat0):
    x_km = np.asarray(x_km, dtype=float)
    y_km = np.asarray(y_km, dtype=float)

    latitude = lat0 + y_km / EARTH_KM_PER_DEG_LAT
    longitude = lon0 + x_km / (
        111.32 * np.cos(np.deg2rad(lat0))
    )
    return longitude, latitude


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def trajectory_summary(trajectory_df):
    """Summarize each SondeHub flight for diagnostics and matching."""
    rows = []

    if trajectory_df is None or trajectory_df.empty:
        return pd.DataFrame()

    for serial, sdf in trajectory_df.groupby("serial", dropna=True):
        sdf = sdf.sort_values("time").dropna(
            subset=["time", "latitude", "longitude", "altitude_m"]
        )
        if sdf.empty:
            continue

        # Prefer a low-altitude point as launch proxy when one exists.
        low = sdf[sdf["altitude_m"] <= 3000]
        launch = low.iloc[0] if not low.empty else sdf.iloc[0]

        start = sdf.iloc[0]
        end = sdf.iloc[-1]

        horizontal_drift = haversine_km(
            start["latitude"],
            start["longitude"],
            end["latitude"],
            end["longitude"],
        )

        rows.append(
            {
                "serial": str(serial),
                "start_time": pd.Timestamp(start["time"]),
                "launch_time_proxy": pd.Timestamp(launch["time"]),
                "start_latitude": float(launch["latitude"]),
                "start_longitude": float(launch["longitude"]),
                "max_altitude_km": float(
                    pd.to_numeric(sdf["altitude_m"], errors="coerce").max()
                    / 1000.0
                ),
                "horizontal_drift_km": float(horizontal_drift),
                "n_points": len(sdf),
            }
        )

    return pd.DataFrame(rows)


def match_raob_to_sondehub(
    raob_df,
    trajectory_df,
    max_launch_distance_km=150.0,
    max_time_difference_hours=3.0,
):
    """
    Match IEM/NWS RAOB profiles to SondeHub flights.

    Matching uses launch-site distance and launch-time difference. A match must
    satisfy both hard thresholds. Among candidates, the score favors close
    launch positions and close launch times.
    """
    if raob_df is None or raob_df.empty:
        return pd.DataFrame()

    out = raob_df.copy()
    out["matched_serial"] = pd.NA
    out["trajectory_match_distance_km"] = np.nan
    out["trajectory_match_time_minutes"] = np.nan
    out["trajectory_latitude"] = np.nan
    out["trajectory_longitude"] = np.nan

    summary = trajectory_summary(trajectory_df)
    if summary.empty:
        return out

    for station, sdf in out.groupby("station_id", dropna=True):
        idx = sdf.index

        launch_lat = sdf["launch_latitude"].dropna()
        launch_lon = sdf["launch_longitude"].dropna()
        if launch_lat.empty or launch_lon.empty:
            launch_lat = sdf["latitude"].dropna()
            launch_lon = sdf["longitude"].dropna()

        if launch_lat.empty or launch_lon.empty:
            continue

        station_lat = float(launch_lat.iloc[0])
        station_lon = float(launch_lon.iloc[0])
        cycle = pd.Timestamp(sdf["time"].iloc[0])

        if cycle.tzinfo is None:
            cycle = cycle.tz_localize("UTC")
        else:
            cycle = cycle.tz_convert("UTC")

        candidates = summary.copy()
        candidates["distance_km"] = haversine_km(
            station_lat,
            station_lon,
            candidates["start_latitude"],
            candidates["start_longitude"],
        )

        launch_times = pd.to_datetime(
            candidates["launch_time_proxy"], utc=True
        )
        candidates["time_minutes"] = (
            (launch_times - cycle).abs().dt.total_seconds() / 60.0
        )

        candidates = candidates[
            (candidates["distance_km"] <= max_launch_distance_km)
            & (
                candidates["time_minutes"]
                <= max_time_difference_hours * 60.0
            )
        ].copy()

        if candidates.empty:
            continue

        # 100 km ~= 60 minutes in the score, just to balance dimensions.
        candidates["score"] = (
            candidates["distance_km"] / 100.0
            + candidates["time_minutes"] / 60.0
        )
        best = candidates.sort_values("score").iloc[0]
        serial = str(best["serial"])

        track = trajectory_df[
            trajectory_df["serial"].astype(str) == serial
        ].dropna(
            subset=["altitude_m", "latitude", "longitude"]
        ).sort_values("altitude_m")

        if len(track) < 3:
            continue

        z = pd.to_numeric(track["altitude_m"], errors="coerce").to_numpy(float)
        track_lat = pd.to_numeric(
            track["latitude"], errors="coerce"
        ).to_numpy(float)
        track_lon = pd.to_numeric(
            track["longitude"], errors="coerce"
        ).to_numpy(float)

        good = np.isfinite(z) & np.isfinite(track_lat) & np.isfinite(track_lon)
        z = z[good]
        track_lat = track_lat[good]
        track_lon = track_lon[good]

        if len(z) < 3:
            continue

        # Ascending and descending data can repeat altitudes. Keep the earliest
        # point at each altitude; for typical sounding launches this follows
        # the ascent used for the thermodynamic profile.
        order = np.argsort(z)
        z = z[order]
        track_lat = track_lat[order]
        track_lon = track_lon[order]

        z_unique, unique_idx = np.unique(z, return_index=True)
        track_lat = track_lat[unique_idx]
        track_lon = track_lon[unique_idx]

        if len(z_unique) < 3:
            continue

        lat_interp = interp1d(
            z_unique,
            track_lat,
            bounds_error=False,
            fill_value=np.nan,
        )
        lon_interp = interp1d(
            z_unique,
            track_lon,
            bounds_error=False,
            fill_value=np.nan,
        )

        target_z = pd.to_numeric(
            sdf["height_m"], errors="coerce"
        ).to_numpy(float)

        matched_lat = lat_interp(target_z)
        matched_lon = lon_interp(target_z)

        out.loc[idx, "matched_serial"] = serial
        out.loc[idx, "trajectory_match_distance_km"] = float(
            best["distance_km"]
        )
        out.loc[idx, "trajectory_match_time_minutes"] = float(
            best["time_minutes"]
        )
        out.loc[idx, "trajectory_latitude"] = matched_lat
        out.loc[idx, "trajectory_longitude"] = matched_lon

    return out


def profile_plot_positions(raob_df):
    """
    Return the best available horizontal coordinates for each RAOB level.

    Matched SondeHub trajectory coordinates are preferred. Otherwise use the
    IEM per-level coordinate when it truly varies, then the launch location.
    """
    out = raob_df.copy()

    lat = pd.to_numeric(out.get("trajectory_latitude"), errors="coerce")
    lon = pd.to_numeric(out.get("trajectory_longitude"), errors="coerce")

    fallback_lat = pd.to_numeric(out["latitude"], errors="coerce")
    fallback_lon = pd.to_numeric(out["longitude"], errors="coerce")

    out["plot_latitude"] = lat.where(lat.notna(), fallback_lat)
    out["plot_longitude"] = lon.where(lon.notna(), fallback_lon)

    return out

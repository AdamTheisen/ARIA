from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .trajectory_match import lonlat_to_xy_km


def profile_points_at_levels(
    raob_df,
    altitude_levels_km,
    variable="air_temperature_c",
):
    """
    Interpolate each sounding vertically to common altitude levels.

    Matched SondeHub trajectory coordinates are used when available so each
    altitude level is located at the estimated balloon position rather than
    always at the launch site.
    """
    rows = []

    if raob_df is None or raob_df.empty:
        return pd.DataFrame(rows)

    for station, sdf in raob_df.groupby("station_id", dropna=True):
        sdf = sdf.dropna(subset=["height_m", variable]).sort_values("height_m")
        if len(sdf) < 2:
            continue

        z = pd.to_numeric(sdf["height_m"], errors="coerce").to_numpy(float)
        value = pd.to_numeric(sdf[variable], errors="coerce").to_numpy(float)

        # Preferred horizontal coordinates: matched trajectory positions.
        traj_lat = pd.to_numeric(
            sdf.get("trajectory_latitude"),
            errors="coerce",
        ) if "trajectory_latitude" in sdf else pd.Series(np.nan, index=sdf.index)

        traj_lon = pd.to_numeric(
            sdf.get("trajectory_longitude"),
            errors="coerce",
        ) if "trajectory_longitude" in sdf else pd.Series(np.nan, index=sdf.index)

        level_lat = pd.to_numeric(sdf["latitude"], errors="coerce")
        level_lon = pd.to_numeric(sdf["longitude"], errors="coerce")

        lat = traj_lat.where(traj_lat.notna(), level_lat).to_numpy(float)
        lon = traj_lon.where(traj_lon.notna(), level_lon).to_numpy(float)

        good = (
            np.isfinite(z)
            & np.isfinite(value)
            & np.isfinite(lat)
            & np.isfinite(lon)
        )

        z = z[good]
        value = value[good]
        lat = lat[good]
        lon = lon[good]

        if len(z) < 2:
            continue

        order = np.argsort(z)
        z = z[order]
        value = value[order]
        lat = lat[order]
        lon = lon[order]

        z_unique, idx = np.unique(z, return_index=True)
        value = value[idx]
        lat = lat[idx]
        lon = lon[idx]

        if len(z_unique) < 2:
            continue

        f_value = interp1d(
            z_unique,
            value,
            bounds_error=False,
            fill_value=np.nan,
        )
        f_lat = interp1d(
            z_unique,
            lat,
            bounds_error=False,
            fill_value=np.nan,
        )
        f_lon = interp1d(
            z_unique,
            lon,
            bounds_error=False,
            fill_value=np.nan,
        )

        levels_m = np.asarray(altitude_levels_km, dtype=float) * 1000.0

        for altitude_km, vv, la, lo in zip(
            altitude_levels_km,
            f_value(levels_m),
            f_lat(levels_m),
            f_lon(levels_m),
        ):
            if np.isfinite(vv) and np.isfinite(la) and np.isfinite(lo):
                rows.append(
                    {
                        "station_id": station,
                        "latitude": float(la),
                        "longitude": float(lo),
                        "altitude_km": float(altitude_km),
                        variable: float(vv),
                    }
                )

    return pd.DataFrame(rows)


def _nan_gaussian(field, sigma=0.8):
    arr = np.asarray(field, dtype=float)
    valid = np.isfinite(arr)

    if not valid.any() or sigma <= 0:
        return arr.copy()

    data = np.where(valid, arr, 0.0)
    weight = valid.astype(float)

    smooth_data = gaussian_filter(data, sigma=sigma, mode="nearest")
    smooth_weight = gaussian_filter(weight, sigma=sigma, mode="nearest")

    out = np.full_like(arr, np.nan)
    good = smooth_weight > 1.0e-8
    out[good] = smooth_data[good] / smooth_weight[good]
    return out


def build_temperature_volume(
    raob_df,
    west,
    east,
    south,
    north,
    altitude_levels_km=None,
    horizontal_resolution_deg=0.5,
    local_radius_km=600,
    background_radius_km=1000,
    local_power=2.0,
    background_power=1.0,
    smoothing_sigma=0.8,
):
    """
    Build a regular 3-D radiosonde-derived temperature field.

    Two horizontal analyses are created at each altitude:
      1) a local IDW field using nearby profiles;
      2) a broad, low-power background IDW field.

    The local field dominates close to observations. It relaxes toward the broad
    field where sounding coverage is sparse. This produces a substantially more
    continuous 3-D field while retaining explicit distance/count/confidence
    diagnostics.

    The field remains observationally derived; it is not a substitute for a
    model analysis such as HRRR.
    """
    if altitude_levels_km is None:
        altitude_levels_km = np.array(
            [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16],
            dtype=float,
        )
    else:
        altitude_levels_km = np.asarray(altitude_levels_km, dtype=float)

    points = profile_points_at_levels(
        raob_df,
        altitude_levels_km,
        variable="air_temperature_c",
    )

    lon = np.arange(
        west,
        east + horizontal_resolution_deg / 2,
        horizontal_resolution_deg,
    )
    lat = np.arange(
        south,
        north + horizontal_resolution_deg / 2,
        horizontal_resolution_deg,
    )

    lon2d, lat2d = np.meshgrid(lon, lat)

    shape = (len(altitude_levels_km), len(lat), len(lon))

    volume = np.full(shape, np.nan, dtype=float)
    nearest = np.full(shape, np.nan, dtype=float)
    contributors = np.zeros(shape, dtype=np.int16)
    confidence = np.zeros(shape, dtype=float)

    if points.empty:
        return (
            lon,
            lat,
            altitude_levels_km,
            volume,
            nearest,
            contributors,
            confidence,
            points,
        )

    lon0 = (west + east) / 2.0
    lat0 = (south + north) / 2.0

    tx, ty = lonlat_to_xy_km(
        lon2d.ravel(),
        lat2d.ravel(),
        lon0,
        lat0,
    )
    target_xy = np.column_stack([tx, ty])

    for zi, altitude in enumerate(altitude_levels_km):
        level = points[np.isclose(points["altitude_km"], altitude)].copy()

        if level.empty:
            continue

        px, py = lonlat_to_xy_km(
            level["longitude"],
            level["latitude"],
            lon0,
            lat0,
        )
        obs_xy = np.column_stack([px, py])

        values = level["air_temperature_c"].to_numpy(float)

        tree = cKDTree(obs_xy)

        k_local = min(6, len(level))
        dist_local, idx_local = tree.query(target_xy, k=k_local)

        if k_local == 1:
            dist_local = dist_local[:, None]
            idx_local = idx_local[:, None]

        local_weights = (
            1.0 / np.maximum(dist_local, 1.0e-6) ** local_power
        )
        local_weights = np.where(
            dist_local <= local_radius_km,
            local_weights,
            0.0,
        )

        local_den = np.sum(local_weights, axis=1)
        local_num = np.sum(
            local_weights * values[idx_local],
            axis=1,
        )
        local_count = np.sum(local_weights > 0, axis=1)

        local_field = np.full(len(target_xy), np.nan)
        local_good = local_den > 0
        local_field[local_good] = (
            local_num[local_good] / local_den[local_good]
        )

        # Broad background: every available profile can contribute, but the
        # low distance power prevents one sounding from creating a giant bullseye.
        k_background = min(12, len(level))
        dist_bg, idx_bg = tree.query(target_xy, k=k_background)

        if k_background == 1:
            dist_bg = dist_bg[:, None]
            idx_bg = idx_bg[:, None]

        bg_weights = (
            1.0 / np.maximum(dist_bg, 1.0e-6) ** background_power
        )
        bg_weights = np.where(
            dist_bg <= background_radius_km,
            bg_weights,
            0.0,
        )

        bg_den = np.sum(bg_weights, axis=1)
        bg_num = np.sum(
            bg_weights * values[idx_bg],
            axis=1,
        )

        bg_field = np.full(len(target_xy), np.nan)
        bg_good = bg_den > 0
        bg_field[bg_good] = bg_num[bg_good] / bg_den[bg_good]

        nearest_distance = np.min(dist_local, axis=1)

        # Blend local -> broad background from 150 to 500 km.
        local_fraction = np.clip(
            1.0 - (nearest_distance - 150.0) / 350.0,
            0.0,
            1.0,
        )

        field = np.full(len(target_xy), np.nan)

        both = np.isfinite(local_field) & np.isfinite(bg_field)
        field[both] = (
            local_fraction[both] * local_field[both]
            + (1.0 - local_fraction[both]) * bg_field[both]
        )

        only_local = np.isfinite(local_field) & ~np.isfinite(bg_field)
        field[only_local] = local_field[only_local]

        only_bg = ~np.isfinite(local_field) & np.isfinite(bg_field)
        field[only_bg] = bg_field[only_bg]

        field2d = field.reshape(lat2d.shape)

        if smoothing_sigma > 0:
            field2d = _nan_gaussian(
                field2d,
                sigma=smoothing_sigma,
            )

        volume[zi] = field2d
        nearest[zi] = nearest_distance.reshape(lat2d.shape)
        contributors[zi] = local_count.reshape(lat2d.shape)

        # Simple visualization confidence score, not formal uncertainty.
        distance_score = np.clip(
            1.0 - nearest_distance / background_radius_km,
            0.0,
            1.0,
        )
        count_score = np.clip(
            local_count / 4.0,
            0.0,
            1.0,
        )
        confidence[zi] = (
            0.65 * distance_score
            + 0.35 * count_score
        ).reshape(lat2d.shape)

    return (
        lon,
        lat,
        altitude_levels_km,
        volume,
        nearest,
        contributors,
        confidence,
        points,
    )


def build_temperature_dataset(
    raob_df,
    west,
    east,
    south,
    north,
    altitude_levels_km=None,
    horizontal_resolution_deg=0.5,
):
    """
    Return the radiosonde-derived 3-D temperature analysis as xarray.
    """
    (
        lon,
        lat,
        altitude,
        temperature,
        nearest,
        contributors,
        confidence,
        points,
    ) = build_temperature_volume(
        raob_df,
        west,
        east,
        south,
        north,
        altitude_levels_km=altitude_levels_km,
        horizontal_resolution_deg=horizontal_resolution_deg,
    )

    dims = ("altitude_km", "latitude", "longitude")

    ds = xr.Dataset(
        {
            "air_temperature": (dims, temperature),
            "nearest_profile_distance_km": (dims, nearest),
            "n_contributing_profiles": (dims, contributors),
            "analysis_confidence": (dims, confidence),
        },
        coords={
            "altitude_km": altitude,
            "latitude": lat,
            "longitude": lon,
        },
        attrs={
            "analysis_type": (
                "radiosonde-derived observational 3-D analysis"
            ),
            "horizontal_interpolation": (
                "local IDW blended with broad IDW background"
            ),
            "trajectory_aware": True,
            "confidence_note": (
                "analysis_confidence is a visualization diagnostic, "
                "not a formal uncertainty estimate"
            ),
        },
    )

    ds["air_temperature"].attrs.update(
        {
            "long_name": "Air temperature",
            "units": "degC",
        }
    )

    ds["nearest_profile_distance_km"].attrs.update(
        {
            "long_name": "Distance to nearest contributing profile",
            "units": "km",
        }
    )

    return ds, points

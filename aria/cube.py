from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
try:
    from shapely import contains_xy as _contains_xy
except ImportError:
    from shapely.vectorized import contains as _contains_xy
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class GridSpec:
    west: float = -105.0
    east: float = -75.0
    south: float = 35.0
    north: float = 50.0
    resolution: float = 0.1

    @property
    def lon(self):
        return np.arange(self.west, self.east + self.resolution / 2, self.resolution)

    @property
    def lat(self):
        return np.arange(self.south, self.north + self.resolution / 2, self.resolution)


class GriddedCubeBuilder:
    """
    Build a time x latitude x longitude xarray cube from point/station observations.

    Two gridding modes:
      nearest   - reproducible, fast, preserves observed values
      linear    - scipy linear interpolation where geometry permits

    max_distance_km limits how far a station can influence a nearest-neighbor cell.
    """

    def __init__(self, grid: GridSpec):
        self.grid = grid
        self._lon2d, self._lat2d = np.meshgrid(grid.lon, grid.lat)

    @staticmethod
    def _haversine_xy(lat, lon):
        # approximate local Cartesian mapping in km, sufficient for regional nearest search
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        lat0 = np.nanmean(lat)
        x = 111.32 * np.cos(np.deg2rad(lat0)) * lon
        y = 110.57 * lat
        return np.column_stack([x, y])

    def _grid_nearest(self, lat, lon, values, max_distance_km=None):
        valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(values)
        if valid.sum() == 0:
            return np.full(self._lat2d.shape, np.nan)

        pts = self._haversine_xy(lat[valid], lon[valid])
        tgt = self._haversine_xy(self._lat2d.ravel(), self._lon2d.ravel())
        tree = cKDTree(pts)
        dist, idx = tree.query(tgt, k=1)
        out = values[valid][idx].astype(float)
        if max_distance_km is not None:
            out[dist > max_distance_km] = np.nan
        return out.reshape(self._lat2d.shape)

    @staticmethod
    def _robust_station_filter(
        lat,
        lon,
        values,
        radius_km=150,
        mad_threshold=4.0,
        min_neighbors=4,
    ):
        """
        Flag isolated station values that are inconsistent with nearby stations.

        For each observation, compute the median and MAD of nearby observations.
        Values farther than ``mad_threshold`` robust standard deviations from
        the local median are excluded before gridding.

        This is intended to suppress isolated bad/biased observations that can
        create artificial cold or warm "holes" in sparse regions.
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        values = np.asarray(values, dtype=float)

        valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(values)
        if valid.sum() < min_neighbors:
            return valid

        xy = GriddedCubeBuilder._haversine_xy(lat[valid], lon[valid])
        vals = values[valid]
        tree = cKDTree(xy)

        keep_local = np.ones(len(vals), dtype=bool)

        for i, point in enumerate(xy):
            neighbors = tree.query_ball_point(point, r=radius_km)

            # Exclude the station itself from the neighborhood statistics.
            neighbors = [j for j in neighbors if j != i]

            if len(neighbors) < min_neighbors:
                continue

            local = vals[neighbors]
            median = np.nanmedian(local)
            mad = np.nanmedian(np.abs(local - median))

            # If the local neighborhood is nearly uniform, use an absolute
            # fallback tolerance so tiny MAD values do not over-flag data.
            robust_sigma = max(1.4826 * mad, 1.0)

            if abs(vals[i] - median) > mad_threshold * robust_sigma:
                keep_local[i] = False

        keep = np.zeros_like(valid, dtype=bool)
        keep[np.where(valid)[0]] = keep_local
        return keep

    def _grid_idw(
        self,
        lat,
        lon,
        values,
        k=8,
        power=2.0,
        max_distance_km=125,
        taper_start_km=None,
        taper_power=2.0,
        min_neighbors=2,
        robust_filter=False,
        robust_radius_km=150,
        robust_mad_threshold=4.0,
        return_diagnostics=False,
    ):
        """
        Inverse-distance weighted interpolation with optional tapering and
        robust station filtering.

        Parameters
        ----------
        taper_start_km : float or None
            Distance where weights begin tapering toward zero.  If omitted,
            tapering begins at 70% of ``max_distance_km``.
        min_neighbors : int
            Require at least this many contributing stations for a valid grid
            value.  This prevents one isolated station from dominating a large
            sparse region.
        robust_filter : bool
            Exclude locally inconsistent station values before interpolation.
        return_diagnostics : bool
            If True, also return distance-to-nearest-station and number of
            contributing stations for each grid cell.
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        values = np.asarray(values, dtype=float)

        valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(values)

        if robust_filter and valid.sum() > 0:
            valid &= self._robust_station_filter(
                lat,
                lon,
                values,
                radius_km=robust_radius_km,
                mad_threshold=robust_mad_threshold,
                min_neighbors=max(min_neighbors, 4),
            )

        if valid.sum() == 0:
            empty = np.full(self._lat2d.shape, np.nan)
            if return_diagnostics:
                return empty, empty.copy(), np.zeros(self._lat2d.shape, dtype=float)
            return empty

        obs_values = values[valid].astype(float)
        pts = self._haversine_xy(lat[valid], lon[valid])
        tgt = self._haversine_xy(self._lat2d.ravel(), self._lon2d.ravel())

        k_use = min(int(k), len(obs_values))
        tree = cKDTree(pts)
        dist, idx = tree.query(tgt, k=k_use)

        if k_use == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        neighbor_values = obs_values[idx]
        exact = dist <= 1.0e-10

        safe_dist = np.maximum(dist, 1.0e-6)
        weights = 1.0 / np.power(safe_dist, power)

        if max_distance_km is not None:
            if taper_start_km is None:
                taper_start_km = 0.70 * max_distance_km

            # Hard cutoff outside max_distance_km.
            inside = dist <= max_distance_km
            weights = np.where(inside, weights, 0.0)

            # Smoothly taper weights to zero near the search-radius edge.
            taper_zone = (dist > taper_start_km) & inside
            if np.any(taper_zone):
                frac = (
                    (max_distance_km - dist[taper_zone])
                    / max(max_distance_km - taper_start_km, 1.0e-6)
                )
                weights[taper_zone] *= np.power(
                    np.clip(frac, 0.0, 1.0),
                    taper_power,
                )

        contributing = np.sum(weights > 0, axis=1)
        numerator = np.sum(weights * neighbor_values, axis=1)
        denominator = np.sum(weights, axis=1)

        out = np.full(len(tgt), np.nan, dtype=float)
        good = (denominator > 0) & (contributing >= min_neighbors)
        out[good] = numerator[good] / denominator[good]

        # Preserve an exact observation only when the grid cell otherwise meets
        # the minimum-neighbor requirement. This avoids one isolated station
        # creating a large artificial feature around itself.
        exact_rows = np.any(exact, axis=1) & good
        if np.any(exact_rows):
            first_exact = np.argmax(exact[exact_rows], axis=1)
            out[exact_rows] = neighbor_values[exact_rows, first_exact]

        out = out.reshape(self._lat2d.shape)

        if return_diagnostics:
            nearest = np.min(dist, axis=1).reshape(self._lat2d.shape)
            ncontrib = contributing.reshape(self._lat2d.shape).astype(float)
            nearest[~np.isfinite(out)] = np.nan
            ncontrib[~np.isfinite(out)] = 0.0
            return out, nearest, ncontrib

        return out

    def _blend_with_background(
        self,
        local_field,
        background_field,
        nearest_distance_km,
        full_local_km=50,
        full_background_km=175,
    ):
        """
        Blend local station interpolation with a broad background field.

        Near dense observations, the local field dominates.  As distance from
        the nearest observation increases, the solution relaxes toward the
        broader background. This helps prevent isolated local extrema from
        dominating data-sparse areas.
        """
        local = np.asarray(local_field, dtype=float)
        background = np.asarray(background_field, dtype=float)
        distance = np.asarray(nearest_distance_km, dtype=float)

        result = local.copy()

        both = np.isfinite(local) & np.isfinite(background) & np.isfinite(distance)
        if not both.any():
            return result

        span = max(full_background_km - full_local_km, 1.0e-6)

        local_weight = 1.0 - (
            (distance - full_local_km) / span
        )
        local_weight = np.clip(local_weight, 0.0, 1.0)

        result[both] = (
            local_weight[both] * local[both]
            + (1.0 - local_weight[both]) * background[both]
        )

        # Where only the background exists, retain the background.
        only_background = ~np.isfinite(local) & np.isfinite(background)
        result[only_background] = background[only_background]

        return result

    @staticmethod
    def _smooth_nan_gaussian(field, sigma=1.0, truncate=3.0):
        """
        NaN-aware Gaussian smoothing.

        The field and a finite-data mask are filtered separately, then divided.
        This prevents missing cells from being treated as zeros and avoids
        smoothing values arbitrarily far into regions without data.
        """
        arr = np.asarray(field, dtype=float)
        finite = np.isfinite(arr)

        if not finite.any() or sigma is None or sigma <= 0:
            return arr.copy()

        data = np.where(finite, arr, 0.0)
        weight = finite.astype(float)

        smooth_data = gaussian_filter(
            data,
            sigma=sigma,
            mode="nearest",
            truncate=truncate,
        )
        smooth_weight = gaussian_filter(
            weight,
            sigma=sigma,
            mode="nearest",
            truncate=truncate,
        )

        result = np.full_like(arr, np.nan, dtype=float)
        good = smooth_weight > 1.0e-8
        result[good] = smooth_data[good] / smooth_weight[good]

        # Keep the original valid-domain mask. This smooths existing coverage
        # without extending the field into cells that had no estimate.
        result[~finite] = np.nan
        return result

    def _grid_linear(self, lat, lon, values):
        valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(values)
        if valid.sum() < 3:
            return np.full(self._lat2d.shape, np.nan)
        return griddata(
            (lon[valid], lat[valid]),
            values[valid],
            (self._lon2d, self._lat2d),
            method="linear",
        )

    def from_station_dataframe(
        self,
        df,
        variables,
        time_freq="1h",
        method="nearest",
        max_distance_km=75,
        idw_k=8,
        idw_power=2.0,
        taper_start_km=None,
        taper_power=2.0,
        min_neighbors=2,
        robust_filter=False,
        robust_radius_km=150,
        robust_mad_threshold=4.0,
        background_blend=False,
        background_k=24,
        background_power=1.0,
        background_radius_km=350,
        background_full_local_km=50,
        background_full_background_km=175,
        smooth_sigma=0.0,
        time_col="time",
        lat_col="latitude",
        lon_col="longitude",
        variable_col="variable",
        value_col="value",
        source_col="source",
    ):
        """
        Expected tidy dataframe columns:
        time, latitude, longitude, variable, value[, source]

        Observations are first aggregated to time_freq, then spatially gridded.
        """
        work = df.copy()

        # Parse as UTC, aggregate in UTC, then remove timezone metadata before
        # creating the xarray object. Zarr expects NumPy datetime64 values.
        work[time_col] = pd.to_datetime(
            work[time_col], utc=True, errors="coerce"
        )
        work = work.dropna(subset=[time_col])

        work["cube_time"] = (
            work[time_col]
            .dt.floor(time_freq)
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )

        arrays = {}
        times = (
            work["cube_time"]
            .drop_duplicates()
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
        )

        for variable in variables:
            vdf = work[work[variable_col] == variable]
            stack = []
            for t in times:
                sdf = vdf[vdf["cube_time"] == t]
                if sdf.empty:
                    field = np.full(self._lat2d.shape, np.nan)
                else:
                    # average duplicate observations from same coordinate/time bin
                    agg = (
                        sdf.groupby([lat_col, lon_col], as_index=False)[value_col]
                        .mean()
                    )
                    lat = agg[lat_col].to_numpy(float)
                    lon = agg[lon_col].to_numpy(float)
                    values = agg[value_col].to_numpy(float)
                    if method == "linear":
                        field = self._grid_linear(lat, lon, values)
                    elif method == "idw":
                        field, nearest_distance, ncontrib = self._grid_idw(
                            lat,
                            lon,
                            values,
                            k=idw_k,
                            power=idw_power,
                            max_distance_km=max_distance_km,
                            taper_start_km=taper_start_km,
                            taper_power=taper_power,
                            min_neighbors=min_neighbors,
                            robust_filter=robust_filter,
                            robust_radius_km=robust_radius_km,
                            robust_mad_threshold=robust_mad_threshold,
                            return_diagnostics=True,
                        )

                        if background_blend:
                            # Broad, low-power IDW acts as a weak regional
                            # background. It is not a substitute for a model
                            # background such as HRRR, but it is useful when
                            # only station data are available.
                            background = self._grid_idw(
                                lat,
                                lon,
                                values,
                                k=background_k,
                                power=background_power,
                                max_distance_km=background_radius_km,
                                taper_start_km=0.75 * background_radius_km,
                                taper_power=2.0,
                                min_neighbors=max(min_neighbors, 3),
                                robust_filter=robust_filter,
                                robust_radius_km=robust_radius_km,
                                robust_mad_threshold=robust_mad_threshold,
                                return_diagnostics=False,
                            )

                            field = self._blend_with_background(
                                field,
                                background,
                                nearest_distance,
                                full_local_km=background_full_local_km,
                                full_background_km=background_full_background_km,
                            )
                    else:
                        field = self._grid_nearest(
                            lat, lon, values, max_distance_km=max_distance_km
                        )

                if smooth_sigma is not None and smooth_sigma > 0:
                    field = self._smooth_nan_gaussian(
                        field,
                        sigma=smooth_sigma,
                    )

                stack.append(field)

            arrays[variable] = (
                ("time", "latitude", "longitude"),
                np.asarray(stack),
            )

        ds = xr.Dataset(
            data_vars=arrays,
            coords={
                "time": ("time", np.asarray(times, dtype="datetime64[ns]")),
                "latitude": self.grid.lat,
                "longitude": self.grid.lon,
            },
            attrs={
                "grid_resolution_degrees": self.grid.resolution,
                "gridding_method": method,
                "time_aggregation": time_freq,
                "max_station_influence_km": max_distance_km,
                "idw_neighbors": idw_k if method == "idw" else "n/a",
                "idw_power": idw_power if method == "idw" else "n/a",
                "idw_min_neighbors": min_neighbors if method == "idw" else "n/a",
                "idw_taper_start_km": taper_start_km if method == "idw" else "n/a",
                "robust_station_filter": robust_filter if method == "idw" else False,
                "background_blend": background_blend if method == "idw" else False,
                "gaussian_smoothing_sigma_grid_cells": smooth_sigma,
            },
        )
        return ds

    def add_native_grid(self, cube, ds, variables=None, method="linear"):
        """
        Regrid an already-gridded xarray Dataset onto the GPGL grid.

        This is appropriate for sources such as satellite/model products that
        already have latitude/longitude coordinates. xarray.interp is used for
        rectilinear grids.
        """
        variables = variables or list(ds.data_vars)
        subset = ds[variables]
        return subset.interp(
            latitude=self.grid.lat,
            longitude=self.grid.lon,
            method=method,
        )

    @staticmethod
    def _state_geometry(states):
        """
        Return a unioned Natural Earth geometry for selected U.S. states.

        ``states`` may contain two-letter postal abbreviations or full names.
        Cartopy will download the Natural Earth admin-1 boundary file on first use.
        """
        requested = {str(s).upper() for s in states}

        shp = shpreader.natural_earth(
            resolution="50m",
            category="cultural",
            name="admin_1_states_provinces_lakes",
        )

        geoms = []

        for record in shpreader.Reader(shp).records():
            attrs = record.attributes

            if attrs.get("adm0_a3") != "USA":
                continue

            name = str(attrs.get("name", "")).upper()
            postal = str(attrs.get("postal", "")).upper()

            if name in requested or postal in requested:
                geoms.append(record.geometry)

        if not geoms:
            raise ValueError(
                f"Could not find Natural Earth state geometries for: {states}"
            )

        return unary_union(geoms)

    def state_mask(self, states):
        """Return a latitude x longitude Boolean mask for selected states."""
        geometry = self._state_geometry(states)
        mask = _contains_xy(
            geometry,
            self._lon2d,
            self._lat2d,
        )
        return np.asarray(mask, dtype=bool)

    def points_in_states(self, latitude, longitude, states):
        """Boolean mask indicating whether arbitrary lon/lat points are in states."""
        geometry = self._state_geometry(states)
        return np.asarray(
            _contains_xy(
                geometry,
                np.asarray(longitude, dtype=float),
                np.asarray(latitude, dtype=float),
            ),
            dtype=bool,
        )

    def apply_state_mask(self, ds, states):
        """
        Mask every latitude/longitude gridded variable outside selected states.

        The rectangular coordinate grid is retained for efficient Zarr storage.
        """
        mask = self.state_mask(states)

        mask_da = xr.DataArray(
            mask,
            dims=("latitude", "longitude"),
            coords={
                "latitude": self.grid.lat,
                "longitude": self.grid.lon,
            },
            name="state_mask",
        )

        out = ds.copy()

        for variable in list(out.data_vars):
            if {"latitude", "longitude"}.issubset(out[variable].dims):
                out[variable] = out[variable].where(mask_da)

        out["state_mask"] = mask_da.astype("uint8")
        out["state_mask"].attrs.update(
            {
                "long_name": "Selected-state analysis mask",
                "flag_values": [0, 1],
                "flag_meanings": "outside_region inside_region",
            }
        )
        out.attrs["states"] = ",".join(states)
        return out


    def coverage_diagnostics(
        self,
        df,
        analysis_time,
        variable,
        time_tolerance="5min",
        variable_col="variable",
        time_col="time",
        lat_col="latitude",
        lon_col="longitude",
        max_distance_km=300,
    ):
        """
        Compute distance-to-nearest-observation and station counts for a
        variable near one analysis time.
        """
        work = df.copy()
        work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
        target = pd.Timestamp(analysis_time)
        if target.tzinfo is None:
            target = target.tz_localize("UTC")
        else:
            target = target.tz_convert("UTC")

        tol = pd.Timedelta(time_tolerance)
        subset = work[
            (work[variable_col] == variable)
            & ((work[time_col] - target).abs() <= tol)
        ].dropna(subset=[lat_col, lon_col])

        if subset.empty:
            shape = self._lat2d.shape
            return xr.Dataset(
                {
                    "nearest_observation_distance_km": (
                        ("latitude", "longitude"),
                        np.full(shape, np.nan),
                    ),
                    "n_observations_within_radius": (
                        ("latitude", "longitude"),
                        np.zeros(shape, dtype=np.int16),
                    ),
                    "analysis_confidence": (
                        ("latitude", "longitude"),
                        np.zeros(shape, dtype=float),
                    ),
                },
                coords={"latitude": self.grid.lat, "longitude": self.grid.lon},
            )

        locations = subset[[lat_col, lon_col]].drop_duplicates()
        lat = locations[lat_col].to_numpy(float)
        lon = locations[lon_col].to_numpy(float)

        pts = self._haversine_xy(lat, lon)
        tgt = self._haversine_xy(self._lat2d.ravel(), self._lon2d.ravel())
        tree = cKDTree(pts)

        dist, _ = tree.query(tgt, k=1)
        counts = np.array(
            [len(v) for v in tree.query_ball_point(tgt, r=max_distance_km)],
            dtype=np.int16,
        )

        distance = dist.reshape(self._lat2d.shape)
        count_grid = counts.reshape(self._lat2d.shape)

        # Simple 0–1 confidence score: close stations and multiple nearby
        # stations increase confidence. This is a diagnostic, not uncertainty.
        distance_score = np.clip(1.0 - distance / max_distance_km, 0.0, 1.0)
        count_score = np.clip(count_grid / 6.0, 0.0, 1.0)
        confidence = 0.6 * distance_score + 0.4 * count_score

        return xr.Dataset(
            {
                "nearest_observation_distance_km": (
                    ("latitude", "longitude"),
                    distance,
                ),
                "n_observations_within_radius": (
                    ("latitude", "longitude"),
                    count_grid,
                ),
                "analysis_confidence": (
                    ("latitude", "longitude"),
                    confidence,
                ),
            },
            coords={"latitude": self.grid.lat, "longitude": self.grid.lon},
            attrs={
                "analysis_time": str(target),
                "variable": variable,
                "time_tolerance": str(tol),
                "count_radius_km": max_distance_km,
                "confidence_note": (
                    "Diagnostic score based on nearest-observation distance "
                    "and observation count; not a formal uncertainty estimate."
                ),
            },
        )

    def write_zarr(self, ds, path, chunks=None):
        """Write the cube to Zarr after normalizing datetime coordinates."""
        ds = ds.copy()

        # Normalize all pandas DatetimeIndex coordinates to timezone-naive UTC.
        for coord_name in list(ds.coords):
            coord = ds[coord_name]
            try:
                idx = coord.to_index()
            except (TypeError, ValueError):
                idx = None

            if isinstance(idx, pd.DatetimeIndex):
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)

                ds = ds.assign_coords(
                    {
                        coord_name: (
                            coord.dims,
                            idx.to_numpy(dtype="datetime64[ns]"),
                        )
                    }
                )

        # Explicit safeguard for the primary time coordinate.
        if "time" in ds.coords:
            idx = ds["time"].to_index()
            if isinstance(idx, pd.DatetimeIndex):
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                ds = ds.assign_coords(
                    time=(
                        ds["time"].dims,
                        idx.to_numpy(dtype="datetime64[ns]"),
                    )
                )

        if chunks:
            ds = ds.chunk(chunks)

        # Fail with a clearer error if a timezone-aware dtype remains.
        for coord_name in ds.coords:
            if isinstance(ds[coord_name].dtype, pd.DatetimeTZDtype):
                raise TypeError(
                    f"Coordinate {coord_name!r} remains timezone-aware: "
                    f"{ds[coord_name].dtype}"
                )

        ds.to_zarr(path, mode="w", consolidated=True)
        return path

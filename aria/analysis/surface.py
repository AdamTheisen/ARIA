from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


@dataclass
class SurfaceAnalysisConfig:
    """
    Configuration for a temporally continuous station analysis.

    ``analysis_interval`` controls output frequency.
    ``lookback`` controls how old an observation may be.
    ``temporal_decay_minutes`` controls how quickly old observations lose
    influence before they age out of the lookback window.
    """

    analysis_interval: str = "5min"
    lookback: str = "15min"
    temporal_decay_minutes: float = 7.5

    method: str = "barnes"
    smoothing_km: float = 140.0
    # Limit Gaussian/Barnes neighbors instead of allocating every station at
    # every grid point. 64 is ample for the regional station density while
    # avoiding the very large Ngrid x Nstation arrays used previously.
    kernel_k: int = 64

    idw_k: int = 12
    idw_power: float = 2.0
    max_distance_km: float = 200.0
    min_neighbors: int = 2

    # Broad background from the same station network. This is intentionally
    # weaker than the local analysis and is only a fallback until a true
    # background analysis (e.g. HRRR) is introduced.
    background_blend: bool = True
    background_k: int = 24
    background_power: float = 1.0
    background_radius_km: float = 400.0
    background_full_local_km: float = 60.0
    background_full_background_km: float = 200.0


class SurfaceAnalysisBuilder:
    """
    Build time-continuous gridded surface analyses from station observations.

    The important difference from a simple "freshest observation" analysis is
    that spatial IDW weights are multiplied by a temporal decay weight:

        weight = distance_weight * exp(-age / temporal_decay)

    Therefore a station gradually loses influence as its observation gets old,
    instead of contributing at full strength until it suddenly ages out.
    """

    def __init__(self, grid, config: SurfaceAnalysisConfig | None = None):
        self.grid = grid
        self.config = config or SurfaceAnalysisConfig()

        self.lon = np.asarray(grid.lon, dtype=float)
        self.lat = np.asarray(grid.lat, dtype=float)
        self.lon2d, self.lat2d = np.meshgrid(self.lon, self.lat)

        self.lat0 = float(np.nanmean(self.lat))
        # Every interpolation pass uses the same analysis grid.
        self._targets_xy = self._xy(self.lat2d, self.lon2d)

    def _xy(self, latitude, longitude):
        latitude = np.asarray(latitude, dtype=float)
        longitude = np.asarray(longitude, dtype=float)

        x = 111.32 * np.cos(np.deg2rad(self.lat0)) * longitude
        y = 110.57 * latitude
        return np.column_stack([x.ravel(), y.ravel()])

    def _latest_per_station(self, observations, analysis_time):
        """
        Keep the latest valid observation for each station/variable while
        retaining its age for temporal weighting.
        """
        work = observations.copy()
        work["time"] = pd.to_datetime(work["time"], utc=True, errors="coerce")

        analysis_time = pd.Timestamp(analysis_time)
        if analysis_time.tzinfo is None:
            analysis_time = analysis_time.tz_localize("UTC")
        else:
            analysis_time = analysis_time.tz_convert("UTC")

        lookback = pd.Timedelta(self.config.lookback)

        work = work[
            (work["time"] <= analysis_time)
            & (work["time"] >= analysis_time - lookback)
        ].copy()

        if work.empty:
            return work

        work["observation_age_minutes"] = (
            analysis_time - work["time"]
        ).dt.total_seconds() / 60.0

        return (
            work.sort_values("observation_age_minutes")
            .drop_duplicates(
                subset=["station_id", "variable"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    def _spatiotemporal_idw(
        self,
        latitude,
        longitude,
        values,
        age_minutes,
        *,
        k,
        power,
        max_distance_km,
        min_neighbors,
    ):
        latitude = np.asarray(latitude, dtype=float)
        longitude = np.asarray(longitude, dtype=float)
        values = np.asarray(values, dtype=float)
        age_minutes = np.asarray(age_minutes, dtype=float)

        valid = (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & np.isfinite(values)
            & np.isfinite(age_minutes)
        )

        shape = self.lat2d.shape

        if valid.sum() == 0:
            return (
                np.full(shape, np.nan),
                np.full(shape, np.nan),
                np.zeros(shape, dtype=np.int16),
                np.full(shape, np.nan),
            )

        latitude = latitude[valid]
        longitude = longitude[valid]
        values = values[valid]
        age_minutes = age_minutes[valid]

        points = self._xy(latitude, longitude)
        targets = self._targets_xy

        tree = cKDTree(points)
        k_use = min(int(k), len(values))

        distance, index = tree.query(targets, k=k_use, workers=-1)

        if k_use == 1:
            distance = distance[:, None]
            index = index[:, None]

        neighbor_values = values[index]
        neighbor_ages = age_minutes[index]

        distance_weight = (
            1.0 / np.maximum(distance, 1.0e-6) ** power
        )
        distance_weight = np.where(
            distance <= max_distance_km,
            distance_weight,
            0.0,
        )

        temporal_weight = np.exp(
            -neighbor_ages
            / max(self.config.temporal_decay_minutes, 1.0e-6)
        )

        weights = distance_weight * temporal_weight

        n_contributing = np.sum(weights > 0, axis=1)
        denominator = np.sum(weights, axis=1)
        numerator = np.sum(weights * neighbor_values, axis=1)

        out = np.full(len(targets), np.nan, dtype=float)
        good = (
            (denominator > 0)
            & (n_contributing >= min_neighbors)
        )
        out[good] = numerator[good] / denominator[good]

        # Effective weighted observation age at each grid cell.
        age_num = np.sum(weights * neighbor_ages, axis=1)
        effective_age = np.full(len(targets), np.nan, dtype=float)
        effective_age[good] = age_num[good] / denominator[good]

        nearest_distance = np.min(distance, axis=1)
        nearest_distance[~good] = np.nan

        return (
            out.reshape(shape),
            nearest_distance.reshape(shape),
            n_contributing.reshape(shape).astype(np.int16),
            effective_age.reshape(shape),
        )


    def _spatiotemporal_kernel(
        self,
        latitude,
        longitude,
        values,
        age_minutes,
        *,
        sigma_km,
        max_distance_km,
        min_neighbors,
    ):
        """Gaussian space/time analysis using a bounded nearest-neighbor query.

        Earlier ARIA versions queried *every* observation for *every* grid point.
        For a regional grid this created large temporary distance/index arrays and
        repeated that cost for every variable, analysis time, and Barnes pass.

        The Gaussian kernel has finite practical support, so only the nearest
        ``kernel_k`` stations inside ``max_distance_km`` are needed here.
        """
        latitude = np.asarray(latitude, float)
        longitude = np.asarray(longitude, float)
        values = np.asarray(values, float)
        age_minutes = np.asarray(age_minutes, float)

        valid = (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & np.isfinite(values)
            & np.isfinite(age_minutes)
        )
        shape = self.lat2d.shape
        n_valid = int(valid.sum())
        if n_valid == 0:
            return (
                np.full(shape, np.nan),
                np.full(shape, np.nan),
                np.zeros(shape, dtype=np.int16),
                np.full(shape, np.nan),
            )

        latitude = latitude[valid]
        longitude = longitude[valid]
        values = values[valid]
        age_minutes = age_minutes[valid]

        points = self._xy(latitude, longitude)
        targets = self._targets_xy
        tree = cKDTree(points)

        k_use = min(max(int(self.config.kernel_k), int(min_neighbors)), n_valid)
        # Query the nearest bounded set of stations everywhere, then apply the
        # support radius using distance to the *nearest* observation.  Applying
        # distance_upper_bound to every neighbor created visible circular lobes
        # and discontinuities in sparse networks (especially Storm Explorer).
        # Gaussian/Barnes weights already decay distant stations smoothly.
        distance, index = tree.query(
            targets,
            k=k_use,
            workers=-1,
        )
        if k_use == 1:
            distance = distance[:, None]
            index = index[:, None]

        missing = (~np.isfinite(distance)) | (index >= n_valid)
        safe_index = np.where(missing, 0, index)
        neighbor_values = values[safe_index]
        neighbor_ages = age_minutes[safe_index]

        spatial = np.exp(
            -0.5 * (distance / max(float(sigma_km), 1.0)) ** 2
        )
        spatial[missing] = 0.0

        temporal = np.exp(
            -neighbor_ages
            / max(self.config.temporal_decay_minutes, 1.0e-6)
        )
        weights = spatial * temporal

        n = np.sum(weights > 0.0, axis=1)
        den = np.sum(weights, axis=1)
        num = np.sum(weights * neighbor_values, axis=1)

        nearest_raw = np.min(distance, axis=1)
        good = (
            (den > 0.0)
            & (n >= int(min_neighbors))
            & np.isfinite(nearest_raw)
            & (nearest_raw <= float(max_distance_km))
        )
        out = np.full(len(targets), np.nan)
        out[good] = num[good] / den[good]

        age = np.full(len(targets), np.nan)
        age_num = np.sum(weights * neighbor_ages, axis=1)
        age[good] = age_num[good] / den[good]

        nearest = nearest_raw.copy()
        nearest[~good] = np.nan

        return (
            out.reshape(shape),
            nearest.reshape(shape),
            n.reshape(shape).astype(np.int16),
            age.reshape(shape),
        )

    def _barnes(self, latitude, longitude, values, age_minutes):
        first,near,n,age=self._spatiotemporal_kernel(latitude,longitude,values,age_minutes,
            sigma_km=self.config.smoothing_km/np.sqrt(2.0),max_distance_km=self.config.max_distance_km,
            min_neighbors=self.config.min_neighbors)
        # Barnes residual correction sampled at nearest analysis-grid point.
        # Vectorized nearest-grid lookup avoids a station-by-station Python loop.
        lon_obs=np.asarray(longitude,float); lat_obs=np.asarray(latitude,float)
        ix=np.abs(self.lon[None,:]-lon_obs[:,None]).argmin(axis=1)
        iy=np.abs(self.lat[None,:]-lat_obs[:,None]).argmin(axis=1)
        sampled=first[iy,ix]
        residual=np.where(np.isfinite(sampled),np.asarray(values,float)-sampled,np.nan)
        corr,_,_,_=self._spatiotemporal_kernel(latitude,longitude,residual,age_minutes,
            sigma_km=max(20.0,self.config.smoothing_km*0.38),max_distance_km=self.config.max_distance_km,
            min_neighbors=self.config.min_neighbors)
        return first+np.nan_to_num(corr,nan=0.0),near,n,age

    def _background(self, observations):
        return self._spatiotemporal_idw(
            observations["latitude"],
            observations["longitude"],
            observations["value"],
            observations["observation_age_minutes"],
            k=self.config.background_k,
            power=self.config.background_power,
            max_distance_km=self.config.background_radius_km,
            min_neighbors=max(self.config.min_neighbors, 2),
        )[0]

    def _blend_background(
        self,
        local,
        background,
        nearest_distance,
    ):
        both = (
            np.isfinite(local)
            & np.isfinite(background)
            & np.isfinite(nearest_distance)
        )

        out = local.copy()

        span = max(
            self.config.background_full_background_km
            - self.config.background_full_local_km,
            1.0e-6,
        )

        local_weight = 1.0 - (
            (
                nearest_distance
                - self.config.background_full_local_km
            )
            / span
        )
        local_weight = np.clip(local_weight, 0.0, 1.0)

        out[both] = (
            local_weight[both] * local[both]
            + (1.0 - local_weight[both]) * background[both]
        )

        only_background = (
            ~np.isfinite(local)
            & np.isfinite(background)
        )
        out[only_background] = background[only_background]

        return out

    def build(
        self,
        observations,
        variables,
        analysis_times,
    ):
        """
        Return a Dataset with dimensions time x latitude x longitude.

        For each science variable, diagnostic layers are also created:
          <var>_nearest_distance_km
          <var>_n_contributing
          <var>_effective_age_minutes
        """
        work = observations.copy()

        if work.empty:
            raise ValueError("No observations were supplied.")

        work["time"] = pd.to_datetime(
            work["time"], utc=True, errors="coerce"
        )

        analysis_times = pd.DatetimeIndex(
            pd.to_datetime(analysis_times, utc=True)
        )

        output = {}

        for variable in variables:
            fields = []
            distances = []
            counts = []
            ages = []

            vdf = work[work["variable"] == variable]

            for analysis_time in analysis_times:
                obs = self._latest_per_station(
                    vdf,
                    analysis_time,
                )

                if obs.empty:
                    shape = self.lat2d.shape
                    fields.append(np.full(shape, np.nan))
                    distances.append(np.full(shape, np.nan))
                    counts.append(np.zeros(shape, dtype=np.int16))
                    ages.append(np.full(shape, np.nan))
                    continue

                method=str(self.config.method).lower()
                if method == "barnes":
                    field,nearest,ncontrib,effective_age=self._barnes(
                        obs["latitude"].to_numpy(float),obs["longitude"].to_numpy(float),
                        obs["value"].to_numpy(float),obs["observation_age_minutes"].to_numpy(float))
                elif method == "gaussian":
                    field,nearest,ncontrib,effective_age=self._spatiotemporal_kernel(
                        obs["latitude"],obs["longitude"],obs["value"],obs["observation_age_minutes"],
                        sigma_km=self.config.smoothing_km,max_distance_km=self.config.max_distance_km,
                        min_neighbors=self.config.min_neighbors)
                else:
                    field,nearest,ncontrib,effective_age=self._spatiotemporal_idw(
                        obs["latitude"],obs["longitude"],obs["value"],obs["observation_age_minutes"],
                        k=self.config.idw_k,power=self.config.idw_power,max_distance_km=self.config.max_distance_km,
                        min_neighbors=self.config.min_neighbors)
                    if self.config.background_blend:
                        background=self._background(obs)
                        field=self._blend_background(field,background,nearest)

                fields.append(field)
                distances.append(nearest)
                counts.append(ncontrib)
                ages.append(effective_age)

            dims = ("time", "latitude", "longitude")

            output[variable] = (
                dims,
                np.asarray(fields),
            )
            output[
                f"{variable}_nearest_distance_km"
            ] = (
                dims,
                np.asarray(distances),
            )
            output[
                f"{variable}_n_contributing"
            ] = (
                dims,
                np.asarray(counts),
            )
            output[
                f"{variable}_effective_age_minutes"
            ] = (
                dims,
                np.asarray(ages),
            )

        ds = xr.Dataset(
            output,
            coords={
                "time": analysis_times.tz_localize(None).to_numpy(
                    dtype="datetime64[ns]"
                ),
                "latitude": self.lat,
                "longitude": self.lon,
            },
            attrs={
                "analysis_interval": self.config.analysis_interval,
                "observation_lookback": self.config.lookback,
                "temporal_decay_minutes": (
                    self.config.temporal_decay_minutes
                ),
                "gridding_method": str(self.config.method),
                "smoothing_km": float(self.config.smoothing_km),
                "time_standard": "UTC",
            },
        )

        return ds

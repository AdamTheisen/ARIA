from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import GPGL_REGION, GridSpec, GriddedCubeBuilder
from .adapters import (
    BulkIEMASOSAdapter,
    AirNowConcentrationAdapter,
    IEMRAOBAdapter,
    SondeHubTrajectoryAdapter,
)
from .adapters.asos_bulk import latest_complete_hour
from .analysis import SurfaceAnalysisBuilder, SurfaceAnalysisConfig
from .upperair import build_upper_air_dataset
from .trajectory_match import (
    match_raob_to_sondehub,
    profile_plot_positions,
)


def fetch_asos_tidy(
    region=GPGL_REGION,
    *,
    start,
    end,
):
    adapter = BulkIEMASOSAdapter(region)

    stations = adapter.fetch(
        start - pd.Timedelta(minutes=15),
        end,
        variables=[
            "tmpf",
            "dwpf",
            "drct",
            "sknt",
        ],
    )

    rows = []

    for station_id, ds in stations.items():
        frame = ds.to_dataframe().reset_index()

        if "lat" not in frame or "lon" not in frame:
            continue

        lat = frame["lat"].dropna()
        lon = frame["lon"].dropna()

        if lat.empty or lon.empty:
            continue

        latitude = float(lat.iloc[0])
        longitude = float(lon.iloc[0])

        for source_var, variable, units in (
            ("tmpf", "air_temperature_f", "degF"),
            ("dwpf", "dew_point_temperature_f", "degF"),
        ):
            if source_var not in frame:
                continue

            for _, row in frame[
                ["time", source_var]
            ].dropna().iterrows():
                rows.append(
                    {
                        "source": "ASOS",
                        "station_id": station_id,
                        "time": row["time"],
                        "latitude": latitude,
                        "longitude": longitude,
                        "variable": variable,
                        "value": float(row[source_var]),
                        "units": units,
                    }
                )

        if {"sknt", "drct"}.issubset(frame.columns):
            wind = frame[
                ["time", "sknt", "drct"]
            ].dropna().copy()

            speed = pd.to_numeric(
                wind["sknt"], errors="coerce"
            )
            direction = pd.to_numeric(
                wind["drct"], errors="coerce"
            )

            good = (
                speed.notna()
                & direction.between(0, 360)
            )

            wind = wind.loc[good]
            speed = speed.loc[good].to_numpy(float)
            direction = np.deg2rad(
                direction.loc[good].to_numpy(float)
            )

            u = -speed * np.sin(direction)
            v = -speed * np.cos(direction)

            for row, uu, vv in zip(
                wind.itertuples(),
                u,
                v,
            ):
                rows.extend(
                    [
                        {
                            "source": "ASOS",
                            "station_id": station_id,
                            "time": row.time,
                            "latitude": latitude,
                            "longitude": longitude,
                            "variable": "u_wind_kt",
                            "value": float(uu),
                            "units": "kt",
                        },
                        {
                            "source": "ASOS",
                            "station_id": station_id,
                            "time": row.time,
                            "latitude": latitude,
                            "longitude": longitude,
                            "variable": "v_wind_kt",
                            "value": float(vv),
                            "units": "kt",
                        },
                    ]
                )

    return pd.DataFrame(rows)


def build_latest_surface(
    *,
    region=GPGL_REGION,
    analysis_interval="5min",
    lookback="15min",
    temporal_decay_minutes=7.5,
    lag_minutes=10,
    method="barnes",
    smoothing_km=140.0,
    max_distance_km=250.0,
    include_history=False,
):
    start, end = latest_complete_hour(
        lag_minutes=lag_minutes
    )

    latest_time = end - pd.Timedelta(analysis_interval)
    if include_history:
        analysis_times = pd.date_range(
            start=start,
            end=latest_time,
            freq=analysis_interval,
        )
    else:
        # The dashboard normally displays only the latest field. Building the
        # entire preceding hour multiplies Barnes work by ~12, so history is
        # opt-in and computed only when the user asks for it.
        analysis_times = pd.DatetimeIndex([latest_time])

    observations = fetch_asos_tidy(
        region,
        start=start,
        end=end,
    )

    grid = GridSpec(
        west=region.west,
        east=region.east,
        south=region.south,
        north=region.north,
        resolution=0.1,
    )

    config = SurfaceAnalysisConfig(
        analysis_interval=analysis_interval,
        lookback=lookback,
        temporal_decay_minutes=temporal_decay_minutes,
        method=method,
        smoothing_km=float(smoothing_km),
        max_distance_km=float(max_distance_km),
        background_blend=(str(method).lower()=="idw"),
    )

    analysis = SurfaceAnalysisBuilder(
        grid,
        config,
    ).build(
        observations,
        variables=[
            "air_temperature_f",
            "dew_point_temperature_f",
            "u_wind_kt",
            "v_wind_kt",
        ],
        analysis_times=analysis_times,
    )

    return analysis, observations, start, end


def build_latest_atmosphere(
    *,
    region=GPGL_REGION,
    fallback_cycles=5,
):
    _, end = latest_complete_hour(lag_minutes=10)

    raob = IEMRAOBAdapter(
        region,
        launch_buffer_deg=2.0,
    )

    raob_df = raob.fetch(
        end=end,
        fallback_cycles=fallback_cycles,
    )

    cycle = (
        raob_df.attrs.get("cycle")
        if not raob_df.empty
        else None
    )

    trajectory = pd.DataFrame()

    if cycle is not None:
        try:
            trajectory = SondeHubTrajectoryAdapter(
                region,
                buffer_deg=2.0,
            ).fetch(
                start=cycle - pd.Timedelta(hours=3),
                end=cycle + pd.Timedelta(hours=6),
                duration="1d",
            )
        except Exception:
            trajectory = pd.DataFrame()

    matched = match_raob_to_sondehub(
        raob_df,
        trajectory,
        max_launch_distance_km=150,
        max_time_difference_hours=3,
    )

    matched = profile_plot_positions(matched)

    atmosphere, points = build_upper_air_dataset(
        matched,
        region.west,
        region.east,
        region.south,
        region.north,
        altitude_levels_km=[
            0.5,
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            12,
            14,
            16,
        ],
        horizontal_resolution_deg=0.5,
    )

    if cycle is not None:
        atmosphere.attrs["raob_cycle"] = str(cycle)

    return atmosphere, matched, trajectory, points



def build_latest_air_quality(
    *,
    region=GPGL_REGION,
    token=None,
    max_distance_km=150.0,
):
    """
    Build a latest AirNow PM2.5 / ozone gridded analysis.

    AirNow is hourly, so this returns the latest available reporting hour rather
    than forcing the field onto the 5-minute meteorological timeline.
    """
    import os

    token = token or os.getenv("AIRNOW_API")
    if not token:
        raise RuntimeError(
            "AIRNOW_API is not set. Export your AirNow API key before "
            "loading the air-quality view."
        )

    start, end = latest_complete_hour(lag_minutes=10)

    airnow = AirNowConcentrationAdapter(
        region,
        token=token,
    )

    observations = airnow.fetch(
        start - pd.Timedelta(hours=2),
        end,
        parameters="OZONE,PM25",
    )

    if observations.empty:
        raise RuntimeError(
            "AirNow returned no PM2.5 or ozone observations "
            "for the current analysis window."
        )

    observations["time"] = pd.to_datetime(
        observations["time"],
        utc=True,
        errors="coerce",
    )

    latest_time = observations["time"].max()

    latest = observations[
        observations["time"] == latest_time
    ].copy()

    grid = GridSpec(
        west=region.west,
        east=region.east,
        south=region.south,
        north=region.north,
        resolution=0.1,
    )

    builder = GriddedCubeBuilder(grid)

    available = [
        variable
        for variable in ("pm25", "ozone")
        if variable in set(latest["variable"])
    ]

    cube = builder.from_station_dataframe(
        latest,
        variables=available,
        time_freq="1h",
        method="idw",
        max_distance_km=float(max_distance_km),
        idw_k=8,
        idw_power=1.5,
        min_neighbors=1,
        taper_start_km=0.75*float(max_distance_km),
        background_blend=False,
        smooth_sigma=1.0,
    )

    cube.attrs.update(
        {
            "source": "EPA AirNow",
            "analysis_time": str(latest_time),
        }
    )

    return cube, latest, observations, latest_time



def load_latest_nexrad(
    radar_id="KMPX",
    *,
    lookback_hours=6,
):
    """
    Download and read the latest available NEXRAD Level-II volume for one
    configured GPGL radar.
    """
    from .adapters.nexrad import NEXRADLevel2Adapter

    adapter = NEXRADLevel2Adapter()
    return adapter.fetch_latest_radar(
        radar_id,
        lookback_hours=lookback_hours,
    )



def build_latest_regional_radar(
    *,
    region=GPGL_REGION,
    radar_ids=None,
):
    """
    Fetch the latest regional Level-II scans and grid them into one 3-D cube.
    """
    from .radar_grid import (
        RegionalRadarConfig,
        fetch_latest_regional_radars,
        grid_regional_reflectivity,
    )

    config = RegionalRadarConfig()

    radars, scans, diagnostics = (
        fetch_latest_regional_radars(
            radar_ids=radar_ids,
            config=config,
            region=region,
        )
    )

    cube = grid_regional_reflectivity(
        radars,
        region=region,
        config=config,
    )

    newest = diagnostics["newest_scan_time"]
    cube.attrs["analysis_time"] = str(newest)
    cube.attrs["radar_count"] = len(scans)

    return cube, scans, diagnostics



def build_latest_regional_radar_fast(
    *,
    region=GPGL_REGION,
    altitude_km=2.0,
    radar_ids=None,
    horizontal_resolution_km=6.0,
):
    from .radar_grid import (
        RegionalRadarConfig,
        fetch_latest_regional_radars,
        grid_regional_reflectivity_altitude,
    )

    config = RegionalRadarConfig()
    radars, scans, diagnostics = fetch_latest_regional_radars(
        radar_ids=radar_ids,
        config=config,
        region=region,
    )

    cube = grid_regional_reflectivity_altitude(
        radars,
        region=region,
        altitude_km=altitude_km,
        horizontal_resolution_km=horizontal_resolution_km,
    )

    cube.attrs["analysis_time"] = str(diagnostics["newest_scan_time"])
    cube.attrs["radar_count"] = len(scans)
    cube.attrs["mode"] = "fast"
    return cube, scans, diagnostics



# ---------------------------------------------------------------------------
# Model workflows (v0.9)
# ---------------------------------------------------------------------------

def build_hrrr_surface(
    *,
    region=GPGL_REGION,
    cycle=None,
    forecast_hour=0,
    variables=None,
):
    """
    Retrieve and normalize selected HRRR surface fields for the GPGL domain.
    """
    from .models import HRRRAdapter

    adapter = HRRRAdapter(region)
    return adapter.open_run(
        cycle=cycle,
        forecast_hour=forecast_hour,
        variables=variables,
        product="sfc",
    )


def build_hrrr_pressure(
    *,
    region=GPGL_REGION,
    cycle=None,
    forecast_hour=0,
    variables=None,
):
    """
    Retrieve and normalize HRRR pressure-level fields for the GPGL domain.
    """
    from .models import HRRRAdapter

    adapter = HRRRAdapter(region)
    return adapter.open_run(
        cycle=cycle,
        forecast_hour=forecast_hour,
        variables=variables,
        product="prs",
    )


def hrrr_available_cycles(
    *,
    region=GPGL_REGION,
    count=12,
):
    from .models import HRRRAdapter
    return HRRRAdapter(region).available_cycles(count=count)


def compare_hrrr_surface_with_asos(
    *,
    region=GPGL_REGION,
    cycle=None,
    forecast_hour=0,
    variable="air_temperature_2m",
):
    from .model_compare import (
        compare_surface_to_asos,
        comparison_metrics,
    )

    model, run = build_hrrr_surface(
        region=region,
        cycle=cycle,
        forecast_hour=forecast_hour,
        variables=[variable],
    )

    start = run.valid_time - pd.Timedelta(minutes=20)
    end = run.valid_time + pd.Timedelta(minutes=20)
    observations = fetch_asos_tidy(
        region,
        start=start,
        end=end,
    )

    comparison = compare_surface_to_asos(
        model,
        observations,
        variable=variable,
    )
    metrics = comparison_metrics(comparison)
    return model, run, comparison, metrics



# ---------------------------------------------------------------------------
# Integrated model/observation workflows (v0.10)
# ---------------------------------------------------------------------------

def build_surface_at_valid_time(
    valid_time,
    *,
    region=GPGL_REGION,
    lookback="30min",
    observation_match_mode="nearest",
):
    """
    Build one regional surface analysis at a requested model valid time.

    ``lookback`` is the maximum allowed time separation. ``nearest`` (default)
    selects the closest observation from each station on either side of valid
    time; ``past`` restricts matching to observations at/before valid time.
    """
    valid_time = pd.Timestamp(valid_time)
    if valid_time.tzinfo is None:
        valid_time = valid_time.tz_localize("UTC")
    else:
        valid_time = valid_time.tz_convert("UTC")

    offset = pd.Timedelta(lookback)
    match_mode = str(observation_match_mode).lower()
    observations = fetch_asos_tidy(
        region,
        start=valid_time - offset,
        end=valid_time + offset if match_mode == "nearest" else valid_time,
    )

    grid = GridSpec(
        west=region.west,
        east=region.east,
        south=region.south,
        north=region.north,
        resolution=0.1,
    )
    config = SurfaceAnalysisConfig(
        analysis_interval="5min",
        lookback=lookback,
        temporal_decay_minutes=max(7.5, pd.Timedelta(lookback).total_seconds() / 120.0),
        observation_match_mode=match_mode,
        method="barnes",
        smoothing_km=140.0,
        max_distance_km=250.0,
        background_blend=False,
    )
    analysis = SurfaceAnalysisBuilder(grid, config).build(
        observations,
        variables=[
            "air_temperature_f",
            "dew_point_temperature_f",
            "u_wind_kt",
            "v_wind_kt",
        ],
        analysis_times=pd.DatetimeIndex([valid_time]),
    )
    analysis.attrs["requested_valid_time"] = str(valid_time)
    return analysis, observations


def build_regional_radar_at_time(
    valid_time,
    *,
    region=GPGL_REGION,
    altitude_km=2.0,
    radar_ids=None,
    horizontal_resolution_km=4.0,
    lookback_hours=1.0,
    max_scan_age_minutes=15.0,
):
    """
    Build the regional NEXRAD selected-altitude mosaic immediately preceding a
    requested valid time. This works for historical HRRR verification as long
    as the NEXRAD archive contains the requested date.
    """
    from .radar_grid import (
        RegionalRadarConfig,
        fetch_latest_regional_radars,
        grid_regional_reflectivity_altitude,
    )

    valid_time = pd.Timestamp(valid_time)
    if valid_time.tzinfo is None:
        valid_time = valid_time.tz_localize("UTC")
    else:
        valid_time = valid_time.tz_convert("UTC")

    config = RegionalRadarConfig(
        lookback_hours=float(lookback_hours),
        max_scan_age_minutes=float(max_scan_age_minutes),
    )
    radars, scans, diagnostics = fetch_latest_regional_radars(
        radar_ids=radar_ids,
        reference_time=valid_time.to_pydatetime(),
        config=config,
    )
    cube = grid_regional_reflectivity_altitude(
        radars,
        region=region,
        altitude_km=altitude_km,
        horizontal_resolution_km=horizontal_resolution_km,
    )
    cube.attrs["analysis_time"] = str(diagnostics["newest_scan_time"])
    cube.attrs["requested_valid_time"] = str(valid_time)
    cube.attrs["radar_count"] = len(scans)
    cube.attrs["mode"] = "valid-time-fast"
    return cube, scans, diagnostics


def build_hrrr_surface_gridded_comparison(
    *,
    region=GPGL_REGION,
    cycle=None,
    forecast_hour=0,
    variable="air_temperature_2m",
    observation_offset_minutes=30,
    observation_match_mode="nearest",
):
    from .integrated import build_surface_difference

    model, run = build_hrrr_surface(
        region=region,
        cycle=cycle,
        forecast_hour=forecast_hour,
        variables=[variable],
    )
    surface, observations = build_surface_at_valid_time(
        run.valid_time,
        region=region,
        lookback=f"{int(observation_offset_minutes)}min",
        observation_match_mode=observation_match_mode,
    )
    comparison = build_surface_difference(
        model,
        surface.isel(time=0),
        variable,
    )
    return model, run, surface, observations, comparison


def build_hrrr_radar_comparison(
    *,
    region=GPGL_REGION,
    cycle=None,
    forecast_hour=0,
    model_variable="composite_reflectivity",
    radar_altitude_km=2.0,
    radar_resolution_km=4.0,
):
    from .integrated import (
        build_radar_difference,
        radar_verification_metrics,
    )

    model, run = build_hrrr_surface(
        region=region,
        cycle=cycle,
        forecast_hour=forecast_hour,
        variables=[model_variable],
    )
    radar, scans, diagnostics = build_regional_radar_at_time(
        run.valid_time,
        region=region,
        altitude_km=radar_altitude_km,
        horizontal_resolution_km=radar_resolution_km,
    )
    comparison = build_radar_difference(
        model,
        radar,
        model_variable=model_variable,
    )
    metrics = radar_verification_metrics(comparison)
    return model, run, radar, scans, diagnostics, comparison, metrics



def match_hrrr_to_observation_time(observation_time):
    """Choose the HRRR initialization immediately before an observation.

    HRRR initialization is strictly earlier than the observation time.  The
    forecast hour is then selected to minimize valid-time offset.
    """
    t = pd.Timestamp(observation_time)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    cycle = t.floor("h")
    if cycle >= t:
        cycle -= pd.Timedelta(hours=1)
    forecast_hour = max(0, int(round((t - cycle).total_seconds() / 3600.0)))
    valid = cycle + pd.Timedelta(hours=forecast_hour)
    return cycle, forecast_hour, valid, (valid - t).total_seconds() / 60.0

def build_latest_hrrr_raob_comparison(
    *,
    region=GPGL_REGION,
    variable="air_temperature",
    station_id=None,
):
    """Compare the newest available regional radiosonde with the preceding HRRR run."""
    from .integrated import match_raob_hrrr_profile

    raob = IEMRAOBAdapter(region, launch_buffer_deg=2.0)
    profiles = raob.fetch(cycle=None, fallback_cycles=5)
    if profiles.empty:
        return None, None, profiles, None, pd.DataFrame(), {}

    sonde_time = pd.Timestamp(profiles.attrs.get("cycle", profiles["time"].max()))
    sonde_time = sonde_time.tz_localize("UTC") if sonde_time.tzinfo is None else sonde_time.tz_convert("UTC")
    cycle, forecast_hour, valid_time, offset_minutes = match_hrrr_to_observation_time(sonde_time)

    fetch_vars = [variable]
    if variable == "wind_speed":
        fetch_vars = ["u_wind", "v_wind"]

    model, run = build_hrrr_pressure(
        region=region, cycle=cycle, forecast_hour=forecast_hour,
        variables=fetch_vars,
    )
    stations = sorted(profiles.station_id.dropna().unique())
    station_id = station_id or stations[0]
    comparison = match_raob_hrrr_profile(model, profiles, station_id, variable)
    match = {
        "sonde_time": sonde_time,
        "hrrr_initialization": cycle,
        "forecast_hour": forecast_hour,
        "hrrr_valid_time": valid_time,
        "offset_minutes": offset_minutes,
    }
    return model, run, profiles, station_id, comparison, match


def build_hrrr_raob_comparison(
    *,
    region=GPGL_REGION,
    cycle=None,
    forecast_hour=0,
    variable="air_temperature",
    station_id=None,
):
    from .integrated import match_raob_hrrr_profile

    fetch_vars=[variable]
    if variable=="wind_speed":
        fetch_vars=["u_wind","v_wind"]

    model, run = build_hrrr_pressure(
        region=region,
        cycle=cycle,
        forecast_hour=forecast_hour,
        variables=fetch_vars,
    )

    raob = IEMRAOBAdapter(region, launch_buffer_deg=2.0)
    profiles = raob.fetch(
        cycle=run.valid_time.floor("12h"),
        fallback_cycles=1,
    )
    if profiles.empty:
        return model, run, profiles, None, pd.DataFrame()

    stations=sorted(profiles.station_id.dropna().unique())
    station_id = station_id or stations[0]
    comparison = match_raob_hrrr_profile(
        model, profiles, station_id, variable
    )
    return model, run, profiles, station_id, comparison


def build_hrrr_lead_time_verification(
    valid_time,
    *,
    region=GPGL_REGION,
    variable="air_temperature_2m",
    forecast_hours=(0,1,3,6,12),
):
    """
    Compare multiple HRRR initialization cycles that verify at one common time
    against the same gridded ASOS analysis.
    """
    from .integrated import build_surface_difference

    valid_time = pd.Timestamp(valid_time)
    if valid_time.tzinfo is None:
        valid_time=valid_time.tz_localize("UTC")
    else:
        valid_time=valid_time.tz_convert("UTC")

    surface, observations = build_surface_at_valid_time(
        valid_time, region=region
    )
    obs_surface=surface.isel(time=0)

    rows=[]
    comparisons={}
    for fxx in forecast_hours:
        cycle=valid_time-pd.Timedelta(hours=int(fxx))
        try:
            model, run=build_hrrr_surface(
                region=region,
                cycle=cycle,
                forecast_hour=int(fxx),
                variables=[variable],
            )
            comp=build_surface_difference(model,obs_surface,variable)
            d=comp.difference.values
            rows.append({
                "forecast_hour":int(fxx),
                "initialization_time":run.initialization_time,
                "valid_time":run.valid_time,
                "bias":float(np.nanmean(d)),
                "mae":float(np.nanmean(np.abs(d))),
                "rmse":float(np.sqrt(np.nanmean(d**2))),
            })
            comparisons[int(fxx)]=comp
        except Exception as exc:
            rows.append({
                "forecast_hour":int(fxx),
                "initialization_time":cycle,
                "valid_time":valid_time,
                "bias":np.nan,"mae":np.nan,"rmse":np.nan,
                "error":str(exc),
            })

    return pd.DataFrame(rows), comparisons, surface, observations


def build_mrms_at_time(valid_time=None, *, region=GPGL_REGION, lookback_hours=3):
    """Load NOAA MRMS quality-controlled composite reflectivity."""
    from .adapters.mrms import MRMSAdapter
    adapter=MRMSAdapter(region)
    radar,scan_time=adapter.open(when=valid_time,lookback_hours=lookback_hours)
    return radar,scan_time


def build_hrrr_mrms_comparison(
    *,
    region=GPGL_REGION,
    cycle=None,
    forecast_hour=0,
    model_variable="composite_reflectivity",
):
    from .integrated import build_radar_difference, radar_verification_metrics, radar_fractions_skill_score
    model,run=build_hrrr_surface(
        region=region,
        cycle=cycle,
        forecast_hour=forecast_hour,
        variables=[model_variable],
    )
    radar,scan_time=build_mrms_at_time(run.valid_time,region=region)
    comparison=build_radar_difference(model,radar,model_variable=model_variable)
    metrics=radar_verification_metrics(comparison)
    fss=radar_fractions_skill_score(comparison)
    return model,run,radar,scan_time,comparison,metrics,fss


def build_hrrr_radar_lead_time_verification(
    valid_time,
    *,
    region=GPGL_REGION,
    model_variable="composite_reflectivity",
    forecast_hours=(0,1,3,6,12),
):
    """Verify multiple HRRR leads against one common MRMS QC composite field."""
    from .integrated import build_radar_difference, radar_verification_metrics, radar_fractions_skill_score
    valid_time=pd.Timestamp(valid_time)
    valid_time=valid_time.tz_localize("UTC") if valid_time.tzinfo is None else valid_time.tz_convert("UTC")
    radar,scan_time=build_mrms_at_time(valid_time,region=region)
    metric_rows=[]
    fss_rows=[]
    comparisons={}
    for fxx in forecast_hours:
        cycle=valid_time-pd.Timedelta(hours=int(fxx))
        try:
            model,run=build_hrrr_surface(
                region=region,
                cycle=cycle,
                forecast_hour=int(fxx),
                variables=[model_variable],
            )
            comp=build_radar_difference(model,radar,model_variable=model_variable)
            met=radar_verification_metrics(comp)
            met.insert(0,"forecast_hour",int(fxx))
            met.insert(1,"initialization_time",run.initialization_time)
            metric_rows.append(met)
            fs=radar_fractions_skill_score(comp)
            fs.insert(0,"forecast_hour",int(fxx))
            fss_rows.append(fs)
            comparisons[int(fxx)]=comp
        except Exception as exc:
            metric_rows.append(pd.DataFrame([{
                "forecast_hour":int(fxx),"initialization_time":cycle,
                "threshold_dbz":np.nan,"POD":np.nan,"FAR":np.nan,"CSI":np.nan,
                "error":str(exc),
            }]))
    metrics=pd.concat(metric_rows,ignore_index=True) if metric_rows else pd.DataFrame()
    fss=pd.concat(fss_rows,ignore_index=True) if fss_rows else pd.DataFrame()
    return metrics,fss,comparisons,radar,scan_time


def build_hrrr_mrms_adapt_objects(
    *,
    region=GPGL_REGION,
    cycle=None,
    forecast_hour=0,
    model_variable="composite_reflectivity",
    threshold_dbz=35.0,
    min_gridpoints=8,
    h_maxima_dbz=5.0,
    max_match_distance_km=100.0,
):
    """HRRR/MRMS valid-time comparison with ADAPT storm-object detection."""
    from .storm_objects import build_adapt_object_comparison

    model,run,radar,scan_time,comparison,grid_metrics,fss = build_hrrr_mrms_comparison(
        region=region,
        cycle=cycle,
        forecast_hour=forecast_hour,
        model_variable=model_variable,
    )
    object_result = build_adapt_object_comparison(
        comparison,
        threshold_dbz=threshold_dbz,
        min_gridpoints=min_gridpoints,
        h_maxima_dbz=h_maxima_dbz,
        max_match_distance_km=max_match_distance_km,
    )
    return model,run,radar,scan_time,comparison,object_result

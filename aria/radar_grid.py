from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr

from .adapters.nexrad import GPGL_NEXRAD_SITES, NEXRADLevel2Adapter, nexrad_sites_for_region
from .region import GPGL_REGION
from .trajectory_match import lonlat_to_xy_km


@dataclass
class RegionalRadarConfig:
    horizontal_resolution_km: float = 4.0
    vertical_resolution_km: float = 1.0
    max_altitude_km: float = 12.0
    lookback_hours: float = 3.0
    max_scan_age_minutes: float = 20.0


def _reflectivity_name(radar):
    for name in (
        "reflectivity",
        "corrected_reflectivity",
        "reflectivity_horizontal",
    ):
        if name in radar.fields:
            return name
    return None


def _standardize_reflectivity(radar):
    name = _reflectivity_name(radar)
    if name is None:
        return False

    if name != "reflectivity":
        radar.add_field(
            "reflectivity",
            radar.fields[name],
            replace_existing=True,
        )

    return True


def fetch_latest_regional_radars(
    *,
    radar_ids=None,
    reference_time=None,
    config=None,
    region=GPGL_REGION,
):
    """
    Download the latest Level-II scan from every configured regional radar.

    Scans too old relative to the newest successfully downloaded volume are
    removed so the mosaic is not built from substantially different times.
    """
    config = config or RegionalRadarConfig()
    radar_ids = radar_ids or sorted(nexrad_sites_for_region(region))
    if not radar_ids:
        raise RuntimeError(f"No NEXRAD sites were found for region {region.name!r}.")

    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    adapter = NEXRADLevel2Adapter()

    loaded = []
    failures = []

    def load_one(radar_id):
        radar, scan = adapter.fetch_latest_radar(
            radar_id,
            when=reference_time,
            lookback_hours=config.lookback_hours,
        )
        if not _standardize_reflectivity(radar):
            raise RuntimeError("No reflectivity field")
        return radar, scan

    max_workers = min(8, max(1, len(radar_ids)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_one, radar_id): radar_id
            for radar_id in radar_ids
        }
        for future in as_completed(futures):
            radar_id = futures[future]
            try:
                loaded.append(future.result())
            except Exception as exc:
                failures.append((radar_id, str(exc)))

    if not loaded:
        raise RuntimeError(
            "No regional NEXRAD volumes could be downloaded. "
            f"Failures: {failures}"
        )

    newest_time = max(
        scan.scan_time
        for _, scan in loaded
        if scan.scan_time is not None
    )

    retained = []
    dropped = []

    for radar, scan in loaded:
        if scan.scan_time is None:
            dropped.append(
                (scan.radar_id, "Unknown scan time")
            )
            continue

        age_minutes = (
            newest_time - scan.scan_time
        ).total_seconds() / 60.0

        if age_minutes <= config.max_scan_age_minutes:
            retained.append((radar, scan))
        else:
            dropped.append(
                (
                    scan.radar_id,
                    f"{age_minutes:.1f} min older than newest scan",
                )
            )

    if not retained:
        raise RuntimeError(
            "All downloaded radar scans were rejected by the "
            "regional scan-age tolerance."
        )

    radars = [item[0] for item in retained]
    scans = [item[1] for item in retained]

    diagnostics = {
        "newest_scan_time": newest_time,
        "retained": [
            {
                "radar_id": scan.radar_id,
                "site_name": scan.site_name,
                "scan_time": scan.scan_time,
            }
            for scan in scans
        ],
        "dropped": dropped,
        "failures": failures,
    }

    return radars, scans, diagnostics


def grid_regional_reflectivity(
    radars,
    *,
    region=GPGL_REGION,
    config=None,
):
    """
    Grid multiple NEXRAD Level-II volumes into one regional Cartesian cube.

    Returns an xarray Dataset with:
        reflectivity[altitude_km, y_km, x_km]

    Latitude/longitude 2-D coordinates are also attached for mapping.
    """
    config = config or RegionalRadarConfig()

    try:
        import pyart
    except ImportError as exc:
        raise RuntimeError(
            "Regional radar gridding requires ARM Py-ART."
        ) from exc

    lon0 = (region.west + region.east) / 2.0
    lat0 = (region.south + region.north) / 2.0

    corner_lon = np.array(
        [region.west, region.east, region.west, region.east]
    )
    corner_lat = np.array(
        [region.south, region.south, region.north, region.north]
    )

    x_corner, y_corner = lonlat_to_xy_km(
        corner_lon,
        corner_lat,
        lon0,
        lat0,
    )

    xmin_km = float(np.nanmin(x_corner))
    xmax_km = float(np.nanmax(x_corner))
    ymin_km = float(np.nanmin(y_corner))
    ymax_km = float(np.nanmax(y_corner))

    nx = int(
        np.ceil(
            (xmax_km - xmin_km)
            / config.horizontal_resolution_km
        )
    ) + 1

    ny = int(
        np.ceil(
            (ymax_km - ymin_km)
            / config.horizontal_resolution_km
        )
    ) + 1

    nz = int(
        np.ceil(
            config.max_altitude_km
            / config.vertical_resolution_km
        )
    ) + 1

    grid = pyart.map.grid_from_radars(
        tuple(radars),
        grid_shape=(nz, ny, nx),
        grid_limits=(
            (0.0, config.max_altitude_km * 1000.0),
            (ymin_km * 1000.0, ymax_km * 1000.0),
            (xmin_km * 1000.0, xmax_km * 1000.0),
        ),
        grid_origin=(lat0, lon0),
        fields=["reflectivity"],
        weighting_function="Barnes2",
        roi_func="dist_beam",
    )

    reflectivity = np.asarray(
        grid.fields["reflectivity"]["data"]
    ).astype(float)

    if np.ma.isMaskedArray(
        grid.fields["reflectivity"]["data"]
    ):
        reflectivity = np.ma.filled(
            grid.fields["reflectivity"]["data"],
            np.nan,
        ).astype(float)

    x_km = grid.x["data"] / 1000.0
    y_km = grid.y["data"] / 1000.0
    altitude_km = grid.z["data"] / 1000.0

    lon2d, lat2d = grid.get_point_longitude_latitude(
        level=0
    )

    ds = xr.Dataset(
        {
            "reflectivity": (
                ("altitude_km", "y_km", "x_km"),
                reflectivity,
            ),
            "latitude": (
                ("y_km", "x_km"),
                lat2d,
            ),
            "longitude": (
                ("y_km", "x_km"),
                lon2d,
            ),
        },
        coords={
            "altitude_km": altitude_km,
            "y_km": y_km,
            "x_km": x_km,
        },
        attrs={
            "analysis_type": "regional multi-NEXRAD Cartesian grid",
            "horizontal_resolution_km": (
                config.horizontal_resolution_km
            ),
            "vertical_resolution_km": (
                config.vertical_resolution_km
            ),
            "max_altitude_km": config.max_altitude_km,
            "radars_used": ",".join(
                radar.metadata.get(
                    "instrument_name",
                    "UNKNOWN",
                )
                for radar in radars
            ),
        },
    )

    ds["reflectivity"].attrs.update(
        {
            "long_name": "Radar reflectivity",
            "units": "dBZ",
        }
    )

    return ds



def grid_regional_reflectivity_altitude(
    radars,
    *,
    region=GPGL_REGION,
    altitude_km=2.0,
    horizontal_resolution_km=6.0,
):
    """
    Fast CAPPI-style regional grid at a single requested altitude.
    """
    try:
        import pyart
    except ImportError as exc:
        raise RuntimeError(
            "Regional radar gridding requires ARM Py-ART."
        ) from exc

    lon0 = (region.west + region.east) / 2.0
    lat0 = (region.south + region.north) / 2.0

    corner_lon = np.array(
        [region.west, region.east, region.west, region.east]
    )
    corner_lat = np.array(
        [region.south, region.south, region.north, region.north]
    )

    x_corner, y_corner = lonlat_to_xy_km(
        corner_lon, corner_lat, lon0, lat0
    )

    xmin_km, xmax_km = float(np.nanmin(x_corner)), float(np.nanmax(x_corner))
    ymin_km, ymax_km = float(np.nanmin(y_corner)), float(np.nanmax(y_corner))

    nx = int(np.ceil((xmax_km - xmin_km) / horizontal_resolution_km)) + 1
    ny = int(np.ceil((ymax_km - ymin_km) / horizontal_resolution_km)) + 1
    z_m = altitude_km * 1000.0

    grid = pyart.map.grid_from_radars(
        tuple(radars),
        grid_shape=(1, ny, nx),
        grid_limits=(
            (z_m - 0.5, z_m + 0.5),
            (ymin_km * 1000.0, ymax_km * 1000.0),
            (xmin_km * 1000.0, xmax_km * 1000.0),
        ),
        grid_origin=(lat0, lon0),
        fields=["reflectivity"],
        weighting_function="Barnes2",
        roi_func="dist_beam",
    )

    raw = grid.fields["reflectivity"]["data"]
    reflectivity = (
        np.ma.filled(raw, np.nan).astype(float)
        if np.ma.isMaskedArray(raw)
        else np.asarray(raw, dtype=float)
    )

    lon2d, lat2d = grid.get_point_longitude_latitude(level=0)

    ds = xr.Dataset(
        {
            "reflectivity": (
                ("altitude_km", "y_km", "x_km"),
                reflectivity,
            ),
            "latitude": (("y_km", "x_km"), lat2d),
            "longitude": (("y_km", "x_km"), lon2d),
        },
        coords={
            "altitude_km": np.array([altitude_km], dtype=float),
            "y_km": grid.y["data"] / 1000.0,
            "x_km": grid.x["data"] / 1000.0,
        },
        attrs={
            "analysis_type": "regional multi-NEXRAD selected-altitude grid",
            "horizontal_resolution_km": horizontal_resolution_km,
            "vertical_resolution_km": 0.0,
            "max_altitude_km": altitude_km,
            "radars_used": ",".join(
                radar.metadata.get("instrument_name", "UNKNOWN")
                for radar in radars
            ),
        },
    )
    ds["reflectivity"].attrs.update(
        {"long_name": "Radar reflectivity", "units": "dBZ"}
    )
    return ds


def grid_single_radar_reflectivity(radar, *, horizontal_resolution_km=2.0, vertical_resolution_km=0.5, max_altitude_km=12.0, max_range_km=180.0):
    """Grid one Py-ART Radar onto a compact Cartesian 3-D reflectivity cube."""
    try:
        import pyart
    except ImportError as exc:
        raise RuntimeError("Single-radar 3-D gridding requires ARM Py-ART.") from exc
    if not _standardize_reflectivity(radar):
        raise RuntimeError("No reflectivity field available for 3-D gridding.")
    nx=int(np.ceil(2*max_range_km/horizontal_resolution_km))+1
    ny=nx; nz=int(np.ceil(max_altitude_km/vertical_resolution_km))+1
    grid=pyart.map.grid_from_radars((radar,),grid_shape=(nz,ny,nx),
        grid_limits=((0,max_altitude_km*1000),(-max_range_km*1000,max_range_km*1000),(-max_range_km*1000,max_range_km*1000)),
        fields=["reflectivity"],weighting_function="Barnes2")
    z=np.asarray(grid.z["data"],float)/1000; y=np.asarray(grid.y["data"],float)/1000; x=np.asarray(grid.x["data"],float)/1000
    data=np.asarray(grid.fields["reflectivity"]["data"].filled(np.nan),float)
    return xr.Dataset({"reflectivity":(("altitude_km","y_km","x_km"),data)},coords={"altitude_km":z,"y_km":y,"x_km":x},
        attrs={"radar_latitude":float(radar.latitude["data"][0]),"radar_longitude":float(radar.longitude["data"][0]),"horizontal_resolution_km":float(horizontal_resolution_km)})

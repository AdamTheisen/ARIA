from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from urllib.parse import quote

import requests
import xarray as xr


@dataclass(frozen=True)
class SSTProduct:
    key: str
    label: str
    source: str


PRODUCTS = {
    "ocean_l4": SSTProduct(
        "ocean_l4",
        "NOAA Geo-Polar Blended SST L4 (diurnally corrected)",
        "NOAA CoastWatch ERDDAP",
    ),
    "great_lakes": SSTProduct(
        "great_lakes",
        "Great Lakes Surface Environmental Analysis (GLSEA)",
        "NOAA GLERL/CoastWatch",
    ),
}

OCEAN_ERDDAP = "https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDsstDLDaily.nc"
GLSEA_ERDDAP = "https://apps.glerl.noaa.gov/erddap/griddap/GLSEA_ACSPO_GCS.nc"
GLSEA_BOUNDS = (-92.4199507342304, -75.8816402880531, 38.8749871947297, 50.6059751976539)


def region_intersects_great_lakes(region):
    # Broad screening envelope only. A future release should use lake polygons.
    return not (
        region.east < -92.5 or region.west > -75.0 or
        region.north < 41.0 or region.south > 49.5
    )


def choose_sst_product(region):
    return PRODUCTS["great_lakes" if region_intersects_great_lakes(region) else "ocean_l4"]


def _open_noaa_ocean_subset(region):
    """Download the latest daily CoastWatch Geo-Polar L4 SST for the ARIA box.

    ERDDAP performs the geographic subset server-side, avoiding a ~150 MB global
    download. The returned Dataset is fully loaded so the temporary file can be
    removed safely.
    """
    variables = ("analysed_sst", "analysis_error", "mask", "sea_ice_fraction")
    south, north = sorted((float(region.south), float(region.north)))
    west, east = sorted((float(region.west), float(region.east)))
    pieces = [f"{v}[(last)][({south}):({north})][({west}):({east})]" for v in variables]
    query = ",".join(pieces)
    url = OCEAN_ERDDAP + "?" + quote(query, safe="[],():,.-")

    cache = Path.home()/".cache"/"aria"/"sst"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/f"ocean_latest_{region.cache_key}.nc"
    tmp = target.with_suffix(".nc.tmp")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    tmp.write_bytes(response.content)
    os.replace(tmp, target)

    with xr.open_dataset(target) as source:
        ds = source.load()
    if "analysed_sst" in ds:
        ds["sst_c"] = ds["analysed_sst"] - 273.15
        ds["sst_c"].attrs.update(long_name="Sea surface temperature", units="degree_C")
        ds["sst_f"] = ds["sst_c"] * 9.0/5.0 + 32.0
        ds["sst_f"].attrs.update(long_name="Sea surface temperature", units="degree_F")
    ds.attrs["aria_source_url"] = OCEAN_ERDDAP
    ds.attrs["aria_product"] = PRODUCTS["ocean_l4"].label
    return ds



def _open_glsea_subset(region):
    """Download the latest ACSPO GLSEA field and subset it locally.

    GLERL ERDDAP has been sensitive to coordinate-value slicing at the exact
    floating-point grid edges. ARIA therefore requests the latest full spatial
    field using index constraints, trying the supported index forms in order,
    then performs the geographic subset locally.
    """
    west0, east0, south0, north0 = GLSEA_BOUNDS
    south = max(float(region.south), south0)
    north = min(float(region.north), north0)
    west = max(float(region.west), west0)
    east = min(float(region.east), east0)
    if south >= north or west >= east:
        raise RuntimeError("The selected region does not overlap the GLSEA analysis domain.")

    cache = Path.home()/".cache"/"aria"/"sst"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/f"glsea_latest_{region.cache_key}.nc"
    tmp = target.with_suffix(".nc.tmp")

    queries = [
        "sst[last][0:1:last][0:1:last]",
        "sst[(last)][0:1:last][0:1:last]",
        "sst[last][][]",
    ]
    failures = []
    used_url = None
    for query in queries:
        url = GLSEA_ERDDAP + "?" + quote(query, safe="[],():,.-")
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            content = response.content
            is_netcdf = content.startswith(b"CDF") or content.startswith(b"\x89HDF")
            if not is_netcdf:
                ctype = response.headers.get("content-type", "")
                raise RuntimeError(
                    f"GLSEA response was not NetCDF ({ctype or 'unknown content type'})."
                )
            tmp.write_bytes(content)
            os.replace(tmp, target)
            used_url = url
            break
        except Exception as exc:
            failures.append(f"{query}: {type(exc).__name__}: {exc}")

    if used_url is None:
        raise RuntimeError(
            "GLSEA retrieval failed for all ERDDAP index-query variants. "
            + " | ".join(failures)
        )

    with xr.open_dataset(target) as source:
        ds = source.load()

    rename = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)

    if "latitude" in ds.coords and "longitude" in ds.coords:
        lat = ds["latitude"]
        lon = ds["longitude"]
        lat_slice = slice(south, north) if float(lat[0]) <= float(lat[-1]) else slice(north, south)
        lon_slice = slice(west, east) if float(lon[0]) <= float(lon[-1]) else slice(east, west)
        ds = ds.sel(latitude=lat_slice, longitude=lon_slice)

    if "sst" in ds:
        units = str(ds["sst"].attrs.get("units", "")).lower()
        ds["sst_c"] = (
            ds["sst"] - 273.15
            if ("kelvin" in units or units in {"k", "degree_k"})
            else ds["sst"]
        )
        ds["sst_c"].attrs.update(long_name="Lake surface temperature", units="degree_C")
        ds["sst_f"] = ds["sst_c"] * 9.0 / 5.0 + 32.0
        ds["sst_f"].attrs.update(long_name="Lake surface temperature", units="degree_F")

    ds.attrs["aria_source_url"] = used_url
    ds.attrs["aria_product"] = PRODUCTS["great_lakes"].label
    ds.attrs["aria_glsea_subset_method"] = "latest_full_field_index_request_then_local_coordinate_subset"
    ds.attrs["aria_subset_bounds"] = f"{west:.5f},{south:.5f},{east:.5f},{north:.5f}"
    return ds


def open_sst(region, *, url=None):
    product = choose_sst_product(region)
    if url:
        return xr.open_dataset(url), product

    if product.key == "ocean_l4":
        # An environment override remains available for offline/private mirrors,
        # but normal use no longer requires configuration.
        endpoint = os.getenv("ARIA_OCEAN_SST_URL")
        if endpoint:
            return xr.open_dataset(endpoint), product
        return _open_noaa_ocean_subset(region), product

    endpoint = os.getenv("ARIA_GLSEA_URL")
    if endpoint:
        return xr.open_dataset(endpoint), product
    return _open_glsea_subset(region), product

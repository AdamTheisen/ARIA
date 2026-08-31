# ARIA Configuration

## Regions

The dashboard provides GPGL, ARM SGP, Upper Midwest, CONUS, and Custom presets.
A custom region is defined by west/east longitude and south/north latitude.

For station-based providers that require state/network queries, ARIA attempts to
infer intersecting states using Cartopy/Natural Earth and then clips returned
observations to the exact bounding box.

## Environment variables

- `AIRNOW_API` — AirNow API key
- `EPA_AQS_EMAIL` / `EPA_AQS_KEY` — optional EPA AQS credentials
- `AIRCRAFT_DATA` — authorized local aircraft CSV
- `ARIA_CUBE_ROOT` — ARIA Icechunk repository root
- `ARIA_DASK_SCHEDULER` — local Dask scheduler, default `threads`
- `ARIA_CUBE_ROOT` — legacy fallback for existing installations
- `ARIA_MRMS_LATEST_TTL` — legacy MRMS latest-cache TTL setting

## Air quality

The default monitor-support radius is **300 km** to provide a useful initial
regional map. The dashboard exposes a 100–500 km control. Unsupported areas
remain subject to monitor-support masking rather than unlimited extrapolation.


## Region definitions

ARIA supports both rectangular bounding boxes and site-centered regions.
Site-centered regions are defined by latitude, longitude, and radius; ARIA
converts them to a provider-compatible bounding box.

The ARM SGP preset now uses a 400 km radius centered near the SGP Central
Facility. This better represents the surrounding mesoscale environment than
the earlier small rectangle.

## Region-aware caching

Region identity is part of ARIA's processed-cache key and is also passed
explicitly into Streamlit cached loader functions. Switching regions therefore
causes source data to be reacquired/subset for the selected domain rather than
reusing a prior region's in-memory result.

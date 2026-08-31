import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aria import GPGL_REGION, GridSpec, GriddedCubeBuilder
from aria.adapters import AirNowConcentrationAdapter
from aria.adapters.asos_bulk import latest_complete_hour


token = os.getenv("AIRNOW_API")
if not token:
    raise RuntimeError(
        "AIRNOW_API is not set. Export your AirNow API key before running."
    )

region = GPGL_REGION
start, end = latest_complete_hour(lag_minutes=10)

output_dir = Path("output") / "air_quality"
output_dir.mkdir(parents=True, exist_ok=True)

airnow = AirNowConcentrationAdapter(region, token=token)
df = airnow.fetch(
    start - pd.Timedelta(hours=1),
    end,
    parameters="OZONE,PM25",
)

if df.empty:
    raise RuntimeError("AirNow returned no PM2.5/ozone observations.")

print(
    df.groupby("variable").agg(
        observations=("value", "count"),
        stations=("station_id", "nunique"),
        latest=("time", "max"),
    )
)

grid = GridSpec(
    west=region.west,
    east=region.east,
    south=region.south,
    north=region.north,
    resolution=0.1,
)

builder = GriddedCubeBuilder(grid)

# Use the latest AirNow reporting time for a clean, dedicated air-quality plot.
latest = pd.to_datetime(df["time"], utc=True).max()
latest_df = df[
    pd.to_datetime(df["time"], utc=True) == latest
].copy()

air_cube = builder.from_station_dataframe(
    latest_df,
    variables=[
        v for v in ("pm25", "ozone")
        if v in set(latest_df["variable"])
    ],
    time_freq="1h",
    method="idw",
    max_distance_km=350,
    idw_k=8,
    idw_power=1.5,
    min_neighbors=1,
    taper_start_km=250,
    background_blend=True,
    background_k=16,
    background_power=1.0,
    background_radius_km=500,
    smooth_sigma=1.0,
)

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(
    [region.west, region.east, region.south, region.north],
    ccrs.PlateCarree(),
)

ax.add_feature(
    cfeature.LAKES.with_scale("50m"),
    facecolor="white",
    edgecolor="0.3",
)
ax.add_feature(
    cfeature.STATES.with_scale("50m"),
    linewidth=0.5,
)
ax.coastlines(resolution="50m", linewidth=0.6)

pm_mesh = None

if "pm25" in air_cube:
    pm = air_cube["pm25"].isel(time=0).values
    if np.isfinite(pm).any():
        pm_mesh = ax.pcolormesh(
            air_cube["longitude"],
            air_cube["latitude"],
            pm,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )

if "ozone" in air_cube:
    ozone = air_cube["ozone"].isel(time=0).values
    finite = ozone[np.isfinite(ozone)]
    if finite.size > 3:
        lo = np.floor(np.nanpercentile(finite, 5) / 5) * 5
        hi = np.ceil(np.nanpercentile(finite, 95) / 5) * 5
        levels = np.arange(lo, hi + 5, 5)

        if len(levels) > 1:
            cs = ax.contour(
                air_cube["longitude"],
                air_cube["latitude"],
                ozone,
                levels=levels,
                colors="black",
                linewidths=0.8,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(cs, inline=True, fontsize=7)

if pm_mesh is not None:
    cbar = fig.colorbar(pm_mesh, ax=ax, pad=0.02)
    cbar.set_label("PM2.5 concentration")

ax.set_title(
    f"EPA/AirNow Air Quality\n{latest:%Y-%m-%d %H:%M} UTC\n"
    "PM2.5 Shading • Ozone Contours"
)

path = output_dir / "latest_air_quality.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
plt.close(fig)

air_cube.to_zarr(
    output_dir / "air_quality.zarr",
    mode="w",
    consolidated=True,
)

print(f"Air-quality output written to {path}")

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aria.plotting.cities import add_mpl_cities


def plot_air_quality(
    cube,
    region,
    *,
    output_path=None,
    show_sites=True,
    observations=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
):
    """
    PM2.5 as filled shading with ozone contour overlay.
    """
    timestamp = pd.Timestamp(cube["time"].values[-1])

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [region.west, region.east, region.south, region.north],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="0.95",
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor="white",
        edgecolor="0.35",
        linewidth=0.6,
        zorder=4,
    )
    ax.add_feature(
        cfeature.STATES.with_scale("50m"),
        linewidth=0.45,
        edgecolor="0.35",
        zorder=5,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.6,
        edgecolor="0.3",
        zorder=5,
    )
    ax.coastlines(
        resolution="50m",
        linewidth=0.6,
        color="0.3",
        zorder=5,
    )

    add_mpl_cities(ax, region)

    pm_mesh = None

    if "pm25" in cube:
        pm = cube["pm25"].isel(time=-1).values
        finite = pm[np.isfinite(pm)]

        if finite.size:
            auto_vmax = max(15.0, float(np.nanpercentile(finite, 98)))
            use_vmin = 0.0 if vmin is None else vmin
            use_vmax = auto_vmax if vmax is None else vmax

            pm_mesh = ax.pcolormesh(
                cube["longitude"],
                cube["latitude"],
                pm,
                shading="auto",
                cmap=cmap,
                vmin=use_vmin,
                vmax=use_vmax,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

    if "ozone" in cube:
        ozone = cube["ozone"].isel(time=-1).values
        finite = ozone[np.isfinite(ozone)]

        if finite.size > 3:
            lo = np.floor(
                np.nanpercentile(finite, 5) / 5
            ) * 5
            hi = np.ceil(
                np.nanpercentile(finite, 95) / 5
            ) * 5

            levels = np.arange(lo, hi + 5, 5)

            if len(levels) > 1:
                contours = ax.contour(
                    cube["longitude"],
                    cube["latitude"],
                    ozone,
                    levels=levels,
                    colors="black",
                    linewidths=0.8,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )
                ax.clabel(
                    contours,
                    inline=True,
                    fontsize=7,
                    fmt="%g",
                )

    if show_sites and observations is not None and not observations.empty:
        sites = observations[
            ["station_id", "latitude", "longitude"]
        ].drop_duplicates()

        ax.scatter(
            sites["longitude"],
            sites["latitude"],
            s=14,
            facecolors="none",
            edgecolors="black",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

    if pm_mesh is not None:
        cbar = fig.colorbar(
            pm_mesh,
            ax=ax,
            pad=0.02,
            shrink=0.86,
        )
        cbar.set_label("PM2.5 concentration")

    ax.set_title(
        "EPA/AirNow Air Quality\n"
        f"{timestamp:%Y-%m-%d %H:%M} UTC\n"
        "PM2.5 Shading • Ozone Contours"
    )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        alpha=0.3,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

    return fig

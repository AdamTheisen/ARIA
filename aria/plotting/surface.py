from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _base_map(ax, region):
    ax.set_extent(
        [
            region.west,
            region.east,
            region.south,
            region.north,
        ],
        ccrs.PlateCarree(),
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


def plot_surface_analysis(
    ds,
    time_index,
    region,
    *,
    output_path=None,
    barb_skip=12,
    temp_vmin=None,
    temp_vmax=None,
    cmap="coolwarm",
    show_confidence=False,
):
    """
    Combined temperature / dew point / wind surface analysis.
    """
    timestamp = pd.Timestamp(
        ds["time"].isel(time=time_index).values
    )

    temp = ds["air_temperature_f"].isel(time=time_index).values
    dew = ds["dew_point_temperature_f"].isel(time=time_index).values
    u = ds["u_wind_kt"].isel(time=time_index).values
    v = ds["v_wind_kt"].isel(time=time_index).values

    if temp_vmin is None or temp_vmax is None:
        finite = temp[np.isfinite(temp)]
        if finite.size:
            if temp_vmin is None:
                temp_vmin = (
                    np.floor(np.nanpercentile(finite, 2) / 5)
                    * 5
                )
            if temp_vmax is None:
                temp_vmax = (
                    np.ceil(np.nanpercentile(finite, 98) / 5)
                    * 5
                )

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    _base_map(ax, region)

    mesh = ax.pcolormesh(
        ds["longitude"],
        ds["latitude"],
        temp,
        cmap=cmap,
        vmin=temp_vmin,
        vmax=temp_vmax,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    finite_dew = dew[np.isfinite(dew)]
    if finite_dew.size:
        lo = (
            np.floor(np.nanpercentile(finite_dew, 5) / 5)
            * 5
        )
        hi = (
            np.ceil(np.nanpercentile(finite_dew, 95) / 5)
            * 5
        )
        levels = np.arange(lo, hi + 5, 5)

        if len(levels) > 1:
            contours = ax.contour(
                ds["longitude"],
                ds["latitude"],
                dew,
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
                fmt="%d",
            )

    lon2d, lat2d = np.meshgrid(
        ds["longitude"].values,
        ds["latitude"].values,
    )

    sl = (
        slice(None, None, barb_skip),
        slice(None, None, barb_skip),
    )

    ax.barbs(
        lon2d[sl],
        lat2d[sl],
        u[sl],
        v[sl],
        length=5,
        linewidth=0.45,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    if show_confidence:
        age_name = "air_temperature_f_effective_age_minutes"
        count_name = "air_temperature_f_n_contributing"

        if age_name in ds and count_name in ds:
            age = ds[age_name].isel(time=time_index).values
            count = ds[count_name].isel(time=time_index).values

            low_confidence = (
                (age > 12)
                | (count < 2)
                | ~np.isfinite(temp)
            )

            ax.contourf(
                ds["longitude"],
                ds["latitude"],
                low_confidence.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["..."],
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        pad=0.02,
        shrink=0.86,
    )
    cbar.set_label("Air Temperature (°F)")

    # Frame-level diagnostics.
    age_name = "air_temperature_f_effective_age_minutes"
    count_name = "air_temperature_f_n_contributing"

    diagnostic = ""
    if age_name in ds:
        age = ds[age_name].isel(time=time_index).values
        finite_age = age[np.isfinite(age)]

        if finite_age.size:
            diagnostic += (
                f" | median effective age "
                f"{np.nanmedian(finite_age):.1f} min"
            )

    if count_name in ds:
        count = ds[count_name].isel(time=time_index).values
        finite_count = count[np.isfinite(temp)]

        if finite_count.size:
            diagnostic += (
                f" | median contributors "
                f"{np.nanmedian(finite_count):.0f}"
            )

    ax.set_title(
        "Great Plains / Great Lakes Surface Analysis\n"
        f"{timestamp:%Y-%m-%d %H:%M} UTC\n"
        "Temperature • Dew Point • Wind"
        + diagnostic,
        fontsize=13,
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


def plot_coverage(
    ds,
    time_index,
    region,
    variable="air_temperature_f",
    output_dir=None,
):
    outputs = {}

    diagnostics = [
        (
            f"{variable}_nearest_distance_km",
            "Distance to Nearest Observation",
            "Distance (km)",
        ),
        (
            f"{variable}_n_contributing",
            "Contributing Observation Count",
            "Contributors",
        ),
        (
            f"{variable}_effective_age_minutes",
            "Effective Observation Age",
            "Age (minutes)",
        ),
    ]

    timestamp = pd.Timestamp(
        ds["time"].isel(time=time_index).values
    )

    for name, title, label in diagnostics:
        if name not in ds:
            continue

        fig = plt.figure(figsize=(11, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        _base_map(ax, region)

        field = ds[name].isel(time=time_index)

        mesh = ax.pcolormesh(
            ds["longitude"],
            ds["latitude"],
            field,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )

        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(label)

        ax.set_title(
            f"{title}\n"
            f"{timestamp:%Y-%m-%d %H:%M} UTC"
        )

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            path = output_dir / (
                f"{name}_{timestamp:%Y%m%d_%H%M}.png"
            )
            fig.savefig(
                path,
                dpi=180,
                bbox_inches="tight",
            )
            outputs[name] = path

        plt.close(fig)

    return outputs

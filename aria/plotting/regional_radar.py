from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from aria.plotting.state_floor import add_state_floor


def plot_regional_reflectivity(
    radar_ds,
    region,
    *,
    altitude_km=1.0,
    output_path=None,
    vmin=-10,
    vmax=70,
    cmap="turbo",
):
    """
    Plot one horizontal altitude slice of the regional multi-radar grid.
    """
    field = radar_ds["reflectivity"].sel(
        altitude_km=altitude_km,
        method="nearest",
    )
    altitude = float(field["altitude_km"].values)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [region.west, region.east, region.south, region.north],
        ccrs.PlateCarree(),
    )

    mesh = ax.pcolormesh(
        radar_ds["longitude"],
        radar_ds["latitude"],
        field,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor="none",
        edgecolor="white",
    )
    ax.add_feature(
        cfeature.STATES.with_scale("50m"),
        linewidth=0.9,
        edgecolor="white",
    )
    ax.coastlines(
        resolution="50m",
        linewidth=0.8,
        color="white",
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        pad=0.02,
    )
    cbar.set_label("Reflectivity (dBZ)")

    ax.set_title(
        f"Regional Multi-NEXRAD Reflectivity — "
        f"{altitude:g} km MSL"
    )

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


def plot_regional_reflectivity_3d(radar_ds, *, threshold_dbz=20.0, surface_count=4, opacity=0.25):
    """Interactive 3-D reflectivity isosurfaces preserving the regular grid."""
    import plotly.graph_objects as go
    z=np.asarray(radar_ds["altitude_km"].values,float)
    y=np.asarray(radar_ds["y_km"].values,float)
    x=np.asarray(radar_ds["x_km"].values,float)
    data=np.asarray(radar_ds["reflectivity"].values,float)
    zz,yy,xx=np.meshgrid(z,y,x,indexing="ij")
    vals=np.nan_to_num(data,nan=-40.0,posinf=70.0,neginf=-40.0)
    above=int(np.count_nonzero(vals >= threshold_dbz))
    max_dbz=float(np.nanmax(data)) if np.isfinite(data).any() else float("nan")
    fig=go.Figure()
    if above:
        fig.add_trace(go.Isosurface(
            x=xx.ravel(),y=yy.ravel(),z=zz.ravel(),value=vals.ravel(),
            isomin=float(threshold_dbz),isomax=70.0,surface_count=max(1,int(surface_count)),
            opacity=float(opacity),colorscale="Turbo",cmin=-30,cmax=70,
            caps=dict(x_show=False,y_show=False,z_show=False),colorbar=dict(title="dBZ"),
            hovertemplate="X %{x:.0f} km<br>Y %{y:.0f} km<br>Z %{z:.1f} km<br>%{value:.1f} dBZ<extra></extra>"))
    fig.update_layout(title=f"3-D Reflectivity | max {max_dbz:.1f} dBZ | voxels ≥ {threshold_dbz:.0f}: {above:,}",height=840,
        scene=dict(
            xaxis=dict(title="East/West (km)", range=[float(x.min()), float(x.max())]),
            yaxis=dict(title="North/South (km)", range=[float(y.min()), float(y.max())]),
            zaxis=dict(title="Altitude (km MSL)", range=[float(z.min()), float(z.max())]),
            aspectmode="manual",aspectratio=dict(x=1.5,y=1,z=.55)
        ),margin=dict(l=0,r=0,t=55,b=0))
    if "radar_longitude" in radar_ds.attrs and "radar_latitude" in radar_ds.attrs:
        add_state_floor(fig,float(z.min()),lon0=float(radar_ds.attrs["radar_longitude"]),lat0=float(radar_ds.attrs["radar_latitude"]),color="black",width=3)
    return fig


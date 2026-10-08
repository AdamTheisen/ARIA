from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from types import SimpleNamespace
from aria.plotting.cities import add_mpl_cities, add_plotly_cities


def _reflectivity_field(radar):
    candidates = [
        "reflectivity",
        "corrected_reflectivity",
        "reflectivity_horizontal",
    ]

    for name in candidates:
        if name in radar.fields:
            return name

    raise RuntimeError(
        "No reflectivity field was found in this NEXRAD volume. "
        f"Available fields: {list(radar.fields)}"
    )



def plot_interactive_nexrad_ppi(
    radar,
    *,
    sweep=0,
    region=None,
    vmin=-30,
    vmax=70,
    colorscale="Turbo",
    max_range_km=230,
    adapt_labels=None,
):
    """Interactive Plotly PPI using a fast geographic max-bin display grid."""
    import plotly.graph_objects as go
    field_name=_reflectivity_field(radar)
    sweep=int(np.clip(sweep,0,radar.nsweeps-1))
    ray0=int(radar.sweep_start_ray_index["data"][sweep])
    ray1=int(radar.sweep_end_ray_index["data"][sweep])+1
    values=np.asarray(np.ma.filled(radar.fields[field_name]["data"][ray0:ray1,:],np.nan),float)
    gate_lat,gate_lon,_=radar.get_gate_lat_lon_alt(sweep)
    gate_lat=np.asarray(gate_lat,float); gate_lon=np.asarray(gate_lon,float)
    ranges=np.asarray(radar.range["data"],float)/1000.0
    if ranges.size==values.shape[1]:
        values=np.where(ranges[None,:]<=float(max_range_km),values,np.nan)

    lon0=float(radar.longitude["data"][0]); lat0=float(radar.latitude["data"][0])
    if region is None:
        west,east=lon0-3.5,lon0+3.5
        south,north=lat0-2.7,lat0+2.7
        display_region=SimpleNamespace(west=west,east=east,south=south,north=north)
    else:
        west,east=float(region.west),float(region.east)
        south,north=float(region.south),float(region.north)
        display_region=region

    n=360
    xedges=np.linspace(west,east,n+1); yedges=np.linspace(south,north,n+1)
    good=np.isfinite(values)&np.isfinite(gate_lon)&np.isfinite(gate_lat)
    xi=np.searchsorted(xedges,gate_lon[good],side="right")-1
    yi=np.searchsorted(yedges,gate_lat[good],side="right")-1
    keep=(xi>=0)&(xi<n)&(yi>=0)&(yi<n)
    xi=xi[keep]; yi=yi[keep]; vv=values[good][keep]
    flat=np.full(n*n,-np.inf,float)
    np.maximum.at(flat,yi*n+xi,vv)
    z=flat.reshape(n,n); z[~np.isfinite(z)]=np.nan
    x=.5*(xedges[:-1]+xedges[1:]); y=.5*(yedges[:-1]+yedges[1:])

    fig=go.Figure(go.Heatmap(
        x=x,y=y,z=z,zmin=float(vmin),zmax=float(vmax),colorscale=colorscale,
        colorbar=dict(title="dBZ"),
        hovertemplate="Lon %{x:.3f}<br>Lat %{y:.3f}<br>%{z:.1f} dBZ<extra></extra>",
    ))

    # State boundaries on top of the radar field.
    try:
        import cartopy.io.shapereader as shpreader
        shp=shpreader.natural_earth(resolution="50m",category="cultural",name="admin_1_states_provinces_lakes")
        for rec in shpreader.Reader(shp).records():
            a=rec.attributes
            if a.get("adm0_a3")!="USA" and a.get("admin")!="United States of America": continue
            boundary=rec.geometry.boundary
            geoms=list(boundary.geoms) if hasattr(boundary,"geoms") else [boundary]
            for g in geoms:
                arr=np.asarray(g.coords)
                if arr[:,0].max()<west or arr[:,0].min()>east or arr[:,1].max()<south or arr[:,1].min()>north:
                    continue
                fig.add_trace(go.Scatter(
                    x=arr[:,0],y=arr[:,1],mode="lines",
                    line=dict(color="black",width=1.4),hoverinfo="skip",showlegend=False,
                ))
    except Exception:
        pass

    if adapt_labels is not None:
        try:
            from aria.storm_objects import extract_object_boundaries
            for ident,coords in extract_object_boundaries(
                adapt_labels,x=gate_lon,y=gate_lat,simplify_stride=2
            ).items():
                fig.add_trace(go.Scatter(
                    x=coords[:,0],y=coords[:,1],mode="lines",
                    line=dict(color="white",width=2),name=f"ADAPT T{ident}",
                    hovertemplate=f"ADAPT T{ident}<extra></extra>",showlegend=False,
                ))
        except Exception:
            pass

    add_plotly_cities(fig,display_region,font_size=9)
    fig.update_layout(
        title=f"{str(getattr(radar,'metadata',{}).get('instrument_name','NEXRAD'))} — Interactive PPI",
        height=760,dragmode="zoom",
        xaxis=dict(title="Longitude",range=[west,east],constrain="domain"),
        yaxis=dict(title="Latitude",range=[south,north],scaleanchor="x",constrain="domain"),
        margin=dict(l=45,r=35,t=65,b=45),
    )
    return fig


def plot_static_nexrad_ppi(
    radar,
    *,
    sweep=0,
    region=None,
    vmin=-30,
    vmax=70,
    cmap="turbo",
    max_range_km=230,
    adapt_labels=None,
):
    """Fast static PPI for animation/playback.

    The radar gates are binned onto a regular lon/lat display grid and rendered
    directly with Matplotlib. This avoids depending on Py-ART RadarMapDisplay
    state during Streamlit animation reruns.
    """
    import cartopy.feature as cfeature

    field_name=_reflectivity_field(radar)
    sweep=int(np.clip(sweep,0,radar.nsweeps-1))
    ray0=int(radar.sweep_start_ray_index["data"][sweep])
    ray1=int(radar.sweep_end_ray_index["data"][sweep])+1

    values=np.asarray(
        np.ma.filled(radar.fields[field_name]["data"][ray0:ray1,:],np.nan),
        dtype=float,
    )
    gate_lat,gate_lon,_=radar.get_gate_lat_lon_alt(sweep)
    gate_lat=np.asarray(gate_lat,dtype=float)
    gate_lon=np.asarray(gate_lon,dtype=float)

    ranges=np.asarray(radar.range["data"],dtype=float)/1000.0
    if ranges.size==values.shape[1]:
        values=np.where(ranges[None,:] <= float(max_range_km), values, np.nan)

    lon0=float(radar.longitude["data"][0])
    lat0=float(radar.latitude["data"][0])
    if region is None:
        west,east=lon0-3.5,lon0+3.5
        south,north=lat0-2.7,lat0+2.7
        display_region=SimpleNamespace(west=west,east=east,south=south,north=north)
    else:
        west,east=float(region.west),float(region.east)
        south,north=float(region.south),float(region.north)
        display_region=region

    n=420
    xedges=np.linspace(west,east,n+1)
    yedges=np.linspace(south,north,n+1)

    good=np.isfinite(values)&np.isfinite(gate_lon)&np.isfinite(gate_lat)
    xi=np.searchsorted(xedges,gate_lon[good],side="right")-1
    yi=np.searchsorted(yedges,gate_lat[good],side="right")-1
    keep=(xi>=0)&(xi<n)&(yi>=0)&(yi<n)

    flat=np.full(n*n,-np.inf,dtype=float)
    if np.any(keep):
        np.maximum.at(flat,yi[keep]*n+xi[keep],values[good][keep])

    grid=flat.reshape(n,n)
    grid[~np.isfinite(grid)]=np.nan

    fig=plt.figure(figsize=(11,8))
    ax=fig.add_subplot(111,projection=ccrs.PlateCarree())
    ax.set_extent([west,east,south,north],crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES.with_scale("50m"),linewidth=0.8,edgecolor="black")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),linewidth=0.6,edgecolor="black")
    ax.coastlines(resolution="50m",linewidth=0.6)

    mesh=ax.pcolormesh(
        xedges,yedges,grid,
        cmap=cmap,vmin=float(vmin),vmax=float(vmax),
        shading="auto",transform=ccrs.PlateCarree(),
    )
    cb=fig.colorbar(mesh,ax=ax,pad=0.02)
    cb.set_label("Reflectivity (dBZ)")

    add_mpl_cities(ax,display_region,font_size=8)

    if adapt_labels is not None:
        try:
            labels=np.asarray(adapt_labels)
            if labels.shape==gate_lat.shape and np.nanmax(labels)>0:
                levels=np.arange(0.5,int(np.nanmax(labels))+0.5,1.0)
                ax.contour(
                    gate_lon,gate_lat,labels,levels=levels,
                    colors="white",linewidths=1.3,
                    transform=ccrs.PlateCarree(),
                )
        except Exception:
            pass

    fixed_angle=float(radar.fixed_angle["data"][sweep])
    radar_name=radar.metadata.get("instrument_name") or "NEXRAD"
    ax.set_title(
        f"{radar_name} Reflectivity — Sweep {sweep} • "
        f"{fixed_angle:.1f}° elevation"
    )
    fig.tight_layout()
    return fig



def plot_nexrad_ppi(
    radar,
    *,
    sweep=0,
    output_path=None,
    max_range_km=230,
    vmin=-10,
    vmax=70,
    cmap="pyart_NWSRef",
    adapt_labels=None,
    show_adapt_ids=True,
):
    """
    Plot one NEXRAD reflectivity PPI sweep using Py-ART.
    """
    try:
        import pyart
    except ImportError as exc:
        raise RuntimeError(
            "Radar plotting requires ARM Py-ART."
        ) from exc

    field = _reflectivity_field(radar)

    sweep = int(
        np.clip(
            sweep,
            0,
            radar.nsweeps - 1,
        )
    )

    display = pyart.graph.RadarMapDisplay(radar)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    display.plot_ppi_map(
        field,
        sweep=sweep,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        min_lon=float(radar.longitude["data"][0]) - 3.5,
        max_lon=float(radar.longitude["data"][0]) + 3.5,
        min_lat=float(radar.latitude["data"][0]) - 2.7,
        max_lat=float(radar.latitude["data"][0]) + 2.7,
        resolution="50m",
        colorbar_label="Reflectivity (dBZ)",
        title_flag=False,
    )

    radar_region=SimpleNamespace(
        west=float(radar.longitude["data"][0])-3.5,
        east=float(radar.longitude["data"][0])+3.5,
        south=float(radar.latitude["data"][0])-2.7,
        north=float(radar.latitude["data"][0])+2.7,
    )
    add_mpl_cities(ax,radar_region,font_size=8)

    if adapt_labels is not None:
        try:
            labels = np.asarray(adapt_labels)
            gate_lat, gate_lon, _ = radar.get_gate_lat_lon_alt(sweep)
            if labels.shape == gate_lat.shape and np.nanmax(labels) > 0:
                levels = np.arange(0.5, int(np.nanmax(labels)) + 0.5, 1.0)
                ax.contour(
                    gate_lon, gate_lat, labels,
                    levels=levels,
                    colors="white",
                    linewidths=1.4,
                    transform=ccrs.PlateCarree(),
                )
                if show_adapt_ids:
                    for object_id in range(1, int(np.nanmax(labels)) + 1):
                        mask = labels == object_id
                        if not np.any(mask):
                            continue
                        lon0 = float(np.nanmean(gate_lon[mask]))
                        lat0 = float(np.nanmean(gate_lat[mask]))
                        ax.text(
                            lon0, lat0, f"T{object_id}",
                            transform=ccrs.PlateCarree(),
                            ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white",
                            bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.55, edgecolor="none"),
                        )
        except Exception:
            # ADAPT overlays are optional and should never prevent the radar plot.
            pass

    fixed_angle = float(
        radar.fixed_angle["data"][sweep]
    )

    radar_name = (
        radar.metadata.get("instrument_name")
        or "NEXRAD"
    )

    ax.set_title(
        f"{radar_name} Reflectivity\n"
        f"Sweep {sweep} • {fixed_angle:.1f}° elevation"
    )

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


def radar_sweep_summary(radar):
    """
    Return a small table describing available elevation sweeps.
    """
    import pandas as pd

    rows = []

    for sweep in range(radar.nsweeps):
        start = int(
            radar.sweep_start_ray_index["data"][sweep]
        )
        end = int(
            radar.sweep_end_ray_index["data"][sweep]
        )

        rows.append(
            {
                "sweep": sweep,
                "elevation_deg": float(
                    radar.fixed_angle["data"][sweep]
                ),
                "rays": end - start + 1,
            }
        )

    return pd.DataFrame(rows)

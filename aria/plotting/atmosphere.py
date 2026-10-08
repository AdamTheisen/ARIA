from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from aria.plotting.cities import add_mpl_cities
from aria.plotting.state_floor import add_state_floor
import pandas as pd
import plotly.graph_objects as go

from aria.trajectory_match import lonlat_to_xy_km


def make_3d_temperature_figure(
    atmosphere_ds,
    region,
    *,
    profiles=None,
    trajectories=None,
    show_volume=False,
    show_isosurfaces=True,
    show_level_surfaces=True,
    confidence_threshold=0.15,
):
    """
    Interactive 3-D atmospheric temperature figure.

    Stacked horizontal temperature surfaces are enabled by default because they
    remain visually useful even when the observational 3-D field is too sparse
    for closed isosurfaces.
    """
    lon0 = (region.west + region.east) / 2.0
    lat0 = (region.south + region.north) / 2.0

    lon = atmosphere_ds["longitude"].values
    lat = atmosphere_ds["latitude"].values
    altitude = atmosphere_ds["altitude_km"].values
    temperature = atmosphere_ds["air_temperature"].values

    confidence = (
        atmosphere_ds["analysis_confidence"].values
        if "analysis_confidence" in atmosphere_ds
        else np.ones_like(temperature)
    )

    # Mask only very-low-confidence voxels; a lenient threshold gives the 3-D
    # renderer enough continuity to form surfaces.
    masked_temperature = np.where(
        confidence >= confidence_threshold,
        temperature,
        np.nan,
    )

    lon3, lat3, z3 = np.meshgrid(
        lon,
        lat,
        altitude,
        indexing="xy",
    )

    temp3 = np.transpose(masked_temperature, (1, 2, 0))
    x3, y3 = lonlat_to_xy_km(
        lon3,
        lat3,
        lon0,
        lat0,
    )

    finite = temp3[np.isfinite(temp3)]
    fig = go.Figure()

    if finite.size:
        tmin = float(np.nanpercentile(finite, 2))
        tmax = float(np.nanpercentile(finite, 98))

        if show_volume:
            fig.add_trace(
                go.Volume(
                    x=x3.ravel(),
                    y=y3.ravel(),
                    z=z3.ravel(),
                    value=temp3.ravel(),
                    isomin=tmin,
                    isomax=tmax,
                    opacity=0.035,
                    surface_count=15,
                    colorscale="RdBu_r",
                    colorbar=dict(title="Temperature (°C)"),
                    caps=dict(
                        x_show=False,
                        y_show=False,
                        z_show=False,
                    ),
                    name="Temperature volume",
                )
            )

        if show_isosurfaces:
            # Use quantiles rather than fixed thresholds. This guarantees that
            # requested isosurfaces fall inside the actual analyzed range.
            iso_values = np.unique(
                np.round(
                    np.nanpercentile(
                        finite,
                        [15, 30, 45, 60, 75, 90],
                    ),
                    1,
                )
            )

            for level in iso_values:
                fig.add_trace(
                    go.Isosurface(
                        x=x3.ravel(),
                        y=y3.ravel(),
                        z=z3.ravel(),
                        value=temp3.ravel(),
                        isomin=float(level - 0.75),
                        isomax=float(level + 0.75),
                        surface_count=1,
                        opacity=0.22,
                        colorscale="RdBu_r",
                        cmin=tmin,
                        cmax=tmax,
                        showscale=False,
                        caps=dict(
                            x_show=False,
                            y_show=False,
                            z_show=False,
                        ),
                        name=f"{level:g} °C isosurface",
                    )
                )

        if show_level_surfaces:
            # Stacked horizontal surfaces are the most robust 3-D representation
            # for a sounding-based analysis. Use every other altitude level.
            lon2d, lat2d = np.meshgrid(lon, lat)
            x2d, y2d = lonlat_to_xy_km(
                lon2d,
                lat2d,
                lon0,
                lat0,
            )

            for zi in range(0, len(altitude), 2):
                field = masked_temperature[zi]
                if not np.isfinite(field).any():
                    continue

                zsurf = np.full_like(field, altitude[zi], dtype=float)

                fig.add_trace(
                    go.Surface(
                        x=x2d,
                        y=y2d,
                        z=zsurf,
                        surfacecolor=field,
                        colorscale="RdBu_r",
                        cmin=tmin,
                        cmax=tmax,
                        opacity=0.58,
                        showscale=(zi == 0),
                        colorbar=(
                            dict(title="Temperature (°C)")
                            if zi == 0
                            else None
                        ),
                        name=f"{altitude[zi]:g} km temperature",
                        hovertemplate=(
                            f"Altitude {altitude[zi]:g} km<br>"
                            "T %{surfacecolor:.1f} °C"
                            "<extra></extra>"
                        ),
                    )
                )

    if profiles is not None and not profiles.empty:
        for station, sdf in profiles.groupby(
            "station_id", dropna=True
        ):
            sdf = sdf.dropna(
                subset=[
                    "plot_longitude",
                    "plot_latitude",
                    "height_m",
                ]
            ).sort_values("height_m")

            if len(sdf) < 2:
                continue

            x, y = lonlat_to_xy_km(
                sdf["plot_longitude"],
                sdf["plot_latitude"],
                lon0,
                lat0,
            )

            matched = sdf["matched_serial"].notna().any()

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=sdf["height_m"] / 1000.0,
                    mode="lines",
                    line=dict(
                        width=5 if matched else 2,
                        dash="solid" if matched else "dash",
                    ),
                    name=(
                        f"{station} matched profile"
                        if matched
                        else f"{station} trajectory unavailable"
                    ),
                )
            )

    if trajectories is not None and not trajectories.empty:
        for serial, sdf in trajectories.groupby("serial"):
            sdf = sdf.dropna(
                subset=[
                    "longitude",
                    "latitude",
                    "altitude_m",
                ]
            ).sort_values("time")

            if len(sdf) < 3:
                continue

            x, y = lonlat_to_xy_km(
                sdf["longitude"],
                sdf["latitude"],
                lon0,
                lat0,
            )

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=sdf["altitude_m"] / 1000.0,
                    mode="lines",
                    line=dict(width=2),
                    name=f"GPS {serial}",
                    visible="legendonly",
                )
            )

    corner_lon = np.array(
        [region.west, region.east, region.west, region.east]
    )
    corner_lat = np.array(
        [region.south, region.south, region.north, region.north]
    )

    corner_x, corner_y = lonlat_to_xy_km(
        corner_lon,
        corner_lat,
        lon0,
        lat0,
    )

    fig.update_layout(
        title="GPGL 3-D Atmospheric Temperature",
        scene=dict(
            xaxis_title="East / West (km)",
            yaxis_title="North / South (km)",
            zaxis_title="Altitude (km MSL)",
            xaxis=dict(
                range=[
                    float(np.nanmin(corner_x)),
                    float(np.nanmax(corner_x)),
                ]
            ),
            yaxis=dict(
                range=[
                    float(np.nanmin(corner_y)),
                    float(np.nanmax(corner_y)),
                ]
            ),
            zaxis=dict(
                range=[
                    0,
                    float(max(18, np.nanmax(altitude))),
                ]
            ),
            aspectmode="data",
        ),
        height=850,
    )

    add_state_floor(fig,0.0,lon0=lon0,lat0=lat0,west=region.west,east=region.east,south=region.south,north=region.north,color="black",width=3)
    return fig


def plot_temperature_slice(
    atmosphere_ds,
    altitude_km,
    region,
    *,
    output_path=None,
    confidence_threshold=0.0,
):
    field = atmosphere_ds["air_temperature"].sel(
        altitude_km=altitude_km,
        method="nearest",
    )

    actual_altitude = float(field["altitude_km"].values)

    confidence = (
        atmosphere_ds["analysis_confidence"].sel(
            altitude_km=actual_altitude
        )
        if "analysis_confidence" in atmosphere_ds
        else None
    )

    plot_field = field.copy()

    if confidence is not None and confidence_threshold > 0:
        plot_field = plot_field.where(
            confidence >= confidence_threshold
        )

    fig = plt.figure(figsize=(11, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [
            region.west,
            region.east,
            region.south,
            region.north,
        ],
        ccrs.PlateCarree(),
    )

    mesh = ax.pcolormesh(
        atmosphere_ds["longitude"],
        atmosphere_ds["latitude"],
        plot_field,
        cmap="coolwarm",
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor="none",
        edgecolor="0.3",
    )
    ax.add_feature(
        cfeature.STATES.with_scale("50m"),
        linewidth=0.5,
    )
    ax.coastlines(
        resolution="50m",
        linewidth=0.6,
    )
    add_mpl_cities(ax,region,font_size=8)

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        pad=0.02,
    )
    cbar.set_label("Air Temperature (°C)")

    if confidence is not None:
        # Show where the field is weakly constrained without blanking it.
        low = confidence.values < 0.25
        if np.any(low):
            ax.contourf(
                atmosphere_ds["longitude"],
                atmosphere_ds["latitude"],
                low.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["..."],
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

    ax.set_title(
        f"Atmospheric Temperature — {actual_altitude:g} km MSL\n"
        "Dotted areas indicate lower observational constraint"
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



def make_3d_layer_explorer(
    atmosphere_ds,
    region,
    *,
    profiles=None,
    selected_altitudes=None,
):
    """
    Easier-to-read 3-D view using only a small number of horizontal temperature
    planes plus optional sounding profiles.

    This avoids the visual clutter and occlusion of full volume rendering.
    """
    lon0 = (region.west + region.east) / 2.0
    lat0 = (region.south + region.north) / 2.0

    lon = atmosphere_ds["longitude"].values
    lat = atmosphere_ds["latitude"].values
    altitude = atmosphere_ds["altitude_km"].values
    temp = atmosphere_ds["air_temperature"].values

    finite = temp[np.isfinite(temp)]
    if finite.size:
        tmin = float(np.nanpercentile(finite, 2))
        tmax = float(np.nanpercentile(finite, 98))
    else:
        tmin, tmax = -40.0, 30.0

    if selected_altitudes is None:
        # Four representative layers by default.
        targets = [1, 3, 6, 10]
    else:
        targets = selected_altitudes

    lon2d, lat2d = np.meshgrid(lon, lat)
    x2d, y2d = lonlat_to_xy_km(
        lon2d,
        lat2d,
        lon0,
        lat0,
    )

    fig = go.Figure()

    for target in targets:
        idx = int(np.nanargmin(np.abs(altitude - target)))
        z = float(altitude[idx])
        field = temp[idx]

        if not np.isfinite(field).any():
            continue

        fig.add_trace(
            go.Surface(
                x=x2d,
                y=y2d,
                z=np.full_like(field, z, dtype=float),
                surfacecolor=field,
                colorscale="RdBu_r",
                cmin=tmin,
                cmax=tmax,
                opacity=0.78,
                showscale=(len(fig.data) == 0),
                colorbar=(
                    dict(title="Temperature (°C)")
                    if len(fig.data) == 0
                    else None
                ),
                name=f"{z:g} km",
                hovertemplate=(
                    f"Altitude {z:g} km<br>"
                    "T %{surfacecolor:.1f} °C"
                    "<extra></extra>"
                ),
            )
        )

    if profiles is not None and not profiles.empty:
        for station, sdf in profiles.groupby(
            "station_id", dropna=True
        ):
            sdf = sdf.dropna(
                subset=[
                    "plot_longitude",
                    "plot_latitude",
                    "height_m",
                ]
            ).sort_values("height_m")

            if len(sdf) < 2:
                continue

            x, y = lonlat_to_xy_km(
                sdf["plot_longitude"],
                sdf["plot_latitude"],
                lon0,
                lat0,
            )

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=sdf["height_m"] / 1000.0,
                    mode="lines",
                    line=dict(width=3),
                    name=f"{station} sounding",
                )
            )

    corner_lon = np.array(
        [region.west, region.east, region.west, region.east]
    )
    corner_lat = np.array(
        [region.south, region.south, region.north, region.north]
    )
    cx, cy = lonlat_to_xy_km(
        corner_lon,
        corner_lat,
        lon0,
        lat0,
    )

    fig.update_layout(
        title="GPGL Atmospheric Temperature Layer Explorer",
        scene=dict(
            xaxis_title="East / West (km)",
            yaxis_title="North / South (km)",
            zaxis_title="Altitude (km MSL)",
            xaxis=dict(
                range=[
                    float(np.nanmin(cx)),
                    float(np.nanmax(cx)),
                ]
            ),
            yaxis=dict(
                range=[
                    float(np.nanmin(cy)),
                    float(np.nanmax(cy)),
                ]
            ),
            zaxis=dict(range=[0, 16]),
            aspectmode="manual",
            aspectratio=dict(x=1.6, y=1.0, z=0.7),
            camera=dict(
                eye=dict(x=1.15, y=-1.75, z=1.05), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=-0.05)
            ),
        ),
        height=850,
    )

    add_state_floor(fig,0.0,lon0=lon0,lat0=lat0,west=region.west,east=region.east,south=region.south,north=region.north,color="black",width=3)
    return fig



def make_3d_slice_explorer(
    atmosphere_ds,
    region,
    *,
    altitude_km=3.0,
    latitude=None,
    longitude=None,
    profiles=None,
):
    """
    3-D orthogonal-slice explorer.

    Instead of stacking flat horizontal planes, this view shows:
      * one horizontal temperature slice;
      * one north-south vertical curtain;
      * one east-west vertical curtain.

    The vertical curtains show actual thermal variation with altitude and are
    generally much easier to interpret than a stack of constant-height planes.
    """
    lon0 = (region.west + region.east) / 2.0
    lat0 = (region.south + region.north) / 2.0

    lon = atmosphere_ds["longitude"].values
    lat = atmosphere_ds["latitude"].values
    altitude = atmosphere_ds["altitude_km"].values
    temp = atmosphere_ds["air_temperature"].values

    if latitude is None:
        latitude = float(np.nanmean(lat))
    if longitude is None:
        longitude = float(np.nanmean(lon))

    zi = int(np.nanargmin(np.abs(altitude - altitude_km)))
    yi = int(np.nanargmin(np.abs(lat - latitude)))
    xi = int(np.nanargmin(np.abs(lon - longitude)))

    actual_alt = float(altitude[zi])
    actual_lat = float(lat[yi])
    actual_lon = float(lon[xi])

    finite = temp[np.isfinite(temp)]
    if finite.size:
        tmin = float(np.nanpercentile(finite, 2))
        tmax = float(np.nanpercentile(finite, 98))
    else:
        tmin, tmax = -50.0, 30.0

    fig = go.Figure()

    # Horizontal slice.
    lon2d, lat2d = np.meshgrid(lon, lat)
    x2d, y2d = lonlat_to_xy_km(
        lon2d,
        lat2d,
        lon0,
        lat0,
    )

    fig.add_trace(
        go.Surface(
            x=x2d,
            y=y2d,
            z=np.full_like(
                temp[zi],
                actual_alt,
                dtype=float,
            ),
            surfacecolor=temp[zi],
            colorscale="RdBu_r",
            cmin=tmin,
            cmax=tmax,
            opacity=0.78,
            colorbar=dict(title="Temperature (°C)"),
            name=f"{actual_alt:g} km horizontal",
            showscale=True,
        )
    )

    # North-south vertical curtain at selected longitude.
    lat_grid, alt_grid = np.meshgrid(
        lat,
        altitude,
    )
    lon_const = np.full_like(
        lat_grid,
        actual_lon,
        dtype=float,
    )
    x_ns, y_ns = lonlat_to_xy_km(
        lon_const,
        lat_grid,
        lon0,
        lat0,
    )

    fig.add_trace(
        go.Surface(
            x=x_ns,
            y=y_ns,
            z=alt_grid,
            surfacecolor=temp[:, :, xi],
            colorscale="RdBu_r",
            cmin=tmin,
            cmax=tmax,
            opacity=0.88,
            showscale=False,
            name=f"N-S curtain @ {actual_lon:.2f}°",
        )
    )

    # East-west vertical curtain at selected latitude.
    lon_grid, alt_grid2 = np.meshgrid(
        lon,
        altitude,
    )
    lat_const = np.full_like(
        lon_grid,
        actual_lat,
        dtype=float,
    )
    x_ew, y_ew = lonlat_to_xy_km(
        lon_grid,
        lat_const,
        lon0,
        lat0,
    )

    fig.add_trace(
        go.Surface(
            x=x_ew,
            y=y_ew,
            z=alt_grid2,
            surfacecolor=temp[:, yi, :],
            colorscale="RdBu_r",
            cmin=tmin,
            cmax=tmax,
            opacity=0.88,
            showscale=False,
            name=f"E-W curtain @ {actual_lat:.2f}°",
        )
    )

    if profiles is not None and not profiles.empty:
        for station, sdf in profiles.groupby(
            "station_id",
            dropna=True,
        ):
            sdf = sdf.dropna(
                subset=[
                    "plot_longitude",
                    "plot_latitude",
                    "height_m",
                ]
            ).sort_values("height_m")

            if len(sdf) < 2:
                continue

            x, y = lonlat_to_xy_km(
                sdf["plot_longitude"],
                sdf["plot_latitude"],
                lon0,
                lat0,
            )

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=sdf["height_m"] / 1000.0,
                    mode="lines",
                    line=dict(width=3),
                    name=f"{station} sounding",
                )
            )

    fig.update_layout(
        title=(
            "GPGL 3-D Temperature Slice Explorer"
            f"<br>{actual_alt:g} km horizontal • "
            f"{actual_lat:.2f}°N • {abs(actual_lon):.2f}°W curtains"
        ),
        scene=dict(
            xaxis_title="East / West (km)",
            yaxis_title="North / South (km)",
            zaxis_title="Altitude (km MSL)",
            xaxis=dict(range=[float(np.nanmin(lonlat_to_xy_km(np.array([region.west,region.east]),np.array([lat0,lat0]),lon0,lat0)[0])),float(np.nanmax(lonlat_to_xy_km(np.array([region.west,region.east]),np.array([lat0,lat0]),lon0,lat0)[0]))]),
            yaxis=dict(range=[float(np.nanmin(lonlat_to_xy_km(np.array([lon0,lon0]),np.array([region.south,region.north]),lon0,lat0)[1])),float(np.nanmax(lonlat_to_xy_km(np.array([lon0,lon0]),np.array([region.south,region.north]),lon0,lat0)[1]))]),
            zaxis=dict(range=[0, 16]),
            aspectmode="manual",
            aspectratio=dict(
                x=1.6,
                y=1.0,
                z=0.85,
            ),
            camera=dict(
                eye=dict(x=1.15, y=-1.75, z=1.05), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=-0.05)
            ),
        ),
        height=850,
    )

    add_state_floor(fig,0.0,lon0=lon0,lat0=lat0,west=region.west,east=region.east,south=region.south,north=region.north,color="black",width=3)
    return fig



VARIABLE_STYLE = {
    "air_temperature": {"label":"Air Temperature","units":"°C","cmap":"coolwarm","profile":"air_temperature_c"},
    "dew_point_temperature": {"label":"Dew Point","units":"°C","cmap":"BrBG","profile":"dew_point_temperature_c"},
    "wind_speed": {"label":"Wind Speed","units":"kt","cmap":"viridis","profile":"wind_speed_kt"},
    "u_wind": {"label":"U Wind","units":"kt","cmap":"coolwarm","profile":"u_wind_kt"},
    "v_wind": {"label":"V Wind","units":"kt","cmap":"coolwarm","profile":"v_wind_kt"},
}

def plot_atmosphere_slice(atmosphere_ds, variable, altitude_km, region,
                          *, cmap=None, vmin=None, vmax=None, output_path=None,
                          wind_barbs=False, barb_skip=4):
    style=VARIABLE_STYLE.get(variable,{"label":variable,"units":"","cmap":"viridis"})
    if variable=="wind_speed" and "u_wind" in atmosphere_ds and "v_wind" in atmosphere_ds:
        field=np.hypot(atmosphere_ds["u_wind"],atmosphere_ds["v_wind"]).sel(altitude_km=altitude_km,method="nearest")
        field.attrs.update(long_name="Wind Speed", units="kt")
    else:
        field=atmosphere_ds[variable].sel(altitude_km=altitude_km,method="nearest")
    actual=float(field.altitude_km.values); finite=field.values[np.isfinite(field.values)]
    if cmap is None: cmap=style["cmap"]
    if finite.size:
        if vmin is None: vmin=0.0 if variable=="wind_speed" else float(np.nanpercentile(finite,2))
        if vmax is None: vmax=float(np.nanpercentile(finite,98))
    fig=plt.figure(figsize=(11,7)); ax=plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([region.west,region.east,region.south,region.north],ccrs.PlateCarree())
    mesh=ax.pcolormesh(atmosphere_ds.longitude,atmosphere_ds.latitude,field,cmap=cmap,
                       vmin=vmin,vmax=vmax,shading="auto",transform=ccrs.PlateCarree())
    if wind_barbs and "u_wind" in atmosphere_ds and "v_wind" in atmosphere_ds:
        u=atmosphere_ds["u_wind"].sel(altitude_km=actual,method="nearest").values
        v=atmosphere_ds["v_wind"].sel(altitude_km=actual,method="nearest").values
        skip=max(1,int(barb_skip)); lon2,lat2=np.meshgrid(atmosphere_ds.longitude.values,atmosphere_ds.latitude.values)
        ax.barbs(lon2[::skip,::skip],lat2[::skip,::skip],u[::skip,::skip],v[::skip,::skip],
                 length=5,linewidth=0.5,color="black",transform=ccrs.PlateCarree(),zorder=7)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),facecolor="none",edgecolor="0.3")
    ax.add_feature(cfeature.STATES.with_scale("50m"),linewidth=0.5); ax.coastlines(resolution="50m",linewidth=0.6)
    add_mpl_cities(ax,region,font_size=8)
    fig.colorbar(mesh,ax=ax,pad=.02,label=f'{style["label"]} ({style["units"]})')
    ax.set_title(f'{style["label"]} — {actual:g} km MSL')
    if output_path:
        output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output_path,dpi=180,bbox_inches="tight")
    return fig

def make_3d_variable_slice_explorer(atmosphere_ds, region, *, variable="air_temperature",
                                    altitude_km=3.0, latitude=None, longitude=None,
                                    profiles=None, cmap=None, vmin=None, vmax=None):
    style=VARIABLE_STYLE.get(variable,{"label":variable,"units":"","cmap":"Viridis","profile":None})
    lon0=(region.west+region.east)/2; lat0=(region.south+region.north)/2
    lon=atmosphere_ds.longitude.values; lat=atmosphere_ds.latitude.values
    altitude=atmosphere_ds.altitude_km.values
    if variable=="wind_speed" and "u_wind" in atmosphere_ds and "v_wind" in atmosphere_ds:
        data=np.hypot(atmosphere_ds["u_wind"].values,atmosphere_ds["v_wind"].values)
    else:
        data=atmosphere_ds[variable].values
    latitude=float(np.nanmean(lat)) if latitude is None else latitude
    longitude=float(np.nanmean(lon)) if longitude is None else longitude
    zi=int(np.nanargmin(abs(altitude-altitude_km))); yi=int(np.nanargmin(abs(lat-latitude))); xi=int(np.nanargmin(abs(lon-longitude)))
    finite=data[np.isfinite(data)]
    if finite.size:
        vmin=(0.0 if variable=="wind_speed" else float(np.nanpercentile(finite,2))) if vmin is None else vmin
        vmax=float(np.nanpercentile(finite,98)) if vmax is None else vmax
    cmap=style["cmap"] if cmap is None else cmap
    # Plotly names differ from matplotlib; map common names.
    plotly_scale={"coolwarm":"RdBu_r","BrBG":"BrBG","viridis":"Viridis","plasma":"Plasma","turbo":"Turbo"}.get(cmap,cmap)
    fig=go.Figure(); lon2,lat2=np.meshgrid(lon,lat); x2,y2=lonlat_to_xy_km(lon2,lat2,lon0,lat0)
    fig.add_trace(go.Surface(x=x2,y=y2,z=np.full_like(data[zi],altitude[zi]),surfacecolor=data[zi],
                             colorscale=plotly_scale,cmin=vmin,cmax=vmax,opacity=.78,
                             colorbar=dict(title=f'{style["label"]} ({style["units"]})'),name="Horizontal"))
    latg,altg=np.meshgrid(lat,altitude); xns,yns=lonlat_to_xy_km(np.full_like(latg,lon[xi]),latg,lon0,lat0)
    fig.add_trace(go.Surface(x=xns,y=yns,z=altg,surfacecolor=data[:,:,xi],colorscale=plotly_scale,cmin=vmin,cmax=vmax,opacity=.88,showscale=False,name="N-S"))
    long,altg2=np.meshgrid(lon,altitude); xew,yew=lonlat_to_xy_km(long,np.full_like(long,lat[yi]),lon0,lat0)
    fig.add_trace(go.Surface(x=xew,y=yew,z=altg2,surfacecolor=data[:,yi,:],colorscale=plotly_scale,cmin=vmin,cmax=vmax,opacity=.88,showscale=False,name="E-W"))
    pvar=style.get("profile")
    if profiles is not None and not profiles.empty and pvar in profiles:
        for station,sdf in profiles.groupby("station_id",dropna=True):
            sdf=sdf.dropna(subset=["plot_longitude","plot_latitude","height_m",pvar]).sort_values("height_m")
            if len(sdf)<2: continue
            x,y=lonlat_to_xy_km(sdf.plot_longitude,sdf.plot_latitude,lon0,lat0)
            vals=pd.to_numeric(sdf[pvar],errors="coerce").to_numpy(float)
            # colored markers along a line; markers communicate selected variable.
            fig.add_trace(go.Scatter3d(x=x,y=y,z=sdf.height_m/1000,mode="lines+markers",
                line=dict(width=2,color="rgba(50,50,50,0.45)"),
                marker=dict(size=3,color=vals,colorscale=plotly_scale,cmin=vmin,cmax=vmax,showscale=False),
                name=f"{station} {style['label']}",
                hovertemplate=f"{station}<br>Alt %{{z:.2f}} km<br>{style['label']} %{{marker.color:.1f}} {style['units']}<extra></extra>"))
    corner_lon=np.array([region.west,region.east,region.west,region.east],float)
    corner_lat=np.array([region.south,region.south,region.north,region.north],float)
    cx,cy=lonlat_to_xy_km(corner_lon,corner_lat,lon0,lat0)
    fig.update_layout(title=f'ARIA 3-D {style["label"]} Slice Explorer',
                      scene=dict(xaxis_title="East/West (km)",yaxis_title="North/South (km)",zaxis_title="Altitude (km MSL)",
                                 xaxis=dict(range=[float(np.nanmin(cx)),float(np.nanmax(cx))]),
                                 yaxis=dict(range=[float(np.nanmin(cy)),float(np.nanmax(cy))]),
                                 zaxis=dict(range=[0,16]),aspectmode="manual",aspectratio=dict(x=1.6,y=1,z=.85),
                                 camera=dict(eye=dict(x=1.15,y=-1.75,z=1.05),
                                             up=dict(x=0,y=0,z=1),
                                             center=dict(x=0,y=0,z=-0.05))),
                      height=850)
    add_state_floor(fig,float(np.nanmin(altitude)),lon0=lon0,lat0=lat0,west=region.west,east=region.east,south=region.south,north=region.north,color="black",width=3)
    return fig

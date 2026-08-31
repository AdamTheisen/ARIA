from __future__ import annotations

from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
from aria.storm_objects import extract_object_boundaries


MODEL_STYLE = {
    "air_temperature_2m": ("2-m Air Temperature", "°C", "coolwarm"),
    "dew_point_temperature_2m": ("2-m Dew Point", "°C", "BrBG"),
    "wind_speed_10m": ("10-m Wind Speed", "m s$^{-1}$", "viridis"),
    "composite_reflectivity": ("Composite Reflectivity", "dBZ", "turbo"),
    "precipitation": ("Accumulated Precipitation", "kg m$^{-2}$", "viridis"),
    "precipitation_rate": ("Precipitation Rate", "kg m$^{-2}$ s$^{-1}$", "viridis"),
    "reflectivity_1km": ("1-km AGL Reflectivity", "dBZ", "turbo"),
    "categorical_rain": ("Categorical Rain", "1", "viridis"),
    "categorical_snow": ("Categorical Snow", "1", "viridis"),
    "categorical_freezing_rain": ("Categorical Freezing Rain", "1", "viridis"),
    "categorical_ice_pellets": ("Categorical Ice Pellets", "1", "viridis"),
    "surface_pressure": ("Surface Pressure", "Pa", "viridis"),
    "air_temperature": ("Air Temperature", "°C", "coolwarm"),
    "dew_point_temperature": ("Dew Point", "°C", "BrBG"),
    "wind_speed": ("Wind Speed", "m s$^{-1}$", "viridis"),
    "u_wind": ("U Wind", "m s$^{-1}$", "coolwarm"),
    "v_wind": ("V Wind", "m s$^{-1}$", "coolwarm"),
    "vertical_velocity": ("Pressure Vertical Velocity", "Pa s$^{-1}$", "coolwarm"),
    "geopotential_height": ("Geopotential Height", "gpm", "viridis"),
}


def _latlon(ds):
    lat = ds["latitude"] if "latitude" in ds.coords else ds["lat"]
    lon = ds["longitude"] if "longitude" in ds.coords else ds["lon"]
    return lat, lon


def _map_base(ax, region):
    ax.set_extent(
        [region.west, region.east, region.south, region.north],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="none", edgecolor="0.35")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.45)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    ax.coastlines(resolution="50m", linewidth=0.5)


def plot_model_field(
    ds,
    variable,
    region,
    *,
    pressure_hpa=None,
    cmap=None,
    vmin=None,
    vmax=None,
    overlay=None,
    output_path=None,
):
    da = ds[variable]
    if pressure_hpa is not None and "pressure_hpa" in da.dims:
        da = da.sel(pressure_hpa=pressure_hpa, method="nearest").squeeze(drop=True)

    da = da.squeeze(drop=True)
    lat, lon = _latlon(ds)
    label, units, default_cmap = MODEL_STYLE.get(
        variable, (variable, da.attrs.get("units", ""), "viridis")
    )
    cmap = cmap or default_cmap

    finite = da.values[np.isfinite(da.values)]
    if finite.size:
        if vmin is None:
            vmin = float(np.nanpercentile(finite, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(finite, 98))

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    _map_base(ax, region)

    mesh = ax.pcolormesh(
        lon, lat, da,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    if overlay is not None:
        oda, ocmap, omin, omax = overlay
        ax.contour(
            lon, lat, oda,
            levels=np.linspace(omin, omax, 8),
            cmap=ocmap,
            linewidths=0.7,
            transform=ccrs.PlateCarree(),
        )

    fig.colorbar(mesh, ax=ax, pad=0.02, label=f"{label} ({units})")
    title = f"{ds.attrs.get('model','MODEL').upper()} — {label}"
    if pressure_hpa is not None and "pressure_hpa" in ds.coords:
        level = float(da.coords.get("pressure_hpa", pressure_hpa))
        title += f" — {level:g} hPa"
    title += (
        f"\nInit {ds.attrs.get('initialization_time','?')}  "
        f"F{int(ds.attrs.get('forecast_hour',0)):02d}  "
        f"Valid {ds.attrs.get('valid_time','?')}"
    )
    ax.set_title(title)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    return fig


def plot_model_cross_section(
    ds,
    variable,
    *,
    start_lat,
    start_lon,
    end_lat,
    end_lon,
    npoints=120,
    cmap=None,
    vmin=None,
    vmax=None,
):
    """
    Nearest-grid cross section through a pressure-level HRRR field.
    """
    da = ds[variable]
    if "pressure_hpa" not in da.dims:
        raise ValueError("Cross section requires a pressure-level model field.")

    lat, lon = _latlon(ds)
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError("Cross section currently expects a curvilinear 2-D model grid.")

    qlat = np.linspace(start_lat, end_lat, npoints)
    qlon = np.linspace(start_lon, end_lon, npoints)

    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([lat.values.ravel(), lon.values.ravel()]))
    _, idx = tree.query(np.column_stack([qlat, qlon]))

    ydim, xdim = lat.dims
    yi, xi = np.unravel_index(idx, lat.shape)

    values = np.stack([
        da.isel({ydim: int(y), xdim: int(x)}).values
        for y, x in zip(yi, xi)
    ], axis=1)

    pressure = da["pressure_hpa"].values
    label, units, default_cmap = MODEL_STYLE.get(
        variable, (variable, da.attrs.get("units", ""), "viridis")
    )
    cmap = cmap or default_cmap

    finite = values[np.isfinite(values)]
    if finite.size:
        if vmin is None:
            vmin = float(np.nanpercentile(finite, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(finite, 98))

    distance = np.linspace(0, 1, npoints)
    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(
        distance,
        pressure,
        values,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Cross-section fraction (start → end)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title(
        f"{ds.attrs.get('model','MODEL').upper()} {label} Cross Section\n"
        f"({start_lat:.2f}, {start_lon:.2f}) → ({end_lat:.2f}, {end_lon:.2f})"
    )
    fig.colorbar(mesh, ax=ax, label=f"{label} ({units})")
    fig.tight_layout()
    return fig



def overlay_regional_radar(fig, radar_ds, *, altitude_km=2.0, min_dbz=10.0):
    """
    Add semi-transparent observed regional NEXRAD reflectivity to an existing
    model map figure.
    """
    ax = fig.axes[0]
    field = radar_ds["reflectivity"].sel(
        altitude_km=altitude_km,
        method="nearest",
    )
    values = np.where(field.values >= min_dbz, field.values, np.nan)
    mesh = ax.pcolormesh(
        radar_ds["longitude"],
        radar_ds["latitude"],
        values,
        shading="auto",
        cmap="turbo",
        vmin=min_dbz,
        vmax=70,
        alpha=0.38,
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    # Separate compact colorbar for the observed radar overlay.
    fig.colorbar(
        mesh,
        ax=ax,
        pad=0.08,
        fraction=0.035,
        label="Observed NEXRAD (dBZ)",
    )
    return fig



def plot_three_panel_comparison(
    comparison,
    region,
    *,
    title=None,
    field_cmap="coolwarm",
    difference_cmap="RdBu_r",
    field_vmin=None,
    field_vmax=None,
    difference_limit=None,
):
    """Model | Observation | Model-minus-observation comparison."""
    lat=comparison["latitude"]
    lon=comparison["longitude"]
    model=comparison["model"]
    obs=comparison["observation"]
    diff=comparison["difference"]

    field_values=np.concatenate([
        model.values[np.isfinite(model.values)],
        obs.values[np.isfinite(obs.values)],
    ])
    if field_values.size:
        if field_vmin is None:
            field_vmin=float(np.nanpercentile(field_values,2))
        if field_vmax is None:
            field_vmax=float(np.nanpercentile(field_values,98))

    finite_diff=np.abs(diff.values[np.isfinite(diff.values)])
    if difference_limit is None:
        difference_limit=(
            float(np.nanpercentile(finite_diff,98))
            if finite_diff.size else 1.0
        )
    difference_limit=max(float(difference_limit),1e-6)

    fig,axes=plt.subplots(
        1,3,figsize=(18,6),
        subplot_kw={"projection":ccrs.PlateCarree()},
    )
    for ax in axes:
        _map_base(ax,region)

    m0=axes[0].pcolormesh(
        lon,lat,model,shading="auto",cmap=field_cmap,
        vmin=field_vmin,vmax=field_vmax,
        transform=ccrs.PlateCarree(),
    )
    axes[0].set_title("Model")
    axes[1].pcolormesh(
        lon,lat,obs,shading="auto",cmap=field_cmap,
        vmin=field_vmin,vmax=field_vmax,
        transform=ccrs.PlateCarree(),
    )
    axes[1].set_title("Observation")
    md=axes[2].pcolormesh(
        lon,lat,diff,shading="auto",cmap=difference_cmap,
        vmin=-difference_limit,vmax=difference_limit,
        transform=ccrs.PlateCarree(),
    )
    axes[2].set_title("Model − Observation")

    units=comparison.attrs.get("units","")
    fig.colorbar(m0,ax=axes[:2],orientation="horizontal",fraction=.05,pad=.06,label=units)
    fig.colorbar(md,ax=axes[2],orientation="horizontal",fraction=.05,pad=.06,label=f"Difference ({units})")
    fig.suptitle(title or comparison.attrs.get("comparison","Model / Observation Comparison"))
    return fig


def plot_radar_three_panel(
    comparison,
    region,
    *,
    difference_limit=30.0,
):
    """HRRR reflectivity | NEXRAD | dBZ difference."""
    lat=comparison.latitude
    lon=comparison.longitude
    fig,axes=plt.subplots(
        1,3,figsize=(18,6),
        subplot_kw={"projection":ccrs.PlateCarree()},
    )
    for ax in axes:
        _map_base(ax,region)

    m=axes[0].pcolormesh(
        lon,lat,comparison.model,shading="auto",
        cmap="turbo",vmin=-10,vmax=70,
        transform=ccrs.PlateCarree(),
    )
    axes[0].set_title("HRRR Simulated Reflectivity")
    axes[1].pcolormesh(
        lon,lat,comparison.observation,shading="auto",
        cmap="turbo",vmin=-10,vmax=70,
        transform=ccrs.PlateCarree(),
    )
    axes[1].set_title("Regional NEXRAD")
    d=axes[2].pcolormesh(
        lon,lat,comparison.difference,shading="auto",
        cmap="RdBu_r",vmin=-difference_limit,vmax=difference_limit,
        transform=ccrs.PlateCarree(),
    )
    axes[2].set_title("HRRR − NEXRAD")

    fig.colorbar(m,ax=axes[:2],orientation="horizontal",fraction=.05,pad=.06,label="Reflectivity (dBZ)")
    fig.colorbar(d,ax=axes[2],orientation="horizontal",fraction=.05,pad=.06,label="Difference (dBZ)")
    fig.suptitle("HRRR / NEXRAD Reflectivity Comparison")
    return fig


def plot_profile_comparison(comparison, label, units):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,7),sharey=True)
    ax1.plot(comparison.observation,comparison.pressure_hpa,label="Radiosonde",marker=".")
    ax1.plot(comparison.model,comparison.pressure_hpa,label="HRRR",marker=".")
    ax1.invert_yaxis()
    ax1.set_xlabel(f"{label} ({units})")
    ax1.set_ylabel("Pressure (hPa)")
    ax1.legend()
    ax1.grid(alpha=.25)

    ax2.plot(comparison.model_minus_observation,comparison.pressure_hpa,marker=".")
    ax2.axvline(0,linewidth=1)
    ax2.set_xlabel(f"HRRR − Sonde ({units})")
    ax2.grid(alpha=.25)
    fig.suptitle(f"HRRR vs Radiosonde — {label}")
    fig.tight_layout()
    return fig


def _interactive_state_lines(region):
    """Return robust Natural Earth U.S. state-boundary line coordinates.

    ``cfeature.STATES.geometries()`` can yield line geometries whose ``boundary``
    consists only of endpoints.  Read the admin-1 polygon shapefile instead and
    draw polygon boundaries, which works consistently for Plotly overlays.
    """
    lines = []
    try:
        shp = shpreader.natural_earth(
            resolution="50m",
            category="cultural",
            name="admin_1_states_provinces_lakes",
        )
        for record in shpreader.Reader(shp).records():
            attrs = record.attributes
            if attrs.get("adm0_a3") != "USA":
                continue
            geom = record.geometry
            if geom.bounds[2] < region.west or geom.bounds[0] > region.east:
                continue
            if geom.bounds[3] < region.south or geom.bounds[1] > region.north:
                continue

            polygons = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
            for poly in polygons:
                boundary = poly.boundary
                parts = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]
                for part in parts:
                    try:
                        xy = np.asarray(part.coords)
                    except Exception:
                        continue
                    if xy.ndim == 2 and xy.shape[0] > 1:
                        lines.append((xy[:, 0], xy[:, 1]))
    except Exception:
        # The raster remains usable if Natural Earth is unavailable on first
        # run; Cartopy may need network access once to cache the shapefile.
        pass
    return lines


def _geographic_y_scale(region):
    """Approximate geographic aspect ratio for lon/lat Cartesian Plotly axes.

    A degree of longitude is shorter than a degree of latitude away from the
    equator.  Plotly Heatmap uses ordinary Cartesian axes, so compensate using
    the cosine of the region's midpoint latitude.
    """
    midlat = 0.5 * (float(region.south) + float(region.north))
    coslat = float(np.cos(np.deg2rad(midlat)))
    return 1.0 / max(coslat, 0.2)


def plot_interactive_field(da, region, *, title, units="", zmin=None, zmax=None, colorscale="Turbo", state_color="rgba(20,20,20,.95)"):
    """Interactive regular-lat/lon field with state boundaries and pan/zoom."""
    import plotly.graph_objects as go
    lat=np.asarray(da.latitude.values); lon=np.asarray(da.longitude.values)
    if lat.ndim!=1 or lon.ndim!=1:
        raise ValueError("Interactive field requires regular 1-D latitude/longitude coordinates.")
    z=np.asarray(da.values,float)
    fig=go.Figure(go.Heatmap(
        x=lon,y=lat,z=z,zmin=zmin,zmax=zmax,colorscale=colorscale,
        colorbar=dict(title=units),
        hovertemplate=f"Lon %{{x:.2f}}<br>Lat %{{y:.2f}}<br>%{{z:.1f}} {units}<extra></extra>",
    ))
    for x,y in _interactive_state_lines(region):
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(color=state_color,width=1.6),hoverinfo="skip",showlegend=False))
    fig.update_layout(
        title=title,height=650,dragmode="zoom",
        xaxis=dict(title="Longitude",range=[region.west,region.east],constrain="domain"),
        yaxis=dict(
            title="Latitude",
            range=[region.south,region.north],
            scaleanchor="x",
            scaleratio=_geographic_y_scale(region),
            constrain="domain",
        ),
        margin=dict(l=50,r=35,t=70,b=45),
    )
    return fig


def plot_interactive_three_panel(
    comparison,
    region,
    *,
    radar=False,
    title=None,
    difference_limit=None,
):
    """Interactive synchronized Model | Observation | Difference maps.

    This is intended for regular latitude/longitude comparison grids such as
    GPGL surface analyses and MRMS. Pan/zoom on any panel is matched across all
    three panels.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    lat=np.asarray(comparison.latitude.values)
    lon=np.asarray(comparison.longitude.values)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Interactive comparison requires a regular 1-D latitude/longitude grid.")
    m=np.asarray(comparison.model.values,float)
    o=np.asarray(comparison.observation.values,float)
    d=np.asarray(comparison.difference.values,float)

    if radar:
        vmin,vmax=-30.0,70.0
        colorscale="Turbo"
        if difference_limit is None: difference_limit=30.0
        panel_titles=("HRRR simulated reflectivity","MRMS QC composite","HRRR − MRMS")
    else:
        vals=np.concatenate([m[np.isfinite(m)],o[np.isfinite(o)]])
        vmin=float(np.nanpercentile(vals,2)) if vals.size else 0.0
        vmax=float(np.nanpercentile(vals,98)) if vals.size else 1.0
        colorscale="RdBu_r" if (vals.size and np.nanmin(vals) < 0 < np.nanmax(vals)) else "Viridis"
        if difference_limit is None:
            finite=np.abs(d[np.isfinite(d)])
            difference_limit=float(np.nanpercentile(finite,98)) if finite.size else 1.0
        panel_titles=("Model","Observation","Model − Observation")

    fig=make_subplots(rows=1,cols=3,shared_yaxes=True,subplot_titles=panel_titles,horizontal_spacing=.035)
    hover="Lon %{x:.2f}<br>Lat %{y:.2f}<br>Value %{z:.1f}<extra></extra>"
    fig.add_trace(go.Heatmap(x=lon,y=lat,z=m,zmin=vmin,zmax=vmax,colorscale=colorscale,
                             colorbar=dict(title=comparison.attrs.get("units",""),x=.30),hovertemplate=hover),row=1,col=1)
    fig.add_trace(go.Heatmap(x=lon,y=lat,z=o,zmin=vmin,zmax=vmax,colorscale=colorscale,showscale=False,
                             hovertemplate=hover),row=1,col=2)
    fig.add_trace(go.Heatmap(x=lon,y=lat,z=d,zmin=-difference_limit,zmax=difference_limit,colorscale="RdBu_r",
                             colorbar=dict(title="Difference",x=1.01),hovertemplate=hover),row=1,col=3)

    for x,y in _interactive_state_lines(region):
        for col in (1,2,3):
            fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(color="rgba(20,20,20,.95)",width=1.4),hoverinfo="skip",showlegend=False),row=1,col=col)

    # Match x/y ranges so zooming one panel synchronizes the others.
    fig.update_xaxes(
        range=[region.west,region.east],
        matches="x",
        title_text="Longitude",
        constrain="domain",
    )
    geo_ratio = _geographic_y_scale(region)
    # Anchor each latitude axis to its own longitude axis so the three maps
    # retain a realistic geographic aspect while x/y zoom remains synchronized.
    for col, xref in ((1, "x"), (2, "x2"), (3, "x3")):
        fig.update_yaxes(
            range=[region.south,region.north],
            matches="y",
            title_text="Latitude" if col == 1 else None,
            scaleanchor=xref,
            scaleratio=geo_ratio,
            constrain="domain",
            row=1,
            col=col,
        )
    fig.update_layout(
        title=title or comparison.attrs.get("comparison","Model / Observation Comparison"),
        height=620,
        dragmode="zoom",
        margin=dict(l=45,r=45,t=85,b=40),
    )
    return fig


def plot_adapt_storm_objects(
    comparison,
    model_objects,
    observed_objects,
    matches,
    region,
    *,
    title="ADAPT storm-object comparison",
    model_labels=None,
    observed_labels=None,
):
    """Interactive HRRR/MRMS object view with ADAPT centroid/object extents."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    lat = np.asarray(comparison.latitude.values, float)
    lon = np.asarray(comparison.longitude.values, float)
    if lat.ndim == 2:
        y = lat[:, 0]
    else:
        y = lat
    if lon.ndim == 2:
        x = lon[0, :]
    else:
        x = lon

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("HRRR simulated reflectivity", "MRMS QC composite"),
        horizontal_spacing=0.06,
    )
    for col, name in ((1, "model"), (2, "observation")):
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=np.asarray(comparison[name].values, float),
                zmin=-30,
                zmax=70,
                colorscale="Turbo",
                colorbar=dict(title="dBZ") if col == 2 else None,
                showscale=col == 2,
                hovertemplate="Lon %{x:.2f}<br>Lat %{y:.2f}<br>%{z:.1f} dBZ<extra></extra>",
            ),
            row=1, col=col,
        )

    def add_objects(df, col, symbol):
        if df is None or df.empty:
            return
        custom = np.column_stack([
            df.area_km2.to_numpy(float),
            df.max_reflectivity_dbz.to_numpy(float),
            df.gridpoints.to_numpy(int),
        ])
        fig.add_trace(
            go.Scatter(
                x=df.centroid_longitude,
                y=df.centroid_latitude,
                mode="markers+text",
                text=df.object_id,
                textposition="top center",
                marker=dict(size=11, symbol=symbol, color="white", line=dict(color="black", width=1.5)),
                customdata=custom,
                hovertemplate=(
                    "%{text}<br>Lon %{x:.2f}<br>Lat %{y:.2f}"
                    "<br>Area %{customdata[0]:.0f} km²"
                    "<br>Max Z %{customdata[1]:.1f} dBZ"
                    "<br>Grid points %{customdata[2]:.0f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=col,
        )
        # Exact segmented outlines are added below from the ADAPT label masks.

    add_objects(model_objects, 1, "circle")
    add_objects(observed_objects, 2, "diamond")

    for labels, col in ((model_labels, 1), (observed_labels, 2)):
        if labels is None:
            continue
        for ident,coords in extract_object_boundaries(labels,x=x,y=y,simplify_stride=3).items():
            fig.add_trace(go.Scatter(
                x=coords[:,0],y=coords[:,1],mode="lines",
                line=dict(color="white",width=2),
                hovertemplate=f"Object {ident}<extra></extra>",showlegend=False),
                row=1,col=col)

    for xx, yy in _interactive_state_lines(region):
        for col in (1, 2):
            fig.add_trace(
                go.Scatter(
                    x=xx, y=yy, mode="lines",
                    line=dict(color="rgba(20,20,20,.95)", width=1.4),
                    hoverinfo="skip", showlegend=False,
                ),
                row=1, col=col,
            )

    fig.update_xaxes(
        range=[region.west, region.east], matches="x", title_text="Longitude", constrain="domain"
    )
    ratio = _geographic_y_scale(region)
    for col, xref in ((1, "x"), (2, "x2")):
        fig.update_yaxes(
            range=[region.south, region.north],
            matches="y",
            title_text="Latitude" if col == 1 else None,
            scaleanchor=xref,
            scaleratio=ratio,
            constrain="domain",
            row=1, col=col,
        )
    fig.update_layout(
        title=title,
        height=650,
        margin=dict(l=30, r=30, t=70, b=35),
        hovermode="closest",
    )
    return fig

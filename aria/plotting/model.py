from __future__ import annotations

from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
from aria.storm_objects import extract_object_boundaries
from aria.plotting.cities import add_plotly_cities, add_mpl_cities


REFLECTIVITY_STYLE = {"cmap":"turbo","plotly":"Turbo","vmin":-30.0,"vmax":70.0,"units":"dBZ"}

def mask_reflectivity_background(da):
    """Mask explicit fills and a repeated minimum/no-echo reflectivity value."""
    out=da.astype(float)
    for key in ("_FillValue","missing_value"):
        try:
            fill=float(da.attrs.get(key))
            out=out.where(~np.isclose(out,fill,rtol=0,atol=1e-6))
        except Exception:
            pass
    vals=np.asarray(out.values,float)
    finite=vals[np.isfinite(vals)]
    if finite.size:
        mn=float(np.nanmin(finite))
        frac=float(np.count_nonzero(np.isclose(finite,mn,rtol=0,atol=1e-6)))/float(finite.size)
        if frac >= 0.005 or mn <= -90:
            out=out.where(~np.isclose(out,mn,rtol=0,atol=1e-6))
    return out

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
    "terrain_height": ("Terrain Elevation", "m", "terrain"),
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
    add_mpl_cities(ax, region)


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
    title=None,
):
    da = ds[variable]
    if "reflect" in str(variable).lower() or "reflect" in str(getattr(da,"name","")).lower():
        da=mask_reflectivity_background(da)
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
    if title is None:
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




def overlay_mrms_reflectivity(fig, mrms_ds, *, min_dbz=10.0):
    """Overlay a lightweight MRMS QC composite on an existing model map."""
    import cartopy.crs as ccrs
    ax = fig.axes[0]
    field = mrms_ds["reflectivity"]
    if "time" in field.dims:
        field = field.isel(time=-1)
    field = field.squeeze(drop=True)
    values = np.where(np.asarray(field.values, float) >= float(min_dbz), field.values, np.nan)
    mesh = ax.pcolormesh(
        mrms_ds["longitude"],
        mrms_ds["latitude"],
        values,
        shading="auto",
        cmap="turbo",
        vmin=float(min_dbz),
        vmax=70.0,
        alpha=0.38,
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    fig.colorbar(
        mesh, ax=ax, pad=0.08, fraction=0.035,
        label="MRMS QC Composite (dBZ)",
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


def mpl_to_plotly_colorscale(cmap_name, samples=21):
    """Convert a Matplotlib colormap name to a Plotly colorscale."""
    import matplotlib as mpl
    cmap = mpl.colormaps.get_cmap(cmap_name)
    values = []
    for i in range(samples):
        x = i / (samples - 1)
        r, g, b, _ = cmap(x)
        values.append([x, f"rgb({int(round(255*r))},{int(round(255*g))},{int(round(255*b))})"])
    return values



def _regularize_interactive_da(da, max_points=320):
    """Return regular 1-D lon/lat axes and a 2-D field for Plotly.

    Native HRRR fields are commonly on a curvilinear Lambert grid.  Surface,
    MRMS, and several ARIA analyses already use regular lat/lon axes.  For a
    curvilinear source we make a bounded nearest-neighbor display grid so the
    scientific source stays untouched while the interactive map remains fast.
    """
    lat=np.asarray(da.latitude.values,float)
    lon=np.asarray(da.longitude.values,float)
    z=np.asarray(da.squeeze(drop=True).values,float)
    if lat.ndim==1 and lon.ndim==1:
        return lon,lat,z
    if lat.ndim!=2 or lon.ndim!=2 or z.ndim!=2:
        raise ValueError("Interactive field requires either 1-D or 2-D latitude/longitude coordinates.")
    good=np.isfinite(lat)&np.isfinite(lon)&np.isfinite(z)
    if not good.any():
        # preserve domain even when the field is empty
        good_geo=np.isfinite(lat)&np.isfinite(lon)
        if not good_geo.any():
            raise ValueError("Interactive field has no finite geographic coordinates.")
        good=good_geo
    west,east=float(np.nanmin(lon[good])),float(np.nanmax(lon[good]))
    south,north=float(np.nanmin(lat[good])),float(np.nanmax(lat[good]))
    ny=min(int(max_points),max(80,z.shape[0]))
    nx=min(int(max_points),max(80,z.shape[1]))
    x=np.linspace(west,east,nx)
    y=np.linspace(south,north,ny)
    xx,yy=np.meshgrid(x,y)
    src=np.column_stack([lon[good],lat[good]])
    vals=z[good]
    try:
        from scipy.spatial import cKDTree
        tree=cKDTree(src)
        _,ind=tree.query(np.column_stack([xx.ravel(),yy.ravel()]),k=1)
        zz=vals[ind].reshape(yy.shape)
    except Exception:
        # Lightweight fallback: nearest index in projected array space.
        # This is display-only and never changes stored/model data.
        rr=np.linspace(0,z.shape[0]-1,ny).round().astype(int)
        cc=np.linspace(0,z.shape[1]-1,nx).round().astype(int)
        zz=z[np.ix_(rr,cc)]
    return x,y,zz


def add_interactive_wind_overlay(fig,u_da,v_da,region,*,mode="Arrows",density="Medium"):
    """Add interactive wind arrows or streamlines to an existing Plotly map."""
    import plotly.graph_objects as go
    if mode in (None,"None"):
        return fig
    ux,uy,u=_regularize_interactive_da(u_da,max_points=220)
    vx,vy,v=_regularize_interactive_da(v_da,max_points=220)
    # Components originate from the same analysis grid; align defensively.
    if u.shape!=v.shape or not np.allclose(ux,vx) or not np.allclose(uy,vy):
        return fig
    density_map={"Sparse":14,"Medium":9,"Dense":6}
    skip=density_map.get(str(density),9)
    coslat=max(np.cos(np.deg2rad((float(region.south)+float(region.north))/2)),0.25)

    if mode=="Arrows":
        xs=[]; ys=[]
        max_speed=float(np.nanpercentile(np.hypot(u,v),95)) if np.isfinite(u).any() and np.isfinite(v).any() else 1.0
        max_speed=max(max_speed,1.0)
        base=0.018*max(float(region.east)-float(region.west),1.0)
        for j in range(skip//2,len(uy),skip):
            for i in range(skip//2,len(ux),skip):
                uu=float(u[j,i]); vv=float(v[j,i])
                sp=float(np.hypot(uu,vv))
                if not np.isfinite(sp) or sp<=0: continue
                scale=base*(0.45+0.75*min(sp/max_speed,1.0))
                dx=(uu/sp)*scale/coslat
                dy=(vv/sp)*scale
                x0=float(ux[i]-dx/2); y0=float(uy[j]-dy/2)
                x1=float(ux[i]+dx/2); y1=float(uy[j]+dy/2)
                # shaft
                xs.extend([x0,x1,None]); ys.extend([y0,y1,None])
                # two short arrowhead wings
                ang=np.arctan2(dy,dx)
                wing=0.28*np.hypot(dx,dy)
                for off in (2.55,-2.55):
                    xs.extend([x1,x1+wing*np.cos(ang+off),None])
                    ys.extend([y1,y1+wing*np.sin(ang+off),None])
        fig.add_trace(go.Scatter(
            x=xs,y=ys,mode="lines",name="Wind",
            line=dict(color="rgba(20,20,20,.72)",width=1.2),
            hoverinfo="skip",showlegend=False,
        ))
        return fig

    # Streamlines: integrate a normalized vector field from a modest seed grid.
    try:
        from scipy.interpolate import RegularGridInterpolator
        fu=RegularGridInterpolator((uy,ux),u,bounds_error=False,fill_value=np.nan)
        fv=RegularGridInterpolator((uy,ux),v,bounds_error=False,fill_value=np.nan)
    except Exception:
        return add_interactive_wind_overlay(fig,u_da,v_da,region,mode="Arrows",density=density)

    nseed={"Sparse":6,"Medium":9,"Dense":12}.get(str(density),9)
    seed_x=np.linspace(float(region.west),float(region.east),nseed+2)[1:-1]
    seed_y=np.linspace(float(region.south),float(region.north),max(4,int(nseed*0.65))+2)[1:-1]
    step=0.012*max(float(region.east)-float(region.west),1.0)
    paths_x=[]; paths_y=[]
    for sy in seed_y:
        for sx in seed_x:
            line=[]
            for direction in (-1.0,1.0):
                x=float(sx); y=float(sy); pts=[]
                for _ in range(55):
                    uu=float(fu((y,x))); vv=float(fv((y,x)))
                    sp=float(np.hypot(uu,vv))
                    if not np.isfinite(sp) or sp<0.2: break
                    pts.append((x,y))
                    x += direction*step*(uu/sp)/max(np.cos(np.deg2rad(y)),0.25)
                    y += direction*step*(vv/sp)
                    if not (region.west<=x<=region.east and region.south<=y<=region.north): break
                if direction<0: pts=pts[::-1]
                if direction<0: line.extend(pts)
                else: line.extend(pts[1:] if line and pts else pts)
            if len(line)>3:
                paths_x.extend([q[0] for q in line]+[None])
                paths_y.extend([q[1] for q in line]+[None])
    fig.add_trace(go.Scatter(
        x=paths_x,y=paths_y,mode="lines",name="Wind streamlines",
        line=dict(color="rgba(20,20,20,.62)",width=1.1),
        hoverinfo="skip",showlegend=False,
    ))
    return fig


def add_interactive_environment_overlay(
    fig, ds, region, *, pressure_hpa=500,
    show_temperature=False, show_wind=True, show_height=True,
    wind_density="Medium", temperature_interval=2.0, height_interval=60.0,
    shaded_field="None", shading_opacity=0.28,
):
    """Overlay HRRR upper-air context on an existing Plotly geographic map.

    Reflectivity remains the primary map field. One environmental variable can
    be shown as semi-transparent shading with its own colorbar, while height
    contours and winds provide independent structural context.
    """
    import plotly.graph_objects as go
    level=float(pressure_hpa)

    def _level(name):
        if name not in ds:
            return None
        da=ds[name]
        if "pressure_hpa" in da.dims or "pressure_hpa" in da.coords:
            da=da.sel(pressure_hpa=level,method="nearest")
        return da.squeeze(drop=True)

    shade=str(shaded_field or "None").lower()
    if shade not in ("none","off"):
        if shade.startswith("temp"):
            da=_level("air_temperature")
            label=f"{int(level)} hPa Temperature"
            units="°C"
            colorscale="RdBu_r"
            zmin=zmax=None
        elif shade.startswith("relative") or shade in ("rh","humidity"):
            da=_level("relative_humidity")
            label=f"{int(level)} hPa Relative Humidity"
            units="%"
            colorscale="Viridis"
            zmin,zmax=0.0,100.0
        else:
            da=None
        if da is not None:
            x,y,z=_regularize_interactive_da(da,max_points=220)
            finite=z[np.isfinite(z)]
            if finite.size:
                if zmin is None:
                    lo=float(np.nanpercentile(finite,2))
                    hi=float(np.nanpercentile(finite,98))
                    pad=max((hi-lo)*0.05,0.5)
                    zmin,zmax=lo-pad,hi+pad
                fig.add_trace(go.Heatmap(
                    x=x,y=y,z=z,zmin=zmin,zmax=zmax,
                    colorscale=colorscale,
                    opacity=float(shading_opacity),
                    name=label,
                    colorbar=dict(
                        title=f"{label}<br>{units}",
                        x=1.13,
                        len=0.62,
                        thickness=14,
                    ),
                    hovertemplate=(
                        f"{label}<br>Lon %{{x:.2f}}<br>Lat %{{y:.2f}}"
                        f"<br>%{{z:.1f}} {units}<extra></extra>"
                    ),
                    showscale=True,
                ))
                fig.update_layout(margin=dict(r=150))

    # Optional temperature line contours remain available independently of the
    # shaded environmental field.
    if show_temperature:
        temp=_level("air_temperature")
        if temp is not None:
            x,y,z=_regularize_interactive_da(temp,max_points=200)
            finite=z[np.isfinite(z)]
            if finite.size:
                step=max(float(temperature_interval),0.1)
                lo=float(np.floor(np.nanmin(finite)/step)*step)
                hi=float(np.ceil(np.nanmax(finite)/step)*step)
                if hi<=lo: hi=lo+step
                fig.add_trace(go.Contour(
                    x=x,y=y,z=z,showscale=False,hoverinfo="skip",
                    contours=dict(
                        start=lo,end=hi,size=step,coloring="none",
                        showlabels=True,labelfont=dict(size=9,color="firebrick"),
                    ),
                    line=dict(color="rgba(180,35,35,.85)",width=1.1),
                    name=f"{int(level)} hPa temperature",showlegend=False,
                ))

    if show_height:
        gh=_level("geopotential_height")
        if gh is not None:
            x,y,z=_regularize_interactive_da(gh,max_points=200)
            finite=z[np.isfinite(z)]
            if finite.size:
                step=max(float(height_interval),1.0)
                lo=float(np.floor(np.nanmin(finite)/step)*step)
                hi=float(np.ceil(np.nanmax(finite)/step)*step)
                if hi<=lo: hi=lo+step
                fig.add_trace(go.Contour(
                    x=x,y=y,z=z,showscale=False,hoverinfo="skip",
                    contours=dict(
                        start=lo,end=hi,size=step,coloring="none",
                        showlabels=True,labelfont=dict(size=9,color="black"),
                    ),
                    line=dict(color="rgba(25,25,25,.90)",width=1.3),
                    name=f"{int(level)} hPa height",showlegend=False,
                ))

    if show_wind:
        u=_level("u_wind"); v=_level("v_wind")
        if u is not None and v is not None:
            fig=add_interactive_wind_overlay(
                fig,u,v,region,mode="Arrows",density=wind_density
            )
    return fig

def add_mpl_environment_overlay(
    ax, ds, region, *, pressure_hpa=500,
    show_temperature=False, show_wind=True, show_height=True,
    wind_density="Medium", temperature_interval=2.0, height_interval=60.0,
    shaded_field="None", shading_opacity=0.28,
):
    """Overlay HRRR upper-air shading, contours and wind on a Cartopy axis."""
    level=float(pressure_hpa)

    def _level(name):
        if name not in ds:
            return None
        da=ds[name]
        if "pressure_hpa" in da.dims or "pressure_hpa" in da.coords:
            da=da.sel(pressure_hpa=level,method="nearest")
        return da.squeeze(drop=True)

    def _geo(da):
        if da is None: return None,None,None
        return (
            np.asarray(da.longitude.values,float),
            np.asarray(da.latitude.values,float),
            np.asarray(da.values,float),
        )

    shade=str(shaded_field or "None").lower()
    if shade not in ("none","off"):
        if shade.startswith("temp"):
            da=_level("air_temperature"); units="°C"; cmap="coolwarm"
            vmin=vmax=None
            cb_label=f"{int(level)} hPa Temperature ({units})"
        elif shade.startswith("relative") or shade in ("rh","humidity"):
            da=_level("relative_humidity"); units="%"; cmap="viridis"
            vmin,vmax=0.0,100.0
            cb_label=f"{int(level)} hPa Relative Humidity ({units})"
        else:
            da=None
        lon,lat,z=_geo(da)
        if z is not None and np.isfinite(z).any():
            mesh=ax.pcolormesh(
                lon,lat,z,cmap=cmap,vmin=vmin,vmax=vmax,
                alpha=float(shading_opacity),
                shading="auto",transform=ccrs.PlateCarree(),zorder=4,
            )
            try:
                cb=ax.figure.colorbar(mesh,ax=ax,pad=.02,fraction=.035)
                cb.set_label(cb_label)
            except Exception:
                pass

    if show_temperature:
        da=_level("air_temperature"); lon,lat,z=_geo(da)
        if z is not None and np.isfinite(z).any():
            vals=z[np.isfinite(z)]
            step=max(float(temperature_interval),0.1)
            lo=np.floor(np.nanmin(vals)/step)*step
            hi=np.ceil(np.nanmax(vals)/step)*step
            if hi<=lo: hi=lo+step
            cs=ax.contour(
                lon,lat,z,levels=np.arange(lo,hi+step*0.5,step),
                colors="firebrick",linewidths=1.0,alpha=.9,
                transform=ccrs.PlateCarree(),zorder=6,
            )
            try: ax.clabel(cs,fmt="%g°C",fontsize=7,inline=True)
            except Exception: pass

    if show_height:
        da=_level("geopotential_height"); lon,lat,z=_geo(da)
        if z is not None and np.isfinite(z).any():
            vals=z[np.isfinite(z)]; step=max(float(height_interval),1.0)
            lo=np.floor(np.nanmin(vals)/step)*step
            hi=np.ceil(np.nanmax(vals)/step)*step
            if hi<=lo: hi=lo+step
            cs=ax.contour(
                lon,lat,z,levels=np.arange(lo,hi+step/2,step),
                colors="black",linewidths=.9,alpha=.9,
                transform=ccrs.PlateCarree(),zorder=6,
            )
            try: ax.clabel(cs,fmt="%g",fontsize=7,inline=True)
            except Exception: pass

    if show_wind:
        u=_level("u_wind"); v=_level("v_wind")
        if u is not None and v is not None:
            lon=np.asarray(u.longitude.values,float)
            lat=np.asarray(u.latitude.values,float)
            uu=np.asarray(u.values,float)*1.94384
            vv=np.asarray(v.values,float)*1.94384
            skip={"Sparse":28,"Medium":20,"Dense":14}.get(str(wind_density),20)
            if lon.ndim==2 and lat.ndim==2:
                sl=(slice(None,None,skip),slice(None,None,skip))
                ax.barbs(
                    lon[sl],lat[sl],uu[sl],vv[sl],length=4.5,linewidth=.45,
                    color="black",transform=ccrs.PlateCarree(),zorder=7,
                )
            elif lon.ndim==1 and lat.ndim==1:
                xx,yy=np.meshgrid(lon,lat)
                sl=(slice(None,None,skip),slice(None,None,skip))
                ax.barbs(
                    xx[sl],yy[sl],uu[sl],vv[sl],length=4.5,linewidth=.45,
                    color="black",transform=ccrs.PlateCarree(),zorder=7,
                )
    return ax

def overlay_interactive_mrms(fig,mrms_ds,*,min_dbz=10.0):
    """Overlay MRMS reflectivity without converting an interactive map to Matplotlib."""
    import plotly.graph_objects as go
    field=mrms_ds["reflectivity"]
    if "time" in field.dims: field=field.isel(time=-1)
    lon,lat,z=_regularize_interactive_da(field)
    z=np.where(z>=float(min_dbz),z,np.nan)
    fig.add_trace(go.Heatmap(
        x=lon,y=lat,z=z,zmin=-30,zmax=70,colorscale="Turbo",
        opacity=.42,showscale=False,name="MRMS QC Composite",
        hovertemplate="MRMS<br>Lon %{x:.2f}<br>Lat %{y:.2f}<br>%{z:.1f} dBZ<extra></extra>",
    ))
    return fig


def plot_interactive_field(da, region, *, title, units="", zmin=None, zmax=None, colorscale="Turbo", state_color="rgba(20,20,20,.95)", show_cities=True, show_contours=False, contour_levels=None):
    """Interactive regular-lat/lon field with state boundaries and pan/zoom."""
    import plotly.graph_objects as go
    if "reflect" in str(getattr(da,"name","")).lower() or "reflect" in str(title).lower():
        da=mask_reflectivity_background(da)
    lon,lat,z=_regularize_interactive_da(da)
    fig=go.Figure(go.Heatmap(
        x=lon,y=lat,z=z,zmin=zmin,zmax=zmax,colorscale=colorscale,
        colorbar=dict(title=units),
        hovertemplate=f"Lon %{{x:.2f}}<br>Lat %{{y:.2f}}<br>%{{z:.1f}} {units}<extra></extra>",
    ))
    if show_contours:
        contours=dict(coloring="lines",showlabels=True,labelfont=dict(size=10,color="black"))
        if contour_levels is not None and len(contour_levels)>=2:
            levels=np.asarray(contour_levels,float); step=float(np.nanmedian(np.diff(levels)))
            contours.update(start=float(levels.min()),end=float(levels.max()),size=step)
        fig.add_trace(go.Contour(x=lon,y=lat,z=z,showscale=False,colorscale=[[0,"rgba(20,20,20,.85)"],[1,"rgba(20,20,20,.85)"]],line=dict(width=1),contours=contours,hoverinfo="skip",name="Contours",showlegend=False))
    for x,y in _interactive_state_lines(region):
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(color=state_color,width=1.6),hoverinfo="skip",showlegend=False))
    if show_cities: add_plotly_cities(fig,region,font_size=10)
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
        for arr in (m,o):
            finite=arr[np.isfinite(arr)]
            if finite.size:
                mn=float(np.nanmin(finite))
                frac=float(np.count_nonzero(np.isclose(finite,mn,rtol=0,atol=1e-6)))/float(finite.size)
                if frac >= 0.005 or mn <= -90:
                    arr[np.isclose(arr,mn,rtol=0,atol=1e-6)]=np.nan
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

    units=comparison.attrs.get("units","")
    shared_bar_title=units if units else "Value"
    difference_bar_title=f"Difference ({units})" if units else "Difference"

    # Model and Observation intentionally share one horizontal colorbar.  The
    # Difference panel uses its own symmetric diverging bar.  Keeping both bars
    # below the maps preserves panel width and prevents colorbars from obscuring
    # the plotted domains.
    fig.add_trace(
        go.Heatmap(
            x=lon,y=lat,z=m,zmin=vmin,zmax=vmax,colorscale=colorscale,
            colorbar=dict(
                title=dict(text=shared_bar_title,side="bottom"),
                orientation="h",
                x=0.325,
                xanchor="center",
                y=-0.16,
                yanchor="top",
                len=0.60,
                thickness=16,
            ),
            hovertemplate=hover,
        ),
        row=1,col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=lon,y=lat,z=o,zmin=vmin,zmax=vmax,colorscale=colorscale,
            showscale=False,hovertemplate=hover,
        ),
        row=1,col=2,
    )
    fig.add_trace(
        go.Heatmap(
            x=lon,y=lat,z=d,zmin=-difference_limit,zmax=difference_limit,
            colorscale="RdBu_r",
            colorbar=dict(
                title=dict(text=difference_bar_title,side="bottom"),
                orientation="h",
                x=0.845,
                xanchor="center",
                y=-0.16,
                yanchor="top",
                len=0.27,
                thickness=16,
            ),
            hovertemplate=hover,
        ),
        row=1,col=3,
    )

    state_line_color = "rgba(20,20,20,.95)"
    for x,y in _interactive_state_lines(region):
        for col in (1,2,3):
            fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(color=state_line_color,width=1.4),hoverinfo="skip",showlegend=False),row=1,col=col)
    for col in (1,2,3): add_plotly_cities(fig,region,font_size=9,row=1,col=col)

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
        height=700,
        dragmode="zoom",
        margin=dict(l=45,r=45,t=85,b=125),
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

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from aria import GPGL_REGION, REGION_PRESETS, Region, __version__
from aria.region import region_from_dict
from aria.region_store import load_saved_regions, save_region, delete_region
from aria.region_stats import region_inventory
from aria.adapters.nexrad import GPGL_NEXRAD_SITES, NEXRAD_DISPLAY_NAMES, nexrad_sites_for_region, nexrad_display_name, normalize_nexrad_id
from aria.adapters.mrms import invalidate_latest_mrms_cache
from aria.adapters.sst import choose_sst_product, open_sst
from aria.adapters.marine import load_latest_marine_observations
from aria.cache import PersistentCache, time_bucket
from aria.storage import (
    load_config as load_storage_config, save_config as save_storage_config,
    load_state as load_storage_state, update_storage,
    status_rows as storage_status_rows, storage_size_bytes, total_storage_size_bytes,
    region_store_root, migrate_legacy_snapshots, DEFAULT_SOURCES, SOURCE_LABELS,
    SOURCE_NATIVE_CADENCE, load_stored_dataset, load_stored_dataset_history, load_stored_time_range,
    load_stored_frame, storage_matches_region, list_storage_profiles,
    create_storage_profile, set_active_profile, all_status_rows, profile_for_region,
    source_time_coverage, load_stored_hrrr_valid_member,
)
from aria.history import historical_plan
from aria.storm_objects import adapt_available, adapt_version, segment_with_adapt, summarize_objects
from aria.plotting import (
    VARIABLE_STYLE,
    MODEL_STYLE,
    make_3d_variable_slice_explorer,
    overlay_regional_radar,
    overlay_mrms_reflectivity,
    overlay_interactive_mrms,
    add_interactive_wind_overlay,
    add_interactive_environment_overlay,
    add_mpl_environment_overlay,
    REFLECTIVITY_STYLE,
    plot_air_quality,
    plot_atmosphere_slice,
    plot_model_cross_section,
    plot_model_field,
    plot_three_panel_comparison,
    plot_radar_three_panel,
    plot_profile_comparison,
    plot_interactive_three_panel,
    plot_interactive_field,
    mpl_to_plotly_colorscale,
    plot_adapt_storm_objects,
    plot_nexrad_ppi,
    plot_static_nexrad_ppi,
    plot_interactive_nexrad_ppi,
    plot_regional_reflectivity,
    plot_regional_reflectivity_3d,
    plot_surface_analysis,
    radar_sweep_summary,
    add_city_labels,
)
from aria.radar_grid import grid_single_radar_reflectivity
from aria.workflows import (
    build_latest_air_quality,
    build_latest_atmosphere,
    build_latest_regional_radar,
    build_latest_regional_radar_fast,
    build_latest_surface,
    build_hrrr_surface,
    build_hrrr_pressure,
    compare_hrrr_surface_with_asos,
    build_hrrr_surface_gridded_comparison,
    build_hrrr_radar_comparison,
    build_hrrr_raob_comparison,
    build_latest_hrrr_raob_comparison,
    build_hrrr_lead_time_verification,
    build_hrrr_mrms_comparison,
    build_hrrr_mrms_adapt_objects,
    build_hrrr_radar_lead_time_verification,
    build_mrms_at_time,
    hrrr_available_cycles,
    latest_safe_hrrr_valid_time,
    load_latest_nexrad,
)

st.set_page_config(page_title="ARIA",layout="wide")
st.title("ARIA — Atmospheric Regional Integration and Analysis")
st.sidebar.caption(f"ARIA v{__version__}")


def _region_preview(region, inventory=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    inventory = inventory or {}
    styles = (
        ("surface", "Surface stations", "circle"),
        ("radars", "NEXRAD", "diamond"),
        ("sondes", "Radiosondes", "triangle-up"),
    )
    for key, label, symbol in styles:
        rows = [r for r in inventory.get(key, []) if r.get("latitude") is not None and r.get("longitude") is not None]
        if not rows:
            continue
        fig.add_trace(go.Scattergeo(
            lon=[r["longitude"] for r in rows], lat=[r["latitude"] for r in rows],
            text=[r.get("id", "") for r in rows], name=label, mode="markers",
            marker=dict(size=7 if key == "surface" else 10, symbol=symbol),
            hovertemplate="%{text}<br>%{lat:.2f}, %{lon:.2f}<extra></extra>",
        ))
    fig.update_geos(
        projection_type="mercator",
        showland=True,
        showcountries=True,
        countrycolor="rgba(40,40,40,0.9)",
        countrywidth=1.0,
        showsubunits=True,
        subunitcolor="rgba(35,35,35,0.95)",
        subunitwidth=1.2,
        showcoastlines=True,
        coastlinecolor="rgba(55,55,55,0.85)",
        coastlinewidth=0.8,
        lonaxis_range=[region.west, region.east],
        lataxis_range=[region.south, region.north],
        fitbounds=False,
    )
    add_city_labels(fig,region,font_size=9)
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=15, b=0), legend=dict(orientation="h"))
    return fig


@st.cache_data(ttl=1800, show_spinner=False)
def _inventory_for_region(name, west, south, east, north):
    region = Region(name=name, west=west, south=south, east=east, north=north)
    return region_inventory(region)


def _show_region_configuration():
    st.subheader("Region Configuration")
    st.caption("Define the analysis domain, inspect observing-system coverage, and launch ARIA. Built-in regions are read-only; duplicate one with a new name to customize it.")

    saved = load_saved_regions()
    choices = list(REGION_PRESETS) + sorted(saved) + ["New Region"]
    choice = st.selectbox("Region", choices, index=0, key="region-config-choice")
    source = REGION_PRESETS.get(choice) or saved.get(choice)

    if st.session_state.get("_region_config_source") != choice:
        st.session_state["_region_config_source"] = choice
        if source is not None:
            st.session_state["cfg_name"] = source.name
            st.session_state["cfg_west"] = float(source.west)
            st.session_state["cfg_east"] = float(source.east)
            st.session_state["cfg_south"] = float(source.south)
            st.session_state["cfg_north"] = float(source.north)
            st.session_state["cfg_center_lat"] = (source.south + source.north) / 2.0
            st.session_state["cfg_center_lon"] = (source.west + source.east) / 2.0
            st.session_state["cfg_radius"] = max(50.0, (source.north-source.south)*111.0/2.0)
        else:
            st.session_state["cfg_name"] = "My Region"
            st.session_state["cfg_center_lat"] = 40.0
            st.session_state["cfg_center_lon"] = -97.0
            st.session_state["cfg_radius"] = 400.0
            st.session_state["cfg_west"] = -102.0
            st.session_state["cfg_east"] = -92.0
            st.session_state["cfg_south"] = 35.0
            st.session_state["cfg_north"] = 45.0

    built_in = choice in REGION_PRESETS
    saved_region = choice in saved
    edit_saved = st.checkbox("Edit saved region",value=False,disabled=not saved_region,key="cfg_edit_saved")
    editable = (not built_in) and (not saved_region or edit_saved)
    mode = st.radio("Region definition", ["Bounding box", "Center + radius"], horizontal=True, key="cfg_mode")
    name = st.text_input("Region name", key="cfg_name", disabled=not editable)

    if mode == "Center + radius":
        c1,c2,c3 = st.columns(3)
        lat = c1.number_input("Center latitude", -90.0, 90.0, key="cfg_center_lat", format="%.3f", disabled=not editable)
        lon = c2.number_input("Center longitude", -180.0, 180.0, key="cfg_center_lon", format="%.3f", disabled=not editable)
        radius = c3.number_input("Radius (km)", 25.0, 2500.0, key="cfg_radius", step=25.0, disabled=not editable)
        candidate = Region.from_center_radius(name or "Custom", lat, lon, radius)
    else:
        c1,c2,c3,c4 = st.columns(4)
        west = c1.number_input("West", -180.0, 180.0, key="cfg_west", format="%.3f", disabled=not editable)
        east = c2.number_input("East", -180.0, 180.0, key="cfg_east", format="%.3f", disabled=not editable)
        south = c3.number_input("South", -90.0, 90.0, key="cfg_south", format="%.3f", disabled=not editable)
        north = c4.number_input("North", -90.0, 90.0, key="cfg_north", format="%.3f", disabled=not editable)
        candidate = source if built_in else Region(name=name or "Custom", west=west, east=east, south=south, north=north)

    if built_in:
        candidate = source

    with st.spinner("Discovering stations and radars in this region..."):
        inv = _inventory_for_region(candidate.name, candidate.west, candidate.south, candidate.east, candidate.north)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Surface stations", len(inv["surface"]), help=f"Stations reporting during the last {inv['surface_minutes']} minutes")
    m2.metric("NEXRAD radars", len(inv["radars"]), help="Operational sites returned by ARIA/Py-ART regional discovery")
    m3.metric("Radiosonde sites", len(inv["sondes"]), help="IEM RAOB launch sites inside the region")
    m4.metric("States queried", len(candidate.query_states))

    st.plotly_chart(_region_preview(candidate, inv), width="stretch", config={"displaylogo":False})

    errors = {k:v for k,v in inv.get("errors",{}).items() if v}
    if errors:
        st.warning("Some inventory sources could not be queried: " + "; ".join(f"{k}: {v}" for k,v in errors.items()))

    with st.expander("Observing sites"):
        t1,t2,t3 = st.tabs(["Surface", "NEXRAD", "Radiosondes"])
        with t1: st.dataframe(pd.DataFrame(inv["surface"]), width="stretch", hide_index=True)
        with t2: st.dataframe(pd.DataFrame(inv["radars"]), width="stretch", hide_index=True)
        with t3: st.dataframe(pd.DataFrame(inv["sondes"]), width="stretch", hide_index=True)

    b1,b2,b3,b4 = st.columns([1,1,1,3])
    if b1.button("Launch ARIA", type="primary", width="stretch"):
        st.session_state["aria_active_region"] = candidate.as_dict()
        st.session_state["aria_region_config_open"] = False
        st.rerun()
    if not built_in and b2.button("Save Changes" if saved_region else "Save Region", width="stretch"):
        if not name.strip():
            st.error("Enter a region name before saving.")
        else:
            save_region(candidate)
            st.success(f"Saved {candidate.name}.")
            st.rerun()
    if built_in and b2.button("Duplicate as Custom Region", width="stretch"):
        duplicate=Region(name=f"{candidate.name} Copy",west=candidate.west,east=candidate.east,south=candidate.south,north=candidate.north,query_states_override=candidate.query_states_override)
        save_region(duplicate); st.session_state.pop("_region_config_source",None); st.success(f"Saved {duplicate.name}."); st.rerun()
    if choice in saved and b3.button("Delete region", width="stretch"):
        delete_region(choice)
        st.session_state.pop("_region_config_source", None)
        st.success(f"Deleted {choice}.")
        st.rerun()


if "aria_region_config_open" not in st.session_state:
    st.session_state["aria_region_config_open"] = True

if st.session_state["aria_region_config_open"]:
    _show_region_configuration()
    st.stop()

ACTIVE_REGION = region_from_dict(st.session_state.get("aria_active_region", GPGL_REGION.as_dict()))
REGION_CACHE_KEY = ACTIVE_REGION.cache_key
_region_changed = st.session_state.get("_aria_region_key") != REGION_CACHE_KEY
if _region_changed:
    st.session_state["_aria_region_key"] = REGION_CACHE_KEY
    st.cache_data.clear()
    st.cache_resource.clear()

st.sidebar.markdown("### Region")
st.sidebar.markdown(f"**{ACTIVE_REGION.name}**")
st.sidebar.caption(
    f"{ACTIVE_REGION.west:.1f}° to {ACTIVE_REGION.east:.1f}° lon • "
    f"{ACTIVE_REGION.south:.1f}° to {ACTIVE_REGION.north:.1f}° lat"
)
if st.sidebar.button("Change region", width="stretch"):
    st.session_state["aria_region_config_open"] = True
    st.rerun()

@st.cache_resource
def processed_cache():
    return PersistentCache()

def disk_cached(namespace, key, builder, max_age_seconds=None):
    value, hit = processed_cache().get_or_build(
        namespace,
        f"v{__version__}-{REGION_CACHE_KEY}-{key}",
        builder,
        max_age_seconds=max_age_seconds,
    )
    return value




SURFACE_FIELD_STYLE = {
    "air_temperature_f": ("Air Temperature","°F","coolwarm"),
    "dew_point_temperature_f": ("Dew Point","°F","BrBG"),
    "relative_humidity_pct": ("Relative Humidity","%","viridis"),
    "wind_speed_kt": ("Wind Speed","kt","viridis"),
    "wind_direction_deg": ("Wind Direction","°","twilight"),
    "wind_gust_kt": ("Wind Gust","kt","viridis"),
    "u_wind_kt": ("U Wind","kt","coolwarm"),
    "v_wind_kt": ("V Wind","kt","coolwarm"),
    "precipitation_1h_in": ("1-h Precipitation","in","Blues"),
    "altimeter_inhg": ("Altimeter","inHg","viridis"),
    "sea_level_pressure_mb": ("Sea-Level Pressure","mb","viridis"),
    "visibility_mi": ("Visibility","mi","viridis"),
    "cloud_base_ft": ("Cloud Base","ft AGL","viridis"),
    "feels_like_f": ("Feels Like Temperature","°F","coolwarm"),
}

def _add_surface_station_overlay(fig, observations, analysis_time, *, show_values=False, show_ids=False):
    """Add latest land-surface station reports to an interactive map."""
    import plotly.graph_objects as go
    if observations is None or observations.empty:
        return fig
    frame=observations.copy()
    frame["time"]=pd.to_datetime(frame["time"],utc=True,errors="coerce")
    target=pd.Timestamp(analysis_time)
    target=target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    frame=frame.dropna(subset=["station_id","latitude","longitude","time","variable","value"])
    if frame.empty:
        return fig
    frame["_offset"]=(frame["time"]-target).abs()
    latest=(frame.sort_values("_offset")
                  .drop_duplicates(["station_id","variable"],keep="first"))
    pivot=latest.pivot_table(index="station_id",columns="variable",values="value",aggfunc="first")
    meta=(latest.sort_values("_offset")
                .drop_duplicates("station_id")
                .set_index("station_id")[["latitude","longitude","time"]])
    joined=meta.join(pivot,how="left").reset_index()
    if joined.empty:
        return fig

    def _numeric_column(name):
        if name in joined:
            return pd.to_numeric(joined[name],errors="coerce")
        return pd.Series(np.nan,index=joined.index,dtype=float)
    temp=_numeric_column("air_temperature_f")
    dew=_numeric_column("dew_point_temperature_f")
    u=_numeric_column("u_wind_kt")
    v=_numeric_column("v_wind_kt")
    wspd=np.hypot(u,v)
    text=[]
    for sid,tv in zip(joined.station_id.astype(str),temp):
        parts=[]
        if show_ids: parts.append(sid)
        if show_values and np.isfinite(tv): parts.append(f"{tv:.0f}°")
        text.append(" ".join(parts))
    custom=[
        [str(sid),
         float(tv) if np.isfinite(tv) else np.nan,
         float(dv) if np.isfinite(dv) else np.nan,
         float(wv) if np.isfinite(wv) else np.nan,
         str(ts)]
        for sid,tv,dv,wv,ts in zip(joined.station_id,temp,dew,wspd,joined["time"])
    ]
    mode="markers+text" if (show_values or show_ids) else "markers"
    fig.add_trace(go.Scatter(
        x=joined.longitude,y=joined.latitude,mode=mode,name="Land stations",
        text=text,textposition="top center",
        marker=dict(size=7,color="white",line=dict(color="black",width=1)),
        customdata=custom,
        hovertemplate=(
            "%{customdata[0]}<br>Lon %{x:.2f}<br>Lat %{y:.2f}"
            "<br>Temp %{customdata[1]:.1f} °F"
            "<br>Dew point %{customdata[2]:.1f} °F"
            "<br>Wind %{customdata[3]:.1f} kt"
            "<br>%{customdata[4]}<extra></extra>"
        ),
    ))
    return fig


def _add_surface_station_overlay_mpl(ax, observations, analysis_time):
    """Add compact latest surface-station context to a Cartopy axis."""
    if observations is None or observations.empty:
        return ax
    frame=observations.copy()
    frame["time"]=pd.to_datetime(frame["time"],utc=True,errors="coerce")
    target=pd.Timestamp(analysis_time)
    target=target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    frame=frame.dropna(subset=["station_id","latitude","longitude","time"])
    if frame.empty: return ax
    frame["_offset"]=(frame["time"]-target).abs()
    latest=frame.sort_values("_offset").drop_duplicates("station_id",keep="first")
    try:
        import cartopy.crs as ccrs
        ax.scatter(latest.longitude,latest.latitude,s=16,facecolor="white",edgecolor="black",linewidth=.7,transform=ccrs.PlateCarree(),zorder=8)
    except Exception:
        pass
    return ax




def _surface_grid_arrays(da):
    """Return lon, lat, z for an ARIA regular surface-analysis field."""
    z=np.asarray(da.values,float)
    if z.ndim>2:
        z=np.squeeze(z)
    lon_name=next((n for n in ("longitude","lon","x") if n in da.coords),None)
    lat_name=next((n for n in ("latitude","lat","y") if n in da.coords),None)
    if lon_name is None or lat_name is None:
        raise ValueError("Surface field is missing latitude/longitude coordinates.")
    lon=np.asarray(da[lon_name].values,float)
    lat=np.asarray(da[lat_name].values,float)
    return lon,lat,z


def _add_surface_analysis_overlay_plotly(
    fig, ds, variable, mode, *, show_labels=True, opacity=0.30,
):
    import plotly.graph_objects as go
    if ds is None or variable not in ds:
        return fig
    da=ds[variable]
    lon,lat,z=_surface_grid_arrays(da)
    finite=z[np.isfinite(z)]
    if not finite.size:
        return fig
    label,units,cmap=SURFACE_FIELD_STYLE.get(variable,(variable,"","viridis"))
    lo=float(np.nanpercentile(finite,5)); hi=float(np.nanpercentile(finite,95))
    if hi<=lo: hi=lo+1.0
    step=max((hi-lo)/8.0,0.1)
    if variable=="sea_level_pressure_mb": step=2.0
    elif variable in ("air_temperature_f","dew_point_temperature_f"): step=5.0
    elif variable=="relative_humidity_pct": step=10.0
    elif variable=="wind_speed_kt": step=5.0

    if mode=="Shaded analysis":
        fig.add_trace(go.Heatmap(
            x=lon,y=lat,z=z,opacity=float(opacity),
            colorscale=mpl_to_plotly_colorscale(cmap),
            zmin=lo,zmax=hi,
            colorbar=dict(title=f"{label} ({units})",len=.45,y=.25),
            hovertemplate=f"{label}: %{{z:.1f}} {units}<extra></extra>",
            name=label,
        ))
    if "Contours" in mode:
        fig.add_trace(go.Contour(
            x=lon,y=lat,z=z,showscale=False,
            contours=dict(
                start=float(np.floor(lo/step)*step),
                end=float(np.ceil(hi/step)*step),
                size=float(step),
                coloring="none",
                showlabels=bool(show_labels),
                labelfont=dict(size=10,color="black"),
            ),
            line=dict(width=1.5,color="black"),
            hovertemplate=f"{label}: %{{z:.1f}} {units}<extra></extra>",
            name=label,showlegend=False,
        ))
    return fig


def _add_surface_analysis_overlay_mpl(ax, ds, variable, mode, *, show_labels=True, opacity=0.30):
    if ds is None or variable not in ds:
        return ax
    try:
        import cartopy.crs as ccrs
        da=ds[variable]
        lon,lat,z=_surface_grid_arrays(da)
        finite=z[np.isfinite(z)]
        if not finite.size:
            return ax
        label,units,cmap=SURFACE_FIELD_STYLE.get(variable,(variable,"","viridis"))
        lo=float(np.nanpercentile(finite,5)); hi=float(np.nanpercentile(finite,95))
        if hi<=lo: hi=lo+1.0
        step=max((hi-lo)/8.0,0.1)
        if variable=="sea_level_pressure_mb": step=2.0
        elif variable in ("air_temperature_f","dew_point_temperature_f"): step=5.0
        elif variable=="relative_humidity_pct": step=10.0
        elif variable=="wind_speed_kt": step=5.0
        levels=np.arange(np.floor(lo/step)*step,np.ceil(hi/step)*step+step*.5,step)
        xx,yy=np.meshgrid(lon,lat)
        if mode=="Shaded analysis":
            ax.contourf(xx,yy,z,levels=levels,cmap=cmap,alpha=float(opacity),
                        transform=ccrs.PlateCarree(),zorder=5)
        if "Contours" in mode:
            cs=ax.contour(xx,yy,z,levels=levels,linewidths=.9,
                          transform=ccrs.PlateCarree(),zorder=7)
            if show_labels:
                ax.clabel(cs,inline=True,fontsize=7,fmt="%g")
    except Exception:
        pass
    return ax


@st.cache_data(ttl=300,show_spinner=False)
def load_surface(region_key, method="barnes", smoothing_km=140.0, max_distance_km=250.0, include_history=False):
    # The persistent collector stores the default latest Barnes analysis. Use
    # that first when it is current; custom interpolation settings still build
    # directly from the source.
    use_store = (method == "barnes" and float(smoothing_km) == 140.0 and float(max_distance_km) == 250.0 and not include_history)
    if use_store:
        ds = load_stored_dataset(ACTIVE_REGION, "surface", "analysis", max_age_minutes=15)
        obs = load_stored_frame(ACTIVE_REGION, "surface", "observations", max_age_minutes=15)
        if ds is None:
            cfg = profile_for_region(ACTIVE_REGION, require_enabled=True)
            if cfg is not None and cfg.get("sources",{}).get("surface",{}).get("enabled"):
                update_storage(source="surface", profile=cfg.get("profile_name"))
                ds = load_stored_dataset(ACTIVE_REGION, "surface", "analysis", max_age_minutes=15)
                obs = load_stored_frame(ACTIVE_REGION, "surface", "observations", max_age_minutes=15)
        if ds is not None:
            times = pd.to_datetime(ds["time"].values, utc=True, errors="coerce") if "time" in ds.coords else pd.DatetimeIndex([])
            start = times.min() if len(times) else pd.NaT
            end = times.max() if len(times) else pd.NaT
            return ds, (obs if obs is not None else pd.DataFrame()), start, end
    bucket=time_bucket(5)
    key=f"latest-{bucket}-{method}-{float(smoothing_km):g}-{float(max_distance_km):g}-hist{int(include_history)}"
    return disk_cached(
        "surface", key,
        lambda: build_latest_surface(region=ACTIVE_REGION,method=method,smoothing_km=smoothing_km,max_distance_km=max_distance_km,include_history=include_history),
        max_age_seconds=15*60,
    )

@st.cache_data(ttl=600,show_spinner=False)
def load_marine(region_key):
    stored = load_stored_frame(ACTIVE_REGION, "marine", "observations", max_age_minutes=60)
    if stored is not None and not stored.empty:
        return stored
    try:
        return load_latest_marine_observations(ACTIVE_REGION)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600,show_spinner=False)
def load_sst(region_key):
    """Return latest locally stored SST without blocking on network I/O."""
    product=choose_sst_product(ACTIVE_REGION)
    stored=load_stored_dataset(
        ACTIVE_REGION,"sst",product.key,max_age_minutes=7*24*60
    )
    if stored is None:
        return None,product,"not_available"
    return stored,product,"persistent_store"


def _display_thin_2d(field, max_points_per_axis=180):
    """Thin a 2-D field for interactive display only."""
    field=field.squeeze(drop=True)
    if field.ndim != 2:
        return field
    ydim,xdim=field.dims[-2],field.dims[-1]
    ystep=max(1,int(np.ceil(field.sizes[ydim]/float(max_points_per_axis))))
    xstep=max(1,int(np.ceil(field.sizes[xdim]/float(max_points_per_axis))))
    if ystep>1 or xstep>1:
        field=field.isel({ydim:slice(None,None,ystep),xdim:slice(None,None,xstep)})
    return field


@st.cache_data(ttl=300,show_spinner=False)
def load_mrms_history(region_key, hours=3.0, max_frames=36):
    return load_stored_dataset_history(
        ACTIVE_REGION, "mrms", "reflectivity",
        hours=float(hours), max_frames=int(max_frames),
    )

@st.cache_data(ttl=3600,show_spinner=False)
def load_atmosphere(region_key):
    bucket=time_bucket(60)
    return disk_cached(
        "atmosphere",
        f"latest-{bucket}",
        lambda: build_latest_atmosphere(region=ACTIVE_REGION),
        max_age_seconds=3*3600,
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_air_quality(region_key, max_distance_km=300.0):
    if float(max_distance_km) == 300.0:
        ds = load_stored_dataset(ACTIVE_REGION, "air_quality", "analysis", max_age_minutes=120)
        latest = load_stored_frame(ACTIVE_REGION, "air_quality", "observations", max_age_minutes=120)
        if ds is None:
            cfg = profile_for_region(ACTIVE_REGION, require_enabled=True)
            if cfg is not None and cfg.get("sources",{}).get("air_quality",{}).get("enabled"):
                update_storage(source="air_quality", profile=cfg.get("profile_name"))
                ds = load_stored_dataset(ACTIVE_REGION, "air_quality", "analysis", max_age_minutes=120)
                latest = load_stored_frame(ACTIVE_REGION, "air_quality", "observations", max_age_minutes=120)
        if ds is not None:
            latest = latest if latest is not None else pd.DataFrame()
            t = pd.Timestamp(ds["time"].values[-1]) if "time" in ds.coords and ds["time"].size else pd.Timestamp.now(tz="UTC")
            return ds, latest, latest.copy(), t
    bucket=time_bucket(30)
    return disk_cached(
        "air_quality", f"latest-{bucket}-r{float(max_distance_km):g}",
        lambda: build_latest_air_quality(region=ACTIVE_REGION,max_distance_km=max_distance_km),
        max_age_seconds=2*3600,
    )

@st.cache_resource(ttl=300,show_spinner=False)
def load_radar(rid):
    # NEXRADLevel2Adapter already persists the raw Level-II files. Keep parsed
    # Py-ART Radar objects in memory while Streamlit is running.
    return load_latest_nexrad(rid)

@st.cache_resource(ttl=300,show_spinner=False)
def load_recent_radars(rid, lookback_minutes=60, max_scans=12):
    from aria.adapters.nexrad import NEXRADLevel2Adapter
    adapter=NEXRADLevel2Adapter()
    scans=adapter.recent_scans(rid,lookback_minutes=lookback_minutes,max_scans=max_scans)
    loaded=[]
    for scan in scans:
        scan=adapter.download_scan(scan)
        loaded.append((adapter.read(scan),scan))
    return loaded

@st.cache_data(ttl=300,show_spinner=False)
def load_regional_radar(region_key):
    bucket=time_bucket(5)
    return disk_cached(
        "regional_radar",
        f"3d-{bucket}",
        lambda: build_latest_regional_radar(region=ACTIVE_REGION),
        max_age_seconds=15*60,
    )

@st.cache_data(ttl=300,show_spinner=False)
def load_regional_radar_fast(region_key, altitude_km, resolution_km):
    bucket=time_bucket(5)
    key=f"cappi-{bucket}-z{float(altitude_km):g}-dx{float(resolution_km):g}"
    return disk_cached(
        "regional_radar",
        key,
        lambda: build_latest_regional_radar_fast(
            region=ACTIVE_REGION,
            altitude_km=altitude_km,
            horizontal_resolution_km=resolution_km,
        ),
        max_age_seconds=15*60,
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_surface(region_key, cycle_iso, forecast_hour, variables):
    var_key="-".join(sorted(variables))
    key=f"{cycle_iso}-f{int(forecast_hour):02d}-sfc-{var_key}"
    return disk_cached(
        "hrrr",
        key,
        lambda: build_hrrr_surface(
            region=ACTIVE_REGION,
            cycle=pd.Timestamp(cycle_iso),
            forecast_hour=forecast_hour,
            variables=list(variables),
        ),
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_pressure(region_key, cycle_iso, forecast_hour, variables):
    var_key="-".join(sorted(variables))
    key=f"{cycle_iso}-f{int(forecast_hour):02d}-prs-{var_key}"
    return disk_cached(
        "hrrr",
        key,
        lambda: build_hrrr_pressure(
            region=ACTIVE_REGION,
            cycle=pd.Timestamp(cycle_iso),
            forecast_hour=forecast_hour,
            variables=list(variables),
        ),
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_environment(region_key, target_time_iso, pressure_hpa, variables):
    """Load HRRR upper-air context, preferring the persistent forecast cube."""
    target=pd.Timestamp(target_time_iso)
    target=target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    valid=target.floor("1h")

    stored,match=load_stored_hrrr_valid_member(
        ACTIVE_REGION,"pressure",valid,max_offset_minutes=30
    )
    if stored is not None and match is not None:
        keep=[v for v in variables if v in stored]
        if keep:
            ds=stored[keep]
            if "pressure_hpa" in ds.coords:
                ds=ds.sel(pressure_hpa=float(pressure_hpa),method="nearest")
            from aria.models.base import ModelRun
            run=ModelRun(
                model="hrrr",
                initialization_time=pd.Timestamp(match["initialization_time"]),
                forecast_hour=int(match["forecast_hour"]),
                valid_time=pd.Timestamp(match["valid_time"]),
                source="ARIA persistent HRRR pressure forecast cube",
                product="prs",
            )
            return ds,run

    # Fallback for profiles that have not yet accumulated the pressure cube.
    cycles=hrrr_available_cycles(count=24)
    candidates=[]
    for cycle in cycles:
        ct=pd.Timestamp(cycle)
        ct=ct.tz_localize("UTC") if ct.tzinfo is None else ct.tz_convert("UTC")
        fxx=int(round((valid-ct).total_seconds()/3600.0))
        if 0 <= fxx <= 18:
            candidates.append((abs((ct+pd.Timedelta(hours=fxx)-valid).total_seconds()),ct,fxx))
    if not candidates:
        raise RuntimeError("No recent HRRR cycle can be matched to this radar time.")
    _,cycle,fxx=min(candidates,key=lambda x:x[0])
    return load_hrrr_pressure(region_key,cycle.isoformat(),fxx,tuple(variables))


@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_asos_comparison(
    region_key, cycle_iso, forecast_hour, variable,
    observation_offset_minutes=30, observation_match_mode="nearest",
):
    key=(
        f"{cycle_iso}-f{int(forecast_hour):02d}-{variable}-stations"
        f"-obs{int(observation_offset_minutes)}-{observation_match_mode}"
    )
    return disk_cached(
        "comparisons",
        key,
        lambda: compare_hrrr_surface_with_asos(
            region=ACTIVE_REGION,
            cycle=pd.Timestamp(cycle_iso),
            forecast_hour=forecast_hour,
            variable=variable,
            observation_offset_minutes=observation_offset_minutes,
            observation_match_mode=observation_match_mode,
        ),
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_gridded_comparison(
    region_key, cycle_iso, forecast_hour, variable,
    observation_offset_minutes=30, observation_match_mode="nearest",
):
    key=(
        f"{cycle_iso}-f{int(forecast_hour):02d}-{variable}-surface-grid"
        f"-obs{int(observation_offset_minutes)}-{observation_match_mode}"
    )
    return disk_cached(
        "comparisons",
        key,
        lambda: build_hrrr_surface_gridded_comparison(
            region=ACTIVE_REGION,
            cycle=pd.Timestamp(cycle_iso),
            forecast_hour=forecast_hour,
            variable=variable,
            observation_offset_minutes=observation_offset_minutes,
            observation_match_mode=observation_match_mode,
        ),
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_radar_comparison(region_key, cycle_iso, forecast_hour, model_variable, radar_resolution):
    key=(
        f"{cycle_iso}-f{int(forecast_hour):02d}-{model_variable}"
        f"-radar-dx{float(radar_resolution):g}"
    )
    return disk_cached(
        "comparisons",
        key,
        lambda: build_hrrr_radar_comparison(
            region=ACTIVE_REGION,
            cycle=pd.Timestamp(cycle_iso),
            forecast_hour=forecast_hour,
            model_variable=model_variable,
            radar_resolution_km=radar_resolution,
        ),
    )

@st.cache_data(ttl=300,show_spinner=False)
def load_mrms(region_key, valid_iso=None):
    if valid_iso is None:
        ds = load_stored_dataset(ACTIVE_REGION, "mrms", "reflectivity", max_age_minutes=15)
        if ds is None:
            cfg = profile_for_region(ACTIVE_REGION, require_enabled=True)
            if cfg is not None and cfg.get("sources",{}).get("mrms",{}).get("enabled"):
                update_storage(source="mrms", profile=cfg.get("profile_name"))
                ds = load_stored_dataset(ACTIVE_REGION, "mrms", "reflectivity", max_age_minutes=15)
        if ds is not None:
            scan_time = ds.attrs.get("analysis_time") or ds.attrs.get("valid_time")
            if scan_time is None and "time" in ds.coords and ds["time"].size:
                scan_time = ds["time"].values[-1]
            return ds, (pd.Timestamp(scan_time) if scan_time is not None else None)
    key=f"mrms-{valid_iso or time_bucket(5)}"
    return disk_cached(
        "mrms",
        key,
        lambda: build_mrms_at_time(pd.Timestamp(valid_iso) if valid_iso else None, region=ACTIVE_REGION),
        max_age_seconds=15*60 if valid_iso is None else None,
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_mrms_comparison(region_key, cycle_iso, forecast_hour, model_variable):
    key=f"{cycle_iso}-f{int(forecast_hour):02d}-{model_variable}-mrms"
    return disk_cached(
        "comparisons",
        key,
        lambda: build_hrrr_mrms_comparison(
            region=ACTIVE_REGION,
            cycle=pd.Timestamp(cycle_iso),
            forecast_hour=forecast_hour,
            model_variable=model_variable,
        ),
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_mrms_adapt_objects(region_key, cycle_iso, forecast_hour, model_variable, threshold_dbz, min_gridpoints, h_maxima_dbz, max_match_distance_km):
    key=(
        f"{cycle_iso}-f{int(forecast_hour):02d}-{model_variable}-adapt"
        f"-z{float(threshold_dbz):g}-n{int(min_gridpoints)}"
        f"-h{float(h_maxima_dbz):g}-d{float(max_match_distance_km):g}"
    )
    return disk_cached(
        "storm_objects",
        key,
        lambda: build_hrrr_mrms_adapt_objects(
            region=ACTIVE_REGION,
            cycle=pd.Timestamp(cycle_iso),
            forecast_hour=forecast_hour,
            model_variable=model_variable,
            threshold_dbz=threshold_dbz,
            min_gridpoints=min_gridpoints,
            h_maxima_dbz=h_maxima_dbz,
            max_match_distance_km=max_match_distance_km,
        ),
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_radar_leads(region_key, valid_iso, model_variable, forecast_hours, thresholds=(20,30,40)):
    leads="-".join(str(int(x)) for x in forecast_hours)
    threshold_key="-".join(str(int(x)) for x in thresholds)
    key=f"{valid_iso}-{model_variable}-radar-leads-{leads}-thr-{threshold_key}"
    return disk_cached(
        "comparisons",
        key,
        lambda: build_hrrr_radar_lead_time_verification(
            pd.Timestamp(valid_iso),
            region=ACTIVE_REGION,
            model_variable=model_variable,
            forecast_hours=tuple(forecast_hours),
            thresholds=tuple(thresholds),
            store_comparisons=False,
        ),
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_lead_verification(region_key, valid_iso, variable, forecast_hours):
    leads="-".join(str(int(x)) for x in forecast_hours)
    key=f"{valid_iso}-{variable}-leads-{leads}"
    return disk_cached(
        "comparisons",
        key,
        lambda: build_hrrr_lead_time_verification(
            pd.Timestamp(valid_iso),
            region=ACTIVE_REGION,
            variable=variable,
            forecast_hours=tuple(forecast_hours),
        ),
    )

@st.cache_data(ttl=3600,show_spinner=False)
def load_hrrr_raob_comparison(region_key, cycle_iso, forecast_hour, variable, station_id=None):
    key=f"{cycle_iso}-f{int(forecast_hour):02d}-{variable}-raob-{station_id or 'auto'}"
    return disk_cached(
        "comparisons",
        key,
        lambda: build_hrrr_raob_comparison(
            region=ACTIVE_REGION,
            cycle=pd.Timestamp(cycle_iso),
            forecast_hour=forecast_hour,
            variable=variable,
            station_id=station_id,
        ),
    )


@st.cache_data(ttl=3600,show_spinner=False)
def load_latest_hrrr_raob(region_key, variable, station_id=None):
    key=f"latest-auto-{variable}-raob-{station_id or 'auto'}"
    return disk_cached(
        "comparisons", key,
        lambda: build_latest_hrrr_raob_comparison(
            region=ACTIVE_REGION, variable=variable, station_id=station_id
        ),
    )

analysis_views=["Data Cube","Surface / Atmosphere","Radar","Regional Radar","Model","Model Evaluation","3-D Atmosphere","Coverage"]
st.sidebar.markdown("### Navigate")
workspace=st.sidebar.radio(
    "Workspace",["Analysis","Configuration"],horizontal=True,key="aria-workspace",
    label_visibility="collapsed",
)
if workspace=="Configuration":
    view=st.sidebar.radio(
        "Configuration page",["Region Configuration","Data Storage"],
        key="aria-configuration-page",
    )
else:
    view=st.sidebar.radio(
        "Analysis view",analysis_views,
        key="aria-analysis-view",
    )

# Keep storage health visible from every page without mixing storage controls
# into the scientific analysis navigation.
_sidebar_storage_cfg=load_storage_config()
_sidebar_profiles=list_storage_profiles(_sidebar_storage_cfg)
_enabled_profiles=[name for name,spec in _sidebar_profiles.items() if spec.get("enabled")]
_sidebar_storage_state=load_storage_state()
if _enabled_profiles:
    _hb=_sidebar_storage_state.get("updated_utc") or _sidebar_storage_state.get("last_collector_finish_utc")
    if _hb:
        try:
            _hb_age=(pd.Timestamp.now(tz="UTC")-pd.Timestamp(_hb)).total_seconds()/60.0
            _storage_badge=(f"● {len(_enabled_profiles)} storage profile(s) enabled" if _hb_age <= 15 else f"⚠ {len(_enabled_profiles)} profile(s); heartbeat stale")
        except Exception:
            _storage_badge=f"● {len(_enabled_profiles)} storage profile(s) enabled"
    else:
        _storage_badge=f"⚠ {len(_enabled_profiles)} profile(s); collector not run"
    st.sidebar.caption(_storage_badge)
else:
    st.sidebar.caption("○ All storage profiles paused")

if view=="Region Configuration":
    _show_region_configuration()
    st.stop()

refresh_minutes={"Data Cube":5,"Surface / Atmosphere":5,"Radar":5,"Regional Radar":5,"Model":30,"Model Evaluation":5,"3-D Atmosphere":60,"Coverage":5}
if view=="Data Storage":
    # Monitoring reruns are lightweight: they reread the state file, not the
    # scientific data sources. Collection remains an external/background job.
    st_autorefresh(interval=15*1000,key="storage-monitor-refresh")
elif view=="Data Cube":
    # Data Cube refreshes only local persistent-store metadata. It never
    # triggers downloads or source collection merely because a browser is open.
    with st.sidebar.expander("Data Cube refresh",expanded=False):
        refresh_choice=st.selectbox(
            "Auto-refresh",["Off","1 min","5 min","15 min"],index=2,
            key="data-cube-refresh-cadence",
        )
        _refresh_lookup={"1 min":1,"5 min":5,"15 min":15}
        if refresh_choice!="Off":
            st_autorefresh(
                interval=_refresh_lookup[refresh_choice]*60*1000,
                key="data-cube-local-refresh",
            )
        if st.button("Refresh now",width="stretch",key="data-cube-refresh-now"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Refresh rereads local ARIA storage/state only; collection remains external.")
else:
    with st.sidebar.expander("App controls",expanded=False):
        auto=st.toggle("Auto-update",value=True,key=f"auto-toggle-{view}")
        if auto:
            st_autorefresh(interval=refresh_minutes[view]*60*1000,key=f"auto-{view}")
        st.caption(f"View refresh cadence: {refresh_minutes[view]} min")
        if st.button("Refresh from sources",width="stretch",key=f"refresh-{view}"):
            invalidate_latest_mrms_cache()
            processed_cache().clear()
            st.cache_data.clear()
            st.rerun()
        cache_bytes=processed_cache().size_bytes()
        st.caption(f"Processed cache: {cache_bytes/1024**2:.1f} MB")
        if st.button("Clear processed cache",width="stretch",key=f"clear-cache-{view}"):
            processed_cache().clear()
            st.cache_data.clear()
            st.rerun()

def _open_data_storage():
    """Navigate from a scientific view directly to Data Storage."""
    st.session_state["aria-workspace"]="Configuration"
    st.session_state["aria-configuration-page"]="Data Storage"


def color_controls(label,default_cmap,finite,default_min=None,default_max=None):
    """Compact display-scale controls used consistently across views."""
    cmaps=["coolwarm","viridis","plasma","turbo","BrBG","RdBu_r","cividis"]
    with st.sidebar.expander(f"Color scale — {label}",expanded=False):
        cmap=st.selectbox(
            "Colormap",cmaps,
            index=cmaps.index(default_cmap) if default_cmap in cmaps else 0,
            key=f"color-{label}-cmap",
        )
        auto_scale=st.toggle("Automatic limits",value=True,key=f"color-{label}-auto")
        f=np.asarray(finite); f=f[np.isfinite(f)]
        amin=float(np.nanpercentile(f,2)) if f.size else (default_min or 0.)
        amax=float(np.nanpercentile(f,98)) if f.size else (default_max or 1.)
        if default_min is not None: amin=default_min
        if default_max is not None: amax=default_max
        if auto_scale:
            return cmap,None,None
        vmin=st.number_input("Minimum",value=float(amin),key=f"color-{label}-min")
        vmax=st.number_input("Maximum",value=float(amax),key=f"color-{label}-max")
        if vmax<=vmin:
            st.warning("Maximum must be greater than minimum.")
        return cmap,vmin,vmax

def status_line(source,data_time,cadence,origin=None):
    now=pd.Timestamp.now(tz="UTC")
    t=pd.Timestamp(data_time)
    t=t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    age=(now-t).total_seconds()/60
    origin_text=" • persistent store" if origin == "persistent_store" else ""
    st.caption(f"**{source}:** {t:%Y-%m-%d %H:%M UTC} • {age:.0f} min old • refresh {cadence} min{origin_text}")

if view=="Data Storage":
    st.subheader("Data Storage")
    st.caption("Persistent regional collection is configured here and runs independently of the scientific views. The monitor refreshes every 15 seconds; source retrieval only occurs when the collector runs.")
    cfg=load_storage_config()
    profiles=list_storage_profiles(cfg)
    active_profile=cfg.get("profile_name") or cfg.get("active_profile")
    profile_names=list(profiles)
    selected_profile=st.selectbox(
        "Storage profile", profile_names,
        index=profile_names.index(active_profile) if active_profile in profile_names else 0,
        format_func=lambda name: profiles[name].get("display_name") or name,
        help="Each profile collects independently. The background collector evaluates all enabled profiles on every run.",
    )
    if selected_profile != active_profile:
        set_active_profile(selected_profile)
        st.rerun()
    cfg=load_storage_config(selected_profile)
    state=load_storage_state(selected_profile)
    configured_region=cfg.get("region",{}).get("name","Unknown")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Profile", selected_profile)
    c2.metric("Collection", "Enabled" if cfg.get("enabled") else "Paused")
    c3.metric("Profile stored", f"{storage_size_bytes(cfg)/1024**3:.2f} GB")
    heartbeat=state.get("last_collector_finish_utc") or "Never"
    c4.metric("Last collector", heartbeat.replace("+00:00","Z")[:19] if heartbeat!="Never" else heartbeat)
    st.caption(f"Persistent store: `{region_store_root(cfg)}` • all profiles: {total_storage_size_bytes(cfg)/1024**3:.2f} GB")
    with st.expander("Create another storage profile"):
        new_profile_name=st.text_input("Profile name",key="storage-new-profile-name")
        st.caption(f"New profile will use the current dashboard region: **{ACTIVE_REGION.name}**. You can change the dashboard region first if needed.")
        copy_settings=st.checkbox("Copy source/cadence settings from this profile",True,key="storage-copy-profile-settings")
        if st.button("Create storage profile",key="storage-create-profile"):
            if not new_profile_name.strip():
                st.warning("Enter a profile name first.")
            else:
                try:
                    create_storage_profile(
                        new_profile_name.strip(),ACTIVE_REGION,
                        copy_from=selected_profile if copy_settings else None,
                        enabled=False,
                    )
                    st.success(f"Created storage profile {new_profile_name.strip()} (paused by default).")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    if cfg.get("enabled"):
        hb=state.get("last_collector_finish_utc")
        if not hb:
            st.warning("Storage is enabled, but the background collector has not run yet. Use **Update Due Sources Now** for a one-time pass or configure cron/launchd/systemd for unattended collection.")
        else:
            try:
                hb_age=(pd.Timestamp.now(tz="UTC")-pd.Timestamp(hb)).total_seconds()/60.0
                if hb_age > 15:
                    st.warning(f"Storage is enabled, but the last collector heartbeat was {hb_age:.0f} minutes ago. The dashboard does not need to stay open, but a background scheduler must invoke `aria storage update`.")
            except Exception:
                pass

    if cfg.get("region",{}).get("name") != ACTIVE_REGION.name or cfg.get("region",{}).get("west") != ACTIVE_REGION.west or cfg.get("region",{}).get("east") != ACTIVE_REGION.east:
        st.info(f"Storage is configured for **{configured_region}** while the dashboard is viewing **{ACTIVE_REGION.name}**.")
        if st.button("Use current dashboard region for storage",width="content"):
            cfg["region"]=ACTIVE_REGION.as_dict()
            save_storage_config(cfg)
            st.success(
                f"Storage region changed to {ACTIVE_REGION.name}. Sources were marked due. "
                "Any incompatible existing Zarr grids will be preserved under _grid_archive and a new active grid will be initialized."
            )
            st.rerun()

    st.markdown("#### Sources and scheduling")
    st.caption("**Check interval** controls how often ARIA looks for something new. **Native cadence** describes how often the upstream product normally changes; the two are intentionally different.")
    interval_options=[5,10,15,30,60,180,360,720,1440]
    def _interval_label(minutes):
        if minutes < 60: return f"Every {minutes} min"
        if minutes == 60: return "Every hour"
        if minutes < 1440: return f"Every {minutes//60} hours"
        return "Daily" if minutes == 1440 else f"Every {minutes/1440:g} days"
    with st.form(f"storage-source-config-{selected_profile}"):
        working=cfg.get("sources",{})
        h1,h2,h3=st.columns([2.2,1.4,1.8])
        h1.markdown("**Data source**"); h2.markdown("**Check interval**"); h3.markdown("**Native product cadence**")
        for key, defaults in DEFAULT_SOURCES.items():
            spec=working.setdefault(key,dict(defaults))
            current=int(spec.get("interval_minutes",defaults["interval_minutes"]))
            options=sorted(set(interval_options+[current]))
            a,b,c=st.columns([2.2,1.4,1.8])
            spec["enabled"]=a.checkbox(SOURCE_LABELS.get(key,key),value=bool(spec.get("enabled",False)),key=f"store-enable-{selected_profile}-{key}")
            spec["interval_minutes"]=int(b.selectbox("Check interval",options,index=options.index(current),format_func=_interval_label,key=f"store-interval-{selected_profile}-{key}",label_visibility="collapsed"))
            c.caption(SOURCE_NATIVE_CADENCE.get(key,""))

        if "hrrr" in working:
            hrrr_spec=working["hrrr"]
            st.markdown("##### HRRR regional forecast cube")
            st.caption(
                "Choose how much HRRR is persistently collected for this regional profile. "
                "Standard is the recommended ARIA data-cube configuration."
            )
            _tier_label={
                "basic":"Basic",
                "standard":"Standard",
                "full_campaign":"Full / Campaign",
            }
            _tier_reverse={v:k for k,v in _tier_label.items()}
            _current_tier=str(hrrr_spec.get("storage_tier","standard")).lower().replace("-","_")
            if _current_tier not in _tier_label:
                _current_tier="standard"
            _chosen=st.selectbox(
                "HRRR storage tier",
                list(_tier_reverse),
                index=list(_tier_reverse).index(_tier_label[_current_tier]),
                key=f"store-hrrr-tier-{selected_profile}",
            )
            hrrr_spec["storage_tier"]=_tier_reverse[_chosen]

            if hrrr_spec["storage_tier"]=="basic":
                hrrr_spec["forecast_hours"]=[0]
                hrrr_spec["pressure_levels_hpa"]=[]
                st.caption(
                    "Basic: F00 surface fields and radar reflectivity only. "
                    "No persistent pressure-level atmosphere."
                )
            else:
                _default_leads=(
                    [0,1,3,6,12]
                    if hrrr_spec["storage_tier"]=="standard"
                    else list(range(0,19))
                )
                _current_leads=[
                    int(x) for x in hrrr_spec.get("forecast_hours",_default_leads)
                    if 0 <= int(x) <= 18
                ]
                hrrr_spec["forecast_hours"]=st.multiselect(
                    "Forecast leads to persist",
                    list(range(0,19)),
                    default=_current_leads or _default_leads,
                    format_func=lambda x:f"F{x:02d}",
                    key=f"store-hrrr-leads-{selected_profile}",
                )

                _standard_levels=[1000,925,850,700,500,300,250]
                _full_levels=[
                    1000,950,925,900,850,800,750,700,650,600,
                    550,500,450,400,350,300,250,200,150,100,
                ]
                _level_options=_full_levels
                _level_default=(
                    _standard_levels
                    if hrrr_spec["storage_tier"]=="standard"
                    else _full_levels
                )
                _current_levels=[
                    int(x) for x in hrrr_spec.get("pressure_levels_hpa",_level_default)
                    if int(x) in _level_options
                ]
                hrrr_spec["pressure_levels_hpa"]=st.multiselect(
                    "Pressure levels to persist",
                    _level_options,
                    default=_current_levels or _level_default,
                    format_func=lambda x:f"{x} hPa",
                    key=f"store-hrrr-levels-{selected_profile}",
                )
                if hrrr_spec["storage_tier"]=="standard":
                    st.caption(
                        "Standard variables: 2-m temperature/dew point, 10-m winds, "
                        "composite and 1-km reflectivity; pressure-level temperature, "
                        "RH, U/V wind, and geopotential height."
                    )
                else:
                    st.caption(
                        "Full / Campaign additionally stores hourly F00–F18 by default, "
                        "more pressure levels, dew point, vertical velocity, surface "
                        "pressure, and precipitation rate. Storage/network use can be substantial."
                    )

        if "nexrad" in working:
            nex=working["nexrad"]
            st.markdown("##### Individual NEXRAD persistence")
            st.caption(
                "Optional native-polar radar persistence follows the Radar DataTree / "
                "Icechunk direction. Keep it off unless you intentionally want individual "
                "Level-II radar storage in addition to MRMS."
            )
            radar_mode=st.radio(
                "Radar storage scope",
                ["Selected radars","All radars in region"],
                index=1 if nex.get("storage_mode")=="all_region" else 0,
                horizontal=True,key=f"store-nexrad-mode-{selected_profile}",
            )
            nex["storage_mode"]="all_region" if radar_mode.startswith("All") else "selected"
            region_radars=sorted(nexrad_sites_for_region(region_from_dict(cfg["region"])))
            if nex["storage_mode"]=="selected":
                defaults=[x for x in nex.get("selected_radars",[]) if x in region_radars]
                nex["selected_radars"]=st.multiselect(
                    "Radars to persist",
                    region_radars,
                    default=defaults,
                    format_func=lambda rid_:nexrad_display_name(rid_, {}),
                    key=f"store-nexrad-sites-{selected_profile}",
                )
            else:
                nex["selected_radars"]=[]
                st.caption(
                    f"{len(region_radars)} regional radar site(s) will be selected automatically."
                )
            nex["native_polar"]=True
            st.info(
                "v0.24 stores this profile configuration but keeps the native Radar DataTree writer "
                "guarded until the xradar/radar-datatree writer is enabled. MRMS collection is unchanged."
            )
        if st.form_submit_button("Save storage configuration",type="primary"):
            cfg["sources"]=working; save_storage_config(cfg); st.success("Storage configuration saved."); st.rerun()

    b1,b2,b3=st.columns(3)
    if not cfg.get("enabled"):
        if b1.button("Start Storage",type="primary",width="stretch"):
            cfg["enabled"]=True; save_storage_config(cfg); st.success(f"Storage profile {selected_profile} enabled. The external scheduler will continue collection after the dashboard closes."); st.rerun()
    else:
        if b1.button("Pause Storage",width="stretch"):
            cfg["enabled"]=False; save_storage_config(cfg); st.warning(f"Storage profile {selected_profile} paused."); st.rerun()
    if b2.button("Update All Due Profiles Now",width="stretch"):
        with st.spinner("Running the storage collector across enabled profiles..."):
            update_storage()
        st.rerun()
    if b3.button("Force Update All Profiles",width="stretch"):
        with st.spinner("Forcing enabled source updates across enabled profiles..."):
            update_storage(force=True)
        st.rerun()

    state=load_storage_state()
    rows=storage_status_rows(load_storage_config(),state)
    display=pd.DataFrame(rows)
    if not display.empty:
        def _fmt_interval(v):
            return _interval_label(int(v)) if pd.notna(v) else "—"
        display["Check interval"]=display["interval_minutes"].map(_fmt_interval)
        display=display[["label","enabled","status","storage_format","product_cadence","Check interval","data_time","last_attempt_utc","last_success_utc","next_check_utc","error"]]
        display.columns=["Source","Enabled","Status","Store format","Native cadence","Check interval","Latest data","Last attempt","Last success","Next due","Error"]
        st.dataframe(display,width="stretch",hide_index=True)

    st.markdown("#### All storage profiles")
    all_rows=pd.DataFrame(all_status_rows())
    if not all_rows.empty:
        # Status rows can contain a mix of ISO strings, pandas timestamps,
        # None, and NaN when a profile/source has never completed. Pandas
        # cannot reliably apply max() to that object-dtype mixture. Normalize
        # first so newly-created profiles are safe to summarize.
        all_rows["last_success_dt"]=pd.to_datetime(
            all_rows.get("last_success_utc"),
            utc=True,
            errors="coerce",
        )
        summary=(all_rows.groupby("profile",as_index=False)
                 .agg(
                     enabled_sources=("enabled","sum"),
                     sources=("source","count"),
                     last_success=("last_success_dt","max"),
                     errors=("status",lambda s:int((s=="error").sum())),
                 ))
        summary["last_success"]=summary["last_success"].apply(
            lambda value: value.isoformat() if pd.notna(value) else "—"
        )
        st.dataframe(summary,width="stretch",hide_index=True)

    with st.expander("Historical retrieval / backfill (v0.21)"):
        st.caption("Historical backfill writes into the same profile stores as live collection. v0.21 initially supports MRMS and HRRR; additional observation archives will use this same interface.")
        hc1,hc2=st.columns(2)
        hstart=hc1.date_input("Start date",key="history-start")
        hend=hc2.date_input("End date",key="history-end")
        hsources=st.multiselect("Historical sources",["mrms","hrrr"],default=["mrms","hrrr"],key="history-sources")
        if st.button("Plan historical retrieval",key="history-plan"):
            if not hsources:
                st.warning("Choose at least one historical source.")
            else:
                try:
                    plan=historical_plan(selected_profile,str(hstart),str(hend),tuple(hsources))
                    st.json(plan)
                    src_arg=",".join(hsources)
                    st.code(f"aria history build --profile {selected_profile} --start {hstart} --end {hend} --sources {src_arg}",language="bash")
                    st.caption("For large backfills, run the command outside Streamlit so it can continue independently of the dashboard.")
                except Exception as exc:
                    st.error(str(exc))

    st.markdown("#### Recent activity")
    st.caption("This is an append-only activity history. Earlier error rows remain visible after a later successful retry; use the source status table above for current state.")
    history=pd.DataFrame(state.get("history",[])[-25:][::-1])
    if history.empty: st.caption("No storage activity recorded yet.")
    else: st.dataframe(history,width="stretch",hide_index=True)

    with st.expander("Cloud-optimized ARIA storage"):
        st.caption(
            "Gridded/multidimensional scientific data are stored as Icechunk-backed Zarr v3. "
            "Point/tabular observations are stored as date-partitioned Parquet datasets. "
            "NetCDF is migration/import only and is not used for active ARIA persistence."
        )
        if st.button("Convert legacy files to cloud-optimized storage",width="content"):
            with st.spinner("Converting NetCDF to Icechunk/Zarr v3 and snapshot Parquet to partitioned Parquet..."):
                migration_result=migrate_legacy_snapshots(archive_legacy=True)
            st.json(migration_result)
            st.success("Conversion complete. Successfully migrated legacy files were moved outside the active storage root.")
        st.code("aria storage migrate",language="bash")

    with st.expander("Run continuously without the dashboard"):
        st.code("*/5 * * * * /full/path/to/aria storage update >> ~/.cache/aria/storage/collector.log 2>&1",language="bash")
        st.caption("The cron entry only wakes ARIA. Each source's cadence is controlled by the dashboard configuration, and source-level locks prevent overlapping collectors.")
        st.code("""aria storage status
aria storage update
aria storage update --force
aria storage enable
aria storage disable""",language="bash")
    st.stop()

if view=="Data Cube":
    st.subheader("Integrated ARCO Data Cube")
    st.caption(
        "See when ARIA's regional observations and models are actually available together. "
        "The timeline reflects timestamps in the persistent local data stores—not simply "
        "whether a collector is enabled."
    )
    _cube_refreshed=pd.Timestamp.now(tz="UTC")
    st.caption(f"Local timeline refreshed: {_cube_refreshed:%Y-%m-%d %H:%M:%S UTC} • collection is independent of this view")
    cfg=profile_for_region(ACTIVE_REGION,require_enabled=False)
    if cfg is None:
        st.info(
            "No storage profile is associated with this region yet. Configure a profile "
            "to begin building a persistent regional data cube."
        )
        st.button(
            "Configure Data Sources & Storage",
            type="primary",
            on_click=_open_data_storage,
            key="data-cube-configure-empty",
        )
    else:
        rows=pd.DataFrame(storage_status_rows(cfg,profile_name=cfg.get("profile_name")))
        coverage=source_time_coverage(cfg,profile_name=cfg.get("profile_name"))
        coverage_df=pd.DataFrame(coverage)

        available=set()
        if not coverage_df.empty:
            available=set(
                coverage_df.loc[coverage_df["count"]>0,"source"].astype(str).tolist()
            )

        # Integrated-readiness summary.
        analysis_count=sum([
            "surface" in available,
            {"hrrr","surface"}.issubset(available),
            {"hrrr","mrms"}.issubset(available),
            {"hrrr","radiosonde"}.issubset(available),
            {"sst","surface"}.issubset(available),
        ])

        # Compute continuous all-source overlap only across sources that
        # actually have continuous coverage segments. Discrete products such
        # as radiosondes and SST are not allowed to imply continuous overlap.
        continuous_sets=[]
        for item in coverage:
            if bool(item.get("discrete",False)):
                continue
            segs=[]
            for a,b in item.get("segments",[]):
                aa=pd.to_datetime(a,utc=True,errors="coerce")
                bb=pd.to_datetime(b,utc=True,errors="coerce")
                if not pd.isna(aa) and not pd.isna(bb) and bb>=aa:
                    segs.append((aa,bb))
            if segs:
                continuous_sets.append(segs)

        common_segments=[]
        if continuous_sets:
            common_segments=continuous_sets[0]
            for segs in continuous_sets[1:]:
                intersections=[]
                for a1,b1 in common_segments:
                    for a2,b2 in segs:
                        a=max(a1,a2)
                        b=min(b1,b2)
                        if b>=a:
                            intersections.append((a,b))
                common_segments=intersections
                if not common_segments:
                    break

        common_start=common_segments[-1][0] if common_segments else None
        common_end=common_segments[-1][1] if common_segments else None
        common_ok=bool(common_segments)

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Region",ACTIVE_REGION.name)
        c2.metric("Datasets available",len(available))
        c3.metric("Integrated analyses",analysis_count)
        c4.metric(
            "Latest all-source overlap",
            f"{common_end:%b %d %H:%M UTC}" if common_ok else "—",
        )

        b1,b2=st.columns([1,3])
        with b1:
            st.button(
                "Configure Data Sources & Storage",
                type="primary",
                on_click=_open_data_storage,
                key="data-cube-configure",
                width="stretch",
            )
        with b2:
            st.caption(
                "Configuration opens the existing Data Storage page for this profile. "
                "The Data Cube page remains focused on scientific availability."
            )

        st.markdown("#### Dataset availability timeline")
        window_label=st.selectbox(
            "Timeline window",
            ["24 hours","3 days","7 days","30 days","All available"],
            index=2,
            key="data-cube-timeline-window",
        )
        now=pd.Timestamp.now(tz="UTC")
        window_hours={
            "24 hours":24,
            "3 days":72,
            "7 days":168,
            "30 days":720,
        }
        window_start=(
            now-pd.Timedelta(hours=window_hours[window_label])
            if window_label in window_hours else None
        )

        try:
            import plotly.graph_objects as go

            fig=go.Figure()
            labels=[]
            for item in coverage:
                if int(item.get("count",0))<=0:
                    continue
                label=item.get("label",item.get("source",""))
                labels.append(label)
                discrete=bool(item.get("discrete",False))

                if discrete:
                    pts=pd.to_datetime(item.get("times",[]),utc=True,errors="coerce")
                    pts=[p for p in pts if not pd.isna(p) and (window_start is None or p>=window_start)]
                    if pts:
                        # Keep the full set for ordinary windows. For unusually
                        # long archives, thin only the rendered markers while
                        # preserving first/last and the reported count.
                        if len(pts)>600:
                            pick=np.linspace(0,len(pts)-1,600).round().astype(int)
                            pts=[pts[i] for i in pick]
                        fig.add_trace(go.Scatter(
                            x=pts,
                            y=[label]*len(pts),
                            mode="markers",
                            marker=dict(size=9),
                            name=label,
                            showlegend=False,
                            hovertemplate=(
                                f"{label}<br>%{{x|%Y-%m-%d %H:%M UTC}}"
                                "<extra></extra>"
                            ),
                        ))
                else:
                    rendered=False
                    for seg_start,seg_end in item.get("segments",[]):
                        a=pd.to_datetime(seg_start,utc=True,errors="coerce")
                        b=pd.to_datetime(seg_end,utc=True,errors="coerce")
                        if pd.isna(a) or pd.isna(b):
                            continue
                        if window_start is not None and b<window_start:
                            continue
                        if window_start is not None:
                            a=max(a,window_start)
                        fig.add_trace(go.Scatter(
                            x=[a,b],
                            y=[label,label],
                            mode="lines",
                            line=dict(width=12),
                            name=label,
                            showlegend=False,
                            hovertemplate=(
                                f"{label}<br>%{{x|%Y-%m-%d %H:%M UTC}}"
                                "<extra></extra>"
                            ),
                        ))
                        rendered=True

                    isolated=pd.to_datetime(
                        item.get("isolated_times",[]),
                        utc=True,
                        errors="coerce",
                    )
                    isolated=[
                        p for p in isolated
                        if not pd.isna(p) and (window_start is None or p>=window_start)
                    ]
                    if isolated:
                        fig.add_trace(go.Scatter(
                            x=isolated,
                            y=[label]*len(isolated),
                            mode="markers",
                            marker=dict(size=9),
                            name=label,
                            showlegend=False,
                            hovertemplate=(
                                f"{label}<br>%{{x|%Y-%m-%d %H:%M UTC}}"
                                "<extra></extra>"
                            ),
                        ))
                        rendered=True

                    if not rendered:
                        last=pd.to_datetime(item.get("last_time"),utc=True,errors="coerce")
                        if not pd.isna(last) and (window_start is None or last>=window_start):
                            fig.add_trace(go.Scatter(
                                x=[last],y=[label],mode="markers",
                                marker=dict(size=9),showlegend=False,
                                hovertemplate=(
                                    f"{label}<br>%{{x|%Y-%m-%d %H:%M UTC}}"
                                    "<extra></extra>"
                                ),
                            ))

            if common_ok:
                for n,(seg_start,seg_end) in enumerate(common_segments):
                    if window_start is not None and seg_end<window_start:
                        continue
                    shade_start=max(seg_start,window_start) if window_start is not None else seg_start
                    fig.add_vrect(
                        x0=shade_start,x1=seg_end,
                        opacity=0.10,line_width=0,
                        annotation_text="Continuous overlap" if n==0 else None,
                        annotation_position="top left",
                    )

            fig.update_layout(
                height=max(360,110+48*max(1,len(labels))),
                margin=dict(l=10,r=20,t=25,b=40),
                hovermode="closest",
                xaxis_title="UTC",
                yaxis_title="",
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=list(reversed(labels)),
                ),
            )
            if window_start is not None:
                fig.update_xaxes(range=[window_start,now])
            st.plotly_chart(fig,width="stretch")

            st.caption(
                "Thick bars show locally stored continuous coverage; markers show "
                "discrete products such as radiosonde launches and daily SST analyses. "
                "Breaks in a bar indicate a gap in the local time series."
            )
        except Exception as exc:
            st.warning(f"Timeline visualization unavailable: {exc}")

        st.markdown("#### Scientific availability")
        if not coverage_df.empty:
            roles={
                "surface":"Regional thermodynamic/wind analysis",
                "mrms":"Observed radar / event context",
                "hrrr":"Model environment and verification",
                "radiosonde":"Upper-air profile validation",
                "sst":"Air–water thermal context",
                "air_quality":"Regional aerosol/gas context",
                "marine":"Marine observations",
            }
            display=coverage_df.copy()
            display=display[display["count"]>0]
            if not display.empty:
                display["Start"]=pd.to_datetime(display["first_time"],utc=True,errors="coerce")
                display["Latest valid data"]=pd.to_datetime(display["last_time"],utc=True,errors="coerce")
                display["Collector checked"]=pd.to_datetime(display["last_success_utc"],utc=True,errors="coerce")
                display["Analysis role"]=display["source"].map(roles).fillna("Regional analysis")
                display["Available times"]=display["count"].astype(int)
                display["Timeline source"]=display["timeline_source"].replace({
                    "parquet_snapshots":"Parquet snapshots",
                    "icechunk_time_coordinate":"Icechunk time",
                    "collector_state":"Collector state",
                })
                display["Rejected future times"]=display["rejected_future_times"].astype(int)
                display=display[
                    ["label","Start","Latest valid data","Available times","Collector checked",
                     "Timeline source","Rejected future times","Analysis role"]
                ]
                display.columns=[
                    "Dataset","Start","Latest valid data","Available times","Collector checked",
                    "Timeline source","Rejected future times","Analysis role"
                ]
                st.dataframe(display,width="stretch",hide_index=True)

        if common_ok:
            st.success(
                "Latest continuous overlap among continuous sources: "
                f"{common_start:%Y-%m-%d %H:%M UTC} → {common_end:%Y-%m-%d %H:%M UTC}. "
                "Radiosondes and SST remain discrete markers and are not treated as "
                "continuous coverage."
            )
        else:
            st.info(
                "There is not yet a common time span shared by every available dataset."
            )

        st.markdown("#### Integrated analyses ready now")
        capabilities=[]
        if "surface" in available:
            capabilities.append("Regional surface analysis and station context")
        if {"hrrr","surface"}.issubset(available):
            capabilities.append("HRRR ↔ surface model–observation comparison")
        if {"hrrr","mrms"}.issubset(available):
            capabilities.append("HRRR ↔ MRMS reflectivity verification and synchronized playback")
        if "mrms" in available:
            capabilities.append("Regional radar playback and event analysis")
        if {"hrrr","radiosonde"}.issubset(available):
            capabilities.append("HRRR ↔ radiosonde profile comparison")
        if {"sst","surface"}.issubset(available):
            capabilities.append("Air + lake/sea temperature regional context")
        if not capabilities:
            capabilities.append("Collect overlapping datasets to unlock integrated analyses.")
        for item in capabilities:
            st.markdown(f"- {item}")
    st.stop()

if view in ("Surface / Atmosphere","Coverage"):
    if view=="Coverage":
        st.sidebar.markdown("#### Data")
        with st.sidebar.expander("Analysis settings",expanded=False):
            surface_method=st.selectbox("Interpolation",["Barnes","Gaussian","IDW"],index=0,key="coverage-interpolation")
            surface_smoothing=st.slider("Smoothing scale (km)",40.0,300.0,140.0,10.0,key="coverage-smoothing")
            surface_support=st.slider("Maximum station support (km)",75.0,400.0,250.0,25.0,key="coverage-support")
        surface,obs,start,end=load_surface(
            REGION_CACHE_KEY,surface_method.lower(),surface_smoothing,surface_support,False
        )
        idx=surface.sizes["time"]-1
        variable=st.sidebar.selectbox("Coverage layer",[
            "air_temperature_f_nearest_distance_km",
            "air_temperature_f_n_contributing",
            "air_temperature_f_effective_age_minutes",
        ])
        fld=surface[variable].isel(time=idx)
        cmap,vmin,vmax=color_controls("Coverage","viridis",fld.values)
        fig=plot_interactive_field(
            fld,ACTIVE_REGION,title=variable,units=fld.attrs.get("units",""),
            zmin=vmin,zmax=vmax,colorscale=mpl_to_plotly_colorscale(cmap),
        )
        fig.update_layout(height=max(fig.layout.height or 0,760))
        st.plotly_chart(fig,width="stretch")
    else:
        st.subheader("Surface / Atmosphere")
        st.sidebar.markdown("#### Data")
        vertical_mode=st.sidebar.selectbox(
            "Vertical level",
            ["Surface","Elevated atmosphere"],
            index=0,
            help="Surface is the default. Choose Elevated atmosphere to move through the observational 3-D analysis by height MSL.",
        )

        if vertical_mode=="Elevated atmosphere":
            atmosphere,profiles,trajectories,points=load_atmosphere(REGION_CACHE_KEY)
            # Some source combinations provide only U/V. Derive speed so Wind
            # Speed always remains a valid visualization choice.
            if "u_wind" in atmosphere and "v_wind" in atmosphere and (
                "wind_speed" not in atmosphere or not np.isfinite(atmosphere["wind_speed"].values).any()
            ):
                atmosphere=atmosphere.copy()
                atmosphere["wind_speed"]=np.hypot(atmosphere["u_wind"],atmosphere["v_wind"])
                atmosphere["wind_speed"].attrs.update(long_name="Wind Speed",units="kt",derived_from="u_wind,v_wind")
            available=[v for v in VARIABLE_STYLE if v in atmosphere]
            variable=st.sidebar.selectbox(
                "Atmospheric variable",available,
                format_func=lambda v:VARIABLE_STYLE[v]["label"],
            )
            altitudes=atmosphere.altitude_km.values.tolist()
            altitude=st.sidebar.select_slider(
                "Height (km MSL)",options=altitudes,
                value=min(altitudes,key=lambda x:abs(x-3.0)),
            )
            style=VARIABLE_STYLE[variable]
            data=atmosphere[variable].sel(altitude_km=altitude,method="nearest").values
            cmap,vmin,vmax=color_controls(style["label"],style["cmap"],data)
            st.sidebar.markdown("#### Display")
            show_barbs=st.sidebar.checkbox("Show wind barbs",value=(variable=="wind_speed"),key="surface-atmos-barbs")
            density=st.sidebar.select_slider(
                "Barb density",options=["Sparse","Medium","Dense"],value="Medium",
                key="surface-atmos-barb-density",
            ) if show_barbs else "Medium"
            fig=plot_atmosphere_slice(
                atmosphere,variable,altitude,ACTIVE_REGION,
                cmap=cmap,vmin=vmin,vmax=vmax,
                wind_barbs=show_barbs,
                barb_skip={"Sparse":7,"Medium":4,"Dense":2}[density],
            )
            st.pyplot(fig,width="stretch"); plt.close(fig)
            cycle=atmosphere.attrs.get("raob_cycle")
            if cycle:
                status_line("RAOB / atmospheric analysis",cycle,60)
        else:
            with st.sidebar.expander("Analysis settings",expanded=False):
                surface_method=st.selectbox("Interpolation",["Barnes","Gaussian","IDW"],index=0,key="surface-interpolation")
                surface_smoothing=st.slider("Smoothing scale (km)",40.0,300.0,140.0,10.0,key="surface-smoothing")
                surface_support=st.slider("Maximum station support (km)",75.0,400.0,250.0,25.0,key="surface-support")
                surface_history=st.checkbox(
                    "Load previous hour",value=False,key="surface-history",
                    help="Off computes only the latest analysis; enable to build the full 5-minute timeline."
                )
            surface,obs,start,end=load_surface(
                REGION_CACHE_KEY,surface_method.lower(),surface_smoothing,surface_support,surface_history
            )
            # Build the menu from fields actually present in the surface
            # analysis so future ASOS additions are not silently dropped by a
            # short hard-coded UI list.
            gridded_surface_fields=[
                key for key in SURFACE_FIELD_STYLE
                if key in surface and not key.endswith(("_nearest_distance_km","_n_contributing","_effective_age_minutes"))
            ]
            surface_options=gridded_surface_fields + [
                "PM2.5","Ozone","Sea/Lake Surface Temperature","Combined Air + Lake Temperature"
            ]
            surface_layer=st.sidebar.selectbox(
                "Surface variable",
                surface_options,
                index=surface_options.index("air_temperature_f") if "air_temperature_f" in surface_options else 0,
                format_func=lambda v:SURFACE_FIELD_STYLE.get(v,(v,"",""))[0],
            )
            idx=surface.sizes["time"]-1
            if surface.sizes["time"]>1:
                idx=st.sidebar.slider("Analysis time",0,surface.sizes["time"]-1,idx)
            st.sidebar.markdown("#### Display")
            show_land_stations=st.sidebar.checkbox("Show land stations",True,key="surface-show-land-stations")
            show_marine_stations=st.sidebar.checkbox("Show marine stations",True,key="surface-show-marine-stations")
            show_station_values=st.sidebar.checkbox("Show station values",False,key="surface-show-station-values")
            show_station_ids=st.sidebar.checkbox("Show station IDs",False,key="surface-show-station-ids")
            wind_overlay=st.sidebar.selectbox(
                "Wind overlay",["None","Arrows","Streamlines"],index=0,key="surface-wind-overlay"
            )
            wind_density=st.sidebar.select_slider(
                "Wind density",options=["Sparse","Medium","Dense"],value="Medium",key="surface-wind-density"
            ) if wind_overlay!="None" else "Medium"

            if surface_layer in ("PM2.5","Ozone"):
                with st.sidebar.expander("Air-quality options",expanded=True):
                    aq_radius=st.slider("Monitor support radius (km)",100.0,500.0,300.0,25.0,key="surface-aq-radius")
                    show_aq_sites=st.checkbox("Show monitoring sites",True,key=f"surface-{surface_layer.lower()}-sites")
                    show_aq_contours=st.checkbox("Show concentration contours",False,key=f"surface-{surface_layer.lower()}-contours")
                try:
                    cube,latest,allobs,aq_time=load_air_quality(REGION_CACHE_KEY, aq_radius)
                    status_line("AirNow",aq_time,30,cube.attrs.get("aria_data_origin"))
                    aq_variable="pm25" if surface_layer=="PM2.5" else "ozone"
                    aq_label="PM2.5" if aq_variable=="pm25" else "Ozone"
                    _unit_values=latest.loc[latest.variable==aq_variable,"units"].dropna().astype(str).unique().tolist() if "units" in latest.columns else []
                    aq_units=_unit_values[0] if _unit_values else ("µg m⁻³" if aq_variable=="pm25" else "ppb")
                    default_cmap="viridis" if aq_variable=="pm25" else "plasma"
                    field=cube[aq_variable].isel(time=-1) if "time" in cube[aq_variable].dims else cube[aq_variable]
                    cmap,vmin,vmax=color_controls(aq_label,default_cmap,field.values,0.0,None)
                    fig=plot_interactive_field(
                        field,ACTIVE_REGION,title=f"Surface {aq_label}",units=aq_units,
                        zmin=vmin,zmax=vmax,colorscale=mpl_to_plotly_colorscale(cmap),
                        show_contours=show_aq_contours,
                    )
                    if show_aq_sites and latest is not None and len(latest):
                        import plotly.graph_objects as go
                        aq_obs=latest[latest.variable==aq_variable].copy()
                        if len(aq_obs):
                            value_col=next((c for c in ("value","concentration",aq_variable) if c in aq_obs.columns),None)
                            custom=np.asarray(aq_obs[value_col],float)[:,None] if value_col else None
                            text=[str(x) for x in aq_obs.get("station_id",pd.Series(index=aq_obs.index,dtype=str))]
                            hover="%{text}<br>Lon %{x:.2f}<br>Lat %{y:.2f}" + (f"<br>%{{customdata[0]:.1f}} {aq_units}" if custom is not None else "") + "<extra></extra>"
                            labels=[]
                            vals=np.asarray(aq_obs[value_col],float) if value_col else np.full(len(aq_obs),np.nan)
                            for sid,val in zip(text,vals):
                                parts=[]
                                if show_station_ids: parts.append(str(sid))
                                if show_station_values and np.isfinite(val):
                                    parts.append(f"{val:.1f}")
                                labels.append(" ".join(parts))
                            mode="markers+text" if (show_station_values or show_station_ids) else "markers"
                            fig.add_trace(go.Scatter(
                                x=aq_obs.longitude,y=aq_obs.latitude,mode=mode,name=f"{aq_label} monitors",
                                marker=dict(size=7,color="white",line=dict(color="black",width=1)),
                                text=labels,textposition="top center",
                                customdata=custom,hovertemplate=hover,
                            ))
                    fig.update_layout(height=max(fig.layout.height or 0,760))
                    st.plotly_chart(fig,width="stretch")
                    st.caption(f"{aq_label} is integrated into Surface / Atmosphere; monitoring-site and contour controls are available here.")
                except Exception as exc:
                    st.error(str(exc))

            elif surface_layer=="Sea/Lake Surface Temperature":
                status_line("ASOS",surface.time.values[idx],5,surface.attrs.get("aria_data_origin"))
                sst_product=choose_sst_product(ACTIVE_REGION)
                try:
                    sst_ds,_,sst_origin=load_sst(REGION_CACHE_KEY)
                    if sst_ds is None:
                        raise RuntimeError(
                            "No locally stored SST is available for this region. "
                            "Enable Sea/Lake Surface Temperature in Data Storage and run the collector."
                        )
                    field_name="sst_f" if "sst_f" in sst_ds else ("sst_c" if "sst_c" in sst_ds else None)
                    if field_name is None:
                        raise RuntimeError("SST source did not contain a recognized temperature field.")
                    field=sst_ds[field_name]
                    if "time" in field.dims:
                        field=field.isel(time=-1)
                    field=_display_thin_2d(field,max_points_per_axis=220)
                    units="°F" if field_name=="sst_f" else "°C"
                    cmap,vmin,vmax=color_controls("Lake / Sea Temperature","turbo",field.values)
                    fig=plot_interactive_field(
                        field,ACTIVE_REGION,title=sst_product.label,units=units,
                        zmin=vmin,zmax=vmax,colorscale=mpl_to_plotly_colorscale(cmap),
                    )
                    marine=load_marine(REGION_CACHE_KEY)
                    if show_marine_stations and marine is not None and not marine.empty:
                        import plotly.graph_objects as go
                        m=marine.copy()
                        custom=np.column_stack([
                            pd.to_numeric(m.get("air_temperature_c"),errors="coerce"),
                            pd.to_numeric(m.get("water_temperature_c"),errors="coerce"),
                        ])
                        fig.add_trace(go.Scatter(
                            x=m.longitude,y=m.latitude,mode="markers",name="Marine stations",
                            marker=dict(size=7,color="white",line=dict(color="black",width=1)),
                            text=m.station_id.astype(str),customdata=custom,
                            hovertemplate="%{text}<br>Air %{customdata[0]:.1f} °C<br>Water %{customdata[1]:.1f} °C<extra></extra>",
                        ))
                    if show_land_stations:
                        fig=_add_surface_station_overlay(
                            fig,obs,surface.time.values[idx],
                            show_values=show_station_values,show_ids=show_station_ids,
                        )
                    if wind_overlay!="None" and "u_wind_kt" in surface and "v_wind_kt" in surface:
                        fig=add_interactive_wind_overlay(
                            fig,surface["u_wind_kt"].isel(time=idx),surface["v_wind_kt"].isel(time=idx),
                            ACTIVE_REGION,mode=wind_overlay,density=wind_density,
                        )
                    fig.update_layout(height=max(fig.layout.height or 0,760))
                    st.plotly_chart(fig,width="stretch")
                    sst_time=(
                        pd.to_datetime(sst_ds["time"].values[-1],utc=True,errors="coerce")
                        if "time" in sst_ds.coords and sst_ds["time"].size else "unknown"
                    )
                    finite_count=int(np.isfinite(np.asarray(field.values,dtype=float)).sum())
                    st.caption(
                        f"SST: {sst_product.label} • {sst_time} • {finite_count:,} valid grid cells • "
                        f"local Icechunk/Zarr store. "
                        "Marine station hover values are shown where available."
                    )
                except Exception as exc:
                    st.warning(str(exc))
            else:
                status_line("ASOS",surface.time.values[idx],5,surface.attrs.get("aria_data_origin"))
                if surface_layer=="Combined Air + Lake Temperature":
                    base=surface.air_temperature_f.isel(time=idx)
                    cmap,vmin,vmax=color_controls("Temperature","coolwarm",base.values)
                    title="Surface Air + Lake Temperature"
                    units="°F"
                    fig=plot_interactive_field(
                        base,ACTIVE_REGION,title=title,units=units,
                        zmin=vmin,zmax=vmax,colorscale=mpl_to_plotly_colorscale(cmap),
                    )
                    sst_overlay_info=None
                    try:
                        import plotly.graph_objects as go
                        sst_ds,sst_product,sst_origin=load_sst(REGION_CACHE_KEY)
                        if sst_ds is None:
                            raise RuntimeError(
                                "No locally stored SST is available. Enable SST collection in Data Storage "
                                "and run or force the storage collector."
                            )
                        if "sst_f" in sst_ds:
                            sst=sst_ds["sst_f"]
                        elif "sst_c" in sst_ds:
                            sst=sst_ds["sst_c"]*9/5+32
                        else:
                            raise RuntimeError("SST dataset has no sst_f or sst_c field.")

                        if "time" in sst.dims:
                            sst=sst.isel(time=-1)
                        sst=sst.squeeze(drop=True)

                        if "longitude" not in sst.coords or "latitude" not in sst.coords:
                            raise RuntimeError("SST dataset is missing latitude/longitude coordinates.")
                        if sst.ndim != 2:
                            raise RuntimeError(f"SST field is {sst.ndim}-D after selection; expected a 2-D surface field.")

                        full_values=np.asarray(sst.values,dtype=float)
                        full_valid=np.isfinite(full_values)
                        if not full_valid.any():
                            raise RuntimeError("SST field contains no finite values in the active region.")

                        sst_display=_display_thin_2d(sst,max_points_per_axis=180)
                        sst_values=np.asarray(sst_display.values,dtype=float)
                        valid=np.isfinite(sst_values)

                        fig.add_trace(go.Heatmap(
                            x=np.asarray(sst_display["longitude"].values),
                            y=np.asarray(sst_display["latitude"].values),
                            z=np.where(valid,sst_values,np.nan),
                            zmin=vmin,zmax=vmax,
                            colorscale=mpl_to_plotly_colorscale(cmap),
                            showscale=False,
                            opacity=1.0,
                            hoverongaps=False,
                            hovertemplate="Lon %{x:.2f}<br>Lat %{y:.2f}<br>Water %{z:.1f} °F<extra></extra>",
                            name="Lake / sea surface temperature",
                        ))
                        sst_time=(
                            pd.to_datetime(sst_ds["time"].values[-1],utc=True,errors="coerce")
                            if "time" in sst_ds.coords and sst_ds["time"].size else "unknown"
                        )
                        sst_overlay_info=(
                            f"{sst_product.label} • {sst_time} • "
                            f"{int(full_valid.sum()):,} valid SST cells in storage • "
                            f"{int(valid.sum()):,} rendered • local Icechunk/Zarr store"
                        )
                    except Exception as exc:
                        st.warning(f"Lake/sea-temperature overlay unavailable: {exc}")
                else:
                    field_key=surface_layer if surface_layer in SURFACE_FIELD_STYLE else "air_temperature_f"
                    base=surface[field_key].isel(time=idx)
                    label,units,default_cmap=SURFACE_FIELD_STYLE[field_key]
                    nonnegative=field_key in {
                        "relative_humidity_pct","wind_speed_kt","wind_gust_kt",
                        "precipitation_1h_in","visibility_mi","cloud_base_ft",
                    }
                    cmap,vmin,vmax=color_controls(
                        label,default_cmap,base.values,0.0 if nonnegative else None,None
                    )
                    title=f"Surface {label}"
                    fig=plot_interactive_field(
                        base,ACTIVE_REGION,title=title,units=units,
                        zmin=vmin,zmax=vmax,colorscale=mpl_to_plotly_colorscale(cmap),
                    )

                marine=load_marine(REGION_CACHE_KEY)
                if show_marine_stations and marine is not None and not marine.empty:
                    import plotly.graph_objects as go
                    m=marine.copy()
                    air_c=pd.to_numeric(m.get("air_temperature_c"),errors="coerce")
                    water_c=pd.to_numeric(m.get("water_temperature_c"),errors="coerce")
                    custom=np.column_stack([air_c,water_c])
                    fig.add_trace(go.Scatter(
                        x=m.longitude,y=m.latitude,mode="markers",name="Marine stations",
                        marker=dict(size=7,color="white",line=dict(color="black",width=1)),
                        text=m.station_id.astype(str),customdata=custom,
                        hovertemplate="%{text}<br>Air %{customdata[0]:.1f} °C<br>Water %{customdata[1]:.1f} °C<extra></extra>",
                    ))
                if show_land_stations:
                    fig=_add_surface_station_overlay(
                        fig,obs,surface.time.values[idx],
                        show_values=show_station_values,show_ids=show_station_ids,
                    )
                if wind_overlay!="None" and "u_wind_kt" in surface and "v_wind_kt" in surface:
                    fig=add_interactive_wind_overlay(
                        fig,
                        surface["u_wind_kt"].isel(time=idx),
                        surface["v_wind_kt"].isel(time=idx),
                        ACTIVE_REGION,mode=wind_overlay,density=wind_density,
                    )
                fig.update_layout(height=max(fig.layout.height or 0,760))
                st.plotly_chart(fig,width="stretch")
                if surface_layer=="Combined Air + Lake Temperature":
                    st.caption(
                        "Air temperature and lake temperature remain distinct physical fields; the combined display uses one temperature scale only for visual context. "
                        "Marine stations report air and/or water temperature according to platform sensor availability."
                    )



elif view=="Radar":
    st.sidebar.markdown("#### Data")
    _radar_meta=nexrad_sites_for_region(ACTIVE_REGION)
    _radar_ids=sorted(_radar_meta) or sorted(GPGL_NEXRAD_SITES)
    # Use explicit human-readable option strings rather than relying on
    # Streamlit format_func so the dropdown visibly shows city/state.
    _radar_labels={}
    for rid_ in _radar_ids:
        _canonical=normalize_nexrad_id(rid_)
        _meta=_radar_meta.get(rid_,{}) or _radar_meta.get(_canonical,{})
        _radar_labels[rid_]=nexrad_display_name(_canonical,_meta)
    _label_to_radar={label:rid_ for rid_,label in _radar_labels.items()}
    _default_radar="KMPX" if "KMPX" in _radar_ids else _radar_ids[0]
    _radar_options=[_radar_labels[rid_] for rid_ in _radar_ids]
    _selected_radar_label=st.sidebar.selectbox(
        "Radar site",
        _radar_options,
        index=_radar_options.index(_radar_labels[_default_radar]),
    )
    rid=_label_to_radar[_selected_radar_label]
    radar_mode=st.sidebar.radio("Radar display",["PPI / Sweep","3-D Volume"],index=0)
    try:
        st.sidebar.markdown("#### Playback")
        radar_playback=st.sidebar.toggle("Animate recent scans",value=False,key=f"radar-playback-{rid}")
        if radar_playback and radar_mode=="PPI / Sweep":
            lookback=st.sidebar.selectbox("Animation window",[30,60,120],index=1,format_func=lambda v:f"Last {v} min",key=f"radar-window-{rid}")
            max_frames=st.sidebar.selectbox("Maximum frames",[6,12,18],index=1,key=f"radar-frames-{rid}")
            speed=st.sidebar.selectbox("Playback speed",[0.5,1.0,2.0],index=1,format_func=lambda v:f"{v:g}×",key=f"radar-speed-{rid}")
            radar_playing=st.sidebar.toggle("▶ Play",value=False,key=f"radar-playing-{rid}")
            with st.spinner("Loading recent NEXRAD volumes for local playback..."):
                recent_loaded=load_recent_radars(rid,int(lookback),int(max_frames))
            if recent_loaded:
                if radar_playing:
                    tick=st_autorefresh(interval=max(400,int(1400/float(speed))),key=f"radar-animation-tick-{rid}")
                    radar_frame=tick % len(recent_loaded)
                else:
                    radar_frame=st.sidebar.slider("Animation frame",0,len(recent_loaded)-1,len(recent_loaded)-1,key=f"radar-frame-{rid}")
                radar,scan=recent_loaded[radar_frame]
                st.sidebar.caption(f"Frame {radar_frame+1}/{len(recent_loaded)} • {pd.Timestamp(scan.scan_time):%H:%M:%S UTC}")
            else:
                radar,scan=load_radar(rid)
        else:
            radar,scan=load_radar(rid)
        status_line(rid,scan.scan_time,5)
        if radar_mode=="PPI / Sweep":
            sweeps=radar_sweep_summary(radar); sw=st.sidebar.selectbox("Elevation sweep",sweeps.sweep.tolist(),
                format_func=lambda s:f"Sweep {s} — {sweeps.loc[sweeps.sweep==s,'elevation_deg'].iloc[0]:.1f}°")
            st.sidebar.markdown("#### Display")
            cmap,vmin,vmax=color_controls("Reflectivity","turbo",np.array([]),-30,70)
            st.sidebar.markdown("#### Environment")
            env_mode=st.sidebar.selectbox("Environmental overlay",["None","HRRR upper air","Surface observations","HRRR + Surface"],key=f"radar-env-mode-{rid}")
            env_ds=None; env_run=None; env_obs=None; env_surface_ds=None
            env_pressure=500; env_temp=False; env_wind=True; env_height=True; env_density="Medium"; env_temp_interval=2.0; env_height_interval=60.0
            env_shaded_field="Temperature"; env_shading_opacity=0.28
            surface_env_mode="Stations"; surface_env_variable="air_temperature_f"; surface_env_labels=True; surface_env_opacity=0.30
            if "HRRR" in env_mode:
                env_pressure=st.sidebar.selectbox("Pressure level",[850,700,500,300],index=2,key=f"radar-env-level-{rid}")
                env_shaded_field=st.sidebar.selectbox(
                    "Environmental shading",
                    ["None","Temperature","Relative humidity"],
                    index=1,key=f"radar-env-shaded-{rid}",
                )
                if env_shaded_field!="None":
                    env_shading_opacity=st.sidebar.slider(
                        "Environmental shading opacity",0.10,0.55,0.28,0.05,
                        key=f"radar-env-shading-opacity-{rid}",
                    )
                env_temp=st.sidebar.checkbox(
                    "Temperature line contours",False,key=f"radar-env-temp-{rid}"
                )
                env_temp_interval=st.sidebar.selectbox(
                    "Temperature contour interval",[1.0,2.0,5.0,10.0],index=1,
                    format_func=lambda v:f"{v:g} °C",
                    key=f"radar-env-temp-step-{rid}",
                ) if env_temp else 2.0
                env_height=st.sidebar.checkbox(
                    "Geopotential-height contours",True,key=f"radar-env-height-{rid}"
                )
                env_height_interval=st.sidebar.selectbox(
                    "Height contour interval",[30.0,60.0,120.0],index=1,
                    format_func=lambda v:f"{v:g} m",
                    key=f"radar-env-height-step-{rid}",
                ) if env_height else 60.0
                env_wind=st.sidebar.checkbox(
                    "Wind barbs/arrows",True,key=f"radar-env-wind-{rid}"
                )
                env_density=st.sidebar.selectbox(
                    "Wind density",["Sparse","Medium","Dense"],index=1,
                    key=f"radar-env-density-{rid}"
                )
                env_vars=[]
                if env_shaded_field=="Temperature" or env_temp:
                    env_vars.append("air_temperature")
                if env_shaded_field=="Relative humidity":
                    env_vars.append("relative_humidity")
                if env_wind:
                    env_vars.extend(["u_wind","v_wind"])
                if env_height:
                    env_vars.append("geopotential_height")
                env_vars=list(dict.fromkeys(env_vars))
                if env_vars:
                    try:
                        with st.spinner(f"Matching {env_pressure}-hPa HRRR environment to radar time..."):
                            env_ds,env_run=load_hrrr_environment(
                                REGION_CACHE_KEY,pd.Timestamp(scan.scan_time).isoformat(),
                                env_pressure,tuple(env_vars)
                            )
                    except Exception as exc:
                        st.warning(f"HRRR environmental overlay unavailable: {exc}")
            if "Surface" in env_mode:
                surface_env_mode=st.sidebar.selectbox(
                    "Surface display",
                    ["Stations","Contours","Shaded analysis","Contours + stations"],
                    index=0,key=f"radar-surface-display-{rid}",
                )
                _surface_env_options=[
                    "air_temperature_f","dew_point_temperature_f","relative_humidity_pct",
                    "wind_speed_kt","sea_level_pressure_mb",
                ]
                surface_env_variable=st.sidebar.selectbox(
                    "Surface variable",_surface_env_options,index=0,
                    format_func=lambda v:SURFACE_FIELD_STYLE.get(v,(v,"",""))[0],
                    key=f"radar-surface-variable-{rid}",
                )
                if "Contours" in surface_env_mode:
                    surface_env_labels=st.sidebar.checkbox(
                        "Show contour labels",True,key=f"radar-surface-contour-labels-{rid}"
                    )
                if surface_env_mode=="Shaded analysis":
                    surface_env_opacity=st.sidebar.slider(
                        "Surface shading opacity",0.10,0.60,0.30,0.05,
                        key=f"radar-surface-opacity-{rid}",
                    )
                env_obs=load_stored_frame(ACTIVE_REGION,"surface","observations",max_age_minutes=180)
                try:
                    target=pd.Timestamp(scan.scan_time)
                    target=target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
                    env_surface_ds=load_stored_time_range(
                        ACTIVE_REGION,"surface","analysis",
                        target-pd.Timedelta(minutes=90),target+pd.Timedelta(minutes=90),
                        max_frames=24,
                    )
                    if env_surface_ds is not None and "time" in env_surface_ds.coords and env_surface_ds.sizes.get("time",0):
                        tt=pd.to_datetime(env_surface_ds["time"].values,utc=True,errors="coerce")
                        nearest=int(np.nanargmin(np.abs((tt-target).total_seconds())))
                        env_surface_ds=env_surface_ds.isel(time=nearest)
                except Exception:
                    env_surface_ds=None
            adapt_labels=None
            st.sidebar.markdown("#### Analysis")
            if adapt_available():
                use_adapt=st.sidebar.checkbox("ADAPT storm-cell detection",value=False,key="radar-adapt-enable")
                if use_adapt:
                    adapt_threshold=st.sidebar.slider("ADAPT threshold (dBZ)",20.0,60.0,35.0,1.0,key="radar-adapt-threshold")
                    adapt_min_points=st.sidebar.slider("ADAPT minimum grid points",1,100,8,1,key="radar-adapt-minpoints")
                    adapt_hmax=st.sidebar.slider("ADAPT h-maxima (dBZ)",1.0,15.0,5.0,1.0,key="radar-adapt-hmax")
                    field_name=next((n for n in ("reflectivity","corrected_reflectivity","reflectivity_horizontal") if n in radar.fields),None)
                    if field_name is not None:
                        ray0=int(radar.sweep_start_ray_index["data"][sw])
                        ray1=int(radar.sweep_end_ray_index["data"][sw])+1
                        sweep_field=np.asarray(np.ma.filled(radar.fields[field_name]["data"][ray0:ray1,:],np.nan),dtype=float)
                        da=xr.DataArray(
                            sweep_field,
                            dims=("y","x"),
                            coords={"y":np.arange(sweep_field.shape[0]),"x":np.arange(sweep_field.shape[1])},
                            name="reflectivity",
                        )
                        try:
                            adapt_labels=segment_with_adapt(
                                da,
                                threshold_dbz=adapt_threshold,
                                min_gridpoints=adapt_min_points,
                                h_maxima_dbz=adapt_hmax,
                            )
                        except Exception as exc:
                            st.warning(f"ADAPT segmentation could not be applied to this sweep: {exc}")
            else:
                st.sidebar.caption("ADAPT is not installed; storm-cell detection is unavailable.")

            radar_plot_type=st.sidebar.radio(
                "PPI plot type",["Interactive","Classic Py-ART"],index=0,key=f"radar-plot-type-{rid}"
            )
            if radar_playback:
                # Playback uses a true static Matplotlib/Py-ART frame. Avoid
                # rebuilding an interactive Plotly widget or depending on
                # Kaleido image export for every animation tick.
                fig=plot_static_nexrad_ppi(
                    radar,sweep=sw,cmap=cmap,
                    vmin=-30 if vmin is None else vmin,
                    vmax=70 if vmax is None else vmax,
                    adapt_labels=adapt_labels,
                )
                if env_ds is not None:
                    add_mpl_environment_overlay(
                        fig.axes[0],env_ds,ACTIVE_REGION,pressure_hpa=env_pressure,
                        show_temperature=env_temp,show_wind=env_wind,show_height=env_height,
                        wind_density=env_density,temperature_interval=env_temp_interval,
                        height_interval=env_height_interval,shaded_field=env_shaded_field,
                        shading_opacity=env_shading_opacity,
                    )
                if env_surface_ds is not None and surface_env_mode!="Stations":
                    _add_surface_analysis_overlay_mpl(
                        fig.axes[0],env_surface_ds,surface_env_variable,surface_env_mode,
                        show_labels=surface_env_labels,opacity=surface_env_opacity,
                    )
                if env_obs is not None and not env_obs.empty and surface_env_mode in ("Stations","Contours + stations"):
                    _add_surface_station_overlay_mpl(fig.axes[0],env_obs,scan.scan_time)
                st.pyplot(fig,width="stretch")
                plt.close(fig)
            elif radar_plot_type=="Interactive":
                fig=plot_interactive_nexrad_ppi(
                    radar,sweep=sw,region=None,
                    vmin=-30 if vmin is None else vmin,
                    vmax=70 if vmax is None else vmax,
                    adapt_labels=adapt_labels,
                )
                if env_ds is not None:
                    fig=add_interactive_environment_overlay(
                        fig,env_ds,ACTIVE_REGION,pressure_hpa=env_pressure,
                        show_temperature=env_temp,show_wind=env_wind,show_height=env_height,
                        wind_density=env_density,temperature_interval=env_temp_interval,
                        height_interval=env_height_interval,shaded_field=env_shaded_field,
                        shading_opacity=env_shading_opacity,
                    )
                if env_surface_ds is not None and surface_env_mode!="Stations":
                    fig=_add_surface_analysis_overlay_plotly(
                        fig,env_surface_ds,surface_env_variable,surface_env_mode,
                        show_labels=surface_env_labels,opacity=surface_env_opacity,
                    )
                if env_obs is not None and not env_obs.empty and surface_env_mode in ("Stations","Contours + stations"):
                    fig=_add_surface_station_overlay(fig,env_obs,scan.scan_time,show_values=True,show_ids=False)
                st.plotly_chart(fig,width="stretch")
            else:
                fig=plot_nexrad_ppi(
                    radar,sweep=sw,cmap=cmap,
                    vmin=-30 if vmin is None else vmin,
                    vmax=70 if vmax is None else vmax,
                    adapt_labels=adapt_labels,
                )
                if env_ds is not None:
                    add_mpl_environment_overlay(
                        fig.axes[0],env_ds,ACTIVE_REGION,pressure_hpa=env_pressure,
                        show_temperature=env_temp,show_wind=env_wind,show_height=env_height,
                        wind_density=env_density,temperature_interval=env_temp_interval,
                        height_interval=env_height_interval,shaded_field=env_shaded_field,
                        shading_opacity=env_shading_opacity,
                    )
                if env_surface_ds is not None and surface_env_mode!="Stations":
                    _add_surface_analysis_overlay_mpl(
                        fig.axes[0],env_surface_ds,surface_env_variable,surface_env_mode,
                        show_labels=surface_env_labels,opacity=surface_env_opacity,
                    )
                if env_obs is not None and not env_obs.empty and surface_env_mode in ("Stations","Contours + stations"):
                    _add_surface_station_overlay_mpl(fig.axes[0],env_obs,scan.scan_time)
                st.pyplot(fig,width="stretch"); plt.close(fig)
            if env_run is not None:
                st.caption(f"Environmental context: HRRR {env_pressure} hPa • init {pd.Timestamp(env_run.initialization_time):%Y-%m-%d %H UTC} • F{int(env_run.forecast_hour):02d} • valid {pd.Timestamp(env_run.valid_time):%Y-%m-%d %H UTC}")
            if adapt_labels is not None:
                gate_lat,gate_lon,_=radar.get_gate_lat_lon_alt(sw)
                objects=summarize_objects(adapt_labels,da,prefix="T",latitude=gate_lat,longitude=gate_lon)
                st.caption(f"ADAPT {adapt_version() or ''} detected {len(objects)} storm cells on sweep {sw}. White outlines and T# labels show detected objects.")
                if len(objects):
                    st.dataframe(objects,width="stretch",hide_index=True)
        else:
            st.sidebar.markdown("#### 3-D settings")
            dx=st.sidebar.selectbox("Grid spacing",[4.0,2.0,1.0],index=1,format_func=lambda v:f"{v:g} km")
            max_range=st.sidebar.slider("3-D range (km)",75.0,230.0,180.0,25.0)
            threshold=st.sidebar.slider("Minimum reflectivity (dBZ)",-10.0,50.0,20.0,5.0)
            opacity=st.sidebar.slider("Isosurface opacity",0.10,0.80,0.25,0.05)
            with st.spinner(f"Gridding {rid} into a 3-D volume..."):
                volume=grid_single_radar_reflectivity(radar,horizontal_resolution_km=dx,max_range_km=max_range)
            fig=plot_regional_reflectivity_3d(volume,threshold_dbz=threshold,surface_count=4,opacity=opacity)
            fig.update_layout(height=850); st.plotly_chart(fig,width="stretch")
    except Exception as exc: st.error(str(exc))

elif view=="Regional Radar":
    st.sidebar.markdown("#### Data")
    mode=st.sidebar.radio(
        "Regional radar mode",
        ["MRMS QC Composite","Fast selected altitude","Detailed 3-D regional volume"],
        index=0,
    )

    try:
        if mode=="MRMS QC Composite":
            st.sidebar.markdown("#### Environment")
            regional_env=st.sidebar.toggle("HRRR upper-air overlay",value=False,key="regional-env-enable")
            reg_env_ds=None; reg_env_run=None
            reg_pressure=500; reg_temp=True; reg_wind=True; reg_height=False; reg_density="Medium"; reg_temp_interval=2.0; reg_height_interval=60.0
            if regional_env:
                reg_pressure=st.sidebar.selectbox("Pressure level",[850,700,500,300],index=2,key="regional-env-level")
                reg_temp=st.sidebar.checkbox("Temperature contours",True,key="regional-env-temp")
                reg_temp_interval=st.sidebar.selectbox("Temperature contour interval",[1.0,2.0,5.0,10.0],index=1,format_func=lambda v:f"{v:g} °C",key="regional-env-temp-step") if reg_temp else 2.0
                reg_wind=st.sidebar.checkbox("Wind barbs/arrows",True,key="regional-env-wind")
                reg_height=st.sidebar.checkbox("Geopotential-height contours",False,key="regional-env-height")
                reg_height_interval=st.sidebar.selectbox("Height contour interval",[30.0,60.0,120.0],index=1,format_func=lambda v:f"{v:g} m",key="regional-env-height-step") if reg_height else 60.0
                reg_density=st.sidebar.selectbox("Wind density",["Sparse","Medium","Dense"],index=1,key="regional-env-density")
            st.sidebar.markdown("#### Playback")
            animate=st.sidebar.toggle("Animate regional radar",value=False,key="regional-mrms-playback")
            cube=None; scan_time=None
            if animate:
                window=st.sidebar.selectbox(
                    "Animation window",[1,3,6,12],index=1,
                    format_func=lambda h:f"Last {h} h",key="regional-mrms-window",
                )
                max_frames=st.sidebar.selectbox(
                    "Maximum frames",[12,24,36,60],index=2,key="regional-mrms-maxframes"
                )
                hist=load_mrms_history(REGION_CACHE_KEY,float(window),int(max_frames))
                if hist is not None and "time" in hist.coords and hist.sizes.get("time",0)>1:
                    times=pd.to_datetime(hist.time.values,utc=True,errors="coerce")
                    speed=st.sidebar.selectbox(
                        "Playback speed",[0.5,1.0,2.0],index=1,
                        format_func=lambda v:f"{v:g}×",key="regional-mrms-speed",
                    )
                    playing=st.sidebar.toggle("▶ Play",value=False,key="regional-mrms-playing")
                    if playing:
                        tick=st_autorefresh(
                            interval=max(400,int(1400/float(speed))),
                            key="regional-mrms-animation-tick",
                        )
                        frame=int(tick % hist.sizes["time"])
                    else:
                        frame=st.sidebar.slider(
                            "Animation frame",0,hist.sizes["time"]-1,hist.sizes["time"]-1,
                            key="regional-mrms-frame",
                        )
                    cube=hist.isel(time=frame,drop=True)
                    scan_time=times[frame]
                    st.sidebar.caption(
                        f"Frame {frame+1}/{hist.sizes['time']} • {pd.Timestamp(scan_time):%Y-%m-%d %H:%M UTC}"
                    )
                else:
                    st.sidebar.warning(
                        "Multiple stored MRMS frames are not available yet. "
                        "Playback becomes available as the background collector accumulates history."
                    )
            if cube is None:
                with st.spinner("Loading NOAA MRMS quality-controlled composite reflectivity..."):
                    cube,scan_time=load_mrms(None)
            status_line("MRMS QC Composite", scan_time or pd.Timestamp.now(tz="UTC"), 5,cube.attrs.get("aria_data_origin"))
            if regional_env and scan_time is not None:
                reg_vars=[]
                if reg_temp: reg_vars.append("air_temperature")
                if reg_wind: reg_vars.extend(["u_wind","v_wind"])
                if reg_height: reg_vars.append("geopotential_height")
                if reg_vars:
                    try:
                        with st.spinner(f"Matching {reg_pressure}-hPa HRRR environment to MRMS time..."):
                            reg_env_ds,reg_env_run=load_hrrr_environment(REGION_CACHE_KEY,pd.Timestamp(scan_time).isoformat(),reg_pressure,tuple(reg_vars))
                    except Exception as exc:
                        st.warning(f"HRRR environmental overlay unavailable: {exc}")
            st.caption("Quality-controlled multi-radar composite. Playback reads accumulated local Icechunk history when available.")
            field=cube["reflectivity"]
            if "time" in field.dims:
                field=field.isel(time=-1,drop=True)
            if animate:
                # Use the existing static map renderer for animation frames.
                # Rename only in-memory for plotting; stored MRMS remains
                # unchanged.
                static_ds=cube.copy()
                if "reflectivity" in static_ds:
                    static_ds=static_ds.rename({"reflectivity":"composite_reflectivity"})
                fig_single=plot_model_field(
                    static_ds,"composite_reflectivity",ACTIVE_REGION,
                    cmap="turbo",vmin=-30,vmax=70,
                    title=(
                        f"NOAA MRMS QC Composite Reflectivity — "
                        f"{pd.Timestamp(scan_time):%Y-%m-%d %H:%M UTC}"
                        if scan_time is not None
                        else "NOAA MRMS QC Composite Reflectivity"
                    ),
                )
                if reg_env_ds is not None:
                    add_mpl_environment_overlay(fig_single.axes[0],reg_env_ds,ACTIVE_REGION,pressure_hpa=reg_pressure,show_temperature=reg_temp,show_wind=reg_wind,show_height=reg_height,wind_density=reg_density,temperature_interval=reg_temp_interval,height_interval=reg_height_interval)
                st.pyplot(fig_single,width="stretch")
                plt.close(fig_single)
            else:
                fig_single=plot_interactive_field(
                    field,ACTIVE_REGION,
                    title=f"NOAA MRMS QC Composite Reflectivity — {pd.Timestamp(scan_time):%Y-%m-%d %H:%M UTC}" if scan_time is not None else "NOAA MRMS Quality-Controlled Composite Reflectivity",
                    units="dBZ",zmin=-30,zmax=70,colorscale="Turbo",state_color="black",
                )
                if reg_env_ds is not None:
                    fig_single=add_interactive_environment_overlay(fig_single,reg_env_ds,ACTIVE_REGION,pressure_hpa=reg_pressure,show_temperature=reg_temp,show_wind=reg_wind,show_height=reg_height,wind_density=reg_density,temperature_interval=reg_temp_interval,height_interval=reg_height_interval)
                st.plotly_chart(fig_single,width="stretch")
            if reg_env_run is not None:
                st.caption(f"Environmental context: HRRR {reg_pressure} hPa • init {pd.Timestamp(reg_env_run.initialization_time):%Y-%m-%d %H UTC} • F{int(reg_env_run.forecast_hour):02d} • valid {pd.Timestamp(reg_env_run.valid_time):%Y-%m-%d %H UTC}")
            c1,c2=st.columns(2); c1.metric("Product","MRMS QC Composite"); c2.metric("Grid","~1 km native")

        elif mode=="Fast selected altitude":
            altitude=st.sidebar.slider(
                "Radar altitude (km MSL)",
                0.0,12.0,2.0,1.0
            )
            resolution=st.sidebar.selectbox(
                "Horizontal grid",
                [8.0,6.0,4.0,1.0],
                index=1,
                format_func=lambda x:f"{x:g} km",
            )

            with st.spinner(
                "Fetching latest radar volumes in parallel and gridding selected altitude..."
            ):
                cube,scans,diag=load_regional_radar_fast(REGION_CACHE_KEY, 
                    altitude,resolution
                )

            status_line("Regional NEXRAD",diag["newest_scan_time"],5)
            cmap,vmin,vmax=color_controls(
                "Reflectivity","turbo",
                cube.reflectivity.values,-30,70
            )

            fig=plot_regional_reflectivity(
                cube,ACTIVE_REGION,
                altitude_km=altitude,
                cmap=cmap,
                vmin=-30 if vmin is None else vmin,
                vmax=70 if vmax is None else vmax,
            )
            st.pyplot(fig,width="stretch")
            plt.close(fig)

            c1,c2,c3=st.columns(3)
            c1.metric("Radars used",len(scans))
            c2.metric("Grid spacing",f"{resolution:g} km")
            c3.metric("Mode","Fast CAPPI")

        else:
            with st.spinner("Building detailed 3-D regional radar cube..."):
                cube,scans,diag=load_regional_radar(REGION_CACHE_KEY)

            status_line("Regional NEXRAD",diag["newest_scan_time"],5)

            alt=cube.altitude_km.values.tolist()
            altitude=st.sidebar.select_slider(
                "Radar altitude (km)",
                options=alt,
                value=min(alt,key=lambda x:abs(x-2)),
            )

            cmap,vmin,vmax=color_controls(
                "Reflectivity","turbo",
                cube.reflectivity.values,-30,70
            )

            if st.sidebar.checkbox("True 3-D volume",True,key="radar-true-3d"):
                iso_threshold=st.sidebar.slider("3-D minimum reflectivity (dBZ)",10.0,50.0,20.0,5.0)
                iso_count=st.sidebar.slider("Isosurface count",1,8,4,1)
                iso_opacity=st.sidebar.slider("3-D opacity",0.10,0.80,0.25,0.05)
                fig=plot_regional_reflectivity_3d(cube,threshold_dbz=iso_threshold,surface_count=iso_count,opacity=iso_opacity)
                fig.update_layout(height=max(fig.layout.height or 0, 760)); st.plotly_chart(fig,width="stretch")
            else:
                fig=plot_regional_reflectivity(cube,ACTIVE_REGION,altitude_km=altitude,cmap=cmap,
                    vmin=-30 if vmin is None else vmin,vmax=70 if vmax is None else vmax)
                st.pyplot(fig,width="stretch"); plt.close(fig)

            c1,c2,c3=st.columns(3)
            c1.metric("Radars used",len(scans))
            c2.metric(
                "Grid spacing",
                f"{cube.attrs['horizontal_resolution_km']:.0f} km",
            )
            c3.metric("Mode","Full 3-D")

    except Exception as exc:
        st.error(str(exc))


elif view=="Model":
    st.subheader("Model Explorer")
    st.sidebar.markdown("#### Data")

    model_name=st.sidebar.selectbox(
        "Model",
        ["HRRR","ERF (local scaffold)","E3SM (local scaffold)"],
        index=0,
    )

    if model_name!="HRRR":
        st.info(
            f"{model_name} uses the new common ModelAdapter framework in v0.10. "
            "The adapter can open local NetCDF/history output programmatically. "
            "The interactive Streamlit workflow is HRRR-first in this release."
        )
        if model_name.startswith("ERF"):
            st.code(
                "from aria.models import ERFAdapter\n"
                "adapter = ERFAdapter(ACTIVE_REGION)\n"
                "ds, run = adapter.open_run('erf_output.nc', variable_map={...})"
            )
        else:
            st.code(
                "from aria.models import E3SMAdapter\n"
                "adapter = E3SMAdapter(ACTIVE_REGION)\n"
                "ds, run = adapter.open_run('eam_history.nc', variable_map={...})"
            )
    else:
        cycles=hrrr_available_cycles(count=12)
        cycle=st.sidebar.selectbox(
            "HRRR initialization",
            cycles,
            index=0,
            format_func=lambda x:pd.Timestamp(x).strftime("%Y-%m-%d %HZ"),
        )
        st.sidebar.markdown("#### Time")
        model_playback=st.sidebar.toggle("Animate forecast hours",value=False,key="model-playback")
        if model_playback:
            model_start,model_end=st.sidebar.slider("Forecast-hour range",0,18,(0,12),1,key="model-playback-range")
            model_speed=st.sidebar.selectbox("Playback speed",[0.5,1.0,2.0],index=1,format_func=lambda v:f"{v:g}×",key="model-playback-speed")
            model_playing=st.sidebar.toggle("▶ Play",value=False,key="model-playing")
            model_frames=list(range(int(model_start),int(model_end)+1))
            if model_playing and model_frames:
                model_tick=st_autorefresh(interval=max(500,int(1600/float(model_speed))),key="model-animation-tick")
                fxx=model_frames[model_tick % len(model_frames)]
                st.sidebar.caption(f"Playing F{fxx:02d}")
            else:
                fxx=st.sidebar.select_slider("Forecast frame",options=model_frames,value=model_frames[0] if model_frames else 0,key="model-playback-frame")
        else:
            fxx=st.sidebar.slider("Forecast hour",0,18,0,1)

        st.sidebar.markdown("#### Product")
        model_mode=st.sidebar.radio(
            "Model product",
            ["Surface","Pressure levels"],
            index=0,
        )

        if model_mode=="Surface":
            surface_options=[
                "air_temperature_2m",
                "dew_point_temperature_2m",
                "wind_speed_10m",
                "composite_reflectivity",
                "reflectivity_1km",
                "precipitation_rate",
                "precipitation",
                "categorical_rain",
                "categorical_snow",
                "categorical_freezing_rain",
                "categorical_ice_pellets",
                "surface_pressure",
                "terrain_height",
            ]
            variable=st.sidebar.selectbox(
                "Variable",
                surface_options,
                format_func=lambda v:MODEL_STYLE.get(v,(v,"",""))[0],
            )

            # Wind speed requires both components.
            fetch_vars=[variable]
            if variable=="wind_speed_10m":
                fetch_vars=["u_wind_10m","v_wind_10m"]

            try:
                with st.spinner(
                    "Retrieving selected HRRR GRIB messages and subsetting the active region..."
                ):
                    model_ds,run=load_hrrr_surface(REGION_CACHE_KEY, 
                        pd.Timestamp(cycle).isoformat(),
                        fxx,
                        tuple(fetch_vars),
                    )

                plot_var=variable
                if variable=="wind_speed_10m":
                    plot_var="wind_speed_10m"

                style=MODEL_STYLE.get(plot_var,(plot_var,"","viridis"))
                if plot_var in {"composite_reflectivity","reflectivity_1km"}:
                    cmap,vmin,vmax=color_controls(
                        style[0],REFLECTIVITY_STYLE["cmap"],model_ds[plot_var].values,
                        REFLECTIVITY_STYLE["vmin"],REFLECTIVITY_STYLE["vmax"],
                    )
                else:
                    cmap,vmin,vmax=color_controls(
                        style[0],style[2],model_ds[plot_var].values,
                    )

                st.sidebar.markdown("#### Display")
                overlay_radar=st.sidebar.checkbox(
                    "Overlay MRMS QC reflectivity",
                    value=False,
                    help="Uses the lightweight ~1-km MRMS composite. Detailed Level-II regional radar belongs in Regional Radar.",
                )

                model_interactive=st.sidebar.checkbox(
                    "Interactive model map",True,key="model-interactive-surface",
                    disabled=bool(model_playback),
                    help="Playback uses pre-rendered static frames for smoother animation."
                )
                if model_playback or not model_interactive:
                    fig=plot_model_field(
                        model_ds,plot_var,ACTIVE_REGION,cmap=cmap,vmin=vmin,vmax=vmax,
                    )
                    if overlay_radar:
                        with st.spinner("Loading MRMS QC composite overlay..."):
                            radar_cube,radar_time=load_mrms(None)
                        fig=overlay_mrms_reflectivity(fig,radar_cube,min_dbz=10.0)
                    st.pyplot(fig,width="stretch")
                    plt.close(fig)
                else:
                    da=model_ds[plot_var].squeeze(drop=True)
                    fig=plot_interactive_field(
                        da,ACTIVE_REGION,
                        title=f"HRRR — {style[0]} — F{int(run.forecast_hour):02d}",
                        units=style[1],zmin=vmin,zmax=vmax,
                        colorscale=mpl_to_plotly_colorscale(cmap),
                    )
                    if overlay_radar:
                        with st.spinner("Loading MRMS QC composite overlay..."):
                            radar_cube,radar_time=load_mrms(None)
                        fig=overlay_interactive_mrms(fig,radar_cube,min_dbz=10.0)
                    fig.update_layout(height=max(fig.layout.height or 0,760))
                    st.plotly_chart(fig,width="stretch")

                c1,c2,c3=st.columns(3)
                c1.metric("Initialization",pd.Timestamp(run.initialization_time).strftime("%H UTC"))
                c2.metric("Forecast",f"F{run.forecast_hour:02d}")
                c3.metric("Valid",pd.Timestamp(run.valid_time).strftime("%m-%d %H UTC"))

                with st.expander("Model provenance"):
                    st.json({
                        "model":model_ds.attrs.get("model"),
                        "initialization_time":model_ds.attrs.get("initialization_time"),
                        "forecast_hour":model_ds.attrs.get("forecast_hour"),
                        "valid_time":model_ds.attrs.get("valid_time"),
                        "source":model_ds.attrs.get("model_source"),
                        "product":model_ds.attrs.get("model_product"),
                        "processing_history":model_ds.attrs.get("processing_history"),
                    })

                st.caption(
                    "Model Explorer shows native HRRR fields and optional lightweight MRMS context. "
                    "Quantitative Model | Observation | Difference comparisons are consolidated under **Model Evaluation**."
                )

            except Exception as exc:
                st.error(str(exc))
                st.caption(
                    "HRRR uses Herbie selective GRIB retrieval. The first request "
                    "for a field can take longer; later requests reuse Herbie's local cache."
                )

        else:
            pressure_options=[
                "air_temperature",
                "dew_point_temperature",
                "wind_speed",
                "u_wind",
                "v_wind",
                "vertical_velocity",
                "geopotential_height",
            ]
            variable=st.sidebar.selectbox(
                "Variable",
                pressure_options,
                format_func=lambda v:MODEL_STYLE.get(v,(v,"",""))[0],
            )

            fetch_vars=[variable]
            if variable=="wind_speed":
                fetch_vars=["u_wind","v_wind"]

            try:
                with st.spinner(
                    "Retrieving HRRR pressure-level GRIB messages..."
                ):
                    model_ds,run=load_hrrr_pressure(REGION_CACHE_KEY, 
                        pd.Timestamp(cycle).isoformat(),
                        fxx,
                        tuple(fetch_vars),
                    )

                pressures=sorted(
                    [float(x) for x in model_ds.pressure_hpa.values],
                    reverse=True,
                )
                pressure=st.sidebar.selectbox(
                    "Pressure level",
                    pressures,
                    index=min(3,len(pressures)-1),
                    format_func=lambda x:f"{x:g} hPa",
                )

                plot_var=variable
                style=MODEL_STYLE.get(plot_var,(plot_var,"","viridis"))
                data=model_ds[plot_var].sel(
                    pressure_hpa=pressure,
                    method="nearest",
                ).values
                cmap,vmin,vmax=color_controls(
                    style[0],style[2],data
                )

                field_tab,cross_tab,sonde_tab=st.tabs(
                    ["Pressure-level field","Vertical cross section","Radiosonde comparison"]
                )

                with field_tab:
                    pressure_interactive=st.checkbox("Interactive pressure-level map",True,key="model-interactive-pressure")
                    if pressure_interactive:
                        da=model_ds[plot_var].sel(pressure_hpa=pressure,method="nearest").squeeze(drop=True)
                        fig=plot_interactive_field(
                            da,ACTIVE_REGION,
                            title=f"HRRR — {style[0]} — {pressure:g} hPa — F{int(run.forecast_hour):02d}",
                            units=style[1],zmin=vmin,zmax=vmax,
                            colorscale=mpl_to_plotly_colorscale(cmap),
                        )
                        fig.update_layout(height=max(fig.layout.height or 0,760))
                        st.plotly_chart(fig,width="stretch")
                    else:
                        fig=plot_model_field(
                            model_ds,plot_var,ACTIVE_REGION,pressure_hpa=pressure,
                            cmap=cmap,vmin=vmin,vmax=vmax,
                        )
                        st.pyplot(fig,width="stretch")
                        plt.close(fig)

                with cross_tab:
                    st.caption(
                        "Cross section uses nearest HRRR grid points along the selected line."
                    )
                    c1,c2=st.columns(2)
                    with c1:
                        start_lat=st.number_input("Start latitude",value=44.0)
                        start_lon=st.number_input("Start longitude",value=-100.0)
                    with c2:
                        end_lat=st.number_input("End latitude",value=44.0)
                        end_lon=st.number_input("End longitude",value=-89.0)

                    if st.button("Build cross section"):
                        figx=plot_model_cross_section(
                            model_ds,
                            plot_var,
                            start_lat=start_lat,
                            start_lon=start_lon,
                            end_lat=end_lat,
                            end_lon=end_lon,
                            cmap=cmap,
                        )
                        st.pyplot(figx,width="stretch")
                        plt.close(figx)

                with sonde_tab:
                    st.caption(
                        "Automatic mode finds the newest available radiosonde, "
                        "uses the HRRR initialization immediately preceding launch, "
                        "and selects the forecast hour nearest the sonde time."
                    )
                    sonde_time_mode=st.radio(
                        "Time matching",["Automatic latest","Manual model selection"],
                        index=0,horizontal=True,key="sonde-time-mode"
                    )
                    sonde_variable=st.selectbox(
                        "Profile variable",
                        ["air_temperature","dew_point_temperature","wind_speed","u_wind","v_wind"],
                        format_func=lambda v:MODEL_STYLE.get(v,(v,"",""))[0],
                        key="sonde-profile-variable",
                    )
                    if st.button("Load HRRR / radiosonde profile",key="load-sonde-profile"):
                        with st.spinner("Loading pressure-level HRRR and radiosonde profile..."):
                            if sonde_time_mode=="Automatic latest":
                                _,run_auto,profiles,station_id,profile_comp,match_info=load_latest_hrrr_raob(REGION_CACHE_KEY, 
                                    sonde_variable,None
                                )
                                if match_info:
                                    c1,c2,c3,c4,c5=st.columns(5)
                                    c1.metric("Sonde launch",pd.Timestamp(match_info["sonde_time"]).strftime("%Y-%m-%d %H:%M UTC"))
                                    c2.metric("HRRR init",pd.Timestamp(match_info["hrrr_initialization"]).strftime("%Y-%m-%d %H:%M UTC"))
                                    c3.metric("Forecast",f'F{int(match_info["forecast_hour"]):02d}')
                                    c4.metric("HRRR valid",pd.Timestamp(match_info["hrrr_valid_time"]).strftime("%Y-%m-%d %H:%M UTC"))
                                    c5.metric("Time offset",f'{match_info["offset_minutes"]:+.0f} min')
                            else:
                                _,_,profiles,station_id,profile_comp=load_hrrr_raob_comparison(REGION_CACHE_KEY, 
                                    pd.Timestamp(cycle).isoformat(),
                                    fxx,
                                    sonde_variable,
                                    None,
                                )
                        if profile_comp.empty:
                            st.warning("No matching radiosonde profile was available for this valid time.")
                        else:
                            stations=sorted(profiles.station_id.dropna().unique())
                            selected=st.selectbox(
                                "Radiosonde station",
                                stations,
                                index=stations.index(station_id) if station_id in stations else 0,
                                key="sonde-station-result",
                            )
                            if selected!=station_id:
                                _,_,profiles,station_id,profile_comp=load_hrrr_raob_comparison(REGION_CACHE_KEY, 
                                    pd.Timestamp(cycle).isoformat(),
                                    fxx,
                                    sonde_variable,
                                    selected,
                                )
                            label,units,_=MODEL_STYLE.get(
                                sonde_variable,(sonde_variable,"","viridis")
                            )
                            figp=plot_profile_comparison(profile_comp,label,units)
                            st.pyplot(figp,width="content")
                            plt.close(figp)
                            st.dataframe(profile_comp,width="stretch")

            except Exception as exc:
                st.error(str(exc))
                st.caption(
                    "Pressure-level retrieval is performed on demand so the "
                    "surface Model view remains lightweight."
                )


elif view=="Model Evaluation":
    st.subheader("Model Evaluation")
    st.caption(
        "Synchronize HRRR and observations by valid time. This view is designed "
        "for event analysis rather than simply showing the newest available layer."
    )

    now_utc=pd.Timestamp.now(tz="UTC")
    try:
        _,latest_mrms_time=load_mrms(None)
        radar_anchor=pd.Timestamp(latest_mrms_time)
        radar_anchor=radar_anchor.tz_localize("UTC") if radar_anchor.tzinfo is None else radar_anchor.tz_convert("UTC")
    except Exception:
        radar_anchor=now_utc.floor("5min")
    st.sidebar.markdown("#### Evaluation time")
    timing_mode=st.sidebar.radio(
        "Evaluation navigation",
        ["Latest observations","Compare Forecast Runs"],
        key="storm-nav-mode",
    )
    observation_step=st.sidebar.number_input(
        "Observation target offset (minutes)",
        min_value=-720,max_value=0,value=0,step=5,key="storm-observation-offset",
    )
    target_radar_time=radar_anchor+pd.Timedelta(minutes=int(observation_step))
    cycles=hrrr_available_cycles(count=24)
    run_back=st.sidebar.slider("Previous HRRR runs",0,17,0,1,key="storm-run-back") if timing_mode=="Compare Forecast Runs" else 0
    candidates=[]
    for cyc in cycles:
        ct=pd.Timestamp(cyc); ct=ct.tz_localize("UTC") if ct.tzinfo is None else ct.tz_convert("UTC")
        fh=int(round((target_radar_time-ct).total_seconds()/3600))
        if 0 <= fh <= 18:
            candidates.append((ct,fh,abs((ct+pd.Timedelta(hours=fh)-target_radar_time).total_seconds())))
    candidates.sort(key=lambda q:(q[2],-q[0].value))
    if candidates:
        cycle,fxx,_=candidates[min(run_back,len(candidates)-1)]
    else:
        cycle=pd.Timestamp(cycles[0]); fxx=0
    valid_time=pd.Timestamp(cycle)+pd.Timedelta(hours=int(fxx))
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Current UTC",now_utc.strftime("%Y-%m-%d %H:%M UTC"))
    c2.metric("Observation target",target_radar_time.strftime("%Y-%m-%d %H:%M UTC"),f"{(target_radar_time-now_utc).total_seconds()/60:+.0f} min")
    c3.metric("HRRR run",pd.Timestamp(cycle).strftime("%Y-%m-%d %H:%M UTC"),f"F{int(fxx):02d}")
    c4.metric("HRRR valid",valid_time.strftime("%Y-%m-%d %H:%M UTC"),f"{(valid_time-target_radar_time).total_seconds()/60:+.0f} min vs obs target")
    mrms_age_min=(now_utc-target_radar_time).total_seconds()/60
    if mrms_age_min > 15:
        st.warning(f"MRMS data are stale: {mrms_age_min:.0f} minutes old.")
    st.caption("Latest observations aligns HRRR to the selected observation target. Compare Forecast Runs holds the observation target fixed while stepping backward through model initializations and adjusting Fxx.")
    surface_tab,radar_tab,lead_tab,objects_tab=st.tabs(
        ["Surface comparison","Radar comparison","Lead-time verification","Storm objects (ADAPT)"]
    )

    with surface_tab:
        sc1,sc2=st.columns(2)
        with sc1:
            surface_obs_offset=st.selectbox(
                "Maximum surface observation offset",
                [10,15,20,30,45,60],
                index=3,
                format_func=lambda v:f"{v} minutes",
                key="surface-observation-offset",
            )
        with sc2:
            surface_match_label=st.radio(
                "Surface observation matching",
                ["Nearest ± offset","Past-only"],
                horizontal=True,
                key="surface-observation-match-mode",
            )
        surface_match_mode="nearest" if surface_match_label.startswith("Nearest") else "past"
        surface_variable=st.selectbox(
            "Surface variable",
            ["air_temperature_2m","dew_point_temperature_2m"],
            format_func=lambda v:MODEL_STYLE[v][0],
            key="storm-surface-variable",
        )
        if st.button("Build surface comparison",key="storm-build-surface"):
            with st.spinner("Building HRRR, ARIA regional surface analysis, and difference field..."):
                _,run,_,observations,comparison=load_hrrr_gridded_comparison(REGION_CACHE_KEY,
                    pd.Timestamp(cycle).isoformat(),
                    fxx,
                    surface_variable,
                    surface_obs_offset,
                    surface_match_mode,
                )
            st.session_state["model_eval_surface_result"]={
                "run":run,"observations":observations,"comparison":comparison,
                "cycle":pd.Timestamp(cycle).isoformat(),"fxx":int(fxx),
                "variable":surface_variable,"offset":int(surface_obs_offset),"match_mode":surface_match_mode,
            }

        _surface_result=st.session_state.get("model_eval_surface_result")
        if _surface_result is not None:
            run=_surface_result["run"]; observations=_surface_result["observations"]; comparison=_surface_result["comparison"]
            _surface_controls_changed=(
                _surface_result.get("cycle")!=pd.Timestamp(cycle).isoformat() or
                _surface_result.get("fxx")!=int(fxx) or
                _surface_result.get("variable")!=surface_variable or
                _surface_result.get("offset")!=int(surface_obs_offset) or
                _surface_result.get("match_mode")!=surface_match_mode
            )
            if _surface_controls_changed:
                st.info("Showing the previously generated surface comparison. Controls or the latest valid time have changed; choose **Build surface comparison** to regenerate it.")
            fig=plot_interactive_three_panel(
                comparison,
                ACTIVE_REGION,
                title=(
                    f"{MODEL_STYLE[_surface_result['variable']][0]} — "
                    f"F{_surface_result['fxx']:02d} valid {pd.Timestamp(run.valid_time):%Y-%m-%d %H UTC}"
                ),
            )
            fig.update_layout(height=max(fig.layout.height or 0, 760)); st.plotly_chart(fig,width='stretch')
            st.caption(
                f"Surface observations use the closest report within {'±' if _surface_result['match_mode'] == 'nearest' else 'the preceding '}{_surface_result['offset']} minutes of HRRR valid time. "
                "The generated result remains visible through Streamlit reruns until you regenerate it."
            )

            diff=comparison.difference.values
            finite=diff[np.isfinite(diff)]
            m1,m2,m3=st.columns(3)
            m1.metric("ASOS observations",len(observations))
            m2.metric("Mean bias",f"{np.nanmean(finite):.2f} {comparison.attrs.get('units','')}" if finite.size else "—")
            m3.metric("Grid RMSE",f"{np.sqrt(np.nanmean(finite**2)):.2f} {comparison.attrs.get('units','')}" if finite.size else "—")
            if len(observations):
                obs_times=pd.to_datetime(observations["time"],utc=True,errors="coerce").dropna()
                if len(obs_times):
                    offsets=(obs_times-pd.Timestamp(run.valid_time)).dt.total_seconds()/60.0
                    st.caption(
                        f"Observation reports retrieved: {obs_times.min():%H:%M}–{obs_times.max():%H:%M} UTC • "
                        f"median absolute offset {offsets.abs().median():.0f} min • "
                        f"maximum absolute offset {offsets.abs().max():.0f} min"
                    )

    with radar_tab:
        reflectivity_variable=st.selectbox(
            "HRRR reflectivity",
            ["composite_reflectivity","reflectivity_1km"],
            format_func=lambda v:MODEL_STYLE[v][0],
            key="storm-reflectivity-variable",
        )
        st.caption(
            "Choose HRRR composite or 1-km AGL reflectivity. The observational side remains NOAA MRMS QC Composite and is explicitly identified as such. "
            "Use box/scroll zoom or pan on any panel; the three panels share the same geographic axes."
        )
        if st.button("Build radar comparison",key="storm-build-radar"):
            with st.spinner("Loading valid-time MRMS QC composite and HRRR reflectivity..."):
                _,run,radar,scan_time,comparison,metrics,fss=load_hrrr_mrms_comparison(REGION_CACHE_KEY,
                    pd.Timestamp(cycle).isoformat(), fxx, reflectivity_variable
                )
            st.session_state["model_eval_radar_result"]={
                "run":run,"radar":radar,"scan_time":scan_time,"comparison":comparison,"metrics":metrics,"fss":fss,
                "cycle":pd.Timestamp(cycle).isoformat(),"fxx":int(fxx),"variable":reflectivity_variable,
            }

        _radar_result=st.session_state.get("model_eval_radar_result")
        if _radar_result is not None:
            run=_radar_result["run"]; scan_time=_radar_result["scan_time"]; comparison=_radar_result["comparison"]; metrics=_radar_result["metrics"]; fss=_radar_result["fss"]
            _radar_controls_changed=(
                _radar_result.get("cycle")!=pd.Timestamp(cycle).isoformat() or
                _radar_result.get("fxx")!=int(fxx) or
                _radar_result.get("variable")!=reflectivity_variable
            )
            if _radar_controls_changed:
                st.info("Showing the previously generated radar comparison. Controls or the latest observation target have changed; choose **Build radar comparison** to regenerate it.")
            fig=plot_interactive_three_panel(
                comparison,ACTIVE_REGION,radar=True,
                title=f"HRRR F{_radar_result['fxx']:02d} vs MRMS QC Composite — valid {pd.Timestamp(run.valid_time):%Y-%m-%d %H UTC}",
            )
            fig.update_layout(height=max(fig.layout.height or 0, 760)); st.plotly_chart(fig,width='stretch')
            m1,m2,m3=st.columns(3)
            m1.metric("Radar product","MRMS QC Composite")
            m2.metric("MRMS time",pd.Timestamp(scan_time).strftime("%H:%M UTC") if scan_time is not None else "latest")
            if scan_time is not None:
                dt=abs((pd.Timestamp(scan_time)-pd.Timestamp(run.valid_time)).total_seconds())/60
                m3.metric("Time offset",f"{dt:.1f} min")
            else:
                m3.metric("Time offset","—")
            st.markdown("**Grid-cell categorical verification**")
            st.dataframe(metrics,width='stretch',hide_index=True)
            st.markdown("**Fractions Skill Score (spatially tolerant)**")
            st.dataframe(fss,width='stretch',hide_index=True)
            st.caption("This generated comparison remains visible through dashboard reruns. Adjacent positive/negative difference couplets often indicate storm displacement rather than a pure reflectivity-amplitude error.")

    with lead_tab:
        verification_type=st.radio("Verification type",["Surface","Radar"],horizontal=True,key="storm-lead-type")
        leads=st.multiselect(
            "Forecast leads", [0,1,2,3,4,6,9,12,15,18], default=[0,1,3,6,12],
            format_func=lambda x:f"F{x:02d}", key="storm-leads",
        )
        safe_valid_time=latest_safe_hrrr_valid_time(valid_time,availability_lag_hours=2)
        if pd.Timestamp(safe_valid_time) != pd.Timestamp(valid_time).floor("1h"):
            st.info(
                f"Latest complete HRRR verification time: {pd.Timestamp(safe_valid_time):%Y-%m-%d %H:%M UTC}. "
                "ARIA keeps real-time verification two hours behind the wall clock so cycles and subset indexes can finish publishing."
            )
        if verification_type=="Surface":
            lead_variable=st.selectbox(
                "Verification variable", ["air_temperature_2m","dew_point_temperature_2m"],
                format_func=lambda v:MODEL_STYLE[v][0], key="storm-lead-variable",
            )
            st.caption("Each HRRR lead verifies against the same ARIA regional surface analysis at the selected valid time.")
            if st.button("Run surface lead-time verification",key="storm-run-leads") and leads:
                with st.spinner("Retrieving HRRR cycles and comparing against one valid-time surface analysis..."):
                    metrics,comparisons,_,_=load_hrrr_lead_verification(REGION_CACHE_KEY, pd.Timestamp(safe_valid_time).isoformat(),lead_variable,tuple(leads))
                st.dataframe(metrics,width='stretch',hide_index=True)
                good=metrics.dropna(subset=["rmse"])
                if not good.empty:
                    figm,axm=plt.subplots(figsize=(8,4.5))
                    axm.plot(good.forecast_hour,good.rmse,marker="o",label="RMSE")
                    axm.plot(good.forecast_hour,good.mae,marker="o",label="MAE")
                    axm.set_xlabel("Forecast lead (hours)"); axm.set_ylabel(MODEL_STYLE[lead_variable][1])
                    axm.set_title(f"{MODEL_STYLE[lead_variable][0]} error vs forecast lead"); axm.grid(alpha=.25); axm.legend()
                    st.pyplot(figm,width='content'); plt.close(figm)
        else:
            lead_radar_variable=st.selectbox("Radar variable",["composite_reflectivity"],format_func=lambda v:MODEL_STYLE[v][0],key="storm-radar-lead-variable")
            lead_thresholds=st.multiselect("Reflectivity thresholds",[10,20,30,40,50],default=[20,30,40],format_func=lambda v:f"≥ {v} dBZ",key="storm-radar-thresholds")
            st.caption("CSI = Hits / (Hits + Misses + False Alarms). POD measures observed echoes detected; FAR measures forecast echoes that did not occur. Select thresholds above; FSS adds spatial tolerance.")
            if st.button("Run radar lead-time verification",key="storm-run-radar-leads") and leads and lead_thresholds:
                with st.spinner("Calculating lead-time metrics. Full comparison grids are not retained in memory..."):
                    metrics,fss,_,radar,scan_time=load_hrrr_radar_leads(
                        REGION_CACHE_KEY,pd.Timestamp(safe_valid_time).isoformat(),lead_radar_variable,
                        tuple(leads),tuple(lead_thresholds),
                    )
                st.session_state["model_eval_lead_radar"]={
                    "metrics":metrics,"fss":fss,"scan_time":scan_time,
                    "thresholds":tuple(lead_thresholds),"leads":tuple(leads),
                    "valid_time":pd.Timestamp(safe_valid_time).isoformat(),"variable":lead_radar_variable,
                }

            _lead_result=st.session_state.get("model_eval_lead_radar")
            if _lead_result:
                metrics=_lead_result["metrics"]; fss=_lead_result["fss"]
                st.markdown("**Categorical scores by lead and threshold**")
                st.dataframe(metrics,width='stretch',hide_index=True)
                st.markdown("**Fractions Skill Score by lead, threshold, and neighborhood**")
                st.dataframe(fss,width='stretch',hide_index=True)
                if not metrics.empty and "CSI" in metrics:
                    good=metrics.dropna(subset=["CSI"])
                    if not good.empty:
                        figc,axc=plt.subplots(figsize=(8,4.5))
                        for threshold,g in good.groupby("threshold_dbz"):
                            axc.plot(g.forecast_hour,g.CSI,marker="o",label=f"{threshold:g} dBZ")
                        axc.set_xlabel("Forecast lead (hours)")
                        axc.set_ylabel("CSI")
                        axc.set_ylim(0,1)
                        axc.grid(alpha=.25)
                        axc.legend(title="Threshold")
                        axc.set_title("HRRR / MRMS CSI vs forecast lead")
                        st.pyplot(figc,width='content')
                        plt.close(figc)

                st.markdown("**Synchronized HRRR / MRMS / Difference playback**")
                st.caption(
                    "Comparison frames are loaded lazily and cached one lead at a time. "
                    "A new forecast lead can take a moment on first load; replay reuses the cached frame."
                )
                available_leads=sorted(int(x) for x in _lead_result["leads"])
                pc1,pc2=st.columns([1,2])
                with pc1:
                    eval_play=st.toggle("▶ Play comparison",value=False,key="model-eval-radar-play")
                    eval_speed=st.selectbox("Playback speed",[0.5,1.0,2.0],index=1,format_func=lambda v:f"{v:g}×",key="model-eval-radar-speed")
                if eval_play:
                    eval_tick=st_autorefresh(interval=max(800,int(2200/float(eval_speed))),key="model-eval-radar-tick")
                    eval_lead=available_leads[eval_tick % len(available_leads)]
                else:
                    eval_lead=pc2.select_slider(
                        "Forecast lead",options=available_leads,value=available_leads[0],
                        format_func=lambda v:f"F{v:02d}",key="model-eval-radar-frame"
                    )

                observation_target=pd.Timestamp(_lead_result["valid_time"])
                if observation_target.tzinfo is None:
                    observation_target=observation_target.tz_localize("UTC")
                else:
                    observation_target=observation_target.tz_convert("UTC")
                hrrr_valid=observation_target.floor("1h")
                frame_cycle=hrrr_valid-pd.Timedelta(hours=int(eval_lead))
                st.caption(
                    f"F{int(eval_lead):02d}: HRRR init {frame_cycle:%Y-%m-%d %H:%M UTC} • "
                    f"HRRR valid {hrrr_valid:%Y-%m-%d %H:%M UTC} • "
                    f"MRMS target {observation_target:%Y-%m-%d %H:%M UTC}"
                )
                try:
                    with st.spinner(f"Loading comparison frame F{int(eval_lead):02d}..."):
                        _,frame_run,_,frame_scan,eval_comp,_,_=load_hrrr_mrms_comparison(
                            REGION_CACHE_KEY,frame_cycle.isoformat(),int(eval_lead),_lead_result["variable"]
                        )
                    if eval_play:
                        eval_fig=plot_three_panel_comparison(
                            eval_comp,ACTIVE_REGION,
                            title=f"Lead-time comparison — F{int(eval_lead):02d} valid {hrrr_valid:%Y-%m-%d %H UTC}",
                            field_cmap="turbo",
                            field_vmin=-30,
                            field_vmax=70,
                        )
                        st.pyplot(eval_fig,width="stretch")
                        plt.close(eval_fig)
                    else:
                        eval_fig=plot_interactive_three_panel(
                            eval_comp,ACTIVE_REGION,radar=True,
                            title=f"Lead-time comparison — F{int(eval_lead):02d} valid {hrrr_valid:%Y-%m-%d %H UTC}",
                        )
                        eval_fig.update_layout(height=max(eval_fig.layout.height or 0,760))
                        st.plotly_chart(eval_fig,width="stretch")
                    if frame_scan is not None:
                        scan_ts=pd.Timestamp(frame_scan)
                        scan_ts=scan_ts.tz_localize("UTC") if scan_ts.tzinfo is None else scan_ts.tz_convert("UTC")
                        st.caption(f"MRMS frame used: {scan_ts:%Y-%m-%d %H:%M UTC} • offset {(scan_ts-hrrr_valid).total_seconds()/60:+.1f} min")
                except Exception as exc:
                    st.warning(
                        f"F{int(eval_lead):02d} is unavailable for this valid time and was skipped instead of crashing playback."
                    )
                    st.caption(str(exc))

    with objects_tab:
        st.caption(
            "Object identification is delegated to ARM-DOE ADAPT's RadarCellSegmenter. "
            "The same ADAPT threshold/watershed detector is applied to HRRR and MRMS on the common comparison grid; "
            "ARIA then performs one-to-one model/observation object matching."
        )
        if not adapt_available():
            st.warning(
                "ADAPT is not installed in the environment running this app. "
                "Recommended for conda environments: `conda install conda-forge::arm-adapt`, then restart Streamlit."
            )
            st.code("conda install conda-forge::arm-adapt", language="bash")
            st.caption(
                "ADAPT is kept optional because it includes compiled radar dependencies; ARIA does not silently install or vendor its detection code."
            )
        else:
            av=adapt_version() or "installed"
            st.success(f"ADAPT available ({av})")
            oc1,oc2,oc3,oc4=st.columns(4)
            threshold=oc1.slider("Object threshold (dBZ)",20.0,55.0,35.0,5.0,key="adapt-threshold")
            min_points=oc2.number_input("Minimum grid points",min_value=1,max_value=500,value=8,step=1,key="adapt-min-points")
            hmax=oc3.slider("Peak separation h (dBZ)",1.0,15.0,5.0,1.0,key="adapt-hmax")
            max_distance=oc4.slider("Max match distance (km)",20.0,250.0,100.0,10.0,key="adapt-match-distance")
            st.caption(
                "White outlines show the actual ADAPT segmented masks; labels mark centroids. "
                "Matching uses minimum centroid distance with a one-to-one assignment."
            )
            if st.button("Identify and match storm objects",key="storm-run-adapt-objects"):
                with st.spinner("Running ADAPT segmentation on HRRR and MRMS, then matching storm objects..."):
                    _,run,_,scan_time,comparison,obj_result=load_hrrr_mrms_adapt_objects(REGION_CACHE_KEY, 
                        pd.Timestamp(cycle).isoformat(), fxx, "composite_reflectivity",
                        threshold, int(min_points), hmax, max_distance,
                    )
                model_labels,obs_labels,model_objects,obs_objects,matches,obj_metrics=obj_result
                figobj=plot_adapt_storm_objects(
                    comparison,model_objects,obs_objects,matches,ACTIVE_REGION,
                    model_labels=model_labels,observed_labels=obs_labels,
                    title=(
                        f"ADAPT storm objects — HRRR F{fxx:02d} vs MRMS "
                        f"valid {pd.Timestamp(run.valid_time):%Y-%m-%d %H UTC}"
                    ),
                )
                st.plotly_chart(figobj,width='stretch')
                m1,m2,m3,m4=st.columns(4)
                m1.metric("HRRR objects",obj_metrics["model_objects"])
                m2.metric("MRMS objects",obj_metrics["observed_objects"])
                m3.metric("Matched",obj_metrics["matched_objects"])
                disp=obj_metrics["mean_displacement_km"]
                m4.metric("Mean displacement",f"{disp:.1f} km" if np.isfinite(disp) else "—")
                s1,s2,s3=st.columns(3)
                s1.metric("Object POD",f"{obj_metrics['object_POD']:.2f}" if np.isfinite(obj_metrics['object_POD']) else "—")
                s2.metric("Object FAR",f"{obj_metrics['object_FAR']:.2f}" if np.isfinite(obj_metrics['object_FAR']) else "—")
                s3.metric("Object CSI",f"{obj_metrics['object_CSI']:.2f}" if np.isfinite(obj_metrics['object_CSI']) else "—")
                st.markdown("**Matched objects**")
                if matches.empty:
                    st.info("No HRRR/MRMS objects met the selected match-distance criterion.")
                else:
                    st.dataframe(matches.round(2),width='stretch',hide_index=True)
                with st.expander("ADAPT object inventories"):
                    st.markdown("**HRRR objects**")
                    st.dataframe(model_objects.round(2),width='stretch',hide_index=True)
                    st.markdown("**MRMS objects**")
                    st.dataframe(obs_objects.round(2),width='stretch',hide_index=True)

else:
    st.sidebar.markdown("#### Data")
    atmosphere,profiles,trajectories,points=load_atmosphere(REGION_CACHE_KEY)
    cycle=atmosphere.attrs.get("raob_cycle","unknown")
    if cycle!="unknown": status_line("RAOB",cycle,60)
    if "u_wind" in atmosphere and "v_wind" in atmosphere and (
        "wind_speed" not in atmosphere or not np.isfinite(atmosphere["wind_speed"].values).any()
    ):
        atmosphere=atmosphere.copy()
        atmosphere["wind_speed"]=np.hypot(atmosphere["u_wind"],atmosphere["v_wind"])
        atmosphere["wind_speed"].attrs.update(long_name="Wind Speed",units="kt",derived_from="u_wind,v_wind")
    available=[v for v in VARIABLE_STYLE if v in atmosphere]
    labels={v:VARIABLE_STYLE[v]["label"] for v in available}
    variable=st.sidebar.selectbox("Upper-air variable",available,format_func=lambda v:labels[v])
    style=VARIABLE_STYLE[variable]; data=atmosphere[variable].values
    cmap,vmin,vmax=color_controls(style["label"],style["cmap"],data)
    altitudes=atmosphere.altitude_km.values.tolist()
    altitude=st.sidebar.select_slider("Altitude (km MSL)",options=altitudes,value=min(altitudes,key=lambda x:abs(x-3)))
    st.sidebar.markdown("#### Slice position")
    latitude=st.sidebar.slider("Curtain latitude",float(atmosphere.latitude.min()),float(atmosphere.latitude.max()),float(atmosphere.latitude.mean()),.25)
    longitude=st.sidebar.slider("Curtain longitude",float(atmosphere.longitude.min()),float(atmosphere.longitude.max()),float(atmosphere.longitude.mean()),.25)
    fig=make_3d_variable_slice_explorer(atmosphere,ACTIVE_REGION,variable=variable,altitude_km=altitude,
        latitude=latitude,longitude=longitude,profiles=profiles,cmap=cmap,vmin=vmin,vmax=vmax)
    fig.update_layout(height=max(fig.layout.height or 0, 760)); st.plotly_chart(fig,width='stretch')
    st.caption("Radiosonde markers are colored by the selected variable using the same scale as the atmospheric slices.")
    with st.expander("Sonde / HRRR profile comparison"):
        prof_var=st.selectbox("Profile variable",["air_temperature","dew_point_temperature","wind_speed","u_wind","v_wind"],key="3d-sonde-model-var")
        stations=sorted(profiles.station_id.dropna().astype(str).unique().tolist()) if profiles is not None and not profiles.empty else []
        if stations:
            station=st.selectbox("Radiosonde station",stations,key="3d-sonde-station")
            cycles3=hrrr_available_cycles(count=12)
            cyc3=st.selectbox("HRRR initialization",cycles3,index=0,format_func=lambda x:pd.Timestamp(x).strftime("%Y-%m-%d %HZ"),key="3d-sonde-cycle")
            f3=st.slider("HRRR forecast hour",0,18,0,1,key="3d-sonde-fxx")
            if st.button("Compare sonde with HRRR",key="3d-sonde-compare"):
                try:
                    _,_,_,_,pc=build_hrrr_raob_comparison(cycle=pd.Timestamp(cyc3),forecast_hour=f3,station_id=station,variable=prof_var)
                    label={"air_temperature":"Temperature","dew_point_temperature":"Dew point","wind_speed":"Wind speed","u_wind":"U wind","v_wind":"V wind"}[prof_var]
                    units="°C" if "temperature" in prof_var else "m s⁻¹"
                    fp=plot_profile_comparison(pc,label,units); st.pyplot(fp,width='content'); plt.close(fp)
                    st.dataframe(pc,width='stretch')
                except Exception as exc: st.error(str(exc))
st.sidebar.markdown("---")
st.sidebar.caption("Source refresh: ASOS/NEXRAD/MRMS 5 min • Marine 10 min • HRRR 30 min • AirNow 30 min • RAOB/SondeHub 60 min • processed products persist on disk")
st.sidebar.caption(f"ARIA v{__version__}")

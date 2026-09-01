from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from aria import GPGL_REGION, REGION_PRESETS, Region, __version__
from aria.region import region_from_dict
from aria.region_store import load_saved_regions, save_region, delete_region
from aria.region_stats import region_inventory
from aria.adapters.nexrad import GPGL_NEXRAD_SITES, nexrad_sites_for_region
from aria.adapters.mrms import invalidate_latest_mrms_cache
from aria.cache import PersistentCache, time_bucket
from aria.storm_objects import adapt_available, adapt_version
from aria.plotting import (
    VARIABLE_STYLE,
    MODEL_STYLE,
    make_3d_variable_slice_explorer,
    overlay_regional_radar,
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
    plot_regional_reflectivity,
    plot_regional_reflectivity_3d,
    plot_surface_analysis,
    radar_sweep_summary,
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
        projection_type="mercator", showland=True, showcountries=True, showsubunits=True,
        lonaxis_range=[region.west, region.east], lataxis_range=[region.south, region.north],
        fitbounds=False,
    )
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
    mode = st.radio("Region definition", ["Bounding box", "Center + radius"], horizontal=True, key="cfg_mode")
    name = st.text_input("Region name", key="cfg_name", disabled=built_in)

    if mode == "Center + radius":
        c1,c2,c3 = st.columns(3)
        lat = c1.number_input("Center latitude", -90.0, 90.0, key="cfg_center_lat", format="%.3f", disabled=built_in)
        lon = c2.number_input("Center longitude", -180.0, 180.0, key="cfg_center_lon", format="%.3f", disabled=built_in)
        radius = c3.number_input("Radius (km)", 25.0, 2500.0, key="cfg_radius", step=25.0, disabled=built_in)
        candidate = Region.from_center_radius(name or "Custom", lat, lon, radius)
    else:
        c1,c2,c3,c4 = st.columns(4)
        west = c1.number_input("West", -180.0, 180.0, key="cfg_west", format="%.3f", disabled=built_in)
        east = c2.number_input("East", -180.0, 180.0, key="cfg_east", format="%.3f", disabled=built_in)
        south = c3.number_input("South", -90.0, 90.0, key="cfg_south", format="%.3f", disabled=built_in)
        north = c4.number_input("North", -90.0, 90.0, key="cfg_north", format="%.3f", disabled=built_in)
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

    st.plotly_chart(_region_preview(candidate, inv), use_container_width=True, config={"displaylogo":False})

    errors = {k:v for k,v in inv.get("errors",{}).items() if v}
    if errors:
        st.warning("Some inventory sources could not be queried: " + "; ".join(f"{k}: {v}" for k,v in errors.items()))

    with st.expander("Observing sites"):
        t1,t2,t3 = st.tabs(["Surface", "NEXRAD", "Radiosondes"])
        with t1: st.dataframe(pd.DataFrame(inv["surface"]), use_container_width=True, hide_index=True)
        with t2: st.dataframe(pd.DataFrame(inv["radars"]), use_container_width=True, hide_index=True)
        with t3: st.dataframe(pd.DataFrame(inv["sondes"]), use_container_width=True, hide_index=True)

    b1,b2,b3,b4 = st.columns([1,1,1,3])
    if b1.button("Launch ARIA", type="primary", use_container_width=True):
        st.session_state["aria_active_region"] = candidate.as_dict()
        st.session_state["aria_region_config_open"] = False
        st.rerun()
    if not built_in and b2.button("Save region", use_container_width=True):
        if not name.strip():
            st.error("Enter a region name before saving.")
        else:
            save_region(candidate)
            st.success(f"Saved {candidate.name}.")
            st.rerun()
    if choice in saved and b3.button("Delete region", use_container_width=True):
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
if st.sidebar.button("Change region", use_container_width=True):
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



@st.cache_data(ttl=300,show_spinner=False)
def load_surface(region_key, method="barnes", smoothing_km=140.0, max_distance_km=250.0, include_history=False):
    bucket=time_bucket(5)
    key=f"latest-{bucket}-{method}-{float(smoothing_km):g}-{float(max_distance_km):g}-hist{int(include_history)}"
    return disk_cached(
        "surface", key,
        lambda: build_latest_surface(region=ACTIVE_REGION,method=method,smoothing_km=smoothing_km,max_distance_km=max_distance_km,include_history=include_history),
        max_age_seconds=15*60,
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
def load_regional_radar_fast(REGION_CACHE_KEY, region_key, altitude_km, resolution_km):
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
def load_hrrr_asos_comparison(region_key, cycle_iso, forecast_hour, variable):
    key=f"{cycle_iso}-f{int(forecast_hour):02d}-{variable}-stations"
    return disk_cached(
        "comparisons",
        key,
        lambda: compare_hrrr_surface_with_asos(
            region=ACTIVE_REGION,
            cycle=pd.Timestamp(cycle_iso),
            forecast_hour=forecast_hour,
            variable=variable,
        ),
    )

@st.cache_data(ttl=1800,show_spinner=False)
def load_hrrr_gridded_comparison(region_key, cycle_iso, forecast_hour, variable):
    key=f"{cycle_iso}-f{int(forecast_hour):02d}-{variable}-surface-grid"
    return disk_cached(
        "comparisons",
        key,
        lambda: build_hrrr_surface_gridded_comparison(
            region=ACTIVE_REGION,
            cycle=pd.Timestamp(cycle_iso),
            forecast_hour=forecast_hour,
            variable=variable,
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
def load_hrrr_radar_leads(region_key, valid_iso, model_variable, forecast_hours):
    leads="-".join(str(int(x)) for x in forecast_hours)
    key=f"{valid_iso}-{model_variable}-radar-leads-{leads}"
    return disk_cached(
        "comparisons",
        key,
        lambda: build_hrrr_radar_lead_time_verification(
            pd.Timestamp(valid_iso),
            region=ACTIVE_REGION,
            model_variable=model_variable,
            forecast_hours=tuple(forecast_hours),
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

view=st.sidebar.radio("View",["Surface","Air Quality","Radar","Regional Radar","Model","Storm Explorer","3-D Atmosphere","Atmosphere Slice","Coverage"])
refresh_minutes={"Surface":5,"Air Quality":30,"Radar":5,"Regional Radar":5,"Model":30,"Storm Explorer":5,"3-D Atmosphere":60,"Atmosphere Slice":60,"Coverage":5}[view]
auto=st.sidebar.toggle("Auto-update",value=True)
if auto:
    st_autorefresh(interval=refresh_minutes*60*1000,key=f"auto-{view}")
st.sidebar.caption(f"Auto-update cadence for this view: {refresh_minutes} min")
if st.sidebar.button("Refresh from sources",use_container_width=True):
    invalidate_latest_mrms_cache()
    # Explicit refresh is the escape hatch for source updates within the same
    # cadence bucket. Normal view switching never clears this disk cache.
    processed_cache().clear()
    st.cache_data.clear()
    st.rerun()

with st.sidebar.expander("Cache"):
    cache_bytes=processed_cache().size_bytes()
    st.caption(f"Persistent processed cache: {cache_bytes/1024**2:.1f} MB")
    if st.button("Clear processed disk cache",use_container_width=True):
        processed_cache().clear()
        st.cache_data.clear()
        st.rerun()

def color_controls(label,default_cmap,finite,default_min=None,default_max=None):
    cmaps=["coolwarm","viridis","plasma","turbo","BrBG","RdBu_r","cividis"]
    cmap=st.sidebar.selectbox(f"{label} colormap",cmaps,index=cmaps.index(default_cmap) if default_cmap in cmaps else 0)
    auto_scale=st.sidebar.toggle(f"Auto {label} scale",value=True)
    f=np.asarray(finite); f=f[np.isfinite(f)]
    amin=float(np.nanpercentile(f,2)) if f.size else (default_min or 0.)
    amax=float(np.nanpercentile(f,98)) if f.size else (default_max or 1.)
    if default_min is not None: amin=default_min
    if default_max is not None: amax=default_max
    if auto_scale: return cmap,None,None
    vmin=st.sidebar.number_input(f"{label} minimum",value=float(amin))
    vmax=st.sidebar.number_input(f"{label} maximum",value=float(amax))
    if vmax<=vmin: st.sidebar.warning("Maximum must be greater than minimum.")
    return cmap,vmin,vmax

def status_line(source,data_time,cadence):
    now=pd.Timestamp.now(tz="UTC")
    t=pd.Timestamp(data_time)
    t=t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    age=(now-t).total_seconds()/60
    st.caption(f"**{source}:** {t:%Y-%m-%d %H:%M UTC} • {age:.0f} min old • refresh {cadence} min")

if view in ("Surface","Coverage"):
    surface_method=st.sidebar.selectbox("Surface interpolation",["Barnes","Gaussian","IDW"],index=0)
    surface_smoothing=st.sidebar.slider("Smoothing scale (km)",40.0,300.0,140.0,10.0)
    surface_support=st.sidebar.slider("Maximum station support (km)",75.0,400.0,250.0,25.0)
    surface_history=st.sidebar.checkbox(
        "Load previous hour",value=False,
        help="Off is much faster: ARIA computes only the latest analysis. Turn on to build the full 5-minute timeline."
    )
    surface,obs,start,end=load_surface(
        REGION_CACHE_KEY, surface_method.lower(),surface_smoothing,surface_support,surface_history
    )
    if surface.sizes["time"] > 1:
        idx=st.sidebar.slider("Analysis time",0,surface.sizes["time"]-1,surface.sizes["time"]-1)
    else:
        idx=0
    status_line("ASOS",surface.time.values[idx],5)
    if view=="Surface":
        cmap,vmin,vmax=color_controls("Temperature","coolwarm",surface.air_temperature_f.values)
        if st.sidebar.checkbox("Interactive map",True,key="surface-interactive"):
            fig=plot_interactive_field(surface.air_temperature_f.isel(time=idx),ACTIVE_REGION,
                title="Surface Air Temperature",units="°F",zmin=vmin,zmax=vmax,colorscale=mpl_to_plotly_colorscale(cmap))
            fig.update_layout(height=max(fig.layout.height or 0, 760)); st.plotly_chart(fig,use_container_width=True)
            st.caption("Hover to inspect values; pan/box/scroll zoom to explore.")
        else:
            fig=plot_surface_analysis(surface,idx,ACTIVE_REGION,cmap=cmap,temp_vmin=vmin,temp_vmax=vmax,
                                      show_confidence=st.sidebar.checkbox("Show low-confidence hatching"))
            st.pyplot(fig,use_container_width=True); plt.close(fig)
    else:
        variable=st.sidebar.selectbox("Coverage layer",["air_temperature_f_nearest_distance_km","air_temperature_f_n_contributing","air_temperature_f_effective_age_minutes"])
        fld=surface[variable].isel(time=idx); cmap,vmin,vmax=color_controls("Coverage","viridis",fld.values)
        fig,ax=plt.subplots(figsize=(11,7)); m=ax.pcolormesh(surface.longitude,surface.latitude,fld,cmap=cmap,vmin=vmin,vmax=vmax,shading="auto")
        fig.colorbar(m,ax=ax); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title(variable)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

elif view=="Air Quality":
    try:
        aq_radius=st.sidebar.slider("Monitor support radius (km)",100.0,500.0,300.0,25.0)
        cube,latest,allobs,t=load_air_quality(REGION_CACHE_KEY, aq_radius); status_line("AirNow",t,30)
        cmap,vmin,vmax=color_controls("PM2.5","viridis",cube.pm25.values if "pm25" in cube else np.array([]),0,None)
        if st.sidebar.checkbox("Interactive map",True,key="aq-interactive") and "pm25" in cube:
            fig=plot_interactive_field(cube.pm25.isel(time=-1),ACTIVE_REGION,title="AirNow PM2.5",
                units="µg m⁻³",zmin=vmin,zmax=vmax,colorscale="Viridis")
            fig.update_layout(height=max(fig.layout.height or 0, 760)); st.plotly_chart(fig,use_container_width=True)
            st.caption("Hover to inspect PM2.5 values; pan/box/scroll zoom to explore.")
        else:
            fig=plot_air_quality(cube,ACTIVE_REGION,observations=latest,show_sites=st.sidebar.checkbox("Show monitoring sites",True),
                                 cmap=cmap,vmin=vmin,vmax=vmax)
            st.pyplot(fig,use_container_width=True); plt.close(fig)
        c1,c2=st.columns(2); c1.metric("PM2.5 sites",latest.loc[latest.variable=="pm25","station_id"].nunique()); c2.metric("Ozone sites",latest.loc[latest.variable=="ozone","station_id"].nunique())
    except Exception as exc: st.error(str(exc))

elif view=="Radar":
    _radar_meta=nexrad_sites_for_region(ACTIVE_REGION)
    _radar_ids=sorted(_radar_meta) or sorted(GPGL_NEXRAD_SITES)
    _default_idx=_radar_ids.index("KMPX") if "KMPX" in _radar_ids else 0
    rid=st.sidebar.selectbox(
        "Radar site",_radar_ids,index=_default_idx,
        format_func=lambda r:f"{r} — {_radar_meta.get(r,{}).get('name',GPGL_NEXRAD_SITES.get(r,r))}"
    )
    radar_mode=st.sidebar.radio("Radar display",["PPI / Sweep","3-D Volume"],index=0)
    try:
        radar,scan=load_radar(rid); status_line(rid,scan.scan_time,5)
        if radar_mode=="PPI / Sweep":
            sweeps=radar_sweep_summary(radar); sw=st.sidebar.selectbox("Elevation sweep",sweeps.sweep.tolist(),
                format_func=lambda s:f"Sweep {s} — {sweeps.loc[sweeps.sweep==s,'elevation_deg'].iloc[0]:.1f}°")
            cmap,vmin,vmax=color_controls("Reflectivity","turbo",np.array([]),-30,70)
            fig=plot_nexrad_ppi(radar,sweep=sw,cmap=cmap,vmin=-30 if vmin is None else vmin,vmax=70 if vmax is None else vmax)
            st.pyplot(fig,use_container_width=True); plt.close(fig)
        else:
            dx=st.sidebar.selectbox("3-D grid spacing",[4.0,2.0,1.0],index=1,format_func=lambda v:f"{v:g} km")
            max_range=st.sidebar.slider("3-D range (km)",75.0,230.0,180.0,25.0)
            threshold=st.sidebar.slider("Minimum reflectivity (dBZ)",-10.0,50.0,20.0,5.0)
            opacity=st.sidebar.slider("Isosurface opacity",0.10,0.80,0.25,0.05)
            with st.spinner(f"Gridding {rid} into a 3-D volume..."):
                volume=grid_single_radar_reflectivity(radar,horizontal_resolution_km=dx,max_range_km=max_range)
            fig=plot_regional_reflectivity_3d(volume,threshold_dbz=threshold,surface_count=4,opacity=opacity)
            fig.update_layout(height=850); st.plotly_chart(fig,use_container_width=True)
    except Exception as exc: st.error(str(exc))

elif view=="Regional Radar":
    mode=st.sidebar.radio(
        "Regional radar mode",
        ["MRMS QC Composite","Fast selected altitude","Detailed 3-D regional volume"],
        index=0,
    )

    try:
        if mode=="MRMS QC Composite":
            with st.spinner("Loading NOAA MRMS quality-controlled composite reflectivity..."):
                cube,scan_time=load_mrms(None)
            status_line("MRMS QC Composite", scan_time or pd.Timestamp.now(tz="UTC"), 5)
            st.caption("Quality-controlled multi-radar composite. Ground clutter and other non-meteorological echoes are substantially reduced compared with raw Level-II mosaics.")
            field=cube.reflectivity.values
            # Show just the middle panel would waste space; use an interactive Plotly heatmap directly.
            fig_single=plot_interactive_field(
                cube.reflectivity,ACTIVE_REGION,
                title="NOAA MRMS Quality-Controlled Composite Reflectivity",
                units="dBZ",zmin=-30,zmax=70,colorscale="Turbo",state_color="white",
            )
            st.plotly_chart(fig_single,use_container_width=True)
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
            st.pyplot(fig,use_container_width=True)
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
                fig.update_layout(height=max(fig.layout.height or 0, 760)); st.plotly_chart(fig,use_container_width=True)
            else:
                fig=plot_regional_reflectivity(cube,ACTIVE_REGION,altitude_km=altitude,cmap=cmap,
                    vmin=-30 if vmin is None else vmin,vmax=70 if vmax is None else vmax)
                st.pyplot(fig,use_container_width=True); plt.close(fig)

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
        fxx=st.sidebar.slider("Forecast hour",0,18,0,1)

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
                cmap,vmin,vmax=color_controls(
                    style[0],
                    style[2],
                    model_ds[plot_var].values,
                )

                overlay_radar=st.sidebar.checkbox(
                    "Overlay observed regional NEXRAD",
                    value=False,
                )

                fig=plot_model_field(
                    model_ds,
                    plot_var,
                    ACTIVE_REGION,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

                if overlay_radar:
                    with st.spinner("Loading regional NEXRAD overlay..."):
                        radar_cube,radar_scans,radar_diag=load_regional_radar_fast(REGION_CACHE_KEY, 
                            2.0,8.0
                        )
                    fig=overlay_regional_radar(
                        fig,
                        radar_cube,
                        altitude_km=2.0,
                        min_dbz=10.0,
                    )

                st.pyplot(fig,use_container_width=True)
                plt.close(fig)

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

                with st.expander("Integrated model / observation comparison",expanded=(fxx==0)):
                    if variable in {"air_temperature_2m","dew_point_temperature_2m"}:
                        st.caption(
                            "Build a common-grid Model | Observation | Difference "
                            "view using the GPGL ASOS surface analysis at HRRR valid time."
                        )
                        if st.button("Build gridded surface comparison",key=f"grid-{variable}-{fxx}"):
                            with st.spinner("Building valid-time ASOS analysis and HRRR difference field..."):
                                _,_,_,_,grid_comp=load_hrrr_gridded_comparison(REGION_CACHE_KEY, 
                                    pd.Timestamp(cycle).isoformat(),
                                    fxx,
                                    variable,
                                )
                            figc=plot_three_panel_comparison(
                                grid_comp,
                                ACTIVE_REGION,
                                title=f"{MODEL_STYLE[variable][0]} — F{fxx:02d} valid-time comparison",
                            )
                            st.pyplot(figc,use_container_width=True)
                            plt.close(figc)

                    elif variable in {"composite_reflectivity","reflectivity_1km"}:
                        st.caption(
                            "Compare HRRR simulated reflectivity with the regional "
                            "NEXRAD mosaic nearest the HRRR valid time."
                        )
                        radar_dx=st.selectbox(
                            "Comparison radar grid",
                            [8.0,4.0,1.0],
                            index=1,
                            format_func=lambda x:f"{x:g} km",
                            key=f"model-radar-dx-{variable}",
                        )
                        if st.button("Build HRRR / NEXRAD comparison",key=f"radar-comp-{variable}-{fxx}"):
                            with st.spinner("Loading historical-valid-time NEXRAD and building difference field..."):
                                _,_,_,_,_,radar_comp,radar_metrics=load_hrrr_radar_comparison(REGION_CACHE_KEY, 
                                    pd.Timestamp(cycle).isoformat(),
                                    fxx,
                                    variable,
                                    radar_dx,
                                )
                            figr=plot_radar_three_panel(radar_comp,ACTIVE_REGION)
                            st.pyplot(figr,use_container_width=True)
                            plt.close(figr)
                            st.markdown("**Threshold verification**")
                            st.dataframe(radar_metrics,use_container_width=True,hide_index=True)

                    else:
                        st.caption(
                            "Direct gridded observation comparison is currently "
                            "implemented for temperature, dew point, and reflectivity."
                        )

                comparison_vars={
                    "air_temperature_2m",
                    "dew_point_temperature_2m",
                    "u_wind_10m",
                    "v_wind_10m",
                }

                if variable in comparison_vars:
                    st.markdown("### Model vs ASOS")
                    if st.button("Run ASOS comparison"):
                        with st.spinner("Matching HRRR to ASOS observations..."):
                            _,_,comp,metrics=load_hrrr_asos_comparison(REGION_CACHE_KEY, 
                                pd.Timestamp(cycle).isoformat(),
                                fxx,
                                variable,
                            )

                        if comp.empty:
                            st.warning(
                                "No ASOS observations were found within ±20 minutes "
                                "of this HRRR valid time."
                            )
                        else:
                            m1,m2,m3,m4=st.columns(4)
                            m1.metric("Stations",metrics["n"])
                            m2.metric("Bias",f'{metrics["bias"]:.2f}')
                            m3.metric("MAE",f'{metrics["mae"]:.2f}')
                            m4.metric("RMSE",f'{metrics["rmse"]:.2f}')

                            fig2,ax2=plt.subplots(figsize=(6,6))
                            ax2.scatter(comp["observation"],comp["model"],s=24)
                            lo=float(np.nanmin([comp.observation.min(),comp.model.min()]))
                            hi=float(np.nanmax([comp.observation.max(),comp.model.max()]))
                            ax2.plot([lo,hi],[lo,hi],"--",linewidth=1)
                            ax2.set_xlabel("ASOS")
                            ax2.set_ylabel("HRRR")
                            ax2.set_title(f"{MODEL_STYLE[variable][0]} — HRRR vs ASOS")
                            st.pyplot(fig2,use_container_width=False)
                            plt.close(fig2)
                            st.dataframe(comp,use_container_width=True)

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
                    fig=plot_model_field(
                        model_ds,
                        plot_var,
                        ACTIVE_REGION,
                        pressure_hpa=pressure,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    st.pyplot(fig,use_container_width=True)
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
                        st.pyplot(figx,use_container_width=True)
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
                            st.pyplot(figp,use_container_width=False)
                            plt.close(figp)
                            st.dataframe(profile_comp,use_container_width=True)

            except Exception as exc:
                st.error(str(exc))
                st.caption(
                    "Pressure-level retrieval is performed on demand so the "
                    "surface Model view remains lightweight."
                )


elif view=="Storm Explorer":
    st.subheader("Storm Explorer")
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
    timing_mode=st.sidebar.radio("Digital Storm navigation",["Follow Radar","Compare Forecast Runs"],key="storm-nav-mode")
    radar_step=st.sidebar.number_input("Radar time offset (minutes)",min_value=-720,max_value=0,value=0,step=5,key="storm-radar-offset")
    target_radar_time=radar_anchor+pd.Timedelta(minutes=int(radar_step))
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
    c2.metric("MRMS target",target_radar_time.strftime("%Y-%m-%d %H:%M UTC"),f"{(target_radar_time-now_utc).total_seconds()/60:+.0f} min")
    c3.metric("HRRR run",pd.Timestamp(cycle).strftime("%Y-%m-%d %H:%M UTC"),f"F{int(fxx):02d}")
    c4.metric("HRRR valid",valid_time.strftime("%Y-%m-%d %H:%M UTC"),f"{(valid_time-target_radar_time).total_seconds()/60:+.0f} min vs radar")
    mrms_age_min=(now_utc-target_radar_time).total_seconds()/60
    if mrms_age_min > 15:
        st.warning(f"MRMS data are stale: {mrms_age_min:.0f} minutes old.")
    st.caption("Follow Radar steps the observation target and automatically aligns HRRR. Compare Forecast Runs holds the radar target and steps backward through model initializations while adjusting Fxx.")
    surface_tab,radar_tab,lead_tab,objects_tab=st.tabs(
        ["Surface comparison","Radar comparison","Lead-time verification","Storm objects (ADAPT)"]
    )

    with surface_tab:
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
                )
            fig=plot_interactive_three_panel(
                comparison,
                ACTIVE_REGION,
                title=(
                    f"{MODEL_STYLE[surface_variable][0]} — "
                    f"F{fxx:02d} valid {pd.Timestamp(run.valid_time):%Y-%m-%d %H UTC}"
                ),
            )
            fig.update_layout(height=max(fig.layout.height or 0, 760)); st.plotly_chart(fig,use_container_width=True)
            st.caption(
                "Surface observations use a 15-minute past-only window ending at the HRRR valid time. "
                "Pan or box/scroll zoom on any panel; Model, Observation, and Difference use synchronized geographic axes."
            )

            diff=comparison.difference.values
            finite=diff[np.isfinite(diff)]
            m1,m2,m3=st.columns(3)
            m1.metric("ASOS observations",len(observations))
            m2.metric("Mean bias",f"{np.nanmean(finite):.2f} {comparison.attrs.get('units','')}" if finite.size else "—")
            m3.metric("Grid RMSE",f"{np.sqrt(np.nanmean(finite**2)):.2f} {comparison.attrs.get('units','')}" if finite.size else "—")

    with radar_tab:
        reflectivity_variable=st.selectbox(
            "HRRR reflectivity",
            ["composite_reflectivity"],
            format_func=lambda v:MODEL_STYLE[v][0],
            key="storm-reflectivity-variable",
        )
        st.caption(
            "HRRR composite reflectivity is compared with NOAA MRMS quality-controlled composite reflectivity at the same valid time. "
            "Use box/scroll zoom or pan on any panel; the three panels share the same geographic axes."
        )
        if st.button("Build radar comparison",key="storm-build-radar"):
            with st.spinner("Loading valid-time MRMS QC composite and HRRR reflectivity..."):
                _,run,radar,scan_time,comparison,metrics,fss=load_hrrr_mrms_comparison(REGION_CACHE_KEY, 
                    pd.Timestamp(cycle).isoformat(), fxx, reflectivity_variable
                )
            fig=plot_interactive_three_panel(
                comparison,ACTIVE_REGION,radar=True,
                title=f"HRRR F{fxx:02d} vs MRMS QC Composite — valid {pd.Timestamp(run.valid_time):%Y-%m-%d %H UTC}",
            )
            fig.update_layout(height=max(fig.layout.height or 0, 760)); st.plotly_chart(fig,use_container_width=True)
            m1,m2,m3=st.columns(3)
            m1.metric("Radar product","MRMS QC Composite")
            m2.metric("MRMS time",pd.Timestamp(scan_time).strftime("%H:%M UTC") if scan_time is not None else "latest")
            if scan_time is not None:
                dt=abs((pd.Timestamp(scan_time)-pd.Timestamp(run.valid_time)).total_seconds())/60
                m3.metric("Time offset",f"{dt:.1f} min")
            else:
                m3.metric("Time offset","—")
            st.markdown("**Grid-cell categorical verification**")
            st.dataframe(metrics,use_container_width=True,hide_index=True)
            st.markdown("**Fractions Skill Score (spatially tolerant)**")
            st.dataframe(fss,use_container_width=True,hide_index=True)
            st.caption("Adjacent positive/negative difference couplets often indicate storm displacement rather than a pure reflectivity-amplitude error.")

    with lead_tab:
        verification_type=st.radio("Verification type",["Surface","Radar"],horizontal=True,key="storm-lead-type")
        leads=st.multiselect(
            "Forecast leads", [0,1,2,3,4,6,9,12,15,18], default=[0,1,3,6,12],
            format_func=lambda x:f"F{x:02d}", key="storm-leads",
        )
        if verification_type=="Surface":
            lead_variable=st.selectbox(
                "Verification variable", ["air_temperature_2m","dew_point_temperature_2m"],
                format_func=lambda v:MODEL_STYLE[v][0], key="storm-lead-variable",
            )
            st.caption("Each HRRR lead verifies against the same ARIA regional surface analysis at the selected valid time.")
            if st.button("Run surface lead-time verification",key="storm-run-leads") and leads:
                with st.spinner("Retrieving HRRR cycles and comparing against one valid-time surface analysis..."):
                    metrics,comparisons,_,_=load_hrrr_lead_verification(REGION_CACHE_KEY, valid_time.isoformat(),lead_variable,tuple(leads))
                st.dataframe(metrics,use_container_width=True,hide_index=True)
                good=metrics.dropna(subset=["rmse"])
                if not good.empty:
                    figm,axm=plt.subplots(figsize=(8,4.5))
                    axm.plot(good.forecast_hour,good.rmse,marker="o",label="RMSE")
                    axm.plot(good.forecast_hour,good.mae,marker="o",label="MAE")
                    axm.set_xlabel("Forecast lead (hours)"); axm.set_ylabel(MODEL_STYLE[lead_variable][1])
                    axm.set_title(f"{MODEL_STYLE[lead_variable][0]} error vs forecast lead"); axm.grid(alpha=.25); axm.legend()
                    st.pyplot(figm,use_container_width=False); plt.close(figm)
        else:
            lead_radar_variable=st.selectbox("Radar variable",["composite_reflectivity"],format_func=lambda v:MODEL_STYLE[v][0],key="storm-radar-lead-variable")
            st.caption("All forecast leads verify against one cached MRMS QC composite at the selected valid time. CSI/POD/FAR are shown by dBZ threshold; FSS adds spatial tolerance.")
            if st.button("Run radar lead-time verification",key="storm-run-radar-leads") and leads:
                with st.spinner("Loading one MRMS field and HRRR reflectivity for each forecast lead..."):
                    metrics,fss,comparisons,radar,scan_time=load_hrrr_radar_leads(REGION_CACHE_KEY, valid_time.isoformat(),lead_radar_variable,tuple(leads))
                st.markdown("**Categorical scores by lead and threshold**")
                st.dataframe(metrics,use_container_width=True,hide_index=True)
                st.markdown("**Fractions Skill Score by lead, threshold, and neighborhood**")
                st.dataframe(fss,use_container_width=True,hide_index=True)
                if not metrics.empty and "CSI" in metrics:
                    good=metrics.dropna(subset=["CSI"])
                    if not good.empty:
                        figc,axc=plt.subplots(figsize=(8,4.5))
                        for threshold,g in good.groupby("threshold_dbz"):
                            axc.plot(g.forecast_hour,g.CSI,marker="o",label=f"{threshold:g} dBZ")
                        axc.set_xlabel("Forecast lead (hours)"); axc.set_ylabel("CSI"); axc.set_ylim(0,1); axc.grid(alpha=.25); axc.legend(title="Threshold")
                        axc.set_title("HRRR / MRMS CSI vs forecast lead")
                        st.pyplot(figc,use_container_width=False); plt.close(figc)

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
                st.plotly_chart(figobj,use_container_width=True)
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
                    st.dataframe(matches.round(2),use_container_width=True,hide_index=True)
                with st.expander("ADAPT object inventories"):
                    st.markdown("**HRRR objects**")
                    st.dataframe(model_objects.round(2),use_container_width=True,hide_index=True)
                    st.markdown("**MRMS objects**")
                    st.dataframe(obs_objects.round(2),use_container_width=True,hide_index=True)

else:
    atmosphere,profiles,trajectories,points=load_atmosphere(REGION_CACHE_KEY)
    cycle=atmosphere.attrs.get("raob_cycle","unknown")
    if cycle!="unknown": status_line("RAOB",cycle,60)
    available=[v for v in VARIABLE_STYLE if v in atmosphere]
    labels={v:VARIABLE_STYLE[v]["label"] for v in available}
    variable=st.sidebar.selectbox("Upper-air variable",available,format_func=lambda v:labels[v])
    style=VARIABLE_STYLE[variable]; data=atmosphere[variable].values
    cmap,vmin,vmax=color_controls(style["label"],style["cmap"],data)
    altitudes=atmosphere.altitude_km.values.tolist()
    altitude=st.sidebar.select_slider("Altitude (km MSL)",options=altitudes,value=min(altitudes,key=lambda x:abs(x-3)))
    if view=="Atmosphere Slice":
        overlay=st.sidebar.selectbox("Overlay",["None","Wind barbs"],key="atmos-overlay")
        density=st.sidebar.select_slider("Barb density",options=["Sparse","Medium","Dense"],value="Medium",key="atmos-barb-density")
        fig=plot_atmosphere_slice(atmosphere,variable,altitude,ACTIVE_REGION,cmap=cmap,vmin=vmin,vmax=vmax,
            wind_barbs=(overlay=="Wind barbs"),barb_skip={"Sparse":7,"Medium":4,"Dense":2}[density])
        st.pyplot(fig,use_container_width=True); plt.close(fig)
    else:
        latitude=st.sidebar.slider("Curtain latitude",float(atmosphere.latitude.min()),float(atmosphere.latitude.max()),float(atmosphere.latitude.mean()),.25)
        longitude=st.sidebar.slider("Curtain longitude",float(atmosphere.longitude.min()),float(atmosphere.longitude.max()),float(atmosphere.longitude.mean()),.25)
        fig=make_3d_variable_slice_explorer(atmosphere,ACTIVE_REGION,variable=variable,altitude_km=altitude,
            latitude=latitude,longitude=longitude,profiles=profiles,cmap=cmap,vmin=vmin,vmax=vmax)
        fig.update_layout(height=max(fig.layout.height or 0, 760)); st.plotly_chart(fig,use_container_width=True)
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
                        fp=plot_profile_comparison(pc,label,units); st.pyplot(fp,use_container_width=False); plt.close(fp)
                        st.dataframe(pc,use_container_width=True)
                    except Exception as exc: st.error(str(exc))

st.sidebar.markdown("---")
st.sidebar.caption("Source refresh: ASOS/NEXRAD/MRMS 5 min • HRRR 30 min • AirNow 30 min • RAOB/SondeHub 60 min • processed products persist on disk")
st.sidebar.caption(f"ARIA v{__version__}")

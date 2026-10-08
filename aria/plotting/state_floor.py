from __future__ import annotations
import numpy as np


def add_state_floor(
    fig,
    z_value,
    *,
    lon0=None,
    lat0=None,
    west=None,
    east=None,
    south=None,
    north=None,
    color="black",
    width=3,
    pad_deg=0.35,
):
    """Add state boundaries to the base plane of a geographic 3-D plot.

    When regional bounds are supplied, geometries are clipped before they are
    transformed. This prevents distant states (notably Alaska) from expanding
    Plotly's automatic 3-D scene limits.
    """
    import plotly.graph_objects as go
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box
        shp=shpreader.natural_earth(
            resolution="50m",
            category="cultural",
            name="admin_1_states_provinces_lakes",
        )
        records=shpreader.Reader(shp).records()
    except Exception:
        return fig

    clip_box=None
    if None not in (west,east,south,north):
        clip_box=box(
            float(west)-float(pad_deg),
            float(south)-float(pad_deg),
            float(east)+float(pad_deg),
            float(north)+float(pad_deg),
        )

    for rec in records:
        a=rec.attributes
        if a.get("adm0_a3")!="USA" and a.get("admin")!="United States of America":
            continue
        geom=rec.geometry
        if clip_box is not None:
            if not geom.intersects(clip_box):
                continue
            geom=geom.intersection(clip_box)
            if geom.is_empty:
                continue
        boundary=geom.boundary
        geoms=list(boundary.geoms) if hasattr(boundary,"geoms") else [boundary]
        for g in geoms:
            if not hasattr(g,"coords"):
                continue
            arr=np.asarray(g.coords,float)
            if arr.ndim!=2 or len(arr)<2:
                continue
            lon=arr[:,0]; lat=arr[:,1]
            if lon0 is not None and lat0 is not None:
                x=(lon-float(lon0))*111.32*np.cos(np.deg2rad(float(lat0)))
                y=(lat-float(lat0))*110.57
            else:
                x=lon; y=lat
            fig.add_trace(go.Scatter3d(
                x=x,y=y,z=np.full(len(x),float(z_value)),
                mode="lines",
                line=dict(color=color,width=width),
                hoverinfo="skip",
                showlegend=False,
            ))
    return fig

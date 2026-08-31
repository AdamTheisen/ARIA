from __future__ import annotations
import numpy as np

def add_state_floor(fig,z_value,*,lon0=None,lat0=None,color="black",width=3):
    import plotly.graph_objects as go
    try:
        import cartopy.io.shapereader as shpreader
        shp=shpreader.natural_earth(resolution="50m",category="cultural",name="admin_1_states_provinces_lakes")
        records=shpreader.Reader(shp).records()
    except Exception:
        return fig
    for rec in records:
        a=rec.attributes
        if a.get("adm0_a3")!="USA" and a.get("admin")!="United States of America": continue
        boundary=rec.geometry.boundary; geoms=list(boundary.geoms) if hasattr(boundary,"geoms") else [boundary]
        for g in geoms:
            arr=np.asarray(g.coords); lon=arr[:,0]; lat=arr[:,1]
            if lon0 is not None and lat0 is not None:
                x=(lon-lon0)*111.32*np.cos(np.deg2rad(lat0)); y=(lat-lat0)*110.57
            else:
                x=lon; y=lat
            fig.add_trace(go.Scatter3d(x=x,y=y,z=np.full(len(x),float(z_value)),mode="lines",line=dict(color=color,width=width),hoverinfo="skip",showlegend=False))
    return fig

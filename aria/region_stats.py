from __future__ import annotations

import pandas as pd

from .adapters.asos_bulk import BulkIEMASOSAdapter
from .adapters.nexrad import nexrad_sites_for_region
from .adapters.raob import IEMRAOBAdapter


def region_inventory(region, *, surface_minutes=30):
    """Discover observing systems available in a region.

    Surface count is the number of ASOS/AWOS/METAR stations that actually
    reported during the recent sampling window. Radiosonde count comes from
    IEM RAOB station metadata. NEXRAD discovery is local Py-ART metadata.
    """
    now = pd.Timestamp.now(tz="UTC")
    surface = []
    surface_error = None
    try:
        stations = BulkIEMASOSAdapter(region).fetch(
            now - pd.Timedelta(minutes=surface_minutes), now,
            variables=["tmpf"],
        )
        for sid, ds in stations.items():
            frame = ds.to_dataframe().reset_index()
            lat = pd.to_numeric(frame.get("lat"), errors="coerce").dropna()
            lon = pd.to_numeric(frame.get("lon"), errors="coerce").dropna()
            surface.append({
                "id": sid,
                "latitude": float(lat.iloc[0]) if not lat.empty else None,
                "longitude": float(lon.iloc[0]) if not lon.empty else None,
            })
    except Exception as exc:
        surface_error = str(exc)

    radars = []
    radar_error = None
    try:
        for rid, meta in sorted(nexrad_sites_for_region(region).items()):
            radars.append({
                "id": rid,
                "name": meta.get("name") or rid,
                "latitude": meta.get("latitude"),
                "longitude": meta.get("longitude"),
            })
    except Exception as exc:
        radar_error = str(exc)

    sondes = []
    sonde_error = None
    try:
        meta = IEMRAOBAdapter(region, launch_buffer_deg=0.0).station_metadata()
        for row in meta.itertuples():
            sondes.append({
                "id": str(row.station_id),
                "name": str(row.station_name or row.station_id),
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
            })
    except Exception as exc:
        sonde_error = str(exc)

    return {
        "surface": surface,
        "radars": radars,
        "sondes": sondes,
        "errors": {
            "surface": surface_error,
            "radars": radar_error,
            "sondes": sonde_error,
        },
        "sample_time": now.isoformat(),
        "surface_minutes": surface_minutes,
    }

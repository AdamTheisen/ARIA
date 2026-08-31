from __future__ import annotations
import pandas as pd
import requests
from .base import BaseAdapter

class USGSWaterAdapter(BaseAdapter):
    name = "usgs"
    def fetch(self, start, end, parameter_codes=("00060","00065","00010","00045"), **kwargs):
        params = {
            "format": "json",
            "bBox": f"{self.region.west},{self.region.south},{self.region.east},{self.region.north}",
            "startDT": str(start),
            "endDT": str(end),
            "parameterCd": ",".join(parameter_codes),
            "siteStatus": "all",
        }
        r = requests.get("https://waterservices.usgs.gov/nwis/iv/", params=params, timeout=120)
        r.raise_for_status()
        js = r.json()
        rows = []
        for series in js.get("value", {}).get("timeSeries", []):
            src = series.get("sourceInfo", {})
            var = series.get("variable", {})
            site = src.get("siteCode", [{}])[0].get("value")
            geo = src.get("geoLocation", {}).get("geogLocation", {})
            code = var.get("variableCode", [{}])[0].get("value")
            units = var.get("unit", {}).get("unitCode")
            for block in series.get("values", []):
                for rec in block.get("value", []):
                    rows.append({
                        "source": "USGS",
                        "station_id": site,
                        "time": rec.get("dateTime"),
                        "latitude": geo.get("latitude"),
                        "longitude": geo.get("longitude"),
                        "variable": code,
                        "value": pd.to_numeric(rec.get("value"), errors="coerce"),
                        "units": units,
                    })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        return df

from __future__ import annotations

import numpy as np
import pandas as pd


def airnow_to_tidy(ds):
    """
    Convert ACT's get_airnow_bounded_obs() xarray Dataset to common tidy rows.

    ACT returns dimensions (time, sites), with site latitude/longitude and
    pollutant variables such as PM25 and OZONE.
    """
    if ds is None or not getattr(ds, "data_vars", None):
        return pd.DataFrame()

    rows = []

    aliases = {
        "PM25": "pm25",
        "OZONE": "ozone",
        "PM10": "pm10",
        "CO": "co",
        "NO2": "no2",
        "SO2": "so2",
    }

    site_names = np.asarray(ds["sites"].values) if "sites" in ds.coords else None
    times = pd.to_datetime(ds["time"].values, utc=True, errors="coerce")

    for source_var, canonical in aliases.items():
        if source_var not in ds:
            continue

        units = ds[source_var].attrs.get("units", "")

        for ti, timestamp in enumerate(times):
            for si in range(ds.sizes.get("sites", 0)):
                value = ds[source_var].values[ti, si]

                if not np.isfinite(value):
                    continue

                station_id = (
                    str(ds["aqs_id"].values[si])
                    if "aqs_id" in ds
                    else str(site_names[si])
                )

                rows.append(
                    {
                        "source": "AirNow",
                        "station_id": station_id,
                        "station_name": str(site_names[si]),
                        "time": timestamp,
                        "latitude": float(ds["latitude"].values[si]),
                        "longitude": float(ds["longitude"].values[si]),
                        "variable": canonical,
                        "value": float(value),
                        "units": units,
                    }
                )

    return pd.DataFrame(rows)

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .base import BaseAdapter


class AircraftCSVAdapter(BaseAdapter):
    """
    Adapter for authorized aircraft meteorological data (e.g., ACARS/AMDAR).

    Airline ACARS/AMDAR observations can carry redistribution restrictions, so
    this package does not scrape or redistribute a restricted feed. Supply a
    locally authorized CSV/Parquet file using the common column mapping below.
    """

    name = "aircraft"

    def __init__(self, region, path):
        super().__init__(region)
        self.path = Path(path)

    def fetch(self, start, end, **kwargs):
        if self.path.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.path)
        else:
            df = pd.read_csv(self.path)

        required = {"time", "latitude", "longitude", "altitude_m"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                "Aircraft file is missing required columns: "
                + ", ".join(sorted(missing))
            )

        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

        df = df[
            df["time"].between(start, end)
            & df["latitude"].between(self.region.south, self.region.north)
            & df["longitude"].between(self.region.west, self.region.east)
        ].copy()

        if {"wind_speed_kt", "wind_direction_deg"}.issubset(df.columns):
            direction = np.deg2rad(pd.to_numeric(df["wind_direction_deg"], errors="coerce"))
            speed = pd.to_numeric(df["wind_speed_kt"], errors="coerce")
            df["u_wind_kt"] = -speed * np.sin(direction)
            df["v_wind_kt"] = -speed * np.cos(direction)

        df["source"] = df.get("source", "Authorized Aircraft")
        return df.reset_index(drop=True)

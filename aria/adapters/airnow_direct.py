from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from .base import BaseAdapter


class AirNowConcentrationAdapter(BaseAdapter):
    """
    Robust AirNow concentration-only adapter.

    This calls the same AirNow bounded-data endpoint used by ACT, but parses the
    CSV directly into a tidy DataFrame and ignores AQI/category metadata. This
    avoids ACT's current mixed-type xarray parsing issue.

    AirNow dataType="C" means concentrations.
    """

    name = "airnow"

    COLUMN_NAMES = [
        "latitude",
        "longitude",
        "time",
        "parameter",
        "concentration",
        "unit",
        "raw_concentration",
        "aqi",
        "category",
        "site_name",
        "site_agency",
        "aqs_id",
        "full_aqs_id",
    ]

    PARAMETER_MAP = {
        "PM25": "pm25",
        "PM2.5": "pm25",
        "OZONE": "ozone",
        "O3": "ozone",
        "PM10": "pm10",
        "CO": "co",
        "NO2": "no2",
        "SO2": "so2",
    }

    def __init__(self, region, token, timeout=120, session=None):
        super().__init__(region)
        self.token = token
        self.timeout = timeout
        self.session = session or requests.Session()

    @staticmethod
    def _hour(value):
        t = pd.Timestamp(value)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        return t.strftime("%Y-%m-%dT%H")

    def fetch(
        self,
        start,
        end,
        parameters="OZONE,PM25",
        monitor_type=0,
        include_raw_concentrations=1,
        **kwargs,
    ):
        url = "https://www.airnowapi.org/aq/data/"

        params = {
            "startDate": self._hour(start),
            "endDate": self._hour(end),
            "parameters": parameters,
            "BBOX": (
                f"{self.region.west},{self.region.south},"
                f"{self.region.east},{self.region.north}"
            ),
            # AirNow/ACT definition: C = concentrations.
            "dataType": "C",
            "format": "text/csv",
            "verbose": 1,
            "monitorType": monitor_type,
            "includerawconcentrations": include_raw_concentrations,
            "API_KEY": self.token,
        }

        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        if not response.text.strip():
            return pd.DataFrame()

        # AirNow's data endpoint is headerless when verbose=1 in the format
        # ACT expects, so assign the same documented column order ACT uses.
        df = pd.read_csv(
            StringIO(response.text),
            names=self.COLUMN_NAMES,
            header=None,
        )

        # Some responses can include a header line. Remove it if present.
        df = df[
            pd.to_numeric(df["latitude"], errors="coerce").notna()
            & pd.to_numeric(df["longitude"], errors="coerce").notna()
        ].copy()

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["concentration"] = pd.to_numeric(df["concentration"], errors="coerce")
        df["raw_concentration"] = pd.to_numeric(
            df["raw_concentration"], errors="coerce"
        )

        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df["parameter"] = df["parameter"].astype(str).str.strip().str.upper()
        df["variable"] = (
            df["parameter"].map(self.PARAMETER_MAP)
            .fillna(df["parameter"].str.lower())
        )

        out = pd.DataFrame(
            {
                "source": "AirNow",
                "station_id": (
                    df["full_aqs_id"]
                    .fillna(df["aqs_id"])
                    .fillna(df["site_name"])
                    .astype(str)
                ),
                "station_name": df["site_name"].astype(str),
                "time": df["time"],
                "latitude": df["latitude"],
                "longitude": df["longitude"],
                "variable": df["variable"],
                "value": df["concentration"],
                "units": df["unit"].astype(str),
                "raw_value": df["raw_concentration"],
                "site_agency": df["site_agency"].astype(str),
            }
        )

        out = out.dropna(
            subset=["time", "latitude", "longitude", "value"]
        )

        out = out[
            out["latitude"].between(self.region.south, self.region.north)
            & out["longitude"].between(self.region.west, self.region.east)
        ]

        return out.reset_index(drop=True)

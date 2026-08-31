from __future__ import annotations

import time
import pandas as pd
import requests

from .base import BaseAdapter


AQS_PARAMETER_CODES = {
    "ozone": "44201",
    "pm25": "88101",
    "pm10": "81102",
    "co": "42101",
    "no2": "42602",
    "so2": "42401",
}


class EPAAQSAdapter(BaseAdapter):
    """
    EPA Air Quality System (AQS) sample-data adapter.

    AQS is the EPA archival system and is not real-time. EPA notes that data can
    take months to appear in AQS. For recent/hourly workflows, use ACTAirNowAdapter.
    """

    name = "epa_aqs"

    def __init__(
        self,
        region,
        email,
        key,
        timeout=120,
        request_pause=0.25,
        session=None,
    ):
        super().__init__(region)
        self.email = email
        self.key = key
        self.timeout = timeout
        self.request_pause = request_pause
        self.session = session or requests.Session()

    @staticmethod
    def _date(value):
        return pd.Timestamp(value).strftime("%Y%m%d")

    def _request_state(self, state, param, start, end):
        url = "https://aqs.epa.gov/data/api/sampleData/byState"
        params = {
            "email": self.email,
            "key": self.key,
            "param": param,
            "bdate": self._date(start),
            "edate": self._date(end),
            "state": state,
        }

        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        header = payload.get("Header", [])
        if header:
            status = header[0].get("status")
            if status not in ("Success", "No data matched your selection"):
                raise RuntimeError(f"EPA AQS request failed: {header[0]}")

        return payload.get("Data", [])

    def fetch(
        self,
        start,
        end,
        parameters=("pm25", "ozone"),
        state_fips=None,
        **kwargs,
    ):
        state_fips = state_fips or getattr(self.region, "state_fips", ())
        rows = []

        for parameter in parameters:
            code = AQS_PARAMETER_CODES.get(parameter.lower(), parameter)

            for state in state_fips:
                rows.extend(
                    self._request_state(
                        state=state,
                        param=code,
                        start=start,
                        end=end,
                    )
                )
                time.sleep(self.request_pause)

        if not rows:
            return pd.DataFrame()

        raw = pd.DataFrame(rows)

        # Build a compact common-schema dataframe from the AQS response.
        out = pd.DataFrame()

        out["source"] = "EPA AQS"
        out["station_id"] = (
            raw["state_code"].astype(str).str.zfill(2)
            + raw["county_code"].astype(str).str.zfill(3)
            + raw["site_number"].astype(str).str.zfill(4)
        )

        # Prefer GMT fields so all cube times are on a common UTC basis.
        out["time"] = pd.to_datetime(
            raw["date_gmt"].astype(str) + " " + raw["time_gmt"].astype(str),
            utc=True,
            errors="coerce",
        )

        out["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
        out["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
        out["value"] = pd.to_numeric(raw["sample_measurement"], errors="coerce")
        out["units"] = raw.get("units_of_measure", "")
        out["parameter_name"] = raw.get("parameter", "")
        out["parameter_code"] = raw.get("parameter_code", "")
        out["method"] = raw.get("method_name", "")
        out["sample_duration"] = raw.get("sample_duration", "")

        code_to_name = {
            "88101": "pm25",
            "44201": "ozone",
            "81102": "pm10",
            "42101": "co",
            "42602": "no2",
            "42401": "so2",
        }

        out["variable"] = (
            out["parameter_code"]
            .astype(str)
            .map(code_to_name)
            .fillna(out["parameter_name"].astype(str).str.lower())
        )

        out = out.dropna(
            subset=["time", "latitude", "longitude", "value"]
        )

        # State-based API queries can return observations outside the desired
        # rectangular domain. The bbox is authoritative for the GPGL cube.
        out = out[
            out["latitude"].between(self.region.south, self.region.north)
            & out["longitude"].between(self.region.west, self.region.east)
        ]

        return out.reset_index(drop=True)

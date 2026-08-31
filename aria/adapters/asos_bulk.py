from __future__ import annotations

from io import StringIO
from datetime import datetime
import time
import requests
import pandas as pd
import xarray as xr

from .base import BaseAdapter

DEFAULT_NETWORKS = [
    "MN_ASOS", "WI_ASOS", "IL_ASOS", "IA_ASOS",
    "ND_ASOS", "SD_ASOS", "NE_ASOS",
]

def _iso(value):
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_complete_hour(now=None, lag_minutes=10):
    """
    Return start/end timestamps for the latest complete UTC hour.

    A small lag is subtracted before flooring to the hour so upstream feeds
    have time to populate the latest observations.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    safe_now = now - pd.Timedelta(minutes=lag_minutes)
    end = safe_now.floor("h")
    start = end - pd.Timedelta(hours=1)
    return start, end


class BulkIEMASOSAdapter(BaseAdapter):
    """
    Bulk ASOS/AWOS fetcher for IEM.

    This intentionally avoids ACT's current station-by-station request loop.
    IEM added a 1-second/IP throttle in April 2026 and explicitly recommends
    requesting multiple stations/networks in a single call.

    Output is a station-keyed dict of xarray.Dataset objects to remain
    compatible with the shape returned by act.discovery.get_asos_data().
    """
    name = "asos"

    def __init__(
        self,
        region,
        networks=None,
        timeout=180,
        max_retries=6,
        backoff=2.0,
        min_request_interval=1.1,
        session=None,
    ):
        super().__init__(region)
        self.networks = list(networks or getattr(region, "asos_networks", DEFAULT_NETWORKS))
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.min_request_interval = min_request_interval
        self.session = session or requests.Session()
        self._last_request = 0.0

    def _wait(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    def _get(self, params):
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        for attempt in range(self.max_retries):
            self._wait()
            response = self.session.get(url, params=params, timeout=self.timeout)
            self._last_request = time.monotonic()

            if response.status_code not in (429, 503):
                response.raise_for_status()
                return response

            retry_after = response.headers.get("Retry-After")
            if retry_after:
                delay = float(retry_after)
            else:
                delay = self.backoff * (2 ** attempt)
            time.sleep(delay)

        response.raise_for_status()

    def fetch(
        self,
        start,
        end,
        variables=None,
        networks=None,
        **kwargs,
    ):
        networks = networks or self.networks
        variables = variables or [
            "tmpf","dwpf","relh","drct","sknt","p01i","alti","mslp",
            "vsby","gust","skyc1","skyl1","feel"
        ]

        params = [
            ("format", "comma"),
            ("tz", "Etc/UTC"),
            ("latlon", "yes"),
            ("elev", "yes"),
            ("missing", "M"),
            ("trace", "T"),
            ("direct", "no"),
            ("sts", _iso(start)),
            ("ets", _iso(end)),
            ("network", ",".join(networks)),
        ]
        for v in variables:
            params.append(("data", v))

        response = self._get(params)
        text = response.text

        # IEM CSV has comment/header lines beginning with '#'
        df = pd.read_csv(StringIO(text), comment="#", na_values=["M", "T"])
        if df.empty:
            return {}

        # Normalize station identifier and time
        station_col = "station" if "station" in df.columns else "station_id"
        df["time"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"])

        # Spatial clip in case network coverage exceeds the GPGL bbox
        if "lat" in df and "lon" in df:
            df = df[
                df["lat"].between(self.region.south, self.region.north)
                & df["lon"].between(self.region.west, self.region.east)
            ]

        out = {}
        for station, sdf in df.groupby(station_col):
            sdf = sdf.sort_values("time").set_index("time")
            ds = xr.Dataset.from_dataframe(sdf)
            ds.attrs.update({
                "source": "Iowa Environmental Mesonet",
                "network_type": "ASOS/AWOS",
                "station_id": str(station),
            })
            out[str(station)] = ds
        return out

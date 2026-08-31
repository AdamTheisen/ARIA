from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET

import requests


# Operational WSR-88D sites that cover or border the GPGL domain.
GPGL_NEXRAD_SITES = {
    "KABR": "Aberdeen, SD",
    "KBIS": "Bismarck, ND",
    "KDLH": "Duluth, MN",
    "KDMX": "Des Moines, IA",
    "KDVN": "Davenport, IA",
    "KFSD": "Sioux Falls, SD",
    "KGRB": "Green Bay, WI",
    "KLOT": "Chicago/Romeoville, IL",
    "KLNX": "North Platte, NE",
    "KMKX": "Milwaukee/Sullivan, WI",
    "KMPX": "Minneapolis/Chanhassen, MN",
    "KMVX": "Grand Forks/Mayville, ND",
    "KOAX": "Omaha/Valley, NE",
    "KUEX": "Hastings, NE",
    "KARX": "La Crosse, WI",
}


def nexrad_site_metadata():
    """Return known NEXRAD locations, preferring Py-ART's national registry."""
    out = {rid: {"name": name, "latitude": None, "longitude": None}
           for rid, name in GPGL_NEXRAD_SITES.items()}
    try:
        import pyart
        locations = pyart.io.nexrad_common.NEXRAD_LOCATIONS
        for rid, meta in locations.items():
            rid = str(rid).upper()
            if not rid.startswith("K"):
                continue
            if isinstance(meta, dict):
                lat = meta.get("lat") or meta.get("latitude")
                lon = meta.get("lon") or meta.get("longitude")
                name = meta.get("name") or meta.get("city") or rid
            else:
                try:
                    lat, lon = float(meta[0]), float(meta[1])
                    name = rid
                except Exception:
                    continue
            out[rid] = {"name": str(name), "latitude": lat, "longitude": lon}
    except Exception:
        pass
    return out

def nexrad_sites_for_region(region, buffer_deg=2.0):
    sites = nexrad_site_metadata()
    selected = {}
    for rid, meta in sites.items():
        lat, lon = meta.get("latitude"), meta.get("longitude")
        if lat is None or lon is None:
            if region.name == "GPGL" and rid in GPGL_NEXRAD_SITES:
                selected[rid] = meta
            continue
        if (region.south-buffer_deg <= float(lat) <= region.north+buffer_deg
                and region.west-buffer_deg <= float(lon) <= region.east+buffer_deg):
            selected[rid] = meta
    return selected


@dataclass
class NEXRADScan:
    radar_id: str
    site_name: str
    key: str
    local_path: Path
    scan_time: datetime | None = None


class NEXRADLevel2Adapter:
    """
    Download recent NEXRAD Level-II volumes from the public NEXRAD Open Data
    AWS bucket without requiring AWS credentials.

    The adapter intentionally downloads one radar volume at a time. This keeps
    the interactive Streamlit workflow responsive and avoids pulling a full
    regional set of large Level-II files merely to inspect one radar.
    """

    bucket_url = "https://unidata-nexrad-level2.s3.amazonaws.com"

    def __init__(
        self,
        cache_dir=None,
        timeout=60,
    ):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "aria" / "nexrad"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

        self.metadata_cache = self.cache_dir / "latest_keys.json"

    def _load_metadata_cache(self):
        if not self.metadata_cache.exists():
            return {}
        try:
            return json.loads(self.metadata_cache.read_text())
        except Exception:
            return {}

    def _save_metadata_cache(self, payload):
        try:
            self.metadata_cache.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass

    def _cached_latest_key(self, radar_id, when, ttl_seconds=240):
        cache = self._load_metadata_cache()
        entry = cache.get(radar_id)
        if not entry:
            return None

        if time.time() - entry.get("fetched_at", 0) > ttl_seconds:
            return None

        scan_time = datetime.fromisoformat(entry["scan_time"])
        if scan_time.tzinfo is None:
            scan_time = scan_time.replace(tzinfo=timezone.utc)

        if scan_time <= when:
            return scan_time, entry["key"]

        return None

    def _store_latest_key(self, radar_id, scan_time, key):
        cache = self._load_metadata_cache()
        cache[radar_id] = {
            "scan_time": scan_time.isoformat(),
            "key": key,
            "fetched_at": time.time(),
        }
        self._save_metadata_cache(cache)

    def _list_keys(self, radar_id, day):
        prefix = f"{day:%Y/%m/%d}/{radar_id}/"

        response = requests.get(
            self.bucket_url,
            params={
                "list-type": "2",
                "prefix": prefix,
                "max-keys": 1000,
            },
            timeout=self.timeout,
        )
        if response.status_code == 403:
            raise RuntimeError(
                "NEXRAD bucket listing returned HTTP 403. The legacy "
                "`noaa-nexrad-level2` bucket was retired in 2025; this "
                "package expects the current `unidata-nexrad-level2` "
                "Open Data bucket. Reinstall/update ARIA if you "
                "are still using the old endpoint."
            )

        response.raise_for_status()

        root = ET.fromstring(response.content)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        keys = [
            node.text
            for node in root.findall(".//s3:Key", namespace)
            if node.text
        ]

        # Metadata files are not radar volumes.
        return [
            key
            for key in keys
            if not key.endswith("_MDM")
            and not key.endswith(".txt")
        ]

    @staticmethod
    def _time_from_key(key):
        name = Path(key).name

        # Common form: KMPX20260825_132345_V06
        try:
            stamp = name[4:19]
            return datetime.strptime(
                stamp,
                "%Y%m%d_%H%M%S",
            ).replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def latest_key(
        self,
        radar_id,
        *,
        when=None,
        lookback_hours=6,
    ):
        radar_id = radar_id.upper()
        if len(radar_id) != 4 or not radar_id.startswith("K"):
            raise ValueError("NEXRAD radar_id must be a four-character US site ID such as KMPX.")

        if when is None:
            when = datetime.now(timezone.utc)

        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        else:
            when = when.astimezone(timezone.utc)

        earliest = when - timedelta(hours=lookback_hours)

        cached = self._cached_latest_key(radar_id, when)
        if cached is not None and cached[0] >= earliest:
            return cached

        # Search today and, when needed, yesterday.
        days = sorted(
            {
                when.date(),
                earliest.date(),
            },
            reverse=True,
        )

        candidates = []

        for day in days:
            day_dt = datetime(
                day.year,
                day.month,
                day.day,
                tzinfo=timezone.utc,
            )

            for key in self._list_keys(radar_id, day_dt):
                scan_time = self._time_from_key(key)

                if scan_time is None:
                    continue

                if earliest <= scan_time <= when:
                    candidates.append((scan_time, key))

        if not candidates:
            raise RuntimeError(
                f"No {radar_id} Level-II scan was found in the "
                f"{lookback_hours}-hour lookback window."
            )

        candidates.sort(key=lambda item: item[0])
        latest = candidates[-1]
        self._store_latest_key(radar_id, latest[0], latest[1])
        return latest

    def download_latest(
        self,
        radar_id,
        *,
        when=None,
        lookback_hours=6,
    ):
        scan_time, key = self.latest_key(
            radar_id,
            when=when,
            lookback_hours=lookback_hours,
        )

        destination = self.cache_dir / Path(key).name

        if not destination.exists():
            response = requests.get(
                f"{self.bucket_url}/{key}",
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()

            with destination.open("wb") as handle:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        handle.write(chunk)

        return NEXRADScan(
            radar_id=radar_id.upper(),
            site_name=nexrad_site_metadata().get(radar_id.upper(), {}).get("name", radar_id.upper()),
            key=key,
            local_path=destination,
            scan_time=scan_time,
        )

    def read(self, scan):
        try:
            import pyart
        except ImportError as exc:
            raise RuntimeError(
                "NEXRAD support requires ARM Py-ART. Install the package "
                "dependencies again with `pip install -e .`."
            ) from exc

        return pyart.io.read_nexrad_archive(
            str(scan.local_path)
        )

    def fetch_latest_radar(
        self,
        radar_id,
        *,
        when=None,
        lookback_hours=6,
    ):
        scan = self.download_latest(
            radar_id,
            when=when,
            lookback_hours=lookback_hours,
        )
        radar = self.read(scan)
        return radar, scan

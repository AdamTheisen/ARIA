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



NEXRAD_DISPLAY_NAMES = {
    **GPGL_NEXRAD_SITES,
    "KTLX":"Oklahoma City, OK","KVNX":"Vance AFB/Enid, OK","KINX":"Tulsa, OK",
    "KICT":"Wichita, KS","KDDC":"Dodge City, KS","KGLD":"Goodland, KS",
    "KFTG":"Denver, CO","KPUX":"Pueblo, CO","KEAX":"Kansas City/Pleasant Hill, MO",
    "KLSX":"St. Louis, MO","KILX":"Lincoln, IL","KGRR":"Grand Rapids, MI",
    "KAPX":"Gaylord, MI","KMQT":"Marquette, MI",
    "KFDR":"Frederick, OK","KTWX":"Topeka, KS","KSGF":"Springfield, MO",
    "KSRX":"Fort Smith, AR","KAMA":"Amarillo, TX","KDYX":"Dyess AFB/Abilene, TX",
    "KLBB":"Lubbock, TX","KFWS":"Dallas/Fort Worth, TX","KSHV":"Shreveport, LA",

    # Southeast U.S. sites commonly included by the SE US regional preset.
    "KFFC":"Atlanta/Peachtree City, GA","KJGX":"Robins AFB, GA",
    "KBMX":"Birmingham/Calera, AL","KHTX":"Huntsville/Hytop, AL",
    "KEOX":"Fort Rucker, AL","KMXX":"Maxwell AFB, AL","KMOB":"Mobile, AL",
    "KGSP":"Greenville-Spartanburg/Greer, SC","KCAE":"Columbia, SC",
    "KCLX":"Charleston, SC","KTLH":"Tallahassee, FL","KEVX":"Eglin AFB, FL",
    "KTBW":"Tampa Bay/Ruskin, FL","KMLB":"Melbourne, FL","KAMX":"Miami, FL",
    "KBYX":"Key West, FL","KJAX":"Jacksonville, FL",
    "KDGX":"Jackson, MS","KGWX":"Columbus AFB, MS",
    "KMRX":"Knoxville/Morristown, TN","KOHX":"Nashville, TN",
    "KHPX":"Fort Campbell, KY","KJKL":"Jackson, KY",
    "KRLX":"Charleston, WV","KFCX":"Blacksburg, VA","KAKQ":"Wakefield, VA",
    "KRAX":"Raleigh, NC","KMHX":"Morehead City, NC","KLTX":"Wilmington, NC",
    "KPOE":"Fort Polk, LA",
}
def normalize_nexrad_id(radar_id):
    """Return a canonical four-character CONUS NEXRAD site ID."""
    rid=str(radar_id or "").strip().upper()
    if len(rid)==3 and rid.isalnum():
        rid="K"+rid
    return rid


def nexrad_display_name(radar_id, metadata=None):
    """Human-friendly radar label with an explicit city/state when available."""
    rid=normalize_nexrad_id(radar_id)
    metadata=metadata or {}
    name=NEXRAD_DISPLAY_NAMES.get(rid)
    if not name:
        city=metadata.get("city") or metadata.get("location")
        state=metadata.get("state") or metadata.get("st")
        candidate=metadata.get("name")
        if city:
            name=f"{city}, {state}" if state else str(city)
        elif candidate and normalize_nexrad_id(candidate)!=rid and str(candidate).upper()!=rid:
            name=str(candidate)
            if state and str(state).lower() not in name.lower():
                name=f"{name}, {state}"
    # Never silently render only an opaque radar ID; make missing metadata visible.
    return f"{rid} — {name}" if name else f"{rid} — location unavailable"


def nexrad_site_metadata():
    """Return known NEXRAD locations using canonical K-prefixed IDs.

    Py-ART commonly exposes the national registry with three-character IDs
    (for example MPX). ARIA normalizes those to KMPX before regional filtering
    and labeling so non-GPGL profiles receive the same city/location metadata.
    """
    out = {
        rid: {
            "name": name,
            "city": name.rsplit(",",1)[0] if "," in name else name,
            "state": name.rsplit(",",1)[1].strip() if "," in name else None,
            "latitude": None,
            "longitude": None,
        }
        for rid, name in GPGL_NEXRAD_SITES.items()
    }
    try:
        import pyart
        locations = pyart.io.nexrad_common.NEXRAD_LOCATIONS
        for raw_id, meta in locations.items():
            rid = normalize_nexrad_id(raw_id)
            if len(rid)!=4 or not rid.startswith("K"):
                continue
            lat=lon=None
            city=state=None
            name=None
            if isinstance(meta, dict):
                lat = meta.get("lat") if meta.get("lat") is not None else meta.get("latitude")
                lon = meta.get("lon") if meta.get("lon") is not None else meta.get("longitude")
                city = meta.get("city") or meta.get("location")
                state = meta.get("state") or meta.get("st")
                name = meta.get("name")
            else:
                try:
                    lat, lon = float(meta[0]), float(meta[1])
                except Exception:
                    continue
                # Some Py-ART registry versions include a descriptive name
                # after latitude/longitude.
                if len(meta) > 2:
                    name = meta[2]

            preferred=NEXRAD_DISPLAY_NAMES.get(rid)
            if preferred:
                display=preferred
                city=city or (preferred.rsplit(",",1)[0] if "," in preferred else preferred)
                state=state or (preferred.rsplit(",",1)[1].strip() if "," in preferred else None)
            elif city:
                display=f"{city}, {state}" if state else str(city)
            elif name and normalize_nexrad_id(name)!=rid:
                display=str(name)
                if state and str(state).lower() not in display.lower():
                    display=f"{display}, {state}"
            else:
                display=None

            out[rid] = {
                "name": display or rid,
                "city": city,
                "state": state,
                "latitude": lat,
                "longitude": lon,
            }
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

    def recent_scans(self, radar_id, *, when=None, lookback_minutes=60, max_scans=12):
        radar_id=radar_id.upper(); when=when or datetime.now(timezone.utc)
        when=when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)
        earliest=when-timedelta(minutes=float(lookback_minutes)); candidates=[]
        for day in sorted({when.date(),earliest.date()}):
            day_dt=datetime(day.year,day.month,day.day,tzinfo=timezone.utc)
            for key in self._list_keys(radar_id,day_dt):
                scan_time=self._time_from_key(key)
                if scan_time is not None and earliest<=scan_time<=when: candidates.append((scan_time,key))
        candidates=sorted(set(candidates),key=lambda x:x[0])[-int(max_scans):]
        return [NEXRADScan(radar_id=radar_id,site_name=nexrad_site_metadata().get(radar_id,{}).get("name",radar_id),key=key,local_path=self.cache_dir/Path(key).name,scan_time=scan_time) for scan_time,key in candidates]

    def download_scan(self, scan):
        destination=self.cache_dir/Path(scan.key).name
        if not destination.exists():
            response=requests.get(f"{self.bucket_url}/{scan.key}",timeout=self.timeout,stream=True); response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk: handle.write(chunk)
        return NEXRADScan(radar_id=scan.radar_id,site_name=scan.site_name,key=scan.key,local_path=destination,scan_time=scan.scan_time)

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

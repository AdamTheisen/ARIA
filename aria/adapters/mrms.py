from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import os
import time
import re
import shutil

import numpy as np
import pandas as pd
import requests
import xarray as xr

LATEST_TTL_SECONDS = int(os.environ.get("ARIA_MRMS_LATEST_TTL", "120"))

def invalidate_latest_mrms_cache(cache_dir=None):
    """Invalidate only rolling MRMS latest files; historical files remain cached."""
    root = Path(cache_dir or Path.home()/".cache"/"aria"/"mrms")
    if not root.exists():
        return
    for p in root.glob("*latest*.grib2*"):
        try: p.unlink()
        except FileNotFoundError: pass

def _fresh(path, ttl=LATEST_TTL_SECONDS):
    return path.exists() and (time.time()-path.stat().st_mtime) < ttl


@dataclass(frozen=True)
class MRMSProduct:
    name: str = "MergedReflectivityQCComposite"
    base_url: str = "https://mrms.ncep.noaa.gov/2D"


class MRMSAdapter:
    """Retrieve NOAA MRMS 2-D GRIB2 products and subset to a lat/lon region."""

    def __init__(self, region, *, cache_dir=None, product=None, timeout=45):
        self.region = region
        self.product = product or MRMSProduct()
        self.cache_dir = Path(cache_dir or Path.home()/".cache"/"aria"/"mrms")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    @property
    def directory_url(self):
        return f"{self.product.base_url}/{self.product.name}/"

    def _listing(self):
        r=requests.get(self.directory_url,timeout=self.timeout)
        r.raise_for_status()
        return r.text

    def available_files(self):
        html=self._listing()
        pat=re.compile(
            rf'(MRMS_{re.escape(self.product.name)}_00\.50_(\d{{8}}-\d{{6}})\.grib2\.gz)'
        )
        rows=[]
        for name, stamp in pat.findall(html):
            t=pd.to_datetime(stamp,format="%Y%m%d-%H%M%S",utc=True)
            rows.append((t,name))
        return sorted(set(rows), key=lambda x:x[0])

    def choose_file(self, when=None, lookback_hours=3):
        if when is None:
            return None, f"MRMS_{self.product.name}.latest.grib2.gz"
        when=pd.Timestamp(when)
        when=when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
        files=self.available_files()
        candidates=[x for x in files if x[0] <= when and x[0] >= when-pd.Timedelta(hours=lookback_hours)]
        if not candidates:
            raise FileNotFoundError(
                f"No {self.product.name} MRMS file found at or before {when} within {lookback_hours} h."
            )
        return candidates[-1]

    def download(self, when=None, lookback_hours=3):
        scan_time,name=self.choose_file(when=when,lookback_hours=lookback_hours)
        gz=self.cache_dir/name
        grib=self.cache_dir/name[:-3]
        is_latest = "latest" in name.lower()
        need_fetch = (is_latest and not _fresh(grib)) or (not is_latest and not grib.exists())
        if need_fetch:
            if is_latest:
                for p in (gz,grib):
                    try: p.unlink()
                    except FileNotFoundError: pass
            if not gz.exists():
                with requests.get(self.directory_url+name,stream=True,timeout=self.timeout) as r:
                    r.raise_for_status()
                    tmp=gz.with_suffix(gz.suffix+".part")
                    with tmp.open("wb") as fh:
                        for chunk in r.iter_content(1024*1024):
                            if chunk: fh.write(chunk)
                    tmp.replace(gz)
            with gzip.open(gz,"rb") as src, grib.open("wb") as dst:
                shutil.copyfileobj(src,dst)
        return grib,scan_time

    def open(self, when=None, lookback_hours=3):
        path,scan_time=self.download(when=when,lookback_hours=lookback_hours)
        ds=xr.open_dataset(path,engine="cfgrib",backend_kwargs={"indexpath":""})
        ds=ds.load()
        # Normalize MRMS coordinates/field to the hub schema.
        rename={}
        if "latitude" not in ds.coords and "lat" in ds.coords: rename["lat"]="latitude"
        if "longitude" not in ds.coords and "lon" in ds.coords: rename["lon"]="longitude"
        if rename: ds=ds.rename(rename)
        if "longitude" in ds.coords:
            lon=ds.longitude
            ds=ds.assign_coords(longitude=xr.where(lon>180,lon-360,lon))
        data_vars=list(ds.data_vars)
        if not data_vars:
            raise ValueError("MRMS GRIB2 contained no data variables.")
        # The product contains a single reflectivity field; use first 2-D field.
        var=next((v for v in data_vars if ds[v].ndim>=2),data_vars[0])
        da=ds[var].squeeze(drop=True).astype(float)
        # MRMS GRIB missing/no-data sentinels can be far below meteorological
        # reflectivity values; keep ordinary weak/negative dBZ but mask sentinels.
        da=da.where(da > -100.0)
        out=da.to_dataset(name="reflectivity")
        if "latitude" not in out.coords or "longitude" not in out.coords:
            raise ValueError("MRMS GRIB2 did not expose latitude/longitude coordinates.")
        lat=out.latitude; lon=out.longitude
        # MRMS is a regular grid, often north-to-south.
        if lat.ndim==1 and lon.ndim==1:
            lat_sel=slice(self.region.south,self.region.north) if lat.values[0] < lat.values[-1] else slice(self.region.north,self.region.south)
            lon_sel=slice(self.region.west,self.region.east) if lon.values[0] < lon.values[-1] else slice(self.region.east,self.region.west)
            out=out.sel(latitude=lat_sel,longitude=lon_sel)
        else:
            mask=(lat>=self.region.south)&(lat<=self.region.north)&(lon>=self.region.west)&(lon<=self.region.east)
            out=out.where(mask,drop=True)
        if scan_time is None:
            # latest files normally carry time in GRIB metadata
            t=ds.coords.get("time")
            if t is not None:
                try: scan_time=pd.Timestamp(t.values).tz_localize("UTC")
                except Exception: scan_time=pd.Timestamp(t.values)
        out.attrs.update({
            "source":"NOAA MRMS",
            "product":self.product.name,
            "analysis_time":str(scan_time) if scan_time is not None else "latest",
            "quality_controlled":True,
        })
        out.reflectivity.attrs.update({"units":"dBZ","long_name":"MRMS quality-controlled composite reflectivity"})
        return out, scan_time

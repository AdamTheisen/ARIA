from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class ModelRun:
    model: str
    initialization_time: pd.Timestamp
    forecast_hour: int
    valid_time: pd.Timestamp
    source: str = ""
    product: str = ""

    @property
    def label(self):
        return (
            f"{self.model.upper()} "
            f"{self.initialization_time:%Y-%m-%d %HZ} "
            f"+{self.forecast_hour:02d} "
            f"({self.valid_time:%Y-%m-%d %H:%MZ})"
        )


class ModelAdapter(ABC):
    """
    Common model interface for GPGL.

    Adapters should return xarray datasets with standardized GPGL variable
    names and preserve model/run/provenance metadata in Dataset.attrs.
    """

    name = "model"

    def __init__(self, region, cache_dir=None):
        self.region = region
        self.cache_dir = Path(
            cache_dir
            or Path.home() / ".cache" / "aria" / "models" / self.name
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def open_run(self, *args, **kwargs) -> tuple[xr.Dataset, ModelRun]:
        raise NotImplementedError

    def subset_region(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Subset either a regular lat/lon dataset or a model grid with 2-D
        latitude/longitude coordinates to the configured GPGL rectangle.
        """
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"

        if lat_name not in ds.coords or lon_name not in ds.coords:
            return ds

        lat = ds[lat_name]
        lon = ds[lon_name]

        if lon.ndim == 1 and lat.ndim == 1:
            lon_values = lon
            if float(lon.max()) > 180:
                west = self.region.west % 360
                east = self.region.east % 360
                if west <= east:
                    ds = ds.sel({lon_name: slice(west, east)})
                else:
                    ds = ds.where(
                        (ds[lon_name] >= west) | (ds[lon_name] <= east),
                        drop=True,
                    )
            else:
                ds = ds.sel({lon_name: slice(self.region.west, self.region.east)})

            # Handle ascending or descending latitude.
            if float(lat[0]) <= float(lat[-1]):
                ds = ds.sel({lat_name: slice(self.region.south, self.region.north)})
            else:
                ds = ds.sel({lat_name: slice(self.region.north, self.region.south)})
            return ds

        # Curvilinear grid: find the smallest y/x index box containing the mask.
        lon_work = lon
        if float(lon.max()) > 180:
            lon_work = ((lon + 180) % 360) - 180

        mask = (
            (lat >= self.region.south)
            & (lat <= self.region.north)
            & (lon_work >= self.region.west)
            & (lon_work <= self.region.east)
        )

        import numpy as np
        where = np.argwhere(mask.values)
        if not len(where):
            return ds

        # Assume final two dimensions correspond to horizontal y/x.
        ydim, xdim = lat.dims[-2], lat.dims[-1]
        y0, x0 = where.min(axis=0)[-2:]
        y1, x1 = where.max(axis=0)[-2:]
        ds = ds.isel({
            ydim: slice(int(y0), int(y1) + 1),
            xdim: slice(int(x0), int(x1) + 1),
        })

        if "longitude" in ds.coords and float(ds.longitude.max()) > 180:
            ds = ds.assign_coords(
                longitude=((ds.longitude + 180) % 360) - 180
            )
        return ds

    @staticmethod
    def attach_provenance(ds, run: ModelRun, *, source_files=None, history=None):
        ds = ds.copy()
        ds.attrs.update({
            "model": run.model,
            "initialization_time": run.initialization_time.isoformat(),
            "forecast_hour": int(run.forecast_hour),
            "valid_time": run.valid_time.isoformat(),
            "model_product": run.product,
            "model_source": run.source,
        })
        if source_files:
            ds.attrs["source_files"] = ";".join(map(str, source_files))
        if history:
            ds.attrs["processing_history"] = history
        return ds

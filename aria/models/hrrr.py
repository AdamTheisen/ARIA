from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import re
import numpy as np
import pandas as pd
import xarray as xr

from .base import ModelAdapter, ModelRun


HRRR_SURFACE_VARIABLES = {
    "air_temperature_2m": {
        "search": ":TMP:2 m above ground",
        "candidates": ("t2m", "t"),
        "units": "degC",
        "label": "2-m Air Temperature",
    },
    "dew_point_temperature_2m": {
        "search": ":DPT:2 m above ground",
        "candidates": ("d2m", "dpt"),
        "units": "degC",
        "label": "2-m Dew Point",
    },
    "u_wind_10m": {
        "search": ":UGRD:10 m above ground",
        "candidates": ("u10", "u"),
        "units": "m s-1",
        "label": "10-m U Wind",
    },
    "v_wind_10m": {
        "search": ":VGRD:10 m above ground",
        "candidates": ("v10", "v"),
        "units": "m s-1",
        "label": "10-m V Wind",
    },
    "composite_reflectivity": {
        "search": ":REFC:entire atmosphere",
        "candidates": ("refc", "unknown"),
        "units": "dBZ",
        "label": "Composite Reflectivity",
    },
    "precipitation": {
        "search": ":APCP:surface",
        "candidates": ("tp", "apcp"),
        "units": "kg m-2",
        "label": "Accumulated Precipitation",
    },
    "precipitation_rate": {
        "search": ":PRATE:surface",
        "candidates": ("prate",),
        "units": "kg m-2 s-1",
        "label": "Precipitation Rate",
    },
    "reflectivity_1km": {
        "search": ":REFD:1000 m above ground",
        "candidates": ("refd", "refd1000"),
        "units": "dBZ",
        "label": "1-km AGL Reflectivity",
    },
    "categorical_rain": {
        "search": ":CRAIN:surface",
        "candidates": ("crain",),
        "units": "1",
        "label": "Categorical Rain",
    },
    "categorical_snow": {
        "search": ":CSNOW:surface",
        "candidates": ("csnow",),
        "units": "1",
        "label": "Categorical Snow",
    },
    "categorical_freezing_rain": {
        "search": ":CFRZR:surface",
        "candidates": ("cfrzr",),
        "units": "1",
        "label": "Categorical Freezing Rain",
    },
    "categorical_ice_pellets": {
        "search": ":CICEP:surface",
        "candidates": ("cicep",),
        "units": "1",
        "label": "Categorical Ice Pellets",
    },
    "surface_pressure": {
        "search": ":PRES:surface",
        "candidates": ("sp", "pres"),
        "units": "Pa",
        "label": "Surface Pressure",
    },
}

HRRR_PRESSURE_VARIABLES = {
    "air_temperature": {
        "search": ":TMP:[0-9]+ mb",
        "candidates": ("t",),
        "units": "degC",
        "label": "Air Temperature",
    },
    "dew_point_temperature": {
        "search": ":DPT:[0-9]+ mb",
        "candidates": ("dpt", "d"),
        "units": "degC",
        "label": "Dew Point",
    },
    "u_wind": {
        "search": ":UGRD:[0-9]+ mb",
        "candidates": ("u",),
        "units": "m s-1",
        "label": "U Wind",
    },
    "v_wind": {
        "search": ":VGRD:[0-9]+ mb",
        "candidates": ("v",),
        "units": "m s-1",
        "label": "V Wind",
    },
    "vertical_velocity": {
        "search": ":VVEL:[0-9]+ mb",
        "candidates": ("w", "vvel"),
        "units": "Pa s-1",
        "label": "Pressure Vertical Velocity",
    },
    "geopotential_height": {
        "search": ":HGT:[0-9]+ mb",
        "candidates": ("gh", "hgt"),
        "units": "gpm",
        "label": "Geopotential Height",
    },
}


class HRRRAdapter(ModelAdapter):
    """
    HRRR access through Herbie.

    Herbie uses NOAA operational/NODD sources and GRIB index files to retrieve
    only requested messages rather than requiring whole HRRR files.
    """

    name = "hrrr"

    def _herbie(self):
        try:
            from herbie import Herbie
        except ImportError as exc:
            raise RuntimeError(
                "HRRR support requires herbie-data. Reinstall this package "
                "with `pip install -e .`."
            ) from exc
        return Herbie

    @staticmethod
    def _latest_cycle(now=None, lag_hours=2):
        now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
        # Conservative operational availability delay.
        return (now - pd.Timedelta(hours=lag_hours)).floor("h")

    def available_cycles(self, count=12, now=None):
        latest = self._latest_cycle(now)
        return [latest - pd.Timedelta(hours=i) for i in range(count)]

    @staticmethod
    def _find_data_var(ds, candidates):
        for candidate in candidates:
            if candidate in ds.data_vars:
                return candidate
        # Herbie/cfgrib generally returns only one requested meteorological
        # variable plus coordinates; use that variable as a safe fallback.
        data_vars = [
            name for name in ds.data_vars
            if name not in {"gribfile_projection"}
        ]
        if len(data_vars) == 1:
            return data_vars[0]
        raise RuntimeError(
            f"Unable to identify requested HRRR field. "
            f"Dataset variables: {list(ds.data_vars)}"
        )

    @staticmethod
    def _normalize_coords(ds):
        rename = {}
        if "latitude" not in ds.coords and "lat" in ds.coords:
            rename["lat"] = "latitude"
        if "longitude" not in ds.coords and "lon" in ds.coords:
            rename["lon"] = "longitude"
        if rename:
            ds = ds.rename(rename)

        if "longitude" in ds.coords:
            lon = ds.longitude
            try:
                if float(lon.max()) > 180:
                    ds = ds.assign_coords(
                        longitude=((lon + 180) % 360) - 180
                    )
            except Exception:
                pass

        for p_name in ("isobaricInhPa", "isobaricInPa"):
            if p_name in ds.coords:
                if p_name == "isobaricInPa":
                    ds = ds.assign_coords(
                        pressure_hpa=ds[p_name] / 100.0
                    ).swap_dims({p_name: "pressure_hpa"})
                else:
                    ds = ds.rename({p_name: "pressure_hpa"})
                break
        return ds

    @staticmethod
    def _convert_units(da, standardized_name):
        attrs = dict(da.attrs)
        units = str(attrs.get("units", ""))

        if "temperature" in standardized_name:
            # HRRR temperatures are normally Kelvin.
            if units.lower() in {"k", "kelvin"} or (
                np.isfinite(da.values).any()
                and float(np.nanmedian(da.values)) > 150
            ):
                da = da - 273.15
            da.attrs["units"] = "degC"

        return da

    def _open_one(self, cycle, fxx, product, meta):
        Herbie = self._herbie()
        H = Herbie(
            pd.Timestamp(cycle).tz_localize(None),
            model="hrrr",
            product=product,
            fxx=int(fxx),
            save_dir=self.cache_dir,
            overwrite=False,
            verbose=False,
        )

        ds = H.xarray(
            meta["search"],
            remove_grib=False,
        )

        if isinstance(ds, list):
            # Merge compatible hypercubes where possible.
            ds = xr.merge(ds, compat="override", join="outer")

        ds = self._normalize_coords(ds)
        source = str(getattr(H, "grib", "") or getattr(H, "grib_source", ""))
        return ds, source

    def open_run(
        self,
        *,
        cycle=None,
        forecast_hour=0,
        variables=None,
        product="sfc",
    ):
        if cycle is None:
            cycle = self._latest_cycle()

        cycle = pd.Timestamp(cycle)
        if cycle.tzinfo is None:
            cycle = cycle.tz_localize("UTC")
        else:
            cycle = cycle.tz_convert("UTC")

        forecast_hour = int(forecast_hour)
        valid = cycle + pd.Timedelta(hours=forecast_hour)

        registry = (
            HRRR_SURFACE_VARIABLES
            if product == "sfc"
            else HRRR_PRESSURE_VARIABLES
        )

        if variables is None:
            variables = list(registry)

        arrays = []
        sources = []

        for standardized_name in variables:
            if standardized_name not in registry:
                raise KeyError(
                    f"Unknown HRRR {product} variable {standardized_name!r}. "
                    f"Available: {list(registry)}"
                )
            meta = registry[standardized_name]
            source_ds, source = self._open_one(
                cycle, forecast_hour, product, meta
            )
            raw_name = self._find_data_var(source_ds, meta["candidates"])
            da = source_ds[raw_name].squeeze(drop=True)
            da = self._convert_units(da, standardized_name)
            da = da.rename(standardized_name)
            da.attrs["long_name"] = meta["label"]
            arrays.append(da)
            if source:
                sources.append(source)

        ds = xr.merge(arrays, compat="override", join="outer")
        ds = self._normalize_coords(ds)
        ds = self.subset_region(ds)

        if "u_wind_10m" in ds and "v_wind_10m" in ds:
            ds["wind_speed_10m"] = np.hypot(
                ds["u_wind_10m"], ds["v_wind_10m"]
            )
            ds["wind_speed_10m"].attrs.update({
                "units": "m s-1",
                "long_name": "10-m Wind Speed",
            })

        if "u_wind" in ds and "v_wind" in ds:
            ds["wind_speed"] = np.hypot(ds["u_wind"], ds["v_wind"])
            ds["wind_speed"].attrs.update({
                "units": "m s-1",
                "long_name": "Wind Speed",
            })

        run = ModelRun(
            model="hrrr",
            initialization_time=cycle,
            forecast_hour=forecast_hour,
            valid_time=valid,
            source="NOAA HRRR via Herbie/NODD",
            product=product,
        )

        ds = self.attach_provenance(
            ds,
            run,
            source_files=sources,
            history=(
                "Herbie selective GRIB retrieval; GPGL rectangular subset; "
                "standardized GPGL variable names."
            ),
        )
        return ds, run

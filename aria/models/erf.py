from __future__ import annotations
import pandas as pd
import xarray as xr
from .base import ModelAdapter, ModelRun


class ERFAdapter(ModelAdapter):
    """
    Adapter for local ERF NetCDF output.

    ERF can retain its native model grid. A variable_map allows an experiment
    to map local ERF output names onto the GPGL standardized names without
    hard-coding one experiment configuration.
    """
    name = "erf"

    def open_run(
        self,
        path,
        *,
        variable_map=None,
        initialization_time=None,
        forecast_hour=0,
    ):
        ds = xr.open_dataset(path)
        if variable_map:
            rename = {
                source: target
                for source, target in variable_map.items()
                if source in ds
            }
            ds = ds.rename(rename)

        ds = self.subset_region(ds)

        init = pd.Timestamp(
            initialization_time
            or ds.attrs.get("initialization_time")
            or ds.attrs.get("start_time")
            or pd.Timestamp.now(tz="UTC")
        )
        if init.tzinfo is None:
            init = init.tz_localize("UTC")
        valid = init + pd.Timedelta(hours=int(forecast_hour))

        run = ModelRun(
            model="erf",
            initialization_time=init,
            forecast_hour=int(forecast_hour),
            valid_time=valid,
            source=str(path),
            product="local-netcdf",
        )
        ds = self.attach_provenance(
            ds, run, source_files=[path],
            history="Opened ERF NetCDF; optional variable normalization; GPGL subset."
        )
        return ds, run

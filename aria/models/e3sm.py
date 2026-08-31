from __future__ import annotations
import pandas as pd
import xarray as xr
from .base import ModelAdapter, ModelRun


class E3SMAdapter(ModelAdapter):
    """
    Initial E3SM adapter for local history files.

    This intentionally leaves component-specific regridding to later releases,
    but establishes the same ModelAdapter/run/provenance contract used by HRRR
    and ERF.
    """
    name = "e3sm"

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
            or pd.Timestamp.now(tz="UTC")
        )
        if init.tzinfo is None:
            init = init.tz_localize("UTC")
        valid = init + pd.Timedelta(hours=int(forecast_hour))

        run = ModelRun(
            model="e3sm",
            initialization_time=init,
            forecast_hour=int(forecast_hour),
            valid_time=valid,
            source=str(path),
            product="local-history",
        )
        ds = self.attach_provenance(
            ds, run, source_files=[path],
            history="Opened E3SM history output; optional variable normalization; GPGL subset."
        )
        return ds, run

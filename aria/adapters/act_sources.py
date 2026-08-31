from __future__ import annotations
from datetime import datetime
from pathlib import Path
import act
from .base import BaseAdapter

def _dt(value):
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)

class ACTAirNowAdapter(BaseAdapter):
    name = "airnow"
    def __init__(self, region, token):
        super().__init__(region)
        self.token = token
    def fetch(
        self,
        start,
        end,
        parameters="OZONE,PM25",
        data_type="B",
        monitor_type=0,
        **kwargs,
    ):
        bbox = getattr(
            self.region,
            "airnow_bbox",
            f"{self.region.west},{self.region.south},{self.region.east},{self.region.north}",
        )
        try:
            return act.discovery.get_airnow_bounded_obs(
                token=self.token,
                start_date=_dt(start).strftime("%Y-%m-%dT%H"),
                end_date=_dt(end).strftime("%Y-%m-%dT%H"),
                latlon_bnds=bbox,
                parameters=parameters,
                data_type=data_type,
                mon_type=monitor_type,
            )
        except ValueError as exc:
            # ACT currently assumes several AirNow fields are numeric and can
            # fail when the "both" response (data_type="C") includes string
            # metadata in a position its parser interprets as numeric.
            #
            # The GPGL cube needs concentrations, not AQI categories, so give
            # the user a clear instruction instead of the lower-level NumPy
            # conversion traceback.
            if str(data_type).upper() == "C":
                raise ValueError(
                    "ACT could not parse the AirNow data_type='C' (AQI + "
                    "concentrations) response. For the GPGL cube use "
                    "data_type='B', which requests concentrations only and "
                    "avoids ACT's mixed-type parsing issue."
                ) from exc
            raise

class ACTAmeriFluxAdapter(BaseAdapter):
    name = "ameriflux"
    def __init__(self, region, user_id, user_email, output_dir="./ameriflux"):
        super().__init__(region)
        self.user_id = user_id
        self.user_email = user_email
        self.output_dir = Path(output_dir)
    def fetch(self, start, end, site_ids, data_product="FLUXNET",
              data_variant="FULLSET", data_policy=None, intended_use=None,
              description="GPGL regional data integration", **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return act.discovery.download_ameriflux_data(
            user_id=self.user_id,
            user_email=self.user_email,
            site_ids=site_ids,
            data_product=data_product,
            data_policy=data_policy,
            data_variant=data_variant,
            agree_policy=True,
            intended_use=intended_use,
            description=description,
            out_dir=str(self.output_dir),
        )

class ACTSURFRADAdapter(BaseAdapter):
    name = "surfrad"
    def __init__(self, region, output_dir="./surfrad"):
        super().__init__(region)
        self.output_dir = Path(output_dir)
    def fetch(self, start, end, sites=("bnd", "sxf"), **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for site in sites:
            files.extend(act.discovery.download_surfrad_data(
                site=site,
                startdate=_dt(start).strftime("%Y%m%d"),
                enddate=_dt(end).strftime("%Y%m%d"),
                output=str(self.output_dir),
            ))
        return files

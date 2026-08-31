from __future__ import annotations
import pandas as pd
import requests
from .base import BaseAdapter

class SondeHubTrajectoryAdapter(BaseAdapter):
    """SondeHub v2 radiosonde telemetry trajectories.

    Used primarily for GPS flight-path geometry. SondeHub atmospheric sensor
    values are not assumed calibrated.
    """
    name = "sondehub"
    def __init__(self, region, timeout=120, session=None, buffer_deg=2.0):
        super().__init__(region); self.timeout=timeout
        self.session=session or requests.Session(); self.buffer_deg=float(buffer_deg)
    def fetch(self, start=None, end=None, duration="1d", **kwargs):
        url='https://api.v2.sondehub.org/sondes/telemetry'
        params={'duration':duration}
        if end is not None:
            t=pd.Timestamp(end); t=t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
            params['datetime']=t.isoformat().replace('+00:00','Z')
        r=self.session.get(url, params=params, timeout=self.timeout); r.raise_for_status(); payload=r.json()
        rows=[]
        for serial, records in (payload or {}).items():
            iterable=list(records.values()) if isinstance(records,dict) else records if isinstance(records,list) else []
            for rec in iterable:
                if not isinstance(rec,dict): continue
                rows.append({
                    'serial':str(serial), 'time':rec.get('datetime',rec.get('time')),
                    'latitude':rec.get('lat',rec.get('latitude')), 'longitude':rec.get('lon',rec.get('longitude')),
                    'altitude_m':rec.get('alt',rec.get('altitude')),
                    'temperature_c_telemetry':rec.get('temp',rec.get('temperature')),
                    'humidity_pct_telemetry':rec.get('humidity',rec.get('rh')),
                    'pressure_hpa_telemetry':rec.get('pressure',rec.get('pressure_hpa')),
                    'sonde_type':rec.get('type',rec.get('sonde_type',''))})
        df=pd.DataFrame(rows)
        if df.empty: return df
        for c in ['latitude','longitude','altitude_m','temperature_c_telemetry','humidity_pct_telemetry','pressure_hpa_telemetry']:
            df[c]=pd.to_numeric(df[c],errors='coerce')
        df['time']=pd.to_datetime(df['time'],utc=True,errors='coerce'); df=df.dropna(subset=['time','latitude','longitude','altitude_m'])
        if start is not None:
            s=pd.Timestamp(start); s=s.tz_localize('UTC') if s.tzinfo is None else s.tz_convert('UTC'); df=df[df.time>=s]
        if end is not None:
            e=pd.Timestamp(end); e=e.tz_localize('UTC') if e.tzinfo is None else e.tz_convert('UTC'); df=df[df.time<=e]
        b=self.buffer_deg
        df=df[df.latitude.between(self.region.south-b,self.region.north+b)&df.longitude.between(self.region.west-b,self.region.east+b)]
        return df.sort_values(['serial','time']).reset_index(drop=True)

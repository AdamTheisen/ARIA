from __future__ import annotations
import numpy as np
import pandas as pd
import requests
from .base import BaseAdapter

class IEMRAOBAdapter(BaseAdapter):
    """NWS/upper-air profiles from IEM, with dynamic RAOB station discovery."""
    name='raob'
    def __init__(self, region, timeout=120, session=None, launch_buffer_deg=2.0):
        super().__init__(region); self.timeout=timeout; self.session=session or requests.Session(); self.launch_buffer_deg=float(launch_buffer_deg); self._station_metadata=None
    @staticmethod
    def latest_cycle(reference_time=None):
        t=pd.Timestamp.now(tz='UTC') if reference_time is None else pd.Timestamp(reference_time)
        t=t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC'); hour=12 if t.hour>=12 else 0
        return t.normalize()+pd.Timedelta(hours=hour)
    @staticmethod
    def prior_cycles(reference_time=None,count=5):
        first=IEMRAOBAdapter.latest_cycle(reference_time); return [first-pd.Timedelta(hours=12*i) for i in range(count)]
    def station_metadata(self, refresh=False):
        if self._station_metadata is not None and not refresh: return self._station_metadata.copy()
        r=self.session.get('https://mesonet.agron.iastate.edu/geojson/network.py',params={'network':'RAOB'},timeout=self.timeout); r.raise_for_status(); p=r.json(); rows=[]
        for f in p.get('features',[]):
            pr=f.get('properties',{}) or {}; g=f.get('geometry',{}) or {}; c=g.get('coordinates',[None,None]); sid=pr.get('sid') or pr.get('id') or pr.get('station')
            if sid is None: continue
            rows.append({'station_id':str(sid).upper(),'station_name':pr.get('sname',pr.get('name','')),'latitude':pd.to_numeric(c[1],errors='coerce'),'longitude':pd.to_numeric(c[0],errors='coerce'),'elevation_m':pd.to_numeric(pr.get('elevation'),errors='coerce')})
        df=pd.DataFrame(rows).dropna(subset=['latitude','longitude']); b=self.launch_buffer_deg
        if not df.empty: df=df[df.latitude.between(self.region.south-b,self.region.north+b)&df.longitude.between(self.region.west-b,self.region.east+b)].reset_index(drop=True)
        self._station_metadata=df; return df.copy()
    @staticmethod
    def _walk(obj,inherited=None):
        inherited=dict(inherited or {})
        if isinstance(obj,dict):
            ctx=inherited.copy()
            for k in ('station','station_id','sid','id','validUTC','valid','time','lat','lon','latitude','longitude'):
                if k in obj and not isinstance(obj[k],(dict,list)): ctx[k]=obj[k]
            level_keys={'pressure_mb','pressure','pres','height_m','height','hght','tmpc','temperature','dwpc','dewpoint','drct','speed_kts','wind_speed'}
            if level_keys.intersection(obj.keys()):
                row=ctx.copy(); row.update({k:v for k,v in obj.items() if not isinstance(v,(dict,list))}); yield row
            for v in obj.values():
                if isinstance(v,(dict,list)): yield from IEMRAOBAdapter._walk(v,ctx)
        elif isinstance(obj,list):
            for item in obj: yield from IEMRAOBAdapter._walk(item,inherited)
    @staticmethod
    def _first(row,*names):
        for n in names:
            if row.get(n) is not None: return row[n]
        return None
    def _request_cycle(self,cycle):
        r=self.session.get('https://mesonet.agron.iastate.edu/json/raob.py',params={'ts':cycle.strftime('%Y-%m-%dT%H:%M:%SZ')},timeout=self.timeout); r.raise_for_status(); return r.json()
    def fetch(self,start=None,end=None,cycle=None,fallback_cycles=5,**kwargs):
        meta=self.station_metadata(); lookup={str(r.station_id).upper():{'latitude':r.latitude,'longitude':r.longitude,'station_name':r.station_name,'elevation_m':r.elevation_m} for r in meta.itertuples()}
        cycles=[pd.Timestamp(cycle)] if cycle is not None else self.prior_cycles(end,count=fallback_cycles); diagnostics=[]
        for cyc in cycles:
            cyc=cyc.tz_localize('UTC') if cyc.tzinfo is None else cyc.tz_convert('UTC'); flat=list(self._walk(self._request_cycle(cyc))); diagnostics.append((cyc,len(flat))); rows=[]
            for item in flat:
                sid=self._first(item,'station','station_id','sid','id'); sid=str(sid).upper() if sid is not None else None; m=lookup.get(sid,{})
                lat=self._first(item,'lat','latitude'); lon=self._first(item,'lon','longitude'); lat=m.get('latitude') if lat is None else lat; lon=m.get('longitude') if lon is None else lon
                rows.append({'source':'NWS Radiosonde / IEM RAOB','station_id':sid,'station_name':m.get('station_name',''),'time':cyc,'latitude':lat,'longitude':lon,'launch_latitude':m.get('latitude'),'launch_longitude':m.get('longitude'),'pressure_hpa':self._first(item,'pressure_mb','pressure','pres'),'height_m':self._first(item,'height_m','height','hght'),'air_temperature_c':self._first(item,'tmpc','temperature_c','temperature'),'dew_point_temperature_c':self._first(item,'dwpc','dewpoint_c','dewpoint'),'wind_direction_deg':self._first(item,'drct','wind_direction'),'wind_speed_kt':self._first(item,'speed_kts','wind_speed')})
            df=pd.DataFrame(rows)
            if df.empty: continue
            for c in ['latitude','longitude','launch_latitude','launch_longitude','pressure_hpa','height_m','air_temperature_c','dew_point_temperature_c','wind_direction_deg','wind_speed_kt']: df[c]=pd.to_numeric(df[c],errors='coerce')
            df=df.dropna(subset=['latitude','longitude','height_m']); df=df[df.air_temperature_c.notna()|df.dew_point_temperature_c.notna()|df.wind_speed_kt.notna()]
            b=self.launch_buffer_deg; good=df.launch_latitude.between(self.region.south-b,self.region.north+b)&df.launch_longitude.between(self.region.west-b,self.region.east+b); df=df.loc[good].copy()
            if df.empty: continue
            d=np.deg2rad(df.wind_direction_deg); s=df.wind_speed_kt; df['u_wind_kt']=-s*np.sin(d); df['v_wind_kt']=-s*np.cos(d); df.attrs['cycle']=cyc; df.attrs['diagnostics']=diagnostics; df.attrs['station_count_metadata']=len(meta); return df.reset_index(drop=True)
        empty=pd.DataFrame(); empty.attrs['diagnostics']=diagnostics; empty.attrs['station_count_metadata']=len(meta); return empty

from __future__ import annotations
from pathlib import Path
import os
import xarray as xr
class CubeManager:
    """Small Icechunk/Zarr backend independent of Streamlit."""
    def __init__(self,root=None):
        self.root=Path(root or os.environ.get("ARIA_CUBE_ROOT", os.environ.get("ARIA_CUBE_ROOT", Path.home()/".cache"/"aria"/"icechunk"))).expanduser(); self.root.mkdir(parents=True,exist_ok=True)
    def _repo(self,name,create=False):
        try: import icechunk
        except ImportError as e: raise RuntimeError("Icechunk is not installed") from e
        path=self.root/name; path.mkdir(parents=True,exist_ok=True)
        storage=icechunk.local_filesystem_storage(str(path))
        try: return icechunk.Repository.open(storage)
        except Exception:
            if not create: raise
            return icechunk.Repository.create(storage)
    def init(self,name): return self._repo(name,True)
    def open(self,name,chunks="auto"):
        repo=self._repo(name); session=repo.readonly_session("main")
        return xr.open_zarr(session.store,chunks=chunks)
    def write(self,name,ds,message="ARIA cube update",append_dim="time"):
        repo=self._repo(name,True); session=repo.writable_session("main")
        try: existing=xr.open_zarr(session.store,chunks=None); has_data=bool(existing.data_vars)
        except Exception: has_data=False
        kwargs=dict(mode="a" if has_data else "w")
        if has_data and append_dim in ds.dims: kwargs["append_dim"]=append_dim
        ds.to_zarr(session.store,**kwargs); return session.commit(message)
    def info(self,name):
        ds=self.open(name,chunks=None); return dict(name=name,sizes=dict(ds.sizes),variables=list(ds.data_vars),attrs=dict(ds.attrs))

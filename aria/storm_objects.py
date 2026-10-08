"""Storm-object identification and verification using ARM-DOE ADAPT.

The ARIA deliberately delegates reflectivity segmentation to ADAPT's
``RadarCellSegmenter`` instead of maintaining a second storm-identification
algorithm.  The threshold detector in ADAPT operates on any 2-D gridded
reflectivity field, which lets the same detector be applied to HRRR simulated
reflectivity and MRMS observations on a common grid.

ADAPT is optional at runtime because it carries compiled radar dependencies.
Install ``arm-adapt`` in the environment running the Data Hub to enable these
functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment


def adapt_available() -> bool:
    """Return True when the ADAPT package can be imported."""
    try:
        from adapt.modules.detection.module import RadarCellSegmenter  # noqa: F401
        return True
    except Exception:
        try:
            from adapt.modules.detection import RadarCellSegmenter  # noqa: F401
            return True
        except Exception:
            return False


def adapt_version() -> str | None:
    """Best-effort installed ADAPT version."""
    try:
        import adapt
        return str(getattr(adapt, "__version__", "installed"))
    except Exception:
        return None


def _segmenter_class():
    try:
        from adapt.modules.detection.module import RadarCellSegmenter
        return RadarCellSegmenter
    except Exception as first:
        try:
            from adapt.modules.detection import RadarCellSegmenter
            return RadarCellSegmenter
        except Exception as second:
            raise RuntimeError(
                "ADAPT is not installed in the Python environment running ARIA. "
                "Install it with conda-forge (recommended): conda install conda-forge::arm-adapt"
            ) from second


def _as_yx_dataset(field: xr.DataArray, name: str = "reflectivity") -> xr.Dataset:
    """Convert an arbitrary 2-D DataArray to ADAPT's expected ``(y, x)`` form."""
    if field.ndim != 2:
        raise ValueError(f"Storm segmentation requires a 2-D field; got {field.dims}.")
    d0, d1 = field.dims
    da = field.astype(float)
    rename = {}
    if d0 != "y":
        rename[d0] = "y"
    if d1 != "x":
        rename[d1] = "x"
    if rename:
        da = da.rename(rename)
    # ADAPT only requires reflectivity + y/x coordinates for the threshold path.
    # Use integer coordinates if renaming latitude/longitude produced geographic
    # coordinate values that are not intended as Cartesian distances.
    da = xr.DataArray(
        da.values,
        dims=("y", "x"),
        coords={"y": np.arange(da.shape[0]), "x": np.arange(da.shape[1])},
        attrs=dict(field.attrs),
        name=name,
    )
    return xr.Dataset({name: da})


def segment_with_adapt(
    field: xr.DataArray,
    *,
    threshold_dbz: float = 35.0,
    min_gridpoints: int = 8,
    h_maxima_dbz: float = 5.0,
    closing_kernel: tuple[int, int] = (1, 1),
) -> xr.DataArray:
    """Identify convective objects using ADAPT's official RadarCellSegmenter.

    The threshold method is intentionally used here because HRRR and MRMS are
    already 2-D composite-reflectivity fields.  It also keeps the segmentation
    method identical on the model and observation sides.
    """
    RadarCellSegmenter = _segmenter_class()
    config = SimpleNamespace(
        method="threshold",
        method_params={"threshold": float(threshold_dbz)},
        closing_kernel=tuple(int(v) for v in closing_kernel),
        filter_by_size=True,
        min_cellsize_gridpoint=max(1, int(min_gridpoints)),
        max_cellsize_gridpoint=None,
        h_maxima=float(h_maxima_dbz),
        reflectivity_var="reflectivity",
        labels_var="cell_labels",
        z_level=0.0,
    )
    ds = _as_yx_dataset(field)
    out = RadarCellSegmenter(config).segment(ds)
    labels = out["cell_labels"].astype(np.int32)
    labels.attrs.update(
        {
            "detector": "ARM-DOE ADAPT RadarCellSegmenter",
            "threshold_dbz": float(threshold_dbz),
            "min_gridpoints": int(min_gridpoints),
            "h_maxima_dbz": float(h_maxima_dbz),
        }
    )
    return labels


def _latlon_mesh(comparison: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "latitude" not in comparison.coords or "longitude" not in comparison.coords:
        raise ValueError("Comparison dataset does not contain latitude/longitude coordinates.")
    lat = np.asarray(comparison.latitude.values, dtype=float)
    lon = np.asarray(comparison.longitude.values, dtype=float)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2, lat2 = np.meshgrid(lon, lat)
        return lat2, lon2
    if lat.shape != lon.shape:
        raise ValueError("Latitude and longitude coordinates are not aligned.")
    return lat, lon


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def _grid_area_km2(lat: np.ndarray, lon: np.ndarray) -> float:
    """Estimate representative horizontal grid-cell area."""
    dx = []
    dy = []
    if lat.shape[1] > 1:
        vals = _haversine_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        dx = vals[np.isfinite(vals)].ravel()
    if lat.shape[0] > 1:
        vals = _haversine_km(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
        dy = vals[np.isfinite(vals)].ravel()
    dxm = float(np.nanmedian(dx)) if len(dx) else 1.0
    dym = float(np.nanmedian(dy)) if len(dy) else 1.0
    return max(dxm * dym, 1e-6)


def summarize_objects(
    labels: xr.DataArray,
    reflectivity: xr.DataArray,
    comparison: xr.Dataset | None = None,
    *,
    prefix: str = "T",
    latitude=None,
    longitude=None,
) -> pd.DataFrame:
    """Summarize ADAPT labels for comparison grids or a single radar sweep."""
    lab = np.asarray(labels.values, dtype=int)
    refl = np.asarray(reflectivity.values, dtype=float)
    if latitude is not None and longitude is not None:
        lat = np.asarray(latitude, dtype=float); lon = np.asarray(longitude, dtype=float)
    elif comparison is not None:
        lat, lon = _latlon_mesh(comparison)
    else:
        raise ValueError("Provide either comparison or explicit latitude/longitude arrays.")
    if lab.shape != refl.shape or lab.shape != lat.shape:
        raise ValueError("Labels, reflectivity, and latitude/longitude must share a shape.")
    cell_area = _grid_area_km2(lat, lon)
    rows = []
    for ident in np.unique(lab):
        if ident <= 0:
            continue
        mask = lab == ident
        finite_geo = mask & np.isfinite(lat) & np.isfinite(lon)
        if not finite_geo.any():
            continue
        vals = refl[mask & np.isfinite(refl)]
        rows.append(
            {
                "object_id": f"{prefix}{int(ident)}",
                "adapt_label": int(ident),
                "centroid_latitude": float(np.nanmean(lat[finite_geo])),
                "centroid_longitude": float(np.nanmean(lon[finite_geo])),
                "latitude_min": float(np.nanmin(lat[finite_geo])),
                "latitude_max": float(np.nanmax(lat[finite_geo])),
                "longitude_min": float(np.nanmin(lon[finite_geo])),
                "longitude_max": float(np.nanmax(lon[finite_geo])),
                "gridpoints": int(mask.sum()),
                "area_km2": float(mask.sum() * cell_area),
                "max_reflectivity_dbz": float(np.nanmax(vals)) if vals.size else np.nan,
                "mean_reflectivity_dbz": float(np.nanmean(vals)) if vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bearing_deg(lat1, lon1, lat2, lon2):
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0


def match_objects(
    model_objects: pd.DataFrame,
    observed_objects: pd.DataFrame,
    *,
    max_distance_km: float = 100.0,
) -> pd.DataFrame:
    """One-to-one centroid matching using the Hungarian assignment algorithm."""
    columns = [
        "model_object", "observed_object", "displacement_km", "bearing_deg",
        "model_area_km2", "observed_area_km2", "area_bias_km2",
        "model_max_dbz", "observed_max_dbz", "max_dbz_bias",
    ]
    if model_objects.empty or observed_objects.empty:
        return pd.DataFrame(columns=columns)
    mlat = model_objects.centroid_latitude.to_numpy(float)[:, None]
    mlon = model_objects.centroid_longitude.to_numpy(float)[:, None]
    olat = observed_objects.centroid_latitude.to_numpy(float)[None, :]
    olon = observed_objects.centroid_longitude.to_numpy(float)[None, :]
    distances = _haversine_km(mlat, mlon, olat, olon)
    ri, ci = linear_sum_assignment(distances)
    rows = []
    for i, j in zip(ri, ci):
        d = float(distances[i, j])
        if not np.isfinite(d) or d > float(max_distance_km):
            continue
        m = model_objects.iloc[int(i)]
        o = observed_objects.iloc[int(j)]
        rows.append(
            {
                "model_object": m.object_id,
                "observed_object": o.object_id,
                "displacement_km": d,
                "bearing_deg": float(_bearing_deg(
                    o.centroid_latitude, o.centroid_longitude,
                    m.centroid_latitude, m.centroid_longitude,
                )),
                "model_area_km2": float(m.area_km2),
                "observed_area_km2": float(o.area_km2),
                "area_bias_km2": float(m.area_km2 - o.area_km2),
                "model_max_dbz": float(m.max_reflectivity_dbz),
                "observed_max_dbz": float(o.max_reflectivity_dbz),
                "max_dbz_bias": float(m.max_reflectivity_dbz - o.max_reflectivity_dbz),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def object_verification_metrics(
    model_objects: pd.DataFrame,
    observed_objects: pd.DataFrame,
    matches: pd.DataFrame,
) -> dict:
    nm = int(len(model_objects))
    no = int(len(observed_objects))
    nh = int(len(matches))
    misses = max(0, no - nh)
    false_alarms = max(0, nm - nh)
    pod = nh / (nh + misses) if nh + misses else np.nan
    far = false_alarms / (nh + false_alarms) if nh + false_alarms else np.nan
    csi = nh / (nh + misses + false_alarms) if nh + misses + false_alarms else np.nan
    return {
        "model_objects": nm,
        "observed_objects": no,
        "matched_objects": nh,
        "missed_objects": misses,
        "false_objects": false_alarms,
        "object_POD": float(pod) if np.isfinite(pod) else np.nan,
        "object_FAR": float(far) if np.isfinite(far) else np.nan,
        "object_CSI": float(csi) if np.isfinite(csi) else np.nan,
        "mean_displacement_km": float(matches.displacement_km.mean()) if nh else np.nan,
        "median_displacement_km": float(matches.displacement_km.median()) if nh else np.nan,
        "mean_area_bias_km2": float(matches.area_bias_km2.mean()) if nh else np.nan,
        "mean_max_dbz_bias": float(matches.max_dbz_bias.mean()) if nh else np.nan,
    }


def build_adapt_object_comparison(
    comparison: xr.Dataset,
    *,
    threshold_dbz: float = 35.0,
    min_gridpoints: int = 8,
    h_maxima_dbz: float = 5.0,
    max_match_distance_km: float = 100.0,
):
    """Segment HRRR and MRMS with ADAPT, then match objects at one valid time."""
    model_labels = segment_with_adapt(
        comparison.model,
        threshold_dbz=threshold_dbz,
        min_gridpoints=min_gridpoints,
        h_maxima_dbz=h_maxima_dbz,
    )
    observed_labels = segment_with_adapt(
        comparison.observation,
        threshold_dbz=threshold_dbz,
        min_gridpoints=min_gridpoints,
        h_maxima_dbz=h_maxima_dbz,
    )
    model_objects = summarize_objects(model_labels, comparison.model, comparison, prefix="M")
    observed_objects = summarize_objects(observed_labels, comparison.observation, comparison, prefix="O")
    matches = match_objects(
        model_objects,
        observed_objects,
        max_distance_km=max_match_distance_km,
    )
    metrics = object_verification_metrics(model_objects, observed_objects, matches)
    return model_labels, observed_labels, model_objects, observed_objects, matches, metrics


def extract_object_boundaries(labels, x=None, y=None, simplify_stride=2):
    """Return cleaned display polygons for positive object labels.

    Quantitative object masks are not modified.  The display outline receives a
    light binary closing/opening pass and contour extraction so one-cell spikes
    and angular point-order artifacts do not dominate the visualization.
    ``x``/``y`` may be either 1-D grid axes or 2-D geographic coordinates.
    """
    import numpy as np
    from scipy import ndimage
    import matplotlib.pyplot as plt

    lab=np.asarray(labels.values if hasattr(labels,"values") else labels)
    if lab.ndim!=2:
        raise ValueError("Object-boundary extraction requires a 2-D label mask.")

    if x is None:
        x=np.arange(lab.shape[1],dtype=float)
    else:
        x=np.asarray(x,float)
    if y is None:
        y=np.arange(lab.shape[0],dtype=float)
    else:
        y=np.asarray(y,float)

    out={}
    structure=np.ones((3,3),dtype=bool)
    for ident in np.unique(lab[np.isfinite(lab)]):
        if ident<=0:
            continue
        raw=(lab==ident)
        if not raw.any():
            continue

        # Display-only cleanup. Preserve the original mask for all metrics.
        cleaned=ndimage.binary_closing(raw,structure=structure,iterations=1)
        cleaned=ndimage.binary_opening(cleaned,structure=structure,iterations=1)
        # Avoid erasing genuinely tiny objects.
        if cleaned.sum()<max(4,int(raw.sum()*0.35)):
            cleaned=raw

        # Matplotlib contour provides an ordered boundary and avoids the former
        # angle-around-centroid ordering that produced long crossing spikes.
        fig,ax=plt.subplots(figsize=(1,1))
        try:
            cs=ax.contour(cleaned.astype(float),levels=[0.5])
            paths=[
                np.asarray(seg,float)
                for seg in (cs.allsegs[0] if cs.allsegs else [])
                if len(seg)>=4
            ]
        finally:
            plt.close(fig)
        if not paths:
            continue

        # Keep the longest external path for this object.
        verts=max(paths,key=len)
        col=verts[:,0]
        row=verts[:,1]
        step=max(1,int(simplify_stride))
        row=row[::step]; col=col[::step]

        if x.ndim==1 and y.ndim==1:
            xx=np.interp(col,np.arange(len(x)),x)
            yy=np.interp(row,np.arange(len(y)),y)
        elif x.ndim==2 and y.ndim==2:
            xx=ndimage.map_coordinates(x,[row,col],order=1,mode="nearest")
            yy=ndimage.map_coordinates(y,[row,col],order=1,mode="nearest")
        else:
            raise ValueError("x and y must both be 1-D axes or both be 2-D coordinate grids.")

        coords=np.column_stack([xx,yy])
        if len(coords):
            coords=np.vstack([coords,coords[0]])
            out[int(ident)]=coords
    return out

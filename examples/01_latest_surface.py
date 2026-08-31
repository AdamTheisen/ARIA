from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aria import GPGL_REGION
from aria.plotting import (
    plot_coverage,
    plot_surface_analysis,
)
from aria.workflows import build_latest_surface


output = Path("output")
surface_dir = output / "surface"
diagnostics_dir = output / "diagnostics"
cube_dir = output / "cube"

for directory in (surface_dir, diagnostics_dir, cube_dir):
    directory.mkdir(parents=True, exist_ok=True)

ds, observations, start, end = build_latest_surface()

finite = ds["air_temperature_f"].values
finite = finite[np.isfinite(finite)]

vmin = np.floor(np.nanpercentile(finite, 2) / 5) * 5
vmax = np.ceil(np.nanpercentile(finite, 98) / 5) * 5

fig = plot_surface_analysis(
    ds,
    ds.sizes["time"] - 1,
    GPGL_REGION,
    output_path=surface_dir / "latest.png",
    temp_vmin=vmin,
    temp_vmax=vmax,
)
plt.close(fig)

plot_coverage(
    ds,
    ds.sizes["time"] - 1,
    GPGL_REGION,
    output_dir=diagnostics_dir,
)

ds.to_zarr(
    cube_dir / "surface.zarr",
    mode="w",
    consolidated=True,
)

print(f"Latest surface analysis written to {surface_dir / 'latest.png'}")

from pathlib import Path

import matplotlib.pyplot as plt

from aria import GPGL_REGION
from aria.plotting import plot_regional_reflectivity
from aria.workflows import build_latest_regional_radar


output = Path("output")
radar_dir = output / "radar"
cube_dir = output / "cube"

radar_dir.mkdir(parents=True, exist_ok=True)
cube_dir.mkdir(parents=True, exist_ok=True)

cube, scans, diagnostics = build_latest_regional_radar()

print(f"Regional grid uses {len(scans)} radars:")
for scan in scans:
    print(
        f"  {scan.radar_id}: "
        f"{scan.scan_time:%Y-%m-%d %H:%M:%S} UTC"
    )

cube.to_zarr(
    cube_dir / "regional_radar.zarr",
    mode="w",
    consolidated=True,
)

for altitude in (1.0, 3.0, 5.0, 8.0):
    fig = plot_regional_reflectivity(
        cube,
        GPGL_REGION,
        altitude_km=altitude,
        output_path=(
            radar_dir
            / f"regional_reflectivity_{altitude:g}km.png"
        ),
    )
    plt.close(fig)

print("Regional radar cube written to output/cube/regional_radar.zarr")

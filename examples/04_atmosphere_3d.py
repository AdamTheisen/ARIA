from pathlib import Path

from aria import GPGL_REGION
from aria.plotting import make_3d_temperature_figure
from aria.workflows import build_latest_atmosphere


output_dir = Path("output") / "atmosphere"
cube_dir = Path("output") / "cube"
output_dir.mkdir(parents=True, exist_ok=True)
cube_dir.mkdir(parents=True, exist_ok=True)

atmosphere, profiles, trajectories, points = build_latest_atmosphere()

fig = make_3d_temperature_figure(
    atmosphere,
    GPGL_REGION,
    profiles=profiles,
    trajectories=trajectories,
    show_volume=False,
    show_isosurfaces=True,
    show_level_surfaces=True,
)

html = output_dir / "atmosphere_3d.html"
fig.write_html(html, include_plotlyjs="cdn")

atmosphere.to_zarr(
    cube_dir / "atmosphere.zarr",
    mode="w",
    consolidated=True,
)

print(
    f"RAOB profiles: "
    f"{profiles['station_id'].nunique() if not profiles.empty else 0}"
)
print(
    f"SondeHub GPS tracks: "
    f"{trajectories['serial'].nunique() if not trajectories.empty else 0}"
)
print(f"3-D atmosphere written to {html}")

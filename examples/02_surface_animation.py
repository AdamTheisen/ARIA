from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from aria import GPGL_REGION
from aria.plotting import plot_surface_analysis
from aria.workflows import build_latest_surface


output = Path("output")
surface_dir = output / "surface"
frames_dir = surface_dir / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)

ds, observations, start, end = build_latest_surface()

finite = ds["air_temperature_f"].values
finite = finite[np.isfinite(finite)]

vmin = np.floor(np.nanpercentile(finite, 2) / 5) * 5
vmax = np.ceil(np.nanpercentile(finite, 98) / 5) * 5

frame_paths = []

for time_index in range(ds.sizes["time"]):
    timestamp = str(ds["time"].values[time_index])[:16].replace(":", "")
    path = frames_dir / f"surface_{timestamp}.png"

    fig = plot_surface_analysis(
        ds,
        time_index,
        GPGL_REGION,
        output_path=path,
        temp_vmin=vmin,
        temp_vmax=vmax,
    )
    plt.close(fig)
    frame_paths.append(path)

images = [
    Image.open(path).convert("P", palette=Image.ADAPTIVE)
    for path in frame_paths
]

if images:
    gif_path = surface_dir / "animation.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=650,
        loop=0,
    )
    print(f"Animation written to {gif_path}")

for image in images:
    image.close()

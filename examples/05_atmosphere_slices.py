from pathlib import Path

import matplotlib.pyplot as plt

from aria import GPGL_REGION
from aria.plotting import plot_temperature_slice
from aria.workflows import build_latest_atmosphere


output_dir = Path("output") / "atmosphere" / "levels"
output_dir.mkdir(parents=True, exist_ok=True)

atmosphere, profiles, trajectories, points = build_latest_atmosphere()

for altitude in atmosphere["altitude_km"].values:
    altitude = float(altitude)

    fig = plot_temperature_slice(
        atmosphere,
        altitude,
        GPGL_REGION,
        output_path=(
            output_dir / f"temperature_{altitude:g}km.png"
        ),
    )
    plt.close(fig)

print(f"Atmospheric temperature slices written to {output_dir}")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from aria.workflows import build_latest_atmosphere
from aria.trajectory_match import haversine_km


output_dir = Path("output") / "atmosphere"
output_dir.mkdir(parents=True, exist_ok=True)

atmosphere, profiles, trajectories, points = build_latest_atmosphere()

lon = atmosphere["longitude"].values
lat = atmosphere["latitude"].values
altitude = atmosphere["altitude_km"].values
temperature = atmosphere["air_temperature"].values

# Default transect: western Nebraska toward central Wisconsin.
start_lon, start_lat = -103.5, 41.5
end_lon, end_lat = -88.5, 44.5

n = 160
transect_lon = np.linspace(start_lon, end_lon, n)
transect_lat = np.linspace(start_lat, end_lat, n)

distance = np.array(
    [
        haversine_km(start_lat, start_lon, la, lo)
        for la, lo in zip(transect_lat, transect_lon)
    ]
)

interpolator = RegularGridInterpolator(
    (altitude, lat, lon),
    temperature,
    bounds_error=False,
    fill_value=np.nan,
)

section = np.full((len(altitude), n), np.nan)

for zi, z in enumerate(altitude):
    query = np.column_stack(
        [
            np.full(n, z),
            transect_lat,
            transect_lon,
        ]
    )
    section[zi] = interpolator(query)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

mesh = ax.pcolormesh(
    distance,
    altitude,
    section,
    cmap="coolwarm",
    shading="auto",
)

if np.isfinite(section).any():
    contours = ax.contour(
        distance,
        altitude,
        section,
        colors="black",
        linewidths=0.6,
    )
    ax.clabel(contours, fontsize=7)

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Air Temperature (°C)")

ax.set_xlabel("Distance along transect (km)")
ax.set_ylabel("Altitude (km MSL)")
ax.set_title(
    "GPGL Atmospheric Temperature Cross Section\n"
    "Western Nebraska → Wisconsin"
)

path = output_dir / "temperature_cross_section.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Cross section written to {path}")

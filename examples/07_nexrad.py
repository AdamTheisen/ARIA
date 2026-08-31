from pathlib import Path

import matplotlib.pyplot as plt

from aria.plotting import plot_nexrad_ppi
from aria.workflows import load_latest_nexrad


RADAR_ID = "KMPX"

output_dir = Path("output") / "radar"
output_dir.mkdir(parents=True, exist_ok=True)

radar, scan = load_latest_nexrad(RADAR_ID)

print(
    f"Loaded {scan.radar_id} ({scan.site_name}) "
    f"at {scan.scan_time:%Y-%m-%d %H:%M:%S} UTC"
)

path = output_dir / f"{RADAR_ID.lower()}_latest_reflectivity.png"

fig = plot_nexrad_ppi(
    radar,
    sweep=0,
    output_path=path,
)
plt.close(fig)

print(f"Radar image written to {path}")

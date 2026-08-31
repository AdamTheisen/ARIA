from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _reflectivity_field(radar):
    candidates = [
        "reflectivity",
        "corrected_reflectivity",
        "reflectivity_horizontal",
    ]

    for name in candidates:
        if name in radar.fields:
            return name

    raise RuntimeError(
        "No reflectivity field was found in this NEXRAD volume. "
        f"Available fields: {list(radar.fields)}"
    )


def plot_nexrad_ppi(
    radar,
    *,
    sweep=0,
    output_path=None,
    max_range_km=230,
    vmin=-10,
    vmax=70,
    cmap="pyart_NWSRef",
):
    """
    Plot one NEXRAD reflectivity PPI sweep using Py-ART.
    """
    try:
        import pyart
    except ImportError as exc:
        raise RuntimeError(
            "Radar plotting requires ARM Py-ART."
        ) from exc

    field = _reflectivity_field(radar)

    sweep = int(
        np.clip(
            sweep,
            0,
            radar.nsweeps - 1,
        )
    )

    display = pyart.graph.RadarMapDisplay(radar)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)

    display.plot_ppi_map(
        field,
        sweep=sweep,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        min_lon=float(radar.longitude["data"][0]) - 3.5,
        max_lon=float(radar.longitude["data"][0]) + 3.5,
        min_lat=float(radar.latitude["data"][0]) - 2.7,
        max_lat=float(radar.latitude["data"][0]) + 2.7,
        resolution="50m",
        colorbar_label="Reflectivity (dBZ)",
        title_flag=False,
    )

    fixed_angle = float(
        radar.fixed_angle["data"][sweep]
    )

    radar_name = (
        radar.metadata.get("instrument_name")
        or "NEXRAD"
    )

    ax.set_title(
        f"{radar_name} Reflectivity\n"
        f"Sweep {sweep} • {fixed_angle:.1f}° elevation"
    )

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

    return fig


def radar_sweep_summary(radar):
    """
    Return a small table describing available elevation sweeps.
    """
    import pandas as pd

    rows = []

    for sweep in range(radar.nsweeps):
        start = int(
            radar.sweep_start_ray_index["data"][sweep]
        )
        end = int(
            radar.sweep_end_ray_index["data"][sweep]
        )

        rows.append(
            {
                "sweep": sweep,
                "elevation_deg": float(
                    radar.fixed_angle["data"][sweep]
                ),
                "rays": end - start + 1,
            }
        )

    return pd.DataFrame(rows)

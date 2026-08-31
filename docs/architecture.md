# ARIA Architecture

ARIA — Atmospheric Regional Integration and Analysis — separates geographic
domain, data acquisition, atmospheric processing, storage, analysis, and
visualization.

## Core flow

    Region
      ↓
    Source adapters
      ↓
    Normalize + quality control
      ↓
    Lazy xarray / Dask processing
      ↓
    Analysis-ready Icechunk/Zarr repositories
      ↓
    Workflows and verification
      ↓
    Dashboard / Python / CLI

`Region` is a first-class object. GPGL is a preset, not a hard-coded processing
domain. Custom rectangular domains can be selected in the dashboard, and future
recipe execution can use the same region definition without Streamlit.

## Source layout

ARIA v0.16.2 uses `aria/` as the actual Python source package. The earlier
`gpgl_data_hub/` compatibility layout has been removed before the first GitHub
release so the repository starts with the intended long-term project structure.

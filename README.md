# ARIA

**Atmospheric Regional Integration and Analysis**

ARIA is a Python framework and interactive dashboard for assembling, harmonizing,
analyzing, comparing, and visualizing regional atmospheric observations and model
data. The original Great Plains / Great Lakes (GPGL) domain is now one built-in
region rather than the identity of the software.


## AI-assisted development disclosure

ARIA has been developed with substantial assistance from **OpenAI ChatGPT**.
The current development work for this release was performed using **GPT-5.6 Sol**
in ChatGPT.

ChatGPT has been used for software architecture, code generation and refactoring,
debugging, documentation, workflow design, and test scaffolding. Human review,
scientific validation, operational testing, and acceptance remain the
responsibility of the ARIA maintainers. AI-generated code should be reviewed and
tested in the same manner as any other contributed code, especially for
scientific calculations, quality control, data provenance, and operational use.

This disclosure should be updated if materially different AI systems or model
versions are used for future development.

## What ARIA does

ARIA integrates surface observations, air quality, radiosondes, NEXRAD,
MRMS, HRRR, ARM data sources, and model output into common regional workflows.
It supports real-time exploration, model/observation comparison, verification,
2-D and 3-D visualization, and analysis-ready data cubes.

## Architecture

    Define region
         |
    Select sources
         |
    Acquire + QC + harmonize
         |
    xarray + Dask processing
         |
    Icechunk / Zarr cubes
         |
    Analysis + verification
         |
    2-D / 3-D exploration

## Install

Create a Python 3.13 environment when possible, then install the repository:

    python -m pip install -e .

Run the dashboard:

    streamlit run aria/dashboard/app.py

The distribution name is `aria-atmos`, the Python package is `aria`, and the command-line interface is `aria`.

## Regions

Built-in presets currently include:

- GPGL
- ARM SGP
- Upper Midwest
- CONUS
- Custom bounding box through the dashboard

List regions:

    aria region list

Inspect a region:

    aria region show GPGL

ARIA's `Region` object is now the authoritative domain passed to data adapters,
gridding, model subsetting, visualization, and persistent-cache keys.

## Cube recipes

ARIA introduces YAML recipes to describe a reproducible regional cube. See
`recipes/gpgl.yaml`.

Validate a recipe:

    aria recipe recipes/gpgl.yaml

The recipe layer is the foundation for a generic regional cube generator. In
v0.16 it defines and validates regions, sources, storage, and compute settings;
source-by-source historical execution will continue to expand.

## Dask

ARIA now includes Dask and Distributed as first-class dependencies. Icechunk
and Zarr datasets can be opened lazily, and `aria.compute` provides
initial chunk recommendations for surface, MRMS, HRRR, atmospheric, and radar
datasets. Station analysis remains optimized with SciPy/KD-tree methods rather
than forcing small station tables through Dask.

## v0.16.0

- Rebranded the project and dashboard as ARIA.
- Added generic region presets and custom dashboard bounds.
- Scoped persistent processed caches by region.
- Added generic NEXRAD site discovery using Py-ART metadata when available.
- Retained detailed 3-D radar in both single-radar and Regional Radar views.
- Fixed empty 3-D radar scenes so state outlines cannot expand the axes to the US.
- Changed AirNow monitor-support default to 300 km.
- Added automatic latest-radiosonde / preceding-HRRR time matching.
- Added ARIA YAML recipe support.
- Added Dask/Distributed compute infrastructure.
- Added the `aria` CLI while retaining the legacy `gpgl` alias.
- Migrated the source package itself to `aria/`.

## Runtime notes

The code is syntax-checked as part of release assembly. Live network access to
HRRR, MRMS, AirNow, IEM, NEXRAD, Icechunk, Py-ART, and Streamlit is environment
dependent and should be exercised after installation.


## v0.16.1

Packaging fix for editable installs. Setuptools package discovery is now explicit:
only `aria*` and `aria_atmos*` are packaged, while `recipes`, `docs`,
`examples`, and `tests` remain repository content and are excluded from package
discovery.


## v0.16.2

- Moved the actual source tree from `gpgl_data_hub/` to `aria/`.
- Updated imports, examples, CLI entry points, tests, and documentation to use `aria`.
- Fixed an import-time `NameError` in `build_latest_regional_radar_fast` where the
  default argument incorrectly referenced `region` before it existed.
- Removed the transitional `aria_atmos` wrapper and legacy `gpgl` CLI alias.


## v0.16.3

Surface-analysis performance release.

- Barnes/Gaussian interpolation now limits each grid point to the nearest 64
  supported stations instead of querying every station in the region.
- `cKDTree.query` uses SciPy worker parallelism and a hard distance upper bound.
- Analysis-grid Cartesian coordinates are cached once per builder.
- Barnes residual sampling is vectorized.
- The Surface and Coverage views now compute only the latest 5-minute analysis
  by default. `Load previous hour` restores the full 5-minute timeline when needed.
- Surface and Air Quality dashboard builders explicitly receive the active ARIA region.


## v0.16.4

- Added center + radius region definitions in addition to rectangular bounds.
- Expanded ARM SGP to a 400 km site-centered default domain.
- Added Custom region controls for center/radius or explicit bounding box.
- Fixed cross-region Streamlit caching by making the region cache key an explicit
  argument to cached loader functions and clearing in-memory resources on region changes.
- Added explicit AI-assisted development provenance documenting use of OpenAI
  ChatGPT GPT-5.6 Sol.

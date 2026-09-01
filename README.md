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


## v0.16.5

- Fixed `Region.from_center_radius()` being accidentally defined at module scope
  instead of as a `Region` class method.
- Restored `Region.as_dict()` as a normal class method; it had also been nested
  incorrectly beneath the misplaced center/radius function.
- Added an import/runtime sanity check for the SGP preset.


## v0.16.6

- Fixed circular/lobed artifacts in Barnes/Gaussian analyses introduced by the
  v0.16.3 bounded-neighbor optimization.
- The nearest 64 stations are still used for performance, but Gaussian/Barnes
  weights are now evaluated continuously; the support radius is applied to the
  nearest-observation distance rather than independently truncating every neighbor.
- Storm Explorer surface comparisons now use a 35-minute past-only observation
  window to better capture routine METAR reports near the model valid time.


## v0.16.7

- Tightened Storm Explorer surface-observation matching to a **15-minute,
  past-only** window ending at the HRRR valid time.
- Restored the intended Model | Observation | Difference comparison layout:
  Model and Observation share one **horizontal colorbar below the first two
  panels**, while Difference has a separate **horizontal symmetric colorbar
  below the third panel**.
- Increased interactive comparison bottom spacing so colorbars do not overlap
  map panels, axes, or each other.

## v0.17.0

- Added a **Region Configuration** start screen. ARIA now opens by defining or
  selecting the analysis region before data workflows are launched.
- Added persistent user-defined regions with **Save** and **Delete** controls;
  built-in ARIA regions remain read-only.
- Added regional observing-system inventory statistics and map preview for
  recently reporting surface stations, operational NEXRAD radars, radiosonde
  sites, and inferred state-query coverage.
- Added expandable station/radar/sonde inventories to make the source coverage
  behind a region visible before analysis begins.
- Added a **Change region** action after launch while retaining explicit
  region-scoped cache behavior.
- Fixed the interactive Surface colormap control so changing the selected
  colormap updates the Plotly map rather than leaving it hard-coded to RdBu_r.
- Radiosonde/HRRR automatic comparison now displays sonde launch time, HRRR
  initialization, forecast lead, HRRR valid time, and model/sonde time offset.
- Retains the v0.16.7 Storm Explorer 15-minute past-only observation window and
  horizontal comparison colorbars.

The planned ~3-km physical surface grid is intentionally not enabled in this
release. At regional scale it needs chunked Barnes evaluation to avoid a large
nearest-neighbor working array; the current 0.1-degree surface grid remains the
safe default until that performance work is completed.


## v0.17.1

- Renamed **Storm Explorer** to **Model Evaluation**.
- Surface model/observation matching now defaults to the nearest station report
  within **±30 minutes** of HRRR valid time, with selectable 10–60 minute
  offsets and an optional past-only mode.
- Replaced radar-specific navigation wording with neutral observation-target
  terminology so Surface Evaluation no longer shows "Follow Radar" or "Radar
  time offset" controls.
- Added observation timing diagnostics to Surface Evaluation.
- Added optional **ADAPT storm-cell detection** to the single-NEXRAD PPI viewer,
  including object outlines, IDs, thresholds, minimum size, h-maxima controls,
  and an object inventory.

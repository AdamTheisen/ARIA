# ARIA

**Atmospheric Regional Integration and Analysis**

ARIA is a Python framework and interactive dashboard for assembling, harmonizing,
analyzing, comparing, and visualizing regional atmospheric observations and model
data. The original Great Plains / Great Lakes (GPGL) domain is now one built-in
region rather than the identity of the software.


## v0.19.1

Storage and dashboard reliability patch following the first v0.19 collector tests:

- fixes NetCDF boolean-attribute serialization for MRMS and Air Quality snapshots
- sanitizes pandas/NumPy timestamps and DataFrame metadata before Parquet writes, fixing radiosonde storage
- adds automatic NOAA GLERL ACSPO GLSEA retrieval through ERDDAP for Great Lakes regions
- moves Data Storage out of scientific Views and into a Configuration workspace
- clarifies source check interval versus native product cadence and refreshes the storage monitor every 15 seconds
- keeps generated Model Evaluation surface and radar/HRRR comparisons visible across Streamlit reruns until explicitly regenerated
- begins local-first dashboard reads from current persistent Surface, Air Quality, and MRMS snapshots, with source fallback when needed
- shows persistent-store provenance in current-data status lines
- fixes Model Evaluation surface matching controls so the selected observation offset and nearest/past mode are actually passed to the workflow

The persistent snapshot format remains a testing-stage interface; backfill, marine buoy ingestion, ADAPT track persistence, and the final ARCO/Icechunk schema are not yet part of this patch.


## v0.19.0 — Persistent Storage Manager (testing release)

ARIA v0.19 introduces a persistent regional storage manager that is independent of Streamlit. The dashboard can enable/pause collection, select sources and cadences, monitor source status and recent activity, select the active storage region, and manually run due/forced updates. The new `aria storage update` CLI is designed to be invoked by cron, launchd, systemd, or a future Kubernetes scheduler so collection continues after the dashboard closes. Source-level locks prevent overlapping scheduled updates, while timestamped NetCDF/Parquet snapshots are stored separately from ARIA's disposable processed cache.

Initial collectors are available for surface analyses/observations, MRMS, HRRR F00 fields, AirNow, radiosondes, and SST. This is a testing architecture: v0.19 uses deduplicated snapshots rather than the final appendable ARCO/Icechunk layout, and historical backfill is not yet implemented. Ocean SST now automatically retrieves a region subset of NOAA CoastWatch's diurnally corrected Geo-Polar Blended L4 product; GLSEA automatic retrieval remains pending. The obsolete `add_plotly_cities`/`city_dataframe` import from v0.18 has also been removed.


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

## v0.18.0
- Editable saved regions and built-in-region duplication.
- Major-city context on Region Configuration map.
- NEXRAD city/state display labels.
- Fixed single-radar ADAPT summaries using native gate coordinates.
- Added HRRR 1-km AGL reflectivity to Model Evaluation radar comparison.
- Initial NOAA SST architecture: GLSEA for Great Lakes, Geo-Polar Blended SST L4 for ocean regions. NOAA endpoints are configurable rather than hard-coded.


## v0.20.0

ARIA v0.20 moves persistent regional scientific storage to **Icechunk-backed
Zarr v3** repositories. Core Xarray datasets append along time in transactional
Icechunk commits; naturally tabular observations and future object catalogs
remain Parquet. v0.19 NetCDF snapshots can be migrated non-destructively with:

```bash
aria storage migrate
```

The regional store is organized by scientific role:

```text
<region>/
  observations/
    surface/
    air_quality/
    radiosonde/
  analyses/
    mrms/
    sst/
  models/
    hrrr/
  derived/
```

Reflectivity interpolation is also corrected in v0.20. HRRR reflectivity is
converted from dBZ to linear Z before interpolation to MRMS/regional grids and
converted back to dBZ before verification. Py-ART Barnes2 gridding for regional
and 3-D NEXRAD products likewise operates on a temporary linear-Z field rather
than averaging dBZ directly. The default computational reflectivity floor is
-30 dBZ and is recorded in dataset provenance.

## v0.20.1

Visualization and retrieval stabilization release:

- GLSEA retrieval now requests the latest full ACSPO GLSEA field by ERDDAP
  dimension index and subsets locally, avoiding fragile floating-point edge
  coordinate requests that could return HTTP 404.
- City labels are wired into the common Plotly Cartesian and Cartopy map
  renderers used by Surface, SST, Air Quality, Model, Model Evaluation, Radar,
  Regional Radar, and atmosphere slices.
- Air Quality supports PM2.5 and ozone selection, filled analyses, optional
  concentration contours, and monitoring-site overlays.
- HRRR adds terrain elevation and forecast-hour playback.
- The Model regional-NEXRAD overlay call is corrected and exposes 1/2/4/8-km
  radar-grid resolution, defaulting to 1 km.
- Single-site NEXRAD PPI supports opt-in recent-volume playback.
- Radar lead-time verification has selectable reflectivity thresholds with a
  clearer CSI definition, plus synchronized HRRR/MRMS/difference playback.


## v0.20.2

Stabilization and workflow-cleanup release.

- Fixes Icechunk append writes by applying variable encoding only when a store
  is first created. Existing variables are appended without re-supplying
  encoding.
- Adds recent-history Icechunk reads for local MRMS playback.
- Fixes the single-radar Py-ART map regression by creating a Cartopy GeoAxes.
- Adds MRMS regional-radar playback using accumulated local storage history.
- Air-quality contours are line-only and default off; PM2.5 and ozone remain
  independently selectable.
- Model Explorer no longer duplicates quantitative Model Evaluation workflows.
  Its regional radar context uses lightweight MRMS QC composite instead of
  expensive multi-Level-II gridding.
- Fixes the HRRR/ASOS comparison loader's observation-offset and matching-mode
  arguments.
- Radar Model Evaluation uses white state outlines and lead-time playback loads
  one cached comparison frame at a time rather than retaining all lead grids.
- Surface and Atmosphere Slice are merged into Surface / Atmosphere. Surface is
  the default; elevated observational slices are selected by height.
- Wind speed is derived from U/V components when an atmospheric wind-speed
  field is absent.
- Adds an initial NOAA/NDBC marine-observation adapter with capability-based
  air temperature, water temperature, wind, pressure, dew point, and wave
  fields where reported.
- Adds a Combined Air + Lake Temperature display with marine-platform hover
  values while retaining air and water temperature as distinct physical fields.


## v0.20.4

Interactive visualization and radar/model-evaluation stabilization release.

- Surface / Atmosphere now has independent land-station and marine-station
  toggles, optional station values/IDs, and interactive surface wind overlays
  using arrows or streamlines.
- Elevated-atmosphere wind speed is derived from U/V whenever the stored wind
  speed is missing or contains no finite values.
- Single Radar defaults to an interactive Plotly PPI with pan/zoom/hover and
  retains Classic Py-ART as a fallback.
- NEXRAD selection continues to use human-readable `RADAR — City, ST` labels.
- Model Explorer defaults to interactive maps for surface and pressure-level
  fields, while retaining static Matplotlib fallbacks.
- HRRR reflectivity defaults to the shared radar convention: Turbo,
  -30 to 70 dBZ.
- Model Explorer's optional MRMS overlay stays interactive.
- Radar lead-time verification now aligns HRRR to hourly model valid times
  independently from the MRMS observation timestamp, and unavailable frames
  are skipped gracefully instead of crashing playback.
- ADAPT object outlines use ordered contours plus light display-only
  morphological cleanup to remove spiky/crossing polygon artifacts; original
  masks remain unchanged for quantitative metrics.
- State outlines are added to 3-D atmosphere and radar geographic scenes.


## v0.20.5

Patch release restoring broader Surface-page variable selection after the Surface / Atmosphere merge.

- Restores Surface-page quick-look variables for Air Temperature, Dew Point, Wind Speed, PM2.5, Ozone, Sea/Lake Surface Temperature, and Combined Air + Lake Temperature.
- Keeps the dedicated Air Quality page intact while again allowing PM2.5 and Ozone quick looks from the Surface page.
- Preserves land-station and marine-station overlays plus surface wind overlays on the restored Surface views.


## v0.20.6

Radar-site selector display patch.

- Radar view now uses explicit `RADAR ID — City, State` dropdown options rather than relying on Streamlit `format_func`, while keeping the radar ID internally for data access.
- Expanded city/state labels for additional central-US NEXRAD sites that may appear in center/radius regions.


## v0.20.7

Model-evaluation styling patch.

- Uses black state outlines in the interactive comparison plots on the Model Evaluation view, including radar/MRMS comparisons, for better visual consistency and readability.


## v0.20.8

3-D atmosphere regional-domain patch.

- Clips 3-D state-outline geometry to the active ARIA region so distant U.S. states, including Alaska, no longer expand the Plotly scene.
- Explicitly constrains 3-D atmosphere x/y scene limits to the configured region.
- Keeps state outlines on the base plane while preserving regional pan/rotate/zoom behavior.


## v0.21.0

Multi-profile persistent storage and historical retrieval foundation.

- Storage configuration schema v3 introduces named storage profiles. Multiple regions can remain enabled simultaneously, each with independent sources, cadences, state, locks, and store directories.
- `aria storage update` now evaluates all enabled profiles; `--profile` limits a run to one profile.
- Data Storage configuration can create/select profiles and shows aggregate status across profiles.
- Existing v0.20 single-region storage configuration is migrated automatically into a named profile while retaining the existing region directory.
- Live data and historical backfill write into the same Icechunk/Parquet storage model.
- Initial historical backfill supports MRMS and HRRR F00 via `aria history plan` and `aria history build`. Backfills are resumable through timestamp de-duplication.
- Added arbitrary stored time-range readers for Icechunk datasets and Parquet snapshot series.
- Historical backfill records ingest mode, requested time, profile, success/failure, and data time in storage provenance/state.


## v0.21.1

Storage-profile status patch.

- Fixes the Data Storage page crash when a newly-created storage profile has
  never completed a source update.
- Normalizes mixed `last_success_utc` values to UTC datetimes before computing
  the per-profile latest-success summary.
- Displays an em dash for profiles that do not yet have a successful ingest.


## v0.21.2

Navigation cleanup patch.

- Removes the redundant top-level Air Quality view.
- PM2.5 and Ozone remain available from Surface / Atmosphere with the AQ-specific
  support-radius, monitoring-site, and contour controls shown when those fields
  are selected.
- Keeps the main navigation focused on Surface / Atmosphere, Radar, Regional
  Radar, Model, Model Evaluation, 3-D Atmosphere, and Coverage.


## v0.21.3

Cloud-optimized storage enforcement patch.

- Active gridded and multidimensional datasets are persisted only as
  Icechunk-backed Zarr v3 repositories.
- Active point/tabular observations are now written as date-partitioned
  Parquet datasets (`year=/month=/day=/part-*.parquet`) instead of accumulating
  flat timestamped Parquet snapshots.
- Legacy NetCDF snapshots are converted into Icechunk/Zarr v3 and are no
  longer used as a normal read fallback.
- Existing snapshot-style Parquet files are converted into the partitioned
  Parquet layout.
- Successfully converted legacy files are moved outside the active storage
  tree into a sibling `storage_legacy_archive` directory.
- `aria storage update` automatically performs the idempotent legacy
  conversion before resuming collection for a profile.
- The Data Storage page now explicitly identifies Icechunk / Zarr v3 and
  partitioned Parquet as the active ARIA persistence formats.


## v0.21.4

Surface-observation restoration patch.

- Restores the broader IEM ASOS/AWOS observation request: temperature, dew
  point, relative humidity, wind speed/direction/gust, precipitation,
  altimeter, sea-level pressure, visibility, first cloud-base height,
  feels-like temperature, and sky-cover observations.
- Grids all appropriate numeric scalar fields alongside U/V wind.
- Surface / Atmosphere now builds its Surface variable selector dynamically
  from the fields present in the analysis dataset instead of maintaining a
  short hard-coded list.
- PM2.5, Ozone, SST, and Combined Air + Lake Temperature remain available in
  the same Surface selector.
- Categorical sky cover is retained in the station-observation table but is
  not spatially interpolated.


## v0.23.4

Storage-grid compatibility patch.

- Detects region/grid/layout changes before appending to an existing
  Icechunk/Zarr v3 repository.
- If a profile's active grid is incompatible with the existing repository,
  ARIA preserves the old repository under the source's `_grid_archive`
  directory and initializes a fresh active repository instead of failing.
- Layout signatures include non-time dimensions, geographic/vertical
  coordinates, and data-variable dimension layouts, so both region changes
  and schema changes are detected.
- Changing a storage profile's region resets its source success state so all
  sources become due immediately for the new domain.
- Prevents the xarray errors where Surface, Air Quality, Radiosonde, or HRRR
  arrays had different latitude/longitude/x/y dimensions than an older store.


## v0.21.6

Surface-observation Parquet schema patch.

- Keeps the shared surface-observation `value` column strictly numeric.
- Stores categorical ASOS sky-cover codes such as CLR/BKN/OVC in a separate
  nullable `text_value` column.
- Hardens the partitioned-Parquet writer against mixed Python object columns,
  preventing PyArrow from attempting to convert categorical strings such as
  `OVC` to floating point.


## v0.21.7

Combined-temperature and storage-status clarification patch.

- Combined Air + Lake Temperature now loads SST local-first from the active
  profile's Icechunk/Zarr v3 store rather than downloading GLSEA/NOAA data on
  every dashboard rerun.
- The combined map validates that the SST field is 2-D, has geographic
  coordinates, and contains finite values before adding the overlay.
- SST is drawn as the top heatmap layer with full opacity over valid water
  pixels and transparent gaps elsewhere.
- Combined and SST-only views report SST product, valid time, valid-cell count,
  and whether the data came from the persistent store or a live source.
- Data Storage now labels Recent Activity as historical/append-only so old
  error rows are not confused with current source state.


## v0.21.8

Combined Air + SST performance patch.

- Dashboard SST reads are persistent-store-only; selecting SST or the Combined
  Air + Lake Temperature view no longer performs a synchronous NOAA/GLSEA
  network request or invokes the storage collector.
- Dense SST fields are thinned only for interactive Plotly rendering (maximum
  about 180 points per axis in the combined view and 220 in the SST-only view).
- Full-resolution SST remains unchanged in Icechunk/Zarr v3.
- Combined-view provenance reports both valid SST cells in storage and the
  number rendered.
- If SST is unavailable locally, the view returns immediately with an
  actionable message rather than blocking on network retrieval.


## v0.21.9

Region Configuration map cleanup.

- Makes U.S. state/subunit outlines explicit and higher contrast in the Region
  Configuration preview map.
- Also strengthens national borders and coastlines so the selected regional
  bounding box is easier to interpret geographically.


## v0.21.10

HRRR/cfgrib cache-recovery patch.

- Detects incomplete/corrupted GRIB reads reported by cfgrib/ecCodes, including
  `PrematureEndOfFileError`, premature end-of-file messages, and invalid GRIB
  message EOF failures.
- Removes only the affected Herbie subset and associated cfgrib index sidecars.
- Retries the HRRR request once with Herbie `overwrite=True`, preventing a
  truncated cached subset from poisoning repeated Model Evaluation requests.
- If the retry is also incomplete, raises one concise ARIA error containing the
  HRRR cycle, forecast lead, product, and field search.
- Updates Model Evaluation Streamlit sizing arguments away from the deprecated
  `use_container_width` form where applicable.


## v0.22.0

Navigation and workflow stabilization release.

- Reorganizes the sidebar around consistent `Data`, `Time`, `Display`,
  `Analysis`, and `Playback` sections instead of interleaving unrelated
  controls.
- Replaces long radio-button navigation lists with compact workspace/view
  selectors.
- Moves auto-refresh, manual refresh, and processed-cache controls into one
  collapsed `App controls` section.
- Moves color-map and min/max controls into collapsed per-variable color-scale
  panels throughout the application.
- Surface / Atmosphere now keeps interpolation/history controls together under
  `Analysis settings`, observation overlays together under `Display`, and AQ
  options together only when PM2.5/Ozone are selected.
- Adds consistent sidebar grouping to Radar, Regional Radar, Model, Model
  Evaluation, Coverage, and 3-D Atmosphere.
- Includes all v0.21.10 HRRR/cfgrib corrupt-cache recovery and the v0.21.x
  cloud-optimized storage, SST-performance, state-outline, and storage-grid
  fixes.
- Updates dashboard calls from deprecated Streamlit `use_container_width` to
  the current `width` API.

Radar architecture direction for the v0.22 series remains xradar as the
canonical I/O/data model, Py-ART as the primary radar-science toolkit, optional
wradlib algorithms where they add value, and LROSE reserved for specialized
external processing backends.


## v0.22.1

Navigation adjustment.

- Restores radio-button navigation for Analysis views instead of the v0.22.0 dropdown.
- Uses radio buttons for Configuration pages as well, keeping navigation visually consistent.
- Retains the v0.22 sidebar organization for Data, Display, Analysis, Playback, and App controls.


## v0.22.2

Playback rendering patch.

- Radar, Regional Radar, Model forecast playback, and Model Evaluation playback
  now render Plotly frames as static PNG images during playback mode.
- This reduces the cost of rebuilding an interactive Plotly widget on every
  animation tick and should make playback smoother.
- If static Plotly export is unavailable, ARIA falls back to interactive Plotly.


## v0.22.3

Native static playback patch.

- Removes the Plotly-to-PNG/Kaleido dependency from playback.
- Single-radar playback now uses the existing static Py-ART/Matplotlib PPI.
- Regional MRMS playback uses a static Matplotlib/Cartopy map.
- HRRR forecast playback uses the static model-field renderer; the interactive
  map control is disabled while playback is active.
- Model Evaluation lead-time playback uses the static three-panel
  Model | Observation | Difference renderer.
- Interactive Plotly remains available when inspecting an individual frame.


## v0.22.4

Radar static-playback fix.

- Fixes `cannot access local variable 'ccrs' where it is not associated with a value`
  when rendering the static Py-ART/Cartopy radar frames used for playback.
- Cartopy CRS is now imported at module scope in `aria.plotting.radar`.
- Removes the nested CRS import inside the optional ADAPT overlay block that
  caused Python to treat `ccrs` as an uninitialized local variable earlier in
  the radar plotting function.


## v0.22.5

Regional-radar playback title fix.

- Static MRMS playback frames now use an observation-specific title rather than
  the generic model-field title.
- Removes the misleading `MODEL`, `Init ?`, and `Valid ?` text from Regional
  Radar playback.
- `plot_model_field` now accepts an optional explicit title, while preserving
  the existing model metadata title behavior for HRRR/model plots.


## v0.22.6

Single-radar playback rendering fix.

- Adds a dedicated static radar playback renderer that bins radar gates onto a
  regular geographic display grid and renders the result directly with
  Matplotlib/Cartopy.
- Playback no longer depends on Py-ART `RadarMapDisplay`, which could render a
  blank frame during Streamlit animation reruns.
- The normal non-playback Classic Py-ART option remains unchanged.
- Static playback preserves the common -30 to 70 dBZ reflectivity scale,
  state boundaries, cities, and optional ADAPT outlines.


## v0.23.0

Integrated ARCO analysis and radar-environment release.

- Adds a Data Cube view summarizing the active region's ARCO-style sources, storage formats, cadence, status, and enabled integrated analyses.
- Adds HRRR upper-air environmental overlays to Single Radar and MRMS Regional Radar.
- Supports 850/700/500/300-hPa temperature contours, wind arrows/barbs, and optional geopotential-height contours.
- Single Radar can also overlay stored surface observations.
- HRRR environmental context is matched to radar/MRMS valid time using an appropriate recent model cycle and forecast lead.
- Static playback and interactive single-frame modes both support the environmental context.


## v0.23.1

Data Cube status fix.

- Fixes a `NameError` in the new Data Cube view caused by calling
  `status_rows` instead of the imported `storage_status_rows` alias.
- The Data Cube view can now build its source/status table from the active
  storage profile.


## v0.23.2

Demo stabilization release.

- NEXRAD labels prefer `RADAR — City, ST`.
- Fixes HRRR environment captions to use ModelRun `initialization_time`.
- Adds configurable temperature and geopotential-height contour intervals.
- Masks repeated minimum/fill reflectivity backgrounds in HRRR and MRMS display.
- Uses black state outlines for Regional Radar over transparent no-echo areas.
- Fixes surface lead-time `store_comparisons` and duplicate error rows.
- Radar lead verification now honors `store_comparisons=False`.
- Handles temporarily missing Herbie subset indexes with a concise retry message.
- Derives upper-air wind speed from U/V and starts wind-speed scales at 0 kt.
- Opens 3-D Atmosphere from the south looking north with explicit vertical up.
- Refocuses Data Cube on scientific availability and integrated-analysis readiness.


## v0.23.3

Data Cube timeline release.

- Makes the Data Cube view timeline-centric rather than duplicating Data Storage.
- Reads actual timestamps from local Icechunk/Zarr and partitioned Parquet stores.
- Shows continuous coverage as bars and discrete radiosonde/SST observations as markers.
- Detects visible gaps in persistent time series based on each source's configured cadence.
- Adds 24-hour, 3-day, 7-day, 30-day, and all-available timeline windows.
- Highlights the all-source overlap span.
- Shows start time, latest valid data time, stored-time count, and last successful collector check separately.
- Adds a direct `Configure Data Sources & Storage` navigation button from Data Cube.
- Keeps collection/storage mechanics on Data Storage while Data Cube focuses on scientific readiness.


## v0.23.4

Data Cube timeline correctness patch.

- Prefers Parquet snapshot timestamps for Surface, Air Quality, Marine, and Radiosonde timelines.
- Uses Icechunk time coordinates for gridded/model sources.
- Rejects impossible future timestamps from Data Cube availability and overlap calculations.
- Exposes the selected timeline source and count of rejected future timestamps.
- Shows isolated continuous-source timestamps as markers instead of zero-width invisible lines.
- Computes continuous overlap from actual coverage segments instead of max(first) / min(last).
- Keeps radiosondes and SST as discrete markers and excludes them from continuous-overlap claims.


## v0.24.0

Integrated-analysis and radar usability patch.

- Data Cube auto-refreshes local storage metadata every 5 minutes by default and adds a local-only Refresh now control.
- NEXRAD site discovery normalizes Py-ART three-character IDs to canonical K-prefixed IDs and exposes explicit human-readable site labels.
- Interactive single-radar state boundaries are black for better visibility.
- PM2.5 and ozone monitoring-site overlays now honor Show station values and Show station IDs.
- Radar environmental context adds Surface Stations, Contours, Shaded analysis, and Contours + stations modes with contour labels.
- HRRR lead-time verification uses a conservative latest-complete valid time so just-started cycles and unpublished subset indexes are not treated as hard failures.
- Standard HRRR↔Surface and HRRR↔MRMS difference products are generated from already-persisted local sources and stored under `derived/model_obs` when temporal matching criteria are met.
- Individual NEXRAD persistence is now represented in storage profiles with Off/selected/all-region configuration groundwork. Native Radar DataTree/Icechunk writing remains deliberately guarded until the writer is enabled rather than silently duplicating raw Level-II files.

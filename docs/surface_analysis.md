# Surface Analysis

GPGL v0.15 uses Barnes objective analysis as the default surface interpolation, with Gaussian and IDW alternatives. The dashboard exposes smoothing scale and maximum station-support distance. Barnes is intended to reduce station-centered "splotchy" artifacts while retaining coherent meteorological gradients.

Terrain/lapse-rate correction is documented as a follow-on refinement rather than silently enabled.


## Performance

ARIA v0.16.3 uses a bounded-neighbor Barnes implementation. The previous
implementation allocated distances to every station for every analysis grid point
twice per Barnes pass and repeated that for every 5-minute time and variable.
The optimized implementation uses the nearest 64 stations within the configured
support radius and parallel SciPy KD-tree queries.

The dashboard also defaults to a latest-only analysis. Enable **Load previous
hour** when the historical 5-minute sequence is needed.

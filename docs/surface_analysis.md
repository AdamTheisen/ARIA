# Surface Analysis

GPGL v0.15 uses Barnes objective analysis as the default surface interpolation, with Gaussian and IDW alternatives. The dashboard exposes smoothing scale and maximum station-support distance. Barnes is intended to reduce station-centered "splotchy" artifacts while retaining coherent meteorological gradients.

Terrain/lapse-rate correction is documented as a follow-on refinement rather than silently enabled.

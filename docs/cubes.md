# ARIA Data Cubes

ARIA uses xarray with Icechunk/Zarr as the persistent cube layer.

The intended production pattern is:

    sources → normalize/QC → Dask graph → Icechunk commit → lazy regional reads

ARIA v0.16 adds YAML recipes and initial Dask chunk recommendations. The cube
manager defaults to `~/.cache/aria/icechunk` or `ARIA_CUBE_ROOT`.

Recommended logical repositories remain source-oriented within a region, for
example:

    regions/gpgl/surface
    regions/gpgl/mrms
    regions/gpgl/hrrr-surface
    regions/sgp/surface

Chunking should follow access patterns. Surface/MRMS generally benefit from
small time chunks and moderate horizontal chunks; 3-D fields additionally chunk
the vertical dimension.

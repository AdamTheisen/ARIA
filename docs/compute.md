# ARIA Compute and Dask

Dask is intended for large, naturally chunked data rather than every workflow.

High-value targets:
1. Icechunk/Zarr lazy reads
2. MRMS time-series processing
3. HRRR regional and multi-cycle processing
4. historical cube generation
5. regional radar task parallelism
6. 3-D atmosphere processing

Surface and AirNow station analyses continue to rely primarily on vectorization
and SciPy KD-trees because Dask overhead is unlikely to help small point tables.

`aria.compute.recommended_chunks()` provides initial defaults. These
are starting points and should be benchmarked against actual ARIA access modes.

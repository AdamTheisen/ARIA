from aria.models.hrrr import HRRR_SURFACE_VARIABLES
from aria.plotting.cities import cities_for_region
from aria.region import Region


def test_hrrr_terrain_registered():
    assert "terrain_height" in HRRR_SURFACE_VARIABLES
    assert HRRR_SURFACE_VARIABLES["terrain_height"]["search"] == ":HGT:surface"


def test_city_filtering():
    region = Region(name="Twin Cities", west=-94.0, east=-92.5, south=44.5, north=45.5)
    names = {row[0] for row in cities_for_region(region)}
    assert "Minneapolis" in names
    assert "St. Paul" in names

from aria.region import Region, REGION_PRESETS

def test_region_cache_key_changes_with_bounds():
    a=Region(name="x",west=-100,south=40,east=-90,north=50)
    b=Region(name="x",west=-99,south=40,east=-90,north=50)
    assert a.cache_key != b.cache_key

def test_presets():
    assert "GPGL" in REGION_PRESETS
    assert REGION_PRESETS["GPGL"].contains(45.0,-95.0)

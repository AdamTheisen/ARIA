import pandas as pd
from aria.workflows import match_hrrr_to_observation_time

def test_exact_hour_uses_preceding_cycle():
    cycle,fxx,valid,offset=match_hrrr_to_observation_time(pd.Timestamp("2026-08-31T12:00:00Z"))
    assert cycle == pd.Timestamp("2026-08-31T11:00:00Z")
    assert fxx == 1
    assert valid == pd.Timestamp("2026-08-31T12:00:00Z")
    assert offset == 0

def test_nonhour_time_matches_nearest_valid():
    cycle,fxx,valid,offset=match_hrrr_to_observation_time(pd.Timestamp("2026-08-31T11:42:00Z"))
    assert cycle == pd.Timestamp("2026-08-31T11:00:00Z")
    assert fxx == 1
    assert valid == pd.Timestamp("2026-08-31T12:00:00Z")
    assert offset == 18

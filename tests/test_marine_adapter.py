from pathlib import Path
import importlib.util

def _module():
    path=Path(__file__).parents[1]/"aria"/"adapters"/"marine.py"
    spec=importlib.util.spec_from_file_location("aria_marine_direct",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_parse_ndbc_air_and_water_temperature():
    marine=_module()
    sample="""#STN LAT LON YYYY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES PTDY ATMP WTMP DEWP VIS TIDE
#yr degN degE yr mo dy hr mn degT m/s m/s m sec sec degT hPa hPa degC degC degC nmi ft
45001 48.1 -87.8 2026 09 18 18 50 220 5.0 7.0 1.2 6 5 230 1012.4 0.2 14.5 17.2 10.1 MM MM
"""
    df=marine._read_ndbc_latest(sample)
    assert len(df)==1
    assert float(df.iloc[0].air_temperature_c)==14.5
    assert float(df.iloc[0].water_temperature_c)==17.2

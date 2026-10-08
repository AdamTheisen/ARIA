from pathlib import Path

ROOT=Path(__file__).parents[1]

def test_air_quality_contours_are_lines():
    text=(ROOT/"aria"/"plotting"/"model.py").read_text()
    assert 'coloring="lines"' in text

def test_radar_ppi_uses_geoaxes():
    text=(ROOT/"aria"/"plotting"/"radar.py").read_text()
    assert "projection=ccrs.PlateCarree()" in text

def test_radar_evaluation_state_lines_white():
    text=(ROOT/"aria"/"plotting"/"model.py").read_text()
    assert 'state_line_color = "rgba(255,255,255,.95)" if radar' in text

def test_model_page_comparisons_consolidated():
    text=(ROOT/"aria"/"dashboard"/"app.py").read_text()
    assert "Integrated model / observation comparison" not in text
    assert "Overlay observed regional NEXRAD" not in text
    assert "Overlay MRMS QC reflectivity" in text

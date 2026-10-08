from pathlib import Path

ROOT=Path(__file__).parents[1]

def test_radar_lead_cycles_are_hour_aligned():
    text=(ROOT/"aria"/"workflows.py").read_text()
    assert 'model_valid_time=valid_time.floor("1h")' in text
    assert "cycle=model_valid_time-pd.Timedelta" in text

def test_single_radar_has_interactive_default():
    text=(ROOT/"aria"/"dashboard"/"app.py").read_text()
    assert '["Interactive","Classic Py-ART"]' in text
    assert "plot_interactive_nexrad_ppi" in text

def test_surface_station_and_wind_controls():
    text=(ROOT/"aria"/"dashboard"/"app.py").read_text()
    for label in ("Show land stations","Show marine stations","Show station values","Show station IDs"):
        assert label in text
    assert '"Wind overlay",["None","Arrows","Streamlines"]' in text

def test_model_explorer_interactive_defaults():
    text=(ROOT/"aria"/"dashboard"/"app.py").read_text()
    assert '"Interactive model map",True' in text
    assert '"Interactive pressure-level map",True' in text

def test_reflectivity_style_is_shared():
    text=(ROOT/"aria"/"plotting"/"model.py").read_text()
    assert 'REFLECTIVITY_STYLE = {"cmap":"turbo","plotly":"Turbo","vmin":-30.0,"vmax":70.0' in text

def test_adapt_boundaries_no_longer_use_centroid_angle_sort():
    text=(ROOT/"aria"/"storm_objects.py").read_text()
    assert "ax.contour(cleaned.astype(float),levels=[0.5])" in text
    assert "np.argsort(np.arctan2" not in text

def test_3d_state_floor_is_used():
    atmosphere=(ROOT/"aria"/"plotting"/"atmosphere.py").read_text()
    radar=(ROOT/"aria"/"plotting"/"regional_radar.py").read_text()
    assert "add_state_floor(fig" in atmosphere
    assert "add_state_floor(fig" in radar

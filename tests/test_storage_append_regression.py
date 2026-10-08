from pathlib import Path

def test_append_does_not_resupply_encoding():
    text=(Path(__file__).parents[1]/"aria"/"storage.py").read_text()
    initialized=text.split("if initialized:",1)[1].split("else:",1)[0]
    assert 'append_dim="time"' in initialized
    assert "encoding=" not in initialized

from pathlib import Path

def test_glsea_uses_index_queries():
    text=(Path(__file__).parents[1]/"aria"/"adapters"/"sst.py").read_text()
    assert "sst[last][0:1:last][0:1:last]" in text
    assert "GLSEA retrieval failed for all ERDDAP index-query variants" in text

import numpy as np
from aria.integrated import dbz_to_linear_z, linear_z_to_dbz

def test_dbz_linear_roundtrip():
    dbz=np.array([-30.0,0.0,20.0,40.0,np.nan])
    z=dbz_to_linear_z(dbz)
    back=linear_z_to_dbz(z)
    assert np.allclose(back[:-1],dbz[:-1])
    assert np.isnan(back[-1])

def test_linear_average_differs_from_dbz_average():
    dbz=np.array([20.0,40.0])
    linear_mean=np.mean(dbz_to_linear_z(dbz))
    result=linear_z_to_dbz(np.array([linear_mean]))[0]
    assert result > 35.0
    assert result < 40.0

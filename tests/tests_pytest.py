import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/wellmastergeomech')
import pytest
from DrawSP import getSP  # Adjust this import based on your actual file organization
from BoreStab import getSigmaTT, getHoop  # Adjust this import based on your actual file organization
import numpy as np
import math

def test_typical_conditions_getSP():
    """
    Test typical conditions to verify if the function returns expected minSH, maxSH, and midSH.
    """
    Sv = 28
    Pp = 19
    bhp = 22
    shmin = 25
    UCS = 46
    phi = 45
    flag = 0
    mu = 0.6
    expected_output = [25, 37.71657072857723, 31.358285364288616]  # Example expected values, adjust based on your calculations
    result = getSP(Sv, Pp, bhp, shmin, UCS, phi, flag, mu)
    for expected, actual in zip(expected_output, result):
        assert math.isclose(expected, actual, abs_tol=1e-2)

def test_edge_case_low_shmin_getSP():
    """
    Test an edge case where shmin is lower than physically possible based on Sv, mu, and other parameters.
    """
    Sv = 48
    Pp = 19
    bhp = 22
    shmin = 15  # Unphysically low shmin value
    UCS = 46
    phi = 45
    flag = 0
    mu = 0.6
    result = getSP(Sv, Pp, bhp, shmin, UCS, phi, flag, mu)
    assert all(math.isnan(x) for x in result), "Expected NaNs for unphysical shmin"

def test_edge_case_high_flag_getSP():
    """
    Test handling of high flag values indicating specific geological observations.
    """
    Sv = 28
    Pp = 19
    bhp = 22
    shmin = 25
    UCS = 46
    phi = 45
    flag = 1  # Indicates both breakouts and tensile fractures on log
    mu = 0.6
    expected_output = [25, 50.32270004382842, 37.66135002191421]
    result = getSP(Sv, Pp, bhp, shmin, UCS, phi, flag, mu)
    for expected, actual in zip(expected_output, result):
        assert math.isclose(expected, actual, abs_tol=1e-2)

def test_typical_conditions_getSigmaTT():
    """
    Test getSigmaTT under typical conditions to verify expected output.
    """
    s1, s2, s3 = 45, 34, 29
    alpha, beta, gamma = 5.4, 2.3, 13.5
    azim, inc = 187, 16
    theta, deltaP, Pp = 37, 3, 19
    nu = 0.35
    expected_output = (30.604213455441034, 7.972412289059704, 7.494520245574544, 32.86098540642659, 5.715640338074149, 75.39892741050409, [89.76444658292043, 90.09424775647605, 0.2356194490192675, 89.95985742720414, 179.89755943841308, -66.69398862185872])
    result = getSigmaTT(s1, s2, s3, alpha, beta, gamma, azim, inc, theta, deltaP, Pp, nu)
    for expected, actual in zip(expected_output, result):
        if isinstance(expected, list):
            for e, a in zip(expected, actual):
                assert math.isclose(e, a, abs_tol=1e-2)
        else:
            assert math.isclose(expected, actual, abs_tol=1e-2)

def test_specific_case_getHoop():
    """
    Test getHoop under a specific case to verify expected output.
    """
    inc, azim = 16, 187
    s1, s2, s3 = 45, 34, 29
    deltaP, Pp, ucs = 3, 19, 46
    alpha, beta, gamma = 5.4, 2.3, 13.5
    nu = 0.35
    expected_crush, expected_frac = ([0] * 361, [0] * 361)
    crush, frac, minstress, maxstress, angle1, angle2  = getHoop(inc, azim, s1, s2, s3, deltaP, Pp, ucs, alpha, beta, gamma, nu)
    for e_crush, a_crush, e_frac, a_frac in zip(expected_crush, crush, expected_frac, frac):
        assert math.isclose(e_crush, a_crush, abs_tol=1e-2)
        assert math.isclose(e_frac, a_frac, abs_tol=1e-2)

# Remember to run your tests with pytest from the command line:
# pytest your_test_file_name.py


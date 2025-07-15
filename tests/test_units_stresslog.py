import pytest
import numpy as np
import stresslog

# --- Test Configuration ---
RTOL = 0.01  # Relative Tolerance
ATOL = 0.051  # Absolute Tolerance for 1-decimal rounding
# ------------------------
# --- Phase 1: Scalar Function Regression Tests ---

@pytest.mark.parametrize("inputs, expected_output", [
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([909.2, 3801.8, 6297.8, 7349.6, 9262.3, 12810.5, 13001.5, 14592.5, 18153.9,
 20549.0])}, 2.1),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([2366.9, 4223.2, 7577.7, 6020.6, 9510.0, 13165.1, 12597.8, 15780.9,
 17919.6, 19066.0])}, 2.2),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1097.2, 5174.0, 5824.9, 7535.2, 8096.7, 11049.8, 13712.1, 16220.4,
 16702.5, 20602.8])}, 2.1),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1703.4, 2977.0, 4788.5, 8110.4, 9125.1, 11923.2, 13231.0, 16285.9,
 19507.4, 19945.5])}, 2.2),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1872.8, 3607.6, 6056.2, 7459.8, 10371.3, 11828.4, 14590.9, 14893.1,
 18591.2, 21427.9])}, 2.2),
])
def test_compute_optimal_gradient(inputs, expected_output):
    """Regression test for stresslog.compute_optimal_gradient."""
    actual_output = stresslog.compute_optimal_gradient(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1507.1, 4182.9, 4142.0, 8134.4, 9551.1, 11783.9, 12754.1, 16508.1,
 19277.5, 19590.3]), 'gradient': 1.0}, 4045.6),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([2962.0, 3311.7, 4687.8, 7435.9, 9963.7, 12115.1, 14161.2, 14856.5,
 18913.4, 19173.5]), 'gradient': 2.3}, -214.7),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1879.4, 4395.2, 7386.3, 6737.3, 9884.5, 11376.6, 15042.9, 16562.7,
 17908.1, 20560.4]), 'gradient': 2.2}, 67.2),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1660.2, 4086.6, 6834.8, 6815.4, 9444.7, 10945.1, 14887.1, 15655.5,
 18728.7, 18946.3]), 'gradient': 1.2}, 2821.6),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1955.6, 3818.9, 3813.3, 6550.2, 10819.6, 12086.3, 14816.6, 16898.6,
 18540.2, 19201.7]), 'gradient': 2.2}, -36.0),
])
def test_compute_optimal_offset(inputs, expected_output):
    """Regression test for stresslog.compute_optimal_offset."""
    actual_output = stresslog.compute_optimal_offset(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'alpha': -110.5, 'strike': -165.0, 'dip': 58.5}, (35.2, 57.0)),
    ({'alpha': -77.8, 'strike': 9.4, 'dip': 5.6}, (-0.3, 5.6)),
    ({'alpha': 6.1, 'strike': 2.0, 'dip': 17.9}, (-17.9, -1.3)),
    ({'alpha': -124.6, 'strike': -123.8, 'dip': 74.2}, (64.3, 52.6)),
    ({'alpha': 2.4, 'strike': -4.5, 'dip': 23.6}, (-23.6, -1.0)),
])
def test_getEuler(inputs, expected_output):
    """Regression test for stresslog.getEuler."""
    actual_output = stresslog.getEuler(**inputs)
    assert isinstance(actual_output, tuple)
    assert len(actual_output) == len(expected_output)
    for actual, expected in zip(actual_output, expected_output):
        if expected is None: continue
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)
        elif isinstance(expected, (list, tuple)):
            assert actual == pytest.approx(expected, rel=RTOL, abs=ATOL, nan_ok=True)
        elif isinstance(expected, complex):
            assert actual.real == pytest.approx(expected.real, rel=RTOL, abs=ATOL)
            assert actual.imag == pytest.approx(expected.imag, rel=RTOL, abs=ATOL)
        else:
            assert actual == pytest.approx(expected, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'p': 13224.2, 't': 147.8}, 0.4),
    ({'p': 16625.5, 't': 133.5}, 0.5),
    ({'p': 8399.4, 't': 119.6}, 0.3),
    ({'p': 15011.3, 't': 31.5}, 0.7),
    ({'p': 17275.3, 't': 159.8}, 0.5),
])
def test_getGasDensity(inputs, expected_output):
    """Regression test for stresslog.getGasDensity."""
    actual_output = stresslog.getGasDensity(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'tvd': 2620.5, 'gradient': 1.9}, 7088.8),
    ({'tvd': 5505.9, 'gradient': 2.3}, 18029.8),
    ({'tvd': 2042.4, 'gradient': 1.1}, 3198.7),
    ({'tvd': 5690.4, 'gradient': 1.4}, 11342.4),
    ({'tvd': 2756.7, 'gradient': 2.3}, 9027.2),
])
def test_getHydrostaticPsi(inputs, expected_output):
    """Regression test for stresslog.getHydrostaticPsi."""
    actual_output = stresslog.getHydrostaticPsi(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 128.0, 's2': 79.3, 's3': 19.5, 'alpha': 168.1, 'beta': 133.9, 'gamma': -76.2}, np.array([[-0.4, 0.6, 0.7],
 [-0.9, -0.4, -0.1],
 [-0.2, 0.7, -0.7]])),
    ({'s1': 124.2, 's2': 112.5, 's3': 33.9, 'alpha': -13.1, 'beta': 139.6, 'gamma': -154.1}, np.array([[-0.5, -0.5, 0.7],
 [0.6, -0.8, -0.2],
 [0.7, 0.3, 0.6]])),
    ({'s1': 167.6, 's2': 93.1, 's3': 18.0, 'alpha': 123.2, 'beta': 95.9, 'gamma': 107.6}, np.array([[1.0, -0.3, -0.1],
 [0.3, 1.0, 0.1],
 [0.0, -0.1, 1.0]])),
    ({'s1': 234.5, 's2': 75.2, 's3': 36.4, 'alpha': 158.9, 'beta': 45.8, 'gamma': -165.6}, np.array([[0.6, 0.5, -0.7],
 [-0.5, 0.8, 0.3],
 [-0.7, -0.2, -0.7]])),
    ({'s1': 109.2, 's2': 126.2, 's3': 42.2, 'alpha': 37.9, 'beta': 99.6, 'gamma': 71.7}, np.array([[0.8, -0.1, 0.5],
 [-0.6, -0.1, 0.8],
 [-0.1, -1.0, -0.2]])),
])
def test_getOrit(inputs, expected_output):
    """Regression test for stresslog.getOrit."""
    actual_output = stresslog.getOrit(**inputs)
    np.testing.assert_allclose(actual_output, expected_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'alpha': -95.1, 'beta': -81.4, 'gamma': 156.9}, np.array([[-0.0, -0.1, 1.0],
 [-0.9, 0.5, 0.1],
 [-0.5, -0.9, -0.1]])),
    ({'alpha': 90.7, 'beta': -80.4, 'gamma': 41.4}, np.array([[-0.0, 0.2, 1.0],
 [-0.7, -0.7, 0.1],
 [0.7, -0.7, 0.1]])),
    ({'alpha': -151.2, 'beta': 3.9, 'gamma': 67.0}, np.array([[-0.9, -0.5, -0.1],
 [0.1, -0.4, 0.9],
 [-0.5, 0.8, 0.4]])),
    ({'alpha': -179.6, 'beta': 140.0, 'gamma': -82.3}, np.array([[0.8, 0.0, -0.6],
 [0.6, -0.1, 0.8],
 [-0.1, -1.0, -0.1]])),
    ({'alpha': -143.6, 'beta': -109.6, 'gamma': 134.5}, np.array([[0.3, 0.2, 0.9],
 [0.1, 1.0, -0.2],
 [-1.0, 0.2, 0.2]])),
])
def test_getRota(inputs, expected_output):
    """Regression test for stresslog.getRota."""
    actual_output = stresslog.getRota(**inputs)
    np.testing.assert_allclose(actual_output, expected_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'Sv': 177.1, 'Pp': 99.3, 'bhp': 57.6, 'shmin': 31.4, 'UCS': 41.0, 'phi': 0.7, 'mu': 0.5, 'nu': 0.4, 'PhiBr': 31.8}, [129.0, 177.1, 153.1]),
    ({'Sv': 76.7, 'Pp': 49.9, 'bhp': 71.2, 'shmin': 29.1, 'UCS': 95.1, 'phi': 0.8, 'mu': 0.5, 'nu': 0.3, 'PhiBr': 30.4}, [60.1, 76.7, 68.4]),
    ({'Sv': 70.9, 'Pp': 86.4, 'bhp': 64.7, 'shmin': 114.8, 'UCS': 84.9, 'phi': 0.7, 'mu': 0.5, 'nu': 0.4, 'PhiBr': 69.8}, [45.8, 70.9, 58.4]),
    ({'Sv': 149.5, 'Pp': 27.7, 'bhp': 58.8, 'shmin': 62.6, 'UCS': 14.3, 'phi': 0.4, 'mu': 0.7, 'nu': 0.4, 'PhiBr': 13.4}, [62.6, 156.4, 109.5]),
    ({'Sv': 65.4, 'Pp': 107.4, 'bhp': 117.3, 'shmin': 70.3, 'UCS': 59.6, 'phi': 0.6, 'mu': 0.9, 'nu': 0.3, 'PhiBr': 48.0}, [-104.3, 65.4, -19.5]),
])
def test_getSP(inputs, expected_output):
    """Regression test for stresslog.getSP."""
    actual_output = stresslog.getSP(**inputs)
    assert isinstance(actual_output, list)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 42.0, 'sy': 43.8, 'sz': 22.5, 'alpha': 120.2, 'beta': -87.5, 'gamma': -135.6}, (np.array([24.0, 5.4, -0.6]), np.array([5.4, 42.3, 0.1]), np.array([-0.6, 0.1, 42.0]))),
    ({'sx': 92.3, 'sy': 148.6, 'sz': 18.5, 'alpha': -25.5, 'beta': 77.0, 'gamma': 98.8}, (np.array([105.5, -59.2, 8.6]), np.array([-59.2, 58.9, -9.0]), np.array([8.6, -9.0, 95.0]))),
    ({'sx': 219.6, 'sy': 139.4, 'sz': 56.8, 'alpha': 113.8, 'beta': -81.1, 'gamma': -26.5}, (np.array([139.5, -4.9, -4.4]), np.array([-4.9, 60.2, 22.5]), np.array([-4.4, 22.5, 216.1]))),
    ({'sx': 186.4, 'sy': 72.3, 'sz': 60.9, 'alpha': -84.0, 'beta': 107.7, 'gamma': 169.7}, (np.array([71.5, 1.8, 4.4]), np.array([1.8, 73.2, -36.0]), np.array([4.4, -36.0, 174.8]))),
    ({'sx': 209.0, 'sy': 107.4, 'sz': 34.1, 'alpha': -78.5, 'beta': 49.4, 'gamma': 142.0}, (np.array([70.8, 16.1, -37.2]), np.array([16.1, 133.0, 66.6]), np.array([-37.2, 66.6, 146.7]))),
])
def test_getStens(inputs, expected_output):
    """Regression test for stresslog.getStens."""
    actual_output = stresslog.getStens(**inputs)
    assert isinstance(actual_output, tuple)
    assert len(actual_output) == len(expected_output)
    for actual, expected in zip(actual_output, expected_output):
        if expected is None: continue
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)
        elif isinstance(expected, (list, tuple)):
            assert actual == pytest.approx(expected, rel=RTOL, abs=ATOL, nan_ok=True)
        elif isinstance(expected, complex):
            assert actual.real == pytest.approx(expected.real, rel=RTOL, abs=ATOL)
            assert actual.imag == pytest.approx(expected.imag, rel=RTOL, abs=ATOL)
        else:
            assert actual == pytest.approx(expected, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'alpha': 120.1, 'beta': 72.3, 'gamma': 69.5}, (229.7, 83.9, 49.7)),
    ({'alpha': -87.2, 'beta': -139.0, 'gamma': 68.0}, (348.0, 106.4, 168.0)),
    ({'alpha': -73.4, 'beta': -65.4, 'gamma': 9.8}, (297.4, 65.8, 117.4)),
    ({'alpha': 134.9, 'beta': -142.8, 'gamma': 158.0}, (281.1, 42.4, 101.1)),
    ({'alpha': -29.9, 'beta': -85.6, 'gamma': -15.4}, (314.7, 85.8, 134.7)),
])
def test_getStrikeDip(inputs, expected_output):
    """Regression test for stresslog.getStrikeDip."""
    actual_output = stresslog.getStrikeDip(**inputs)
    assert isinstance(actual_output, tuple)
    assert len(actual_output) == len(expected_output)
    for actual, expected in zip(actual_output, expected_output):
        if expected is None: continue
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)
        elif isinstance(expected, (list, tuple)):
            assert actual == pytest.approx(expected, rel=RTOL, abs=ATOL, nan_ok=True)
        elif isinstance(expected, complex):
            assert actual.real == pytest.approx(expected.real, rel=RTOL, abs=ATOL)
            assert actual.imag == pytest.approx(expected.imag, rel=RTOL, abs=ATOL)
        else:
            assert actual == pytest.approx(expected, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 213.2, 'sy': 46.6, 'sz': 64.1, 'alpha': -176.3, 'beta': -25.4, 'gamma': -151.5}, (88.28087981945+184.06939657419537j)),
    ({'sx': 166.4, 'sy': 120.1, 'sz': 39.2, 'alpha': 168.0, 'beta': 160.3, 'gamma': -101.6}, (122.46193370602431+0j)),
    ({'sx': 155.4, 'sy': 63.9, 'sz': 23.3, 'alpha': 61.4, 'beta': 121.1, 'gamma': 152.9}, (122.40277970837967+0j)),
    ({'sx': 136.1, 'sy': 30.4, 'sz': 64.5, 'alpha': -112.5, 'beta': 143.8, 'gamma': -35.1}, (82.13336122374571+104.296777268603j)),
    ({'sx': 233.0, 'sy': 99.1, 'sz': 23.0, 'alpha': -9.0, 'beta': 177.2, 'gamma': 104.0}, (94.97630968769508+0j)),
])
def test_getVertical(inputs, expected_output):
    """Regression test for stresslog.getVertical."""
    actual_output = stresslog.getVertical(**inputs)
    assert actual_output.real == pytest.approx(expected_output.real, rel=RTOL, abs=ATOL)
    assert actual_output.imag == pytest.approx(expected_output.imag, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ROP': 379.6, 'RPM': 100.8, 'WOB': 272.4, 'BTDIA': 18.1, 'ECD': 0.9, 'pn': 1.4}, 0.5),
    ({'ROP': 88.5, 'RPM': 190.3, 'WOB': 22.1, 'BTDIA': 7.7, 'ECD': 1.3, 'pn': 1.2}, 0.4),
    ({'ROP': 431.7, 'RPM': 76.2, 'WOB': 466.6, 'BTDIA': 16.6, 'ECD': 1.0, 'pn': 1.0}, 0.3),
    ({'ROP': 90.0, 'RPM': 308.2, 'WOB': 143.5, 'BTDIA': 15.8, 'ECD': 1.4, 'pn': 1.7}, 0.7),
    ({'ROP': 475.5, 'RPM': 4.2, 'WOB': 279.7, 'BTDIA': 11.6, 'ECD': 1.5, 'pn': 2.1}, np.nan),
])
def test_get_Dxc(inputs, expected_output):
    """Regression test for stresslog.get_Dxc."""
    actual_output = stresslog.get_Dxc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ObgTgcc': 2.1, 'pn': 2.2, 'd': 1.6, 'nde': 0.8, 'tvdbgl': 3864.5, 'D0': 0.6, 'Dxc': 0.6}, 2.1),
    ({'ObgTgcc': 2.3, 'pn': 1.5, 'd': 3.8, 'nde': 1.5, 'tvdbgl': 5909.4, 'D0': 0.5, 'Dxc': 1.3}, 2.3),
    ({'ObgTgcc': 2.4, 'pn': 1.5, 'd': 19.4, 'nde': 1.2, 'tvdbgl': 2288.4, 'D0': 1.4, 'Dxc': 0.6}, 2.4),
    ({'ObgTgcc': 1.9, 'pn': 1.2, 'd': 16.3, 'nde': 0.9, 'tvdbgl': 1143.8, 'D0': 0.6, 'Dxc': 1.2}, 1.9),
    ({'ObgTgcc': 2.1, 'pn': 1.2, 'd': 10.4, 'nde': 0.8, 'tvdbgl': 2189.6, 'D0': 1.3, 'Dxc': 0.8}, 2.1),
])
def test_get_PPgrad_Dxc_gcc(inputs, expected_output):
    """Regression test for stresslog.get_PPgrad_Dxc_gcc."""
    actual_output = stresslog.get_PPgrad_Dxc_gcc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ObgTgcc': 2.1, 'pn': 1.4, 'be': 7.9, 'ne': 1.1, 'tvdbgl': 5911.6, 'res0': 0.5, 'resdeep': 0.8}, 2.1),
    ({'ObgTgcc': 1.2, 'pn': 2.3, 'be': 19.5, 'ne': 1.2, 'tvdbgl': 4700.6, 'res0': 1.1, 'resdeep': 0.9}, 1.2),
    ({'ObgTgcc': 1.9, 'pn': 1.7, 'be': 2.7, 'ne': 0.8, 'tvdbgl': 4661.5, 'res0': 1.2, 'resdeep': 0.7}, 1.9),
    ({'ObgTgcc': 2.0, 'pn': 1.9, 'be': 1.2, 'ne': 1.0, 'tvdbgl': 2202.3, 'res0': 1.1, 'resdeep': 0.7}, 2.0),
    ({'ObgTgcc': 2.1, 'pn': 1.9, 'be': 6.2, 'ne': 0.5, 'tvdbgl': 3474.9, 'res0': 1.0, 'resdeep': 1.4}, 2.1),
])
def test_get_PPgrad_Eaton_gcc(inputs, expected_output):
    """Regression test for stresslog.get_PPgrad_Eaton_gcc."""
    actual_output = stresslog.get_PPgrad_Eaton_gcc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ObgTgcc': 1.5, 'pn': 1.1, 'b': 6.0, 'tvdbgl': 1381.3, 'c': 14.0, 'mudline': 189.0, 'matrick': 53.2, 'deltmu0': 185.0, 'dalm': 171.0}, 1.5),
    ({'ObgTgcc': 1.9, 'pn': 1.7, 'b': 5.5, 'tvdbgl': 2231.0, 'c': 3.5, 'mudline': 201.7, 'matrick': 60.3, 'deltmu0': 123.3, 'dalm': 178.5}, 1.9),
    ({'ObgTgcc': 1.4, 'pn': 1.7, 'b': 11.2, 'tvdbgl': 5378.9, 'c': 19.5, 'mudline': 217.0, 'matrick': 56.5, 'deltmu0': 64.1, 'dalm': 184.0}, 1.4),
    ({'ObgTgcc': 1.8, 'pn': 0.9, 'b': 10.0, 'tvdbgl': 5386.2, 'c': 12.1, 'mudline': 203.0, 'matrick': 57.2, 'deltmu0': 67.8, 'dalm': 99.5}, 1.8),
    ({'ObgTgcc': 1.7, 'pn': 1.4, 'b': 1.2, 'tvdbgl': 2099.6, 'c': 15.0, 'mudline': 187.0, 'matrick': 62.0, 'deltmu0': 161.1, 'dalm': 137.6}, 1.7),
])
def test_get_PPgrad_Zhang_gcc(inputs, expected_output):
    """Regression test for stresslog.get_PPgrad_Zhang_gcc."""
    actual_output = stresslog.get_PPgrad_Zhang_gcc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'nu2': 0.3, 'ObgTppg': 18.4, 'biot': 0.8, 'ppgZhang': 18.2, 'tecB': 0.6}, 27.2),
    ({'nu2': 0.3, 'ObgTppg': 11.5, 'biot': 1.1, 'ppgZhang': 18.7, 'tecB': 0.8}, 25.9),
    ({'nu2': 0.4, 'ObgTppg': 15.3, 'biot': 1.3, 'ppgZhang': 16.0, 'tecB': 1.1}, 34.0),
    ({'nu2': 0.3, 'ObgTppg': 18.6, 'biot': 1.1, 'ppgZhang': 14.0, 'tecB': 1.1}, 37.2),
    ({'nu2': 0.4, 'ObgTppg': 16.4, 'biot': 0.9, 'ppgZhang': 19.7, 'tecB': 1.0}, 33.2),
])
def test_get_Shmin_grad_Daine_ppg(inputs, expected_output):
    """Regression test for stresslog.get_Shmin_grad_Daine_ppg."""
    actual_output = stresslog.get_Shmin_grad_Daine_ppg(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 237.6, 'sy': 131.3, 'sz': 45.3, 'txy': 11.3, 'tyz': 15.5, 'tzx': 5.5, 'phi': 0.3, 'cohesion': 10.2, 'pp': 129.9}, 30.8),
    ({'sx': 184.8, 'sy': 71.6, 'sz': 68.5, 'txy': 20.0, 'tyz': 11.9, 'tzx': 1.7, 'phi': 0.5, 'cohesion': 7.5, 'pp': 52.8}, 39.9),
    ({'sx': 132.1, 'sy': 134.0, 'sz': 17.1, 'txy': 10.0, 'tyz': 11.6, 'tzx': 18.2, 'phi': 0.6, 'cohesion': 3.0, 'pp': 61.9}, 48.7),
    ({'sx': 124.9, 'sy': 118.6, 'sz': 42.8, 'txy': 2.9, 'tyz': 17.9, 'tzx': 4.9, 'phi': 0.5, 'cohesion': 10.2, 'pp': 13.6}, 39.9),
    ({'sx': 117.0, 'sy': 135.0, 'sz': 27.9, 'txy': 1.4, 'tyz': 17.6, 'tzx': 1.9, 'phi': 0.3, 'cohesion': 9.7, 'pp': 129.6}, 30.8),
])
def test_lade(inputs, expected_output):
    """Regression test for stresslog.lade."""
    actual_output = stresslog.lade(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 57.2, 'sy': 95.5, 'sz': 47.9, 'txy': 5.7, 'tyz': 17.5, 'tzx': 3.6, 'phi': 0.3, 'cohesion': 10.1, 'pp': 15.6}, 26.9),
    ({'sx': 30.1, 'sy': 98.9, 'sz': 24.2, 'txy': 11.6, 'tyz': 5.8, 'tzx': 19.7, 'phi': 0.7, 'cohesion': 12.5, 'pp': 130.6}, 22.0),
    ({'sx': 153.6, 'sy': 74.7, 'sz': 41.9, 'txy': 4.9, 'tyz': 18.6, 'tzx': 15.5, 'phi': 0.7, 'cohesion': 15.1, 'pp': 96.2}, 93.6),
    ({'sx': 118.8, 'sy': 141.5, 'sz': 57.5, 'txy': 19.1, 'tyz': 17.0, 'tzx': 17.6, 'phi': 0.7, 'cohesion': 6.1, 'pp': 62.5}, 248.2),
    ({'sx': 246.1, 'sy': 47.2, 'sz': 36.0, 'txy': 11.1, 'tyz': 7.8, 'tzx': 16.4, 'phi': 0.4, 'cohesion': 16.1, 'pp': 128.3}, 60.8),
])
def test_lade_failure(inputs, expected_output):
    """Regression test for stresslog.lade_failure."""
    actual_output = stresslog.lade_failure(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 236.9, 'sy': 125.1, 'sz': 31.4}, 50.1),
    ({'sx': 111.0, 'sy': 130.0, 'sz': 65.8}, 71.0),
    ({'sx': 157.9, 'sy': 141.7, 'sz': 49.3}, 55.8),
    ({'sx': 240.7, 'sy': 57.2, 'sz': 44.7}, 53.1),
    ({'sx': 199.9, 'sy': 130.2, 'sz': 43.8}, 58.0),
])
def test_mogi(inputs, expected_output):
    """Regression test for stresslog.mogi."""
    actual_output = stresslog.mogi(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 176.7, 's2': 21.0, 's3': 28.4}, 30.8),
    ({'s1': 184.6, 's2': 22.5, 's3': 50.1}, 46.5),
    ({'s1': 131.0, 's2': 122.3, 's3': 43.7}, 48.1),
    ({'s1': 75.6, 's2': 51.1, 's3': 11.4}, 17.0),
    ({'s1': 94.6, 's2': 130.5, 's3': 54.7}, 43.7),
])
def test_mogi_failure(inputs, expected_output):
    """Regression test for stresslog.mogi_failure."""
    actual_output = stresslog.mogi_failure(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 43.8, 's3': 45.0, 'cohesion': 7.3, 'phi': 0.8}, 37.5),
    ({'s1': 179.0, 's3': 23.4, 'cohesion': 9.7, 'phi': 0.5}, -20.8),
    ({'s1': 85.1, 's3': 34.5, 'cohesion': 5.1, 'phi': 0.5}, 7.8),
    ({'s1': 154.2, 's3': 67.4, 'cohesion': 17.9, 'phi': 0.5}, 25.4),
    ({'s1': 219.5, 's3': 14.2, 'cohesion': 9.3, 'phi': 0.4}, -48.6),
])
def test_mohr_failure(inputs, expected_output):
    """Regression test for stresslog.mohr_failure."""
    actual_output = stresslog.mohr_failure(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sigmamax': 163.7, 'sigmamin': 43.5, 'pp': 45.5, 'ucs': 90.5, 'nu': 0.2}, 258.4),
    ({'sigmamax': 218.2, 'sigmamin': 46.1, 'pp': 132.1, 'ucs': 20.1, 'nu': 0.4}, 326.6),
    ({'sigmamax': 149.2, 'sigmamin': 41.4, 'pp': 111.9, 'ucs': 97.0, 'nu': 0.4}, 163.1),
    ({'sigmamax': 161.1, 'sigmamin': 57.9, 'pp': 18.1, 'ucs': 36.2, 'nu': 0.3}, 265.2),
    ({'sigmamax': 75.8, 'sigmamin': 45.7, 'pp': 30.4, 'ucs': 98.7, 'nu': 0.3}, 45.9),
])
def test_willson_sanding_cwf(inputs, expected_output):
    """Regression test for stresslog.willson_sanding_cwf."""
    actual_output = stresslog.willson_sanding_cwf(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sigmamax': 84.5, 'sigmamin': 54.4, 'pp': 23.8, 'ucs': 61.5, 'k0': 1.2, 'nu': 0.3}, 104.2),
    ({'sigmamax': 219.8, 'sigmamin': 35.7, 'pp': 54.2, 'ucs': 66.1, 'k0': 0.8, 'nu': 0.4}, 259.0),
    ({'sigmamax': 170.4, 'sigmamin': 58.9, 'pp': 51.0, 'ucs': 50.3, 'k0': 0.6, 'nu': 0.4}, 138.6),
    ({'sigmamax': 190.1, 'sigmamin': 58.2, 'pp': 85.3, 'ucs': 94.6, 'k0': 1.5, 'nu': 0.2}, 424.2),
    ({'sigmamax': 194.4, 'sigmamin': 67.9, 'pp': 109.2, 'ucs': 56.6, 'k0': 0.6, 'nu': 0.2}, 180.9),
])
def test_zhang_sanding_cwf(inputs, expected_output):
    """Regression test for stresslog.zhang_sanding_cwf."""
    actual_output = stresslog.zhang_sanding_cwf(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)

# --- Phase 2: Vectorized Function Consistency Tests ---

@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ROP': np.array([115.1, 493.5, 438.9, 355.6, 488.0]), 'RPM': np.array([302.5, 166.1, 298.1, 457.2, 1.3]), 'WOB': np.array([210.5, 269.0, 473.6, 260.2, 462.0]), 'BTDIA': np.array([6.9, 16.1, 12.6, 10.6, 16.8]), 'ECD': np.array([1.6, 1.4, 1.5, 2.1, 2.1]), 'pn': np.array([1.0, 2.0, 1.7, 2.1, 1.5])}, np.array([0.4, 0.5, 0.5, 0.5, np.nan])),
    ({'ROP': np.array([455.7, 396.7, 173.9, 54.5, 256.9]), 'RPM': np.array([258.6, 405.8, 285.0, 381.1, 2.6]), 'WOB': np.array([120.0, 377.9, 172.3, 237.5, 217.8]), 'BTDIA': np.array([18.6, 13.9, 15.0, 11.1, 1.7]), 'ECD': np.array([1.3, 1.9, 2.1, 1.7, 1.5]), 'pn': np.array([1.1, 2.2, 1.3, 1.8, 1.4])}, np.array([0.3, 0.6, 0.3, 0.8, np.nan])),
    ({'ROP': np.array([61.4, 349.7, 380.3, 471.3, 430.6]), 'RPM': np.array([126.7, 156.2, 127.4, 218.3, 4.5]), 'WOB': np.array([420.1, 356.3, 280.0, 72.6, 175.5]), 'BTDIA': np.array([17.9, 16.2, 12.4, 16.3, 2.8]), 'ECD': np.array([1.5, 2.2, 1.3, 2.0, 1.1]), 'pn': np.array([0.9, 1.2, 2.0, 1.0, 1.8])}, np.array([0.4, 0.2, 0.6, 0.2, np.nan])),
    ({'ROP': np.array([324.9, 177.2, 315.5, 171.7, 490.2]), 'RPM': np.array([447.5, 285.8, 294.7, 279.3, 2.3]), 'WOB': np.array([319.9, 83.6, 49.0, 19.5, 215.8]), 'BTDIA': np.array([14.4, 11.2, 14.2, 7.6, 7.4]), 'ECD': np.array([2.1, 1.1, 1.8, 1.1, 1.5]), 'pn': np.array([1.9, 1.8, 1.6, 1.6, 1.4])}, np.array([0.5, 0.8, 0.4, 0.6, np.nan])),
    ({'ROP': np.array([56.0, 24.1, 456.0, 57.9, 219.1]), 'RPM': np.array([370.9, 447.1, 123.8, 257.8, 1.2]), 'WOB': np.array([374.7, 339.1, 328.0, 202.0, 385.8]), 'BTDIA': np.array([19.3, 19.3, 19.6, 2.4, 5.9]), 'ECD': np.array([1.6, 1.9, 1.0, 2.3, 1.5]), 'pn': np.array([2.2, 2.0, 2.0, 1.6, 2.1])}, np.array([1.0, 0.9, 0.7, 0.6, np.nan])),
])
def test_get_Dxc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_Dxc_vec."""
    actual_vector_output = stresslog.get_Dxc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ObgTgcc': np.array([1.5, 2.3, 0.9, 1.2, 1.1]), 'pn': np.array([1.5, 2.3, 1.0, 1.5, 1.1]), 'd': np.array([18.7, 16.4, 16.0, 9.6, 2.4]), 'nde': np.array([1.2, 1.4, 0.8, 0.6, 1.0]), 'tvdbgl': np.array([2396.8, 1541.6, 4526.3, 3960.6, 2652.0]), 'D0': np.array([1.0, 0.5, 0.7, 1.0, 0.6]), 'Dxc': np.array([0.6, 0.9, 1.2, 1.5, 1.3])}, np.array([1.5, 2.3, 0.9, 1.2, 1.1])),
    ({'ObgTgcc': np.array([1.7, 1.0, 2.3, 1.8, 1.5]), 'pn': np.array([2.0, 2.4, 1.7, 1.2, 2.0]), 'd': np.array([13.8, 4.8, 5.9, 19.9, 1.1]), 'nde': np.array([0.7, 0.8, 0.9, 1.2, 0.9]), 'tvdbgl': np.array([1757.7, 4513.0, 1395.3, 5372.7, 1983.2]), 'D0': np.array([0.6, 1.2, 1.3, 1.1, 1.0]), 'Dxc': np.array([1.1, 1.3, 0.8, 1.4, 1.0])}, np.array([1.7, 1.0, 2.3, 1.8, 1.5])),
    ({'ObgTgcc': np.array([1.6, 1.2, 1.5, 1.1, 1.0]), 'pn': np.array([1.9, 1.1, 1.2, 1.0, 1.7]), 'd': np.array([3.1, 10.1, 9.0, 7.6, 4.3]), 'nde': np.array([1.0, 1.3, 0.6, 0.6, 1.0]), 'tvdbgl': np.array([2502.6, 1834.6, 3950.6, 5107.7, 1766.0]), 'D0': np.array([1.3, 1.1, 1.0, 0.9, 0.8]), 'Dxc': np.array([1.0, 1.2, 1.2, 1.5, 0.9])}, np.array([1.6, 1.2, 1.5, 1.1, 1.0])),
    ({'ObgTgcc': np.array([1.6, 1.5, 1.9, 0.9, 1.7]), 'pn': np.array([2.3, 2.0, 2.3, 2.0, 1.6]), 'd': np.array([14.8, 16.1, 8.0, 12.8, 11.9]), 'nde': np.array([1.2, 0.8, 1.0, 1.2, 0.9]), 'tvdbgl': np.array([5500.7, 2626.6, 1901.7, 3718.9, 5249.8]), 'D0': np.array([0.8, 0.7, 1.0, 0.9, 1.4]), 'Dxc': np.array([1.5, 0.6, 1.2, 0.6, 0.8])}, np.array([1.6, 1.5, 1.9, 0.9, 1.7])),
    ({'ObgTgcc': np.array([2.0, 0.9, 2.1, 2.2, 1.3]), 'pn': np.array([1.7, 1.7, 1.6, 1.9, 1.7]), 'd': np.array([18.3, 2.8, 3.8, 14.8, 3.0]), 'nde': np.array([0.7, 0.9, 0.7, 0.9, 1.2]), 'tvdbgl': np.array([4461.6, 3771.0, 4078.4, 2845.2, 3365.3]), 'D0': np.array([1.4, 1.0, 1.1, 1.0, 1.1]), 'Dxc': np.array([1.2, 0.6, 1.0, 1.2, 0.9])}, np.array([2.0, 0.9, 2.1, 2.2, 1.3])),
])
def test_get_PPgrad_Dxc_gcc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_PPgrad_Dxc_gcc_vec."""
    actual_vector_output = stresslog.get_PPgrad_Dxc_gcc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ObgTgcc': np.array([2.3, 0.9, 1.0, 1.8, 2.2]), 'pn': np.array([1.9, 1.0, 2.4, 1.0, 1.6]), 'be': np.array([7.1, 2.4, 14.0, 6.9, 3.6]), 'ne': np.array([1.1, 0.7, 0.9, 0.5, 0.9]), 'tvdbgl': np.array([3612.0, 2418.1, 4090.6, 3831.7, 3556.9]), 'res0': np.array([1.5, 1.1, 1.4, 1.4, 1.3]), 'resdeep': np.array([1.4, 1.2, 1.4, 0.8, 1.4])}, np.array([2.3, 0.9, 1.0, 1.8, 2.2])),
    ({'ObgTgcc': np.array([1.9, 1.0, 1.3, 1.9, 1.9]), 'pn': np.array([2.0, 2.3, 1.0, 2.2, 2.1]), 'be': np.array([3.6, 9.6, 8.7, 3.4, 11.6]), 'ne': np.array([0.7, 0.6, 0.6, 1.4, 0.7]), 'tvdbgl': np.array([3449.5, 4893.6, 3444.0, 5924.6, 4828.0]), 'res0': np.array([0.6, 0.9, 0.8, 1.2, 0.6]), 'resdeep': np.array([0.8, 0.5, 0.8, 0.5, 0.5])}, np.array([1.9, 1.0, 1.3, 1.9, 1.9])),
    ({'ObgTgcc': np.array([1.1, 1.9, 2.3, 1.9, 1.5]), 'pn': np.array([2.1, 1.9, 1.9, 1.3, 1.9]), 'be': np.array([14.0, 7.3, 10.2, 3.5, 7.6]), 'ne': np.array([0.8, 1.3, 1.3, 0.7, 0.8]), 'tvdbgl': np.array([3629.5, 3097.4, 1480.0, 1214.4, 3849.6]), 'res0': np.array([0.9, 1.4, 1.0, 1.1, 1.1]), 'resdeep': np.array([1.0, 1.5, 1.3, 0.8, 1.2])}, np.array([1.1, 1.9, 2.3, 1.9, 1.5])),
    ({'ObgTgcc': np.array([1.1, 1.5, 2.2, 0.9, 1.4]), 'pn': np.array([0.9, 2.0, 1.8, 2.2, 0.9]), 'be': np.array([15.4, 19.4, 16.9, 14.4, 13.3]), 'ne': np.array([0.8, 1.4, 0.8, 0.6, 0.6]), 'tvdbgl': np.array([5390.9, 2286.1, 1114.6, 2558.6, 4931.7]), 'res0': np.array([1.5, 0.8, 1.0, 0.5, 0.8]), 'resdeep': np.array([0.9, 1.0, 0.5, 1.1, 1.5])}, np.array([1.1, 1.5, 2.2, 0.9, 1.4])),
    ({'ObgTgcc': np.array([1.3, 1.5, 2.4, 1.0, 1.4]), 'pn': np.array([1.4, 1.1, 2.0, 1.4, 0.9]), 'be': np.array([15.5, 14.6, 16.8, 4.6, 17.3]), 'ne': np.array([0.5, 1.3, 1.1, 0.9, 1.2]), 'tvdbgl': np.array([4974.4, 5624.8, 3384.6, 1541.4, 3921.0]), 'res0': np.array([1.4, 1.4, 1.5, 1.3, 0.6]), 'resdeep': np.array([0.8, 1.0, 0.9, 1.2, 1.5])}, np.array([1.3, 1.5, 2.4, 1.0, 1.4])),
])
def test_get_PPgrad_Eaton_gcc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_PPgrad_Eaton_gcc_vec."""
    actual_vector_output = stresslog.get_PPgrad_Eaton_gcc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ObgTgcc': np.array([1.3, 1.2, 1.7, 1.0, 1.6]), 'pn': np.array([1.1, 2.1, 1.4, 2.3, 2.1]), 'b': np.array([16.2, 18.2, 3.5, 11.5, 17.6]), 'tvdbgl': np.array([4137.5, 3234.6, 4903.0, 4771.5, 3845.3]), 'c': np.array([7.7, 11.7, 19.0, 7.6, 4.1]), 'mudline': np.array([211.6, 210.0, 194.9, 219.2, 215.8]), 'matrick': np.array([52.3, 51.9, 60.0, 53.5, 53.1]), 'deltmu0': np.array([205.3, 112.0, 96.2, 207.9, 167.7]), 'dalm': np.array([109.4, 137.8, 112.0, 128.8, 212.2])}, np.array([1.3, 1.2, 1.7, 1.0, 1.6])),
    ({'ObgTgcc': np.array([2.1, 0.9, 2.3, 1.2, 1.0]), 'pn': np.array([2.2, 1.2, 1.9, 1.9, 1.7]), 'b': np.array([19.2, 15.6, 11.7, 10.9, 18.6]), 'tvdbgl': np.array([5510.1, 3750.5, 2636.8, 5709.6, 2440.5]), 'c': np.array([13.8, 5.3, 12.8, 5.5, 8.5]), 'mudline': np.array([192.5, 211.5, 182.2, 189.2, 186.4]), 'matrick': np.array([51.0, 67.7, 56.2, 52.5, 59.2]), 'deltmu0': np.array([63.7, 204.9, 99.5, 96.7, 169.9]), 'dalm': np.array([126.6, 129.6, 72.8, 68.8, 138.7])}, np.array([2.1, 0.9, 2.3, 1.2, 1.0])),
    ({'ObgTgcc': np.array([2.2, 1.6, 2.1, 1.4, 1.4]), 'pn': np.array([1.8, 1.6, 1.1, 1.6, 1.1]), 'b': np.array([16.6, 13.6, 14.0, 16.3, 17.5]), 'tvdbgl': np.array([5616.8, 4398.6, 1676.0, 2054.2, 2863.2]), 'c': np.array([3.8, 11.7, 13.5, 6.6, 10.5]), 'mudline': np.array([198.4, 195.6, 210.9, 180.5, 196.2]), 'matrick': np.array([59.7, 65.6, 55.6, 66.9, 64.9]), 'deltmu0': np.array([76.5, 188.6, 138.9, 110.6, 175.2]), 'dalm': np.array([132.8, 173.2, 142.4, 151.5, 142.3])}, np.array([2.2, 1.6, 2.1, 1.4, 1.4])),
    ({'ObgTgcc': np.array([1.1, 1.2, 1.7, 2.4, 2.3]), 'pn': np.array([1.0, 1.5, 1.2, 1.1, 2.3]), 'b': np.array([2.5, 19.4, 17.4, 16.2, 12.9]), 'tvdbgl': np.array([2900.8, 4294.3, 1187.5, 3727.2, 3851.8]), 'c': np.array([15.1, 19.1, 1.5, 12.8, 15.3]), 'mudline': np.array([208.5, 209.3, 218.5, 201.3, 188.0]), 'matrick': np.array([53.7, 56.7, 66.6, 57.0, 53.6]), 'deltmu0': np.array([94.8, 94.2, 71.5, 123.2, 183.3]), 'dalm': np.array([192.0, 206.4, 214.1, 67.7, 63.1])}, np.array([1.1, 1.2, 1.7, 2.4, 2.3])),
    ({'ObgTgcc': np.array([1.1, 1.6, 2.2, 2.2, 2.1]), 'pn': np.array([1.7, 2.3, 1.2, 1.9, 1.0]), 'b': np.array([6.5, 18.4, 9.2, 15.5, 19.1]), 'tvdbgl': np.array([5359.5, 4377.9, 2188.6, 4024.0, 4627.4]), 'c': np.array([14.3, 7.2, 8.3, 10.7, 13.2]), 'mudline': np.array([181.8, 180.8, 184.4, 189.9, 215.4]), 'matrick': np.array([65.2, 57.7, 68.0, 60.5, 55.6]), 'deltmu0': np.array([168.0, 62.7, 103.7, 157.6, 127.8]), 'dalm': np.array([78.9, 120.1, 117.9, 147.4, 98.4])}, np.array([1.1, 1.6, 2.2, 2.2, 2.1])),
])
def test_get_PP_grad_Zhang_gcc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_PP_grad_Zhang_gcc_vec."""
    actual_vector_output = stresslog.get_PP_grad_Zhang_gcc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'nu2': np.array([0.2, 0.3, 0.4, 0.4, 0.2]), 'ObgTppg': np.array([9.2, 8.3, 19.1, 11.2, 10.9]), 'biot': np.array([1.2, 1.2, 1.2, 1.3, 0.6]), 'ppgZhang': np.array([18.2, 19.6, 17.1, 16.2, 19.9]), 'tecB': np.array([0.6, 1.1, 1.1, 0.7, 0.7])}, np.array([24.2, 26.1, 40.6, 22.3, 19.3])),
    ({'nu2': np.array([0.4, 0.3, 0.4, 0.4, 0.4]), 'ObgTppg': np.array([13.0, 8.5, 9.7, 19.3, 14.3]), 'biot': np.array([0.8, 1.3, 1.2, 1.5, 0.8]), 'ppgZhang': np.array([12.5, 10.2, 17.7, 11.0, 12.3]), 'tecB': np.array([1.3, 1.5, 1.5, 1.0, 1.2])}, np.array([28.9, 24.0, 28.1, 37.7, 30.0])),
    ({'nu2': np.array([0.2, 0.3, 0.3, 0.2, 0.4]), 'ObgTppg': np.array([15.4, 18.9, 17.5, 9.7, 10.0]), 'biot': np.array([0.8, 1.4, 0.7, 1.5, 1.1]), 'ppgZhang': np.array([18.2, 15.0, 12.2, 16.8, 19.7]), 'tecB': np.array([1.0, 1.1, 1.1, 0.9, 1.0])}, np.array([30.2, 40.9, 31.6, 30.1, 23.9])),
    ({'nu2': np.array([0.4, 0.2, 0.3, 0.2, 0.3]), 'ObgTppg': np.array([19.8, 9.8, 19.9, 15.6, 15.5]), 'biot': np.array([1.2, 0.8, 1.2, 1.1, 1.4]), 'ppgZhang': np.array([10.6, 14.6, 18.0, 9.9, 8.8]), 'tecB': np.array([0.5, 0.9, 0.7, 1.3, 0.5])}, np.array([27.3, 20.0, 34.8, 32.3, 21.4])),
    ({'nu2': np.array([0.2, 0.4, 0.3, 0.4, 0.3]), 'ObgTppg': np.array([16.7, 16.9, 15.3, 9.2, 7.8]), 'biot': np.array([1.5, 1.2, 0.9, 0.9, 0.5]), 'ppgZhang': np.array([13.3, 17.4, 8.2, 13.5, 17.6]), 'tecB': np.array([0.5, 0.6, 0.7, 0.6, 1.1])}, np.array([27.5, 28.4, 21.5, 15.7, 17.0])),
])
def test_get_Shmin_grad_Daine_ppg_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_Shmin_grad_Daine_ppg_vec."""
    actual_vector_output = stresslog.get_Shmin_grad_Daine_ppg_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)

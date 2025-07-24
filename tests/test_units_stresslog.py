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
 6000.0]), 'porepressures': np.array([3200.0, 3464.8, 6070.9, 8790.7, 10603.2, 12037.5, 13928.6, 15543.7,
 18926.5, 21039.8])}, 2.3),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1108.1, 2316.3, 6557.0, 6792.8, 11063.0, 12879.0, 15308.5, 17338.0,
 16869.5, 20910.7])}, 2.2),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([2320.9, 2439.8, 5828.6, 8851.9, 9860.2, 11649.9, 11761.3, 16160.0,
 18580.8, 20585.7])}, 2.2),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([612.8, 3622.1, 5396.3, 6268.1, 9224.5, 12296.7, 14860.8, 15723.8, 18909.8,
 20765.6])}, 2.2),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([2700.1, 3690.9, 6820.5, 8388.6, 10468.3, 10381.0, 12522.3, 14158.6,
 17927.8, 19984.7])}, 2.1),
])
def test_compute_optimal_gradient(inputs, expected_output):
    """Regression test for stresslog.compute_optimal_gradient."""
    actual_output = stresslog.compute_optimal_gradient(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([762.0, 3489.9, 6230.9, 8255.3, 10901.9, 11269.8, 13732.6, 14388.7,
 18304.8, 20811.5]), 'gradient': 2.0}, 298.0),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1842.4, 5264.6, 7362.5, 7268.2, 9127.8, 13844.7, 14178.9, 16879.8,
 18745.5, 21383.4]), 'gradient': 1.8}, 1022.4),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([574.4, 3440.4, 6978.3, 7451.3, 9908.2, 11794.8, 12343.0, 15659.8, 17392.7,
 20786.9]), 'gradient': 1.9}, 430.7),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1552.4, 2846.5, 6206.2, 7006.4, 10181.3, 13389.2, 14262.5, 15051.4,
 17966.6, 18823.8]), 'gradient': 1.9}, 466.0),
    ({'tvds': np.array([1000.0, 1555.6, 2111.1, 2666.7, 3222.2, 3777.8, 4333.3, 4888.9, 5444.4,
 6000.0]), 'porepressures': np.array([1752.2, 4519.7, 6064.5, 7443.8, 10051.4, 12863.0, 13134.7, 14897.9,
 17757.4, 19203.4]), 'gradient': 1.2}, 2803.0),
])
def test_compute_optimal_offset(inputs, expected_output):
    """Regression test for stresslog.compute_optimal_offset."""
    actual_output = stresslog.compute_optimal_offset(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 178.3, 'sy': 143.2, 'sz': 52.9, 'alpha': 102.4, 'beta': -73.5, 'gamma': 102.0, 'azim': 16.2, 'inc': -174.2}, np.array([[54.9, 12.8, 6.3],
 [12.8, 144.4, 9.0],
 [6.3, 9.0, 175.1]])),
    ({'sx': 157.7, 'sy': 86.7, 'sz': 45.1, 'alpha': 71.2, 'beta': 152.5, 'gamma': 147.8, 'azim': -87.0, 'inc': -77.1}, np.array([[61.2, -6.2, -30.6],
 [-6.2, 89.3, -26.7],
 [-30.6, -26.7, 139.0]])),
    ({'sx': 110.3, 'sy': 47.3, 'sz': 38.2, 'alpha': -1.9, 'beta': -145.1, 'gamma': -92.3, 'azim': 153.3, 'inc': 71.7}, np.array([[53.3, -5.9, 17.6],
 [-5.9, 47.1, -22.5],
 [17.6, -22.5, 95.4]])),
    ({'sx': 172.1, 'sy': 89.9, 'sz': 17.7, 'alpha': -15.7, 'beta': -59.1, 'gamma': -156.2, 'azim': 41.0, 'inc': -46.8}, np.array([[142.5, -23.1, -37.4],
 [-23.1, 91.4, 41.1],
 [-37.4, 41.1, 45.8]])),
    ({'sx': 86.7, 'sy': 25.6, 'sz': 21.5, 'alpha': -1.6, 'beta': 94.0, 'gamma': 141.1, 'azim': -101.6, 'inc': 139.2}, np.array([[49.3, -4.3, -31.1],
 [-4.3, 24.1, 2.0],
 [-31.1, 2.0, 60.3]])),
])
def test_getAlignedStress(inputs, expected_output):
    """Regression test for stresslog.getAlignedStress."""
    actual_output = stresslog.getAlignedStress(**inputs)
    np.testing.assert_allclose(actual_output, expected_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'alpha': -179.0, 'strike': -96.7, 'dip': 41.4}, (41.4, 0.7)),
    ({'alpha': 76.0, 'strike': 142.4, 'dip': 76.4}, (-58.9, 63.0)),
    ({'alpha': -35.8, 'strike': 77.7, 'dip': 84.7}, (76.9, 65.9)),
    ({'alpha': 47.5, 'strike': 179.7, 'dip': 27.7}, (19.4, 20.1)),
    ({'alpha': 114.1, 'strike': -8.3, 'dip': 83.8}, (75.1, -65.2)),
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
    ({'p': 3008.4, 't': 113.5}, 0.1),
    ({'p': 9546.4, 't': 175.8}, 0.3),
    ({'p': 11668.8, 't': 129.8}, 0.4),
    ({'p': 4132.3, 't': 72.7}, 0.2),
    ({'p': 5800.7, 't': 108.1}, 0.2),
])
def test_getGasDensity(inputs, expected_output):
    """Regression test for stresslog.getGasDensity."""
    actual_output = stresslog.getGasDensity(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'tvd': 5892.7, 'gradient': 1.6}, 13423.6),
    ({'tvd': 3533.3, 'gradient': 2.0}, 10061.1),
    ({'tvd': 3224.7, 'gradient': 2.0}, 9182.4),
    ({'tvd': 5147.0, 'gradient': 1.9}, 13923.3),
    ({'tvd': 2216.9, 'gradient': 1.7}, 5365.8),
])
def test_getHydrostaticPsi(inputs, expected_output):
    """Regression test for stresslog.getHydrostaticPsi."""
    actual_output = stresslog.getHydrostaticPsi(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 37.9, 's2': 119.2, 's3': 29.1, 'alpha': 156.3, 'beta': -168.4, 'gamma': -105.5}, np.array([[0.4, 0.9, -0.1],
 [0.9, -0.4, 0.3],
 [-0.3, 0.2, 0.9]])),
    ({'s1': 126.1, 's2': 86.1, 's3': 20.7, 'alpha': 126.5, 'beta': 135.1, 'gamma': 153.1}, np.array([[0.7, 0.5, -0.4],
 [-0.2, 0.8, 0.6],
 [0.6, -0.3, 0.7]])),
    ({'s1': 43.3, 's2': 126.1, 's3': 62.3, 'alpha': -85.4, 'beta': -12.6, 'gamma': -133.4}, np.array([[0.1, -0.7, -0.7],
 [-1.0, 0.1, -0.2],
 [0.2, 0.7, -0.7]])),
    ({'s1': 149.8, 's2': 37.0, 's3': 57.9, 'alpha': -48.1, 'beta': 155.5, 'gamma': 145.2}, np.array([[0.5, 0.7, -0.6],
 [0.7, 0.1, 0.7],
 [0.5, -0.7, -0.4]])),
    ({'s1': 237.4, 's2': 66.2, 's3': 21.5, 'alpha': 157.7, 'beta': -127.4, 'gamma': 134.1}, np.array([[-0.2, 0.8, -0.6],
 [0.9, 0.4, 0.2],
 [0.4, -0.4, -0.8]])),
])
def test_getOrit(inputs, expected_output):
    """Regression test for stresslog.getOrit."""
    actual_output = stresslog.getOrit(**inputs)
    np.testing.assert_allclose(actual_output, expected_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'alpha': 55.1, 'beta': 64.7, 'gamma': 81.9}, np.array([[0.2, 0.4, -0.9],
 [0.4, 0.8, 0.4],
 [0.9, -0.5, 0.1]])),
    ({'alpha': 49.5, 'beta': -4.7, 'gamma': 63.8}, np.array([[0.6, 0.8, 0.1],
 [-0.4, 0.2, 0.9],
 [0.7, -0.6, 0.4]])),
    ({'alpha': 32.5, 'beta': -51.4, 'gamma': -92.7}, np.array([[0.5, 0.3, 0.8],
 [0.7, 0.4, -0.6],
 [-0.5, 0.9, -0.0]])),
    ({'alpha': 69.0, 'beta': -148.6, 'gamma': -73.2}, np.array([[-0.3, -0.8, 0.5],
 [-0.1, 0.6, 0.8],
 [-0.9, 0.2, -0.2]])),
    ({'alpha': 7.9, 'beta': 47.7, 'gamma': -76.5}, np.array([[0.7, 0.1, -0.7],
 [-0.7, 0.1, -0.7],
 [0.0, 1.0, 0.2]])),
])
def test_getRota(inputs, expected_output):
    """Regression test for stresslog.getRota."""
    actual_output = stresslog.getRota(**inputs)
    np.testing.assert_allclose(actual_output, expected_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'Sv': 234.0, 'Pp': 50.8, 'bhp': 124.8, 'shmin': 96.2, 'UCS': 66.7, 'phi': 0.5, 'mu': 0.7, 'nu': 0.2, 'PhiBr': 22.0}, [100.5, 234.0, 167.2]),
    ({'Sv': 34.6, 'Pp': 14.0, 'bhp': 79.7, 'shmin': 44.9, 'UCS': 27.7, 'phi': 0.5, 'mu': 0.6, 'nu': 0.2, 'PhiBr': 32.4}, [44.9, 78.3, 61.6]),
    ({'Sv': 135.6, 'Pp': 34.5, 'bhp': 123.4, 'shmin': 40.1, 'UCS': 24.2, 'phi': 0.4, 'mu': 0.6, 'nu': 0.2, 'PhiBr': 24.2}, [66.9, 135.6, 101.3]),
    ({'Sv': 65.9, 'Pp': 66.6, 'bhp': 95.7, 'shmin': 21.8, 'UCS': 46.5, 'phi': 0.4, 'mu': 0.7, 'nu': 0.3, 'PhiBr': 27.4}, [64.0, 65.9, 65.0]),
    ({'Sv': 143.9, 'Pp': 61.5, 'bhp': 49.3, 'shmin': 29.6, 'UCS': 34.0, 'phi': 0.4, 'mu': 0.9, 'nu': 0.4, 'PhiBr': 18.6}, [77.8, 143.9, 110.9]),
])
def test_getSP(inputs, expected_output):
    """Regression test for stresslog.getSP."""
    actual_output = stresslog.getSP(**inputs)
    assert isinstance(actual_output, list)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 217.5, 's2': 70.6, 's3': 28.7, 'alpha': -101.7, 'beta': 64.0, 'gamma': -110.7, 'azim': -10.3, 'inc': -48.3, 'theta': 69.6, 'deltaP': 62.5, 'Pp': 109.6, 'nu': 0.2}, (-108.4, -15.3, 109.2, 56.8, -180.6, 33.4, [88.1, 88.2, -1.9, 88.9, 177.9, 55.9])),
    ({'s1': 139.0, 's2': 88.7, 's3': 65.9, 'alpha': -97.0, 'beta': -173.6, 'gamma': -142.0, 'azim': -87.9, 'inc': 95.9, 'theta': 149.0, 'deltaP': 104.3, 'Pp': 59.7, 'nu': 0.2}, (-22.5, 88.3, -13.3, 89.9, -24.1, -6.8, [87.4, 88.3, -177.5, 93.0, 176.5, 31.7])),
    ({'s1': 187.0, 's2': 94.7, 's3': 49.9, 'alpha': 85.3, 'beta': -5.4, 'gamma': -7.4, 'azim': 173.8, 'inc': -62.5, 'theta': -34.3, 'deltaP': 98.9, 'Pp': 23.3, 'nu': 0.2}, (143.2, 94.3, 21.6, 151.4, 86.2, 69.3, [89.9, 91.5, -0.1, 90.1, 178.5, -93.8])),
    ({'s1': 220.4, 's2': 59.0, 's3': 60.4, 'alpha': -143.2, 'beta': 0.6, 'gamma': 118.0, 'azim': -73.1, 'inc': -73.5, 'theta': 82.1, 'deltaP': 80.2, 'Pp': 56.8, 'nu': 0.4}, (-188.3, -191.1, -24.0, -165.6, -213.8, -46.6, [-2.1, 87.5, 87.9, 90.0, 177.5, -1.8])),
    ({'s1': 120.3, 's2': 81.9, 's3': 26.5, 'alpha': -6.2, 'beta': -179.6, 'gamma': -123.9, 'azim': 79.6, 'inc': -158.6, 'theta': -49.9, 'deltaP': 121.4, 'Pp': 118.8, 'nu': 0.4}, (-176.1, -82.3, 35.1, -70.7, -187.7, 18.4, [-87.8, 89.9, 2.2, 93.1, 176.9, -175.9])),
])
def test_getSigmaTT(inputs, expected_output):
    """Regression test for stresslog.getSigmaTT."""
    actual_output = stresslog.getSigmaTT(**inputs)
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
    ({'sx': 58.8, 'sy': 73.7, 'sz': 20.5, 'alpha': 38.6, 'beta': 137.1, 'gamma': -101.7}, (np.array([41.3, 22.1, -0.1]), np.array([22.1, 46.0, -10.0]), np.array([-0.1, -10.0, 65.6]))),
    ({'sx': 160.8, 'sy': 128.5, 'sz': 41.4, 'alpha': -29.2, 'beta': 64.5, 'gamma': 64.3}, (np.array([132.2, -10.9, -9.4]), np.array([-10.9, 46.7, 22.0]), np.array([-9.4, 22.0, 151.8]))),
    ({'sx': 79.3, 'sy': 52.0, 'sz': 59.9, 'alpha': -174.5, 'beta': -45.0, 'gamma': -57.7}, (np.array([67.2, -1.6, -12.2]), np.array([-1.6, 57.2, -3.7]), np.array([-12.2, -3.7, 66.8]))),
    ({'sx': 221.3, 'sy': 85.0, 'sz': 29.7, 'alpha': 96.3, 'beta': -37.0, 'gamma': 41.8}, (np.array([58.0, 5.2, -30.6]), np.array([5.2, 163.2, 77.4]), np.array([-30.6, 77.4, 114.8]))),
    ({'sx': 146.6, 'sy': 37.1, 'sz': 32.4, 'alpha': 67.6, 'beta': -61.0, 'gamma': -126.2}, (np.array([39.4, 11.1, 17.0]), np.array([11.1, 56.2, 44.0]), np.array([17.0, 44.0, 120.5]))),
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
    ({'alpha': -89.5, 'beta': 81.7, 'gamma': 73.8}, (16.5, 87.7, -163.5)),
    ({'alpha': 50.6, 'beta': -108.4, 'gamma': -99.0}, (312.1, 87.2, 132.1)),
    ({'alpha': 51.7, 'beta': 82.3, 'gamma': 135.6}, (96.4, 95.5, -83.6)),
    ({'alpha': 44.9, 'beta': 63.5, 'gamma': 24.2}, (198.2, 66.0, 18.2)),
    ({'alpha': 39.2, 'beta': -34.5, 'gamma': -142.3}, (273.0, 130.7, 93.0)),
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
    ({'sx': 41.6, 'sy': 118.2, 'sz': 40.0, 'alpha': -171.8, 'beta': -48.6, 'gamma': 178.6}, (40.920681322278156+0j)),
    ({'sx': 218.0, 'sy': 31.2, 'sz': 37.3, 'alpha': -132.7, 'beta': -147.8, 'gamma': 145.1}, (87.18122918937385+103.46921618017099j)),
    ({'sx': 190.5, 'sy': 60.4, 'sz': 61.7, 'alpha': 112.5, 'beta': -101.6, 'gamma': -118.5}, (185.25172116678672+64.90176336601297j)),
    ({'sx': 95.9, 'sy': 66.8, 'sz': 68.6, 'alpha': 7.7, 'beta': -16.4, 'gamma': -2.0}, (70.7742482748011+93.24496611982863j)),
    ({'sx': 154.4, 'sy': 92.2, 'sz': 34.0, 'alpha': 24.3, 'beta': -164.9, 'gamma': 85.1}, (96.02524001099222+0j)),
])
def test_getVertical(inputs, expected_output):
    """Regression test for stresslog.getVertical."""
    actual_output = stresslog.getVertical(**inputs)
    assert actual_output.real == pytest.approx(expected_output.real, rel=RTOL, abs=ATOL)
    assert actual_output.imag == pytest.approx(expected_output.imag, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ROP': 331.5, 'RPM': 394.7, 'WOB': 384.6, 'BTDIA': 16.0, 'ECD': 1.3, 'pn': 1.8}, 0.7),
    ({'ROP': 384.6, 'RPM': 311.1, 'WOB': 333.8, 'BTDIA': 14.7, 'ECD': 1.5, 'pn': 2.2}, 0.7),
    ({'ROP': 397.9, 'RPM': 413.8, 'WOB': 124.8, 'BTDIA': 14.6, 'ECD': 2.2, 'pn': 2.0}, 0.4),
    ({'ROP': 172.4, 'RPM': 229.3, 'WOB': 312.6, 'BTDIA': 2.2, 'ECD': 1.9, 'pn': 1.9}, 0.7),
    ({'ROP': 297.2, 'RPM': 2.6, 'WOB': 133.7, 'BTDIA': 17.5, 'ECD': 2.2, 'pn': 1.1}, np.nan),
])
def test_get_Dxc(inputs, expected_output):
    """Regression test for stresslog.get_Dxc."""
    actual_output = stresslog.get_Dxc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ObgTgcc': 1.8, 'pn': 1.4, 'd': 2.1, 'nde': 1.1, 'tvdbgl': 4847.2, 'D0': 0.9, 'Dxc': 1.3}, 1.8),
    ({'ObgTgcc': 1.7, 'pn': 1.6, 'd': 1.3, 'nde': 0.8, 'tvdbgl': 2744.8, 'D0': 1.1, 'Dxc': 1.3}, 1.7),
    ({'ObgTgcc': 1.0, 'pn': 2.1, 'd': 19.7, 'nde': 1.4, 'tvdbgl': 4742.0, 'D0': 0.6, 'Dxc': 1.1}, 1.0),
    ({'ObgTgcc': 1.9, 'pn': 1.5, 'd': 19.5, 'nde': 1.4, 'tvdbgl': 4568.9, 'D0': 1.3, 'Dxc': 0.7}, 1.9),
    ({'ObgTgcc': 1.9, 'pn': 1.3, 'd': 10.0, 'nde': 1.1, 'tvdbgl': 4937.0, 'D0': 0.5, 'Dxc': 0.6}, 1.9),
])
def test_get_PPgrad_Dxc_gcc(inputs, expected_output):
    """Regression test for stresslog.get_PPgrad_Dxc_gcc."""
    actual_output = stresslog.get_PPgrad_Dxc_gcc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ObgTgcc': 1.8, 'pn': 1.8, 'be': 3.6, 'ne': 0.9, 'tvdbgl': 5897.0, 'res0': 0.7, 'resdeep': 1.3}, 1.8),
    ({'ObgTgcc': 1.0, 'pn': 1.8, 'be': 6.8, 'ne': 1.3, 'tvdbgl': 5457.2, 'res0': 1.5, 'resdeep': 0.6}, 1.0),
    ({'ObgTgcc': 1.5, 'pn': 1.6, 'be': 9.4, 'ne': 0.7, 'tvdbgl': 5970.3, 'res0': 0.8, 'resdeep': 0.9}, 1.5),
    ({'ObgTgcc': 1.3, 'pn': 2.3, 'be': 19.4, 'ne': 0.9, 'tvdbgl': 4155.2, 'res0': 0.8, 'resdeep': 1.0}, 1.3),
    ({'ObgTgcc': 1.1, 'pn': 2.1, 'be': 4.7, 'ne': 1.2, 'tvdbgl': 3490.3, 'res0': 1.0, 'resdeep': 0.5}, 1.1),
])
def test_get_PPgrad_Eaton_gcc(inputs, expected_output):
    """Regression test for stresslog.get_PPgrad_Eaton_gcc."""
    actual_output = stresslog.get_PPgrad_Eaton_gcc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ObgTgcc': 2.0, 'pn': 1.8, 'b': 7.6, 'tvdbgl': 3186.8, 'c': 9.6, 'mudline': 213.9, 'matrick': 58.2, 'deltmu0': 128.4, 'dalm': 64.4}, 2.0),
    ({'ObgTgcc': 1.6, 'pn': 1.9, 'b': 4.0, 'tvdbgl': 4694.1, 'c': 11.2, 'mudline': 211.5, 'matrick': 58.0, 'deltmu0': 157.5, 'dalm': 112.7}, 1.6),
    ({'ObgTgcc': 1.3, 'pn': 2.2, 'b': 15.0, 'tvdbgl': 2435.5, 'c': 11.3, 'mudline': 184.2, 'matrick': 54.2, 'deltmu0': 57.1, 'dalm': 162.7}, 1.3),
    ({'ObgTgcc': 2.1, 'pn': 2.0, 'b': 7.5, 'tvdbgl': 2613.0, 'c': 2.9, 'mudline': 215.6, 'matrick': 50.1, 'deltmu0': 162.2, 'dalm': 117.0}, 2.1),
    ({'ObgTgcc': 1.5, 'pn': 1.5, 'b': 19.7, 'tvdbgl': 5463.6, 'c': 12.9, 'mudline': 195.9, 'matrick': 58.3, 'deltmu0': 154.2, 'dalm': 84.7}, 1.5),
])
def test_get_PPgrad_Zhang_gcc(inputs, expected_output):
    """Regression test for stresslog.get_PPgrad_Zhang_gcc."""
    actual_output = stresslog.get_PPgrad_Zhang_gcc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'nu2': 0.4, 'ObgTppg': 19.3, 'biot': 0.8, 'ppgZhang': 18.9, 'tecB': 0.9}, 35.3),
    ({'nu2': 0.4, 'ObgTppg': 8.3, 'biot': 1.2, 'ppgZhang': 9.4, 'tecB': 0.6}, 14.3),
    ({'nu2': 0.4, 'ObgTppg': 9.6, 'biot': 1.0, 'ppgZhang': 17.4, 'tecB': 0.5}, 17.0),
    ({'nu2': 0.2, 'ObgTppg': 8.2, 'biot': 1.4, 'ppgZhang': 9.6, 'tecB': 1.5}, 24.4),
    ({'nu2': 0.4, 'ObgTppg': 10.0, 'biot': 1.0, 'ppgZhang': 9.2, 'tecB': 1.4}, 23.7),
])
def test_get_Shmin_grad_Daine_ppg(inputs, expected_output):
    """Regression test for stresslog.get_Shmin_grad_Daine_ppg."""
    actual_output = stresslog.get_Shmin_grad_Daine_ppg(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 98.5, 'sy': 96.6, 'sz': 32.2, 'txy': 1.7, 'tyz': 2.3, 'tzx': 15.3, 'phi': 0.6, 'cohesion': 6.1, 'pp': 92.5}, 48.7),
    ({'sx': 64.7, 'sy': 42.1, 'sz': 22.7, 'txy': 17.3, 'tyz': 18.4, 'tzx': 15.6, 'phi': 0.5, 'cohesion': 2.5, 'pp': 96.7}, 39.9),
    ({'sx': 166.0, 'sy': 50.9, 'sz': 11.6, 'txy': 12.9, 'tyz': 5.8, 'tzx': 10.5, 'phi': 0.7, 'cohesion': 14.1, 'pp': 103.5}, 62.8),
    ({'sx': 191.1, 'sy': 133.9, 'sz': 13.6, 'txy': 7.3, 'tyz': 11.3, 'tzx': 8.0, 'phi': 0.6, 'cohesion': 12.4, 'pp': 52.9}, 48.7),
    ({'sx': 217.8, 'sy': 148.0, 'sz': 57.4, 'txy': 2.3, 'tyz': 10.1, 'tzx': 15.1, 'phi': 0.4, 'cohesion': 4.8, 'pp': 20.1}, 34.3),
])
def test_lade(inputs, expected_output):
    """Regression test for stresslog.lade."""
    actual_output = stresslog.lade(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 172.6, 'sy': 97.0, 'sz': 53.5, 'txy': 11.0, 'tyz': 12.7, 'tzx': 7.7, 'phi': 0.6, 'cohesion': 7.9, 'pp': 136.4}, 77.1),
    ({'sx': 113.4, 'sy': 112.9, 'sz': 33.4, 'txy': 16.1, 'tyz': 9.3, 'tzx': 18.1, 'phi': 0.4, 'cohesion': 9.7, 'pp': 85.2}, 66.0),
    ({'sx': 139.3, 'sy': 134.5, 'sz': 43.1, 'txy': 7.9, 'tyz': 14.2, 'tzx': 3.5, 'phi': 0.7, 'cohesion': 5.4, 'pp': 105.1}, 89.9),
    ({'sx': 55.5, 'sy': 43.6, 'sz': 15.8, 'txy': 1.4, 'tyz': 1.3, 'tzx': 9.9, 'phi': 0.8, 'cohesion': 2.1, 'pp': 133.4}, 85.1),
    ({'sx': 182.9, 'sy': 42.8, 'sz': 56.9, 'txy': 17.4, 'tyz': 5.6, 'tzx': 1.3, 'phi': 0.7, 'cohesion': 17.1, 'pp': 91.1}, 82.1),
])
def test_lade_failure(inputs, expected_output):
    """Regression test for stresslog.lade_failure."""
    actual_output = stresslog.lade_failure(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 87.8, 'sy': 64.3, 'sz': 16.9}, 22.9),
    ({'sx': 145.0, 'sy': 129.7, 'sz': 25.0}, 31.7),
    ({'sx': 113.9, 'sy': 51.7, 'sz': 57.2}, 54.7),
    ({'sx': 97.6, 'sy': 145.5, 'sz': 44.0}, 53.3),
    ({'sx': 149.5, 'sy': 126.0, 'sz': 46.3}, 53.7),
])
def test_mogi(inputs, expected_output):
    """Regression test for stresslog.mogi."""
    actual_output = stresslog.mogi(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 110.6, 's2': 132.0, 's3': 14.7}, 11.6),
    ({'s1': 186.4, 's2': 72.9, 's3': 69.0}, 73.3),
    ({'s1': 240.2, 's2': 60.5, 's3': 56.5}, 62.7),
    ({'s1': 58.4, 's2': 78.4, 's3': 11.5}, 6.9),
    ({'s1': 248.2, 's2': 49.4, 's3': 53.1}, 57.8),
])
def test_mogi_failure(inputs, expected_output):
    """Regression test for stresslog.mogi_failure."""
    actual_output = stresslog.mogi_failure(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 242.2, 's3': 17.5, 'cohesion': 1.2, 'phi': 0.6}, -38.0),
    ({'s1': 206.5, 's3': 12.8, 'cohesion': 10.6, 'phi': 0.6}, -26.2),
    ({'s1': 157.8, 's3': 22.8, 'cohesion': 14.7, 'phi': 0.5}, -11.3),
    ({'s1': 62.9, 's3': 58.9, 'cohesion': 16.8, 'phi': 0.6}, 46.3),
    ({'s1': 199.5, 's3': 44.1, 'cohesion': 8.2, 'phi': 0.5}, -12.1),
])
def test_mohr_failure(inputs, expected_output):
    """Regression test for stresslog.mohr_failure."""
    actual_output = stresslog.mohr_failure(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sigmamax': 231.8, 'sigmamin': 21.8, 'pp': 44.9, 'ucs': 84.3, 'nu': 0.4}, 344.6),
    ({'sigmamax': 181.5, 'sigmamin': 64.8, 'pp': 55.4, 'ucs': 26.7, 'nu': 0.3}, 294.9),
    ({'sigmamax': 151.4, 'sigmamin': 24.3, 'pp': 44.6, 'ucs': 53.0, 'nu': 0.4}, 217.2),
    ({'sigmamax': 185.7, 'sigmamin': 25.6, 'pp': 125.8, 'ucs': 51.6, 'nu': 0.2}, 308.4),
    ({'sigmamax': 172.8, 'sigmamin': 20.7, 'pp': 129.3, 'ucs': 73.6, 'nu': 0.4}, 228.6),
])
def test_willson_sanding_cwf(inputs, expected_output):
    """Regression test for stresslog.willson_sanding_cwf."""
    actual_output = stresslog.willson_sanding_cwf(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sigmamax': 199.2, 'sigmamin': 63.4, 'pp': 61.5, 'ucs': 92.9, 'k0': 1.2, 'nu': 0.2}, 379.4),
    ({'sigmamax': 165.2, 'sigmamin': 55.5, 'pp': 17.2, 'ucs': 62.0, 'k0': 0.7, 'nu': 0.4}, 156.4),
    ({'sigmamax': 149.4, 'sigmamin': 18.0, 'pp': 36.3, 'ucs': 67.2, 'k0': 1.2, 'nu': 0.4}, 252.6),
    ({'sigmamax': 49.4, 'sigmamin': 17.0, 'pp': 120.8, 'ucs': 98.7, 'k0': 1.3, 'nu': 0.2}, -60.4),
    ({'sigmamax': 101.2, 'sigmamin': 17.6, 'pp': 139.7, 'ucs': 29.0, 'k0': 0.9, 'nu': 0.4}, 113.6),
])
def test_zhang_sanding_cwf(inputs, expected_output):
    """Regression test for stresslog.zhang_sanding_cwf."""
    actual_output = stresslog.zhang_sanding_cwf(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)

# --- Phase 2: Vectorized Function Consistency Tests ---

@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ROP': np.array([234.8, 463.5, 397.0, 499.3, 192.2]), 'RPM': np.array([204.8, 164.9, 447.3, 478.9, 2.5]), 'WOB': np.array([275.9, 154.2, 213.6, 252.3, 165.0]), 'BTDIA': np.array([10.0, 13.2, 17.5, 1.6, 15.9]), 'ECD': np.array([1.3, 1.8, 1.6, 2.2, 1.7]), 'pn': np.array([1.2, 1.8, 2.2, 1.5, 1.8])}, np.array([0.5, 0.3, 0.7, 0.4, np.nan])),
    ({'ROP': np.array([36.3, 195.3, 189.0, 308.5, 282.6]), 'RPM': np.array([76.1, 113.5, 295.1, 257.6, 2.8]), 'WOB': np.array([406.0, 488.5, 245.2, 374.7, 239.6]), 'BTDIA': np.array([8.5, 4.3, 16.2, 7.4, 2.3]), 'ECD': np.array([1.8, 1.1, 1.6, 1.9, 2.3]), 'pn': np.array([1.9, 1.8, 1.5, 2.4, 1.9])}, np.array([0.7, 0.9, 0.5, 0.7, np.nan])),
    ({'ROP': np.array([463.9, 35.6, 208.5, 241.4, 241.7]), 'RPM': np.array([465.9, 474.6, 8.0, 397.2, 1.1]), 'WOB': np.array([172.4, 440.0, 340.0, 345.7, 473.7]), 'BTDIA': np.array([14.3, 6.9, 14.1, 5.4, 19.1]), 'ECD': np.array([1.5, 1.9, 1.6, 1.1, 1.1]), 'pn': np.array([1.2, 1.6, 0.9, 1.4, 2.0])}, np.array([0.4, 0.8, np.nan, 0.8, np.nan])),
    ({'ROP': np.array([43.8, 144.4, 262.2, 369.8, 426.1]), 'RPM': np.array([266.2, 366.6, 387.9, 259.1, 3.6]), 'WOB': np.array([30.7, 201.1, 411.1, 482.1, 312.0]), 'BTDIA': np.array([18.0, 14.2, 19.2, 14.2, 14.6]), 'ECD': np.array([1.3, 1.0, 1.1, 1.7, 1.1]), 'pn': np.array([2.3, 1.5, 1.4, 1.3, 0.9])}, np.array([1.0, 0.9, 0.7, 0.4, np.nan])),
    ({'ROP': np.array([281.7, 293.5, 353.3, 464.2, 292.7]), 'RPM': np.array([262.5, 194.0, 68.9, 10.5, 4.4]), 'WOB': np.array([120.2, 400.2, 87.5, 126.7, 151.1]), 'BTDIA': np.array([16.8, 12.5, 18.8, 16.4, 14.9]), 'ECD': np.array([2.2, 1.3, 1.0, 1.7, 1.5]), 'pn': np.array([1.3, 2.0, 1.5, 1.0, 2.3])}, np.array([0.3, 0.7, 0.4, np.nan, np.nan])),
])
def test_get_Dxc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_Dxc_vec."""
    actual_vector_output = stresslog.get_Dxc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ObgTgcc': np.array([0.9, 1.6, 1.3, 1.7, 1.5]), 'pn': np.array([1.3, 1.7, 1.5, 1.3, 2.4]), 'd': np.array([7.9, 18.2, 3.0, 10.8, 19.6]), 'nde': np.array([1.3, 1.1, 0.6, 1.2, 0.9]), 'tvdbgl': np.array([3996.9, 2878.5, 4983.0, 2618.7, 5471.6]), 'D0': np.array([0.9, 1.5, 0.7, 1.1, 1.1]), 'Dxc': np.array([0.9, 0.5, 0.6, 1.1, 0.6])}, np.array([0.9, 1.6, 1.3, 1.7, 1.5])),
    ({'ObgTgcc': np.array([2.3, 2.3, 2.1, 1.2, 2.2]), 'pn': np.array([2.0, 1.7, 2.4, 1.7, 2.1]), 'd': np.array([19.1, 10.2, 6.5, 2.8, 12.1]), 'nde': np.array([0.7, 1.1, 0.8, 1.0, 1.3]), 'tvdbgl': np.array([3801.4, 3908.7, 4461.8, 2932.1, 2085.3]), 'D0': np.array([0.5, 1.3, 0.8, 1.3, 1.0]), 'Dxc': np.array([1.3, 1.0, 1.3, 1.1, 1.1])}, np.array([2.3, 2.3, 2.1, 1.2, 2.2])),
    ({'ObgTgcc': np.array([0.9, 1.4, 1.4, 1.0, 1.1]), 'pn': np.array([2.3, 1.0, 1.8, 1.8, 1.2]), 'd': np.array([11.2, 10.4, 12.0, 11.5, 10.3]), 'nde': np.array([0.8, 1.3, 1.4, 0.9, 0.8]), 'tvdbgl': np.array([2793.0, 1429.9, 4924.7, 4739.6, 2857.0]), 'D0': np.array([1.2, 1.0, 0.9, 0.6, 1.5]), 'Dxc': np.array([0.7, 0.7, 0.6, 0.9, 1.0])}, np.array([0.9, 1.4, 1.4, 1.0, 1.1])),
    ({'ObgTgcc': np.array([1.4, 1.6, 2.0, 1.0, 1.9]), 'pn': np.array([1.2, 0.9, 1.0, 1.1, 1.4]), 'd': np.array([4.1, 18.8, 10.8, 9.2, 12.3]), 'nde': np.array([1.1, 1.4, 1.2, 0.6, 1.2]), 'tvdbgl': np.array([3151.9, 4301.9, 4934.2, 3814.8, 3430.6]), 'D0': np.array([0.7, 0.6, 1.4, 1.2, 0.6]), 'Dxc': np.array([1.1, 0.5, 1.3, 1.4, 0.5])}, np.array([1.4, 1.6, 2.0, 1.0, 1.9])),
    ({'ObgTgcc': np.array([2.1, 2.1, 1.7, 2.0, 1.3]), 'pn': np.array([2.4, 2.1, 1.6, 2.2, 0.9]), 'd': np.array([3.2, 8.3, 3.8, 15.3, 5.0]), 'nde': np.array([0.8, 0.7, 1.3, 1.5, 0.6]), 'tvdbgl': np.array([5651.0, 1317.2, 1907.3, 3904.2, 1999.5]), 'D0': np.array([1.3, 1.4, 0.6, 0.9, 1.2]), 'Dxc': np.array([0.8, 0.5, 0.6, 0.7, 1.3])}, np.array([2.1, 2.1, 1.7, 2.0, 1.3])),
])
def test_get_PPgrad_Dxc_gcc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_PPgrad_Dxc_gcc_vec."""
    actual_vector_output = stresslog.get_PPgrad_Dxc_gcc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ObgTgcc': np.array([1.3, 2.3, 1.4, 1.4, 2.2]), 'pn': np.array([2.2, 1.3, 1.4, 1.8, 2.0]), 'be': np.array([18.3, 14.6, 17.1, 12.3, 17.8]), 'ne': np.array([1.1, 0.9, 1.1, 0.7, 1.0]), 'tvdbgl': np.array([2694.7, 4282.7, 3500.9, 2198.8, 5984.5]), 'res0': np.array([1.4, 1.4, 0.6, 0.7, 1.3]), 'resdeep': np.array([1.5, 1.4, 1.4, 1.2, 0.8])}, np.array([1.3, 2.3, 1.4, 1.4, 2.2])),
    ({'ObgTgcc': np.array([1.8, 1.5, 1.4, 2.2, 2.2]), 'pn': np.array([1.3, 2.3, 1.1, 2.1, 1.9]), 'be': np.array([4.3, 8.5, 19.8, 4.2, 17.7]), 'ne': np.array([1.4, 1.2, 0.9, 0.9, 1.4]), 'tvdbgl': np.array([2163.3, 1036.4, 2573.1, 5131.3, 4502.6]), 'res0': np.array([1.3, 1.5, 0.5, 1.1, 1.1]), 'resdeep': np.array([0.5, 0.6, 0.6, 1.0, 0.7])}, np.array([1.8, 1.5, 1.4, 2.2, 2.2])),
    ({'ObgTgcc': np.array([1.8, 2.1, 2.2, 1.3, 2.1]), 'pn': np.array([2.0, 2.3, 1.0, 1.5, 2.2]), 'be': np.array([3.7, 14.5, 4.8, 9.5, 1.6]), 'ne': np.array([1.2, 0.9, 1.4, 0.7, 0.8]), 'tvdbgl': np.array([3549.0, 4699.0, 2605.3, 1767.5, 5582.3]), 'res0': np.array([0.6, 1.5, 1.1, 1.5, 0.9]), 'resdeep': np.array([0.7, 1.5, 0.9, 0.8, 1.1])}, np.array([1.8, 2.1, 2.2, 1.3, 2.1])),
    ({'ObgTgcc': np.array([1.1, 1.6, 1.4, 2.1, 1.0]), 'pn': np.array([1.7, 2.0, 2.1, 1.7, 1.9]), 'be': np.array([9.1, 1.4, 10.8, 19.4, 2.3]), 'ne': np.array([0.6, 1.0, 0.8, 1.0, 0.9]), 'tvdbgl': np.array([2222.4, 5457.0, 4485.5, 4315.2, 2110.5]), 'res0': np.array([1.1, 0.9, 0.6, 1.0, 1.0]), 'resdeep': np.array([0.8, 0.6, 1.5, 0.8, 1.1])}, np.array([1.1, 1.6, 1.4, 2.1, 1.0])),
    ({'ObgTgcc': np.array([1.5, 2.0, 1.9, 1.0, 1.6]), 'pn': np.array([1.3, 0.9, 2.2, 2.4, 1.2]), 'be': np.array([1.3, 12.9, 3.7, 9.3, 16.8]), 'ne': np.array([0.6, 1.4, 0.5, 0.5, 1.1]), 'tvdbgl': np.array([4093.4, 1515.8, 3868.9, 4349.6, 2434.4]), 'res0': np.array([0.7, 1.4, 0.8, 0.9, 1.4]), 'resdeep': np.array([0.6, 0.7, 0.6, 1.0, 0.9])}, np.array([1.5, 2.0, 1.9, 1.0, 1.6])),
])
def test_get_PPgrad_Eaton_gcc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_PPgrad_Eaton_gcc_vec."""
    actual_vector_output = stresslog.get_PPgrad_Eaton_gcc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ObgTgcc': np.array([1.1, 1.7, 1.7, 1.6, 1.4]), 'pn': np.array([2.0, 2.1, 1.9, 2.3, 1.5]), 'b': np.array([8.1, 14.1, 18.5, 13.5, 11.2]), 'tvdbgl': np.array([3848.4, 1631.0, 5401.9, 4090.6, 1601.7]), 'c': np.array([15.5, 1.4, 7.7, 10.7, 5.7]), 'mudline': np.array([185.9, 193.1, 209.0, 188.1, 191.0]), 'matrick': np.array([64.5, 69.1, 57.6, 64.1, 60.6]), 'deltmu0': np.array([124.4, 155.0, 181.4, 171.6, 180.5]), 'dalm': np.array([93.4, 148.8, 136.3, 130.9, 63.8])}, np.array([1.1, 1.7, 1.7, 1.6, 1.4])),
    ({'ObgTgcc': np.array([2.2, 0.9, 1.8, 2.2, 1.7]), 'pn': np.array([1.7, 1.8, 1.1, 1.0, 1.5]), 'b': np.array([4.3, 12.4, 12.3, 9.3, 10.1]), 'tvdbgl': np.array([2328.1, 3882.0, 4304.6, 1962.6, 5439.6]), 'c': np.array([17.1, 1.9, 6.3, 18.2, 8.5]), 'mudline': np.array([184.1, 206.6, 205.6, 202.4, 205.9]), 'matrick': np.array([64.0, 60.1, 59.0, 58.7, 57.8]), 'deltmu0': np.array([154.4, 193.9, 99.2, 63.2, 100.3]), 'dalm': np.array([158.0, 72.5, 187.4, 190.4, 69.6])}, np.array([2.2, 0.9, 1.8, 2.2, 1.7])),
    ({'ObgTgcc': np.array([0.9, 1.0, 1.0, 2.3, 1.3]), 'pn': np.array([2.1, 2.2, 1.4, 2.2, 2.0]), 'b': np.array([18.5, 11.2, 4.0, 19.4, 12.4]), 'tvdbgl': np.array([1679.5, 1026.1, 5526.3, 1927.6, 3204.1]), 'c': np.array([16.6, 7.7, 16.7, 9.7, 16.6]), 'mudline': np.array([196.8, 192.7, 205.9, 199.0, 186.6]), 'matrick': np.array([51.4, 52.5, 60.5, 51.4, 65.8]), 'deltmu0': np.array([54.1, 62.3, 148.5, 116.2, 109.6]), 'dalm': np.array([147.4, 159.8, 151.3, 145.8, 70.4])}, np.array([0.9, 1.0, 1.0, 2.3, 1.3])),
    ({'ObgTgcc': np.array([0.9, 1.4, 2.1, 2.3, 2.1]), 'pn': np.array([1.8, 1.4, 1.4, 2.3, 1.2]), 'b': np.array([6.8, 8.1, 11.5, 1.2, 12.9]), 'tvdbgl': np.array([4064.7, 2668.7, 2342.3, 4290.0, 1184.7]), 'c': np.array([1.8, 3.7, 9.9, 2.6, 7.7]), 'mudline': np.array([191.5, 191.6, 181.3, 194.7, 203.6]), 'matrick': np.array([60.4, 64.3, 65.4, 64.6, 51.7]), 'deltmu0': np.array([70.4, 121.0, 97.8, 140.3, 68.0]), 'dalm': np.array([62.8, 116.6, 130.5, 116.4, 170.4])}, np.array([0.9, 1.4, 2.1, 2.3, 2.1])),
    ({'ObgTgcc': np.array([2.0, 2.2, 1.0, 1.9, 1.9]), 'pn': np.array([1.6, 1.3, 1.3, 2.0, 2.2]), 'b': np.array([5.6, 15.5, 14.7, 7.6, 7.0]), 'tvdbgl': np.array([3981.3, 4921.5, 4314.3, 2812.6, 4984.6]), 'c': np.array([13.1, 9.2, 16.4, 14.3, 15.0]), 'mudline': np.array([184.5, 189.9, 194.2, 219.4, 185.6]), 'matrick': np.array([60.0, 62.3, 56.8, 52.4, 69.1]), 'deltmu0': np.array([78.5, 104.9, 77.5, 131.2, 98.1]), 'dalm': np.array([80.6, 110.6, 120.3, 152.1, 163.7])}, np.array([2.0, 2.2, 1.0, 1.9, 1.9])),
])
def test_get_PP_grad_Zhang_gcc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_PP_grad_Zhang_gcc_vec."""
    actual_vector_output = stresslog.get_PP_grad_Zhang_gcc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'nu2': np.array([0.3, 0.3, 0.4, 0.3, 0.3]), 'ObgTppg': np.array([17.0, 10.9, 7.9, 16.1, 13.1]), 'biot': np.array([1.0, 1.5, 1.1, 0.9, 0.5]), 'ppgZhang': np.array([13.9, 8.7, 7.5, 15.7, 10.5]), 'tecB': np.array([1.2, 0.7, 0.9, 1.1, 0.6])}, np.array([35.6, 19.8, 15.1, 32.7, 16.5])),
    ({'nu2': np.array([0.3, 0.3, 0.2, 0.3, 0.3]), 'ObgTppg': np.array([9.3, 14.8, 16.5, 12.6, 17.2]), 'biot': np.array([0.5, 0.9, 0.7, 1.1, 0.6]), 'ppgZhang': np.array([14.9, 15.8, 14.3, 11.4, 16.7]), 'tecB': np.array([0.6, 0.5, 1.3, 0.9, 1.4])}, np.array([13.8, 21.9, 33.1, 23.9, 37.2])),
    ({'nu2': np.array([0.2, 0.4, 0.2, 0.3, 0.2]), 'ObgTppg': np.array([9.1, 15.0, 14.6, 12.7, 9.9]), 'biot': np.array([0.7, 0.8, 0.6, 1.2, 1.2]), 'ppgZhang': np.array([9.6, 10.6, 9.7, 19.3, 9.7]), 'tecB': np.array([0.8, 0.7, 0.9, 0.6, 0.9])}, np.array([14.6, 23.3, 21.2, 26.3, 20.1])),
    ({'nu2': np.array([0.3, 0.2, 0.4, 0.3, 0.2]), 'ObgTppg': np.array([17.9, 12.0, 11.8, 13.6, 11.9]), 'biot': np.array([0.7, 0.8, 1.3, 1.3, 1.2]), 'ppgZhang': np.array([13.7, 14.1, 18.8, 17.4, 17.3]), 'tecB': np.array([0.6, 1.2, 1.2, 1.3, 1.3])}, np.array([23.9, 25.9, 30.2, 36.4, 34.0])),
    ({'nu2': np.array([0.3, 0.3, 0.4, 0.4, 0.3]), 'ObgTppg': np.array([16.8, 15.9, 8.2, 16.0, 18.2]), 'biot': np.array([0.9, 0.7, 1.2, 0.9, 1.1]), 'ppgZhang': np.array([18.9, 18.3, 15.3, 10.9, 18.2]), 'tecB': np.array([1.2, 0.5, 1.0, 0.6, 1.5])}, np.array([37.1, 22.1, 19.8, 23.5, 46.5])),
])
def test_get_Shmin_grad_Daine_ppg_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_Shmin_grad_Daine_ppg_vec."""
    actual_vector_output = stresslog.get_Shmin_grad_Daine_ppg_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)

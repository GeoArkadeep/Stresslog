import pytest
import numpy as np
import stresslog
import pandas as pd
# --- Test Configuration ---
RTOL = 0.01  # Relative Tolerance
ATOL = 0.051  # Absolute Tolerance for 1-decimal rounding
# ------------------------
# --- Phase 1: Scalar Function Regression Tests ---

@pytest.mark.parametrize("inputs, expected_output", [
    ({'tvds': np.array([1000.0000, 1555.5556, 2111.1111, 2666.6667, 3222.2222, 3777.7778,
 4333.3333, 4888.8889, 5444.4444, 6000.0000]), 'porepressures': np.array([605.7176, 2552.0289, 4674.7912, 8224.7070, 9658.6526, 11795.4031,
 14345.6867, 16588.5969, 16450.7648, 19613.6491])}, 2.0973),
    ({'tvds': np.array([1000.0000, 1555.5556, 2111.1111, 2666.6667, 3222.2222, 3777.7778,
 4333.3333, 4888.8889, 5444.4444, 6000.0000]), 'porepressures': np.array([581.2288, 3177.5257, 5688.3837, 7422.5514, 9579.3750, 10764.5609,
 14923.7025, 15970.1787, 18425.2006, 21308.3180])}, 2.1641),
    ({'tvds': np.array([1000.0000, 1555.5556, 2111.1111, 2666.6667, 3222.2222, 3777.7778,
 4333.3333, 4888.8889, 5444.4444, 6000.0000]), 'porepressures': np.array([2457.0878, 3150.8340, 5718.8465, 8260.3096, 9370.9079, 13017.7722,
 12673.3184, 15860.9319, 18298.2248, 20951.6791])}, 2.2026),
    ({'tvds': np.array([1000.0000, 1555.5556, 2111.1111, 2666.6667, 3222.2222, 3777.7778,
 4333.3333, 4888.8889, 5444.4444, 6000.0000]), 'porepressures': np.array([723.9032, 2918.2663, 5132.1410, 9276.2792, 10012.8829, 12105.7924,
 13198.4224, 15213.7371, 16122.8259, 19093.6123])}, 2.0830),
    ({'tvds': np.array([1000.0000, 1555.5556, 2111.1111, 2666.6667, 3222.2222, 3777.7778,
 4333.3333, 4888.8889, 5444.4444, 6000.0000]), 'porepressures': np.array([1166.9345, 2471.8941, 6874.5809, 8357.9413, 9732.3939, 11314.5895,
 13708.1363, 15657.6030, 16902.8261, 21096.1711])}, 2.1529),
])
def test_compute_optimal_gradient(inputs, expected_output):
    """Regression test for stresslog.compute_optimal_gradient."""
    actual_output = stresslog.compute_optimal_gradient(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'tvds': np.array([1000.0000, 1555.5556, 2111.1111, 2666.6667, 3222.2222, 3777.7778,
 4333.3333, 4888.8889, 5444.4444, 6000.0000]), 'porepressures': np.array([1760.1036, 2936.9768, 5965.8311, 8085.2926, 9034.1041, 13526.8372,
 12760.2171, 18462.6946, 18874.2288, 21208.2730]), 'gradient': 1.1697}, 3262.2558),
    ({'tvds': np.array([1000.0000, 1555.5556, 2111.1111, 2666.6667, 3222.2222, 3777.7778,
 4333.3333, 4888.8889, 5444.4444, 6000.0000]), 'porepressures': np.array([2164.7872, 4242.8755, 4429.3309, 7300.7286, 9433.2740, 10097.1754,
 13663.6354, 15611.9875, 18789.1143, 20651.9113]), 'gradient': 1.7252}, 831.0355),
    ({'tvds': np.array([1000.0000, 1555.5556, 2111.1111, 2666.6667, 3222.2222, 3777.7778,
 4333.3333, 4888.8889, 5444.4444, 6000.0000]), 'porepressures': np.array([1872.7494, 4570.7485, 4997.7100, 8108.5604, 8255.4156, 13390.6286,
 13685.1340, 14983.7473, 17698.2777, 18780.7164]), 'gradient': 2.0709}, 106.7555),
    ({'tvds': np.array([1000.0000, 1555.5556, 2111.1111, 2666.6667, 3222.2222, 3777.7778,
 4333.3333, 4888.8889, 5444.4444, 6000.0000]), 'porepressures': np.array([501.9965, 3418.5795, 5805.6819, 8221.7400, 8299.0341, 12445.1001,
 13667.8695, 15861.3049, 16103.1466, 21829.3969]), 'gradient': 1.3261}, 2122.2120),
    ({'tvds': np.array([1000.0000, 1555.5556, 2111.1111, 2666.6667, 3222.2222, 3777.7778,
 4333.3333, 4888.8889, 5444.4444, 6000.0000]), 'porepressures': np.array([1834.4736, 3603.0528, 5815.8701, 7114.8873, 9668.1088, 12197.4688,
 14191.2557, 18354.2215, 18313.0132, 20196.2058]), 'gradient': 1.5390}, 1579.0733),
])
def test_compute_optimal_offset(inputs, expected_output):
    """Regression test for stresslog.compute_optimal_offset."""
    actual_output = stresslog.compute_optimal_offset(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 214.1281, 'sy': 133.7049, 'sz': 24.0118, 'alpha': 31.9500, 'beta': 137.6914, 'gamma': 156.4129, 'azim': 17.1944, 'inc': 48.0984}, np.array([[57.3088, -50.1741, -5.8894],
 [-50.1741, 104.0882, -19.9151],
 [-5.8894, -19.9151, 210.4478]])),
    ({'sx': 177.7444, 'sy': 78.1221, 'sz': 33.4195, 'alpha': 174.4133, 'beta': 24.4748, 'gamma': -168.7745, 'azim': -61.6053, 'inc': -48.1411}, np.array([[112.6015, 52.8587, 22.6608],
 [52.8587, 132.5382, -1.3271],
 [22.6608, -1.3271, 44.1463]])),
    ({'sx': 65.0527, 'sy': 41.0732, 'sz': 59.7610, 'alpha': -51.8836, 'beta': 13.0606, 'gamma': 94.3250, 'azim': -145.9021, 'inc': -94.2694}, np.array([[42.6427, -5.2784, -2.2722],
 [-5.2784, 63.7687, -1.0023],
 [-2.2722, -1.0023, 59.4755]])),
    ({'sx': 194.8521, 'sy': 49.1840, 'sz': 43.4497, 'alpha': 4.9629, 'beta': 174.9108, 'gamma': -74.4608, 'azim': 100.8616, 'inc': 40.4969}, np.array([[47.2097, 20.1820, 2.1763],
 [20.1820, 192.0932, -0.1527],
 [2.1763, -0.1527, 48.1829]])),
    ({'sx': 85.0716, 'sy': 77.4262, 'sz': 13.6165, 'alpha': -125.9013, 'beta': -23.4850, 'gamma': 146.7243, 'azim': 64.6218, 'inc': -177.5348}, np.array([[81.4641, -6.2618, 12.1350],
 [-6.2618, 54.6471, 30.1788],
 [12.1350, 30.1788, 40.0031]])),
])
def test_getAlignedStress(inputs, expected_output):
    """Regression test for stresslog.getAlignedStress."""
    actual_output = stresslog.getAlignedStress(**inputs)
    np.testing.assert_allclose(actual_output, expected_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'alpha': -72.1168, 'strike': -170.8765, 'dip': 45.7887}, (-17.5188, 43.0121)),
    ({'alpha': 43.1436, 'strike': -147.2238, 'dip': 19.5076}, (-19.7049, -17.5368)),
    ({'alpha': -172.6124, 'strike': 66.4269, 'dip': 49.6650}, (31.2103, -40.8185)),
    ({'alpha': 157.9129, 'strike': -135.0615, 'dip': 35.2235}, (33.1937, -12.5255)),
    ({'alpha': -105.3905, 'strike': 16.7348, 'dip': 81.4054}, (74.1338, 56.8642)),
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
    ({'p': 17850.7501, 't': 195.0588}, 0.5073),
    ({'p': 12479.4943, 't': 186.0080}, 0.3617),
    ({'p': 3392.3825, 't': 42.2604}, 0.1431),
    ({'p': 1603.3619, 't': 76.3476}, 0.0611),
    ({'p': 14446.0916, 't': 94.7149}, 0.5226),
])
def test_getGasDensity(inputs, expected_output):
    """Regression test for stresslog.getGasDensity."""
    actual_output = stresslog.getGasDensity(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'tvd': 3289.0122, 'gradient': 1.4435}, 6759.6493),
    ({'tvd': 4608.9498, 'gradient': 2.3071}, 15138.9846),
    ({'tvd': 5963.7122, 'gradient': 1.0998}, 9338.6064),
    ({'tvd': 1595.4501, 'gradient': 2.0357}, 4624.1832),
    ({'tvd': 1024.7394, 'gradient': 0.9019}, 1315.7811),
])
def test_getHydrostaticPsi(inputs, expected_output):
    """Regression test for stresslog.getHydrostaticPsi."""
    actual_output = stresslog.getHydrostaticPsi(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 88.6585, 's2': 56.5636, 's3': 60.7171, 'alpha': -115.7769, 'beta': -41.5454, 'gamma': 97.4112}, np.array([[0.1698, -0.9302, -0.3255],
 [0.6483, 0.3542, -0.6740],
 [0.7422, -0.0965, 0.6632]])),
    ({'s1': 83.2750, 's2': 24.5579, 's3': 37.5261, 'alpha': -173.2391, 'beta': -97.7900, 'gamma': 114.9120}, np.array([[0.8427, -0.5212, 0.1346],
 [0.5241, 0.8515, 0.0160],
 [-0.1229, 0.0571, 0.9908]])),
    ({'s1': 148.1516, 's2': 125.0573, 's3': 32.8879, 'alpha': -66.4486, 'beta': -30.8757, 'gamma': -5.3530}, np.array([[-0.1186, 0.9318, -0.3429],
 [0.5057, 0.3539, 0.7868],
 [0.8545, -0.0801, -0.5132]])),
    ({'s1': 140.4412, 's2': 38.5408, 's3': 58.2245, 'alpha': 24.9672, 'beta': -4.4205, 'gamma': -131.2968}, np.array([[0.3311, 0.2710, -0.9039],
 [-0.5738, -0.7026, -0.4208],
 [-0.7491, 0.6580, -0.0771]])),
    ({'s1': 100.8658, 's2': 127.1914, 's3': 25.1994, 'alpha': -10.2930, 'beta': 68.6843, 'gamma': 9.4542}, np.array([[0.8748, 0.3577, 0.3268],
 [-0.3258, -0.0650, 0.9432],
 [0.3586, -0.9316, 0.0597]])),
])
def test_getOrit(inputs, expected_output):
    """Regression test for stresslog.getOrit."""
    actual_output = stresslog.getOrit(**inputs)
    np.testing.assert_allclose(actual_output, expected_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'alpha': -60.4550, 'beta': 1.1118, 'gamma': 78.0589}, np.array([[0.4930, -0.8698, -0.0194],
 [0.1894, 0.0855, 0.9782],
 [-0.8492, -0.4859, 0.2069]])),
    ({'alpha': 145.0433, 'beta': -130.2936, 'gamma': 155.5664}, np.array([[0.5300, -0.3705, 0.7627],
 [0.7802, 0.5654, -0.2675],
 [-0.3321, 0.7369, 0.5888]])),
    ({'alpha': -176.2154, 'beta': 24.6480, 'gamma': 8.8869}, np.array([[-0.9069, -0.0600, -0.4170],
 [0.0009, -0.9901, 0.1404],
 [-0.4213, 0.1270, 0.8980]])),
    ({'alpha': 113.2234, 'beta': -55.1932, 'gamma': -79.0591}, np.array([[-0.2251, 0.5246, 0.8211],
 [-0.4923, 0.6660, -0.5604],
 [-0.8408, -0.5304, 0.1083]])),
    ({'alpha': -130.1694, 'beta': -66.2135, 'gamma': 102.1851}, np.array([[-0.2602, -0.3082, 0.9151],
 [0.4157, 0.8196, 0.3942],
 [-0.8715, 0.4829, -0.0851]])),
])
def test_getRota(inputs, expected_output):
    """Regression test for stresslog.getRota."""
    actual_output = stresslog.getRota(**inputs)
    np.testing.assert_allclose(actual_output, expected_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'Sv': 248.2212, 'Pp': 22.2024, 'bhp': 19.9046, 'shmin': 149.1666, 'UCS': 94.9405, 'phi': 0.4255, 'mu': 0.8924, 'nu': 0.3077, 'PhiBr': 71.8114}, [149.1666, 655.1543, 402.1605]),
    ({'Sv': 232.1905, 'Pp': 92.7974, 'bhp': 78.9861, 'shmin': 89.7344, 'UCS': 49.8774, 'phi': 0.6034, 'mu': 0.8971, 'nu': 0.1586, 'PhiBr': 75.9624}, [120.5661, 232.1905, 176.3783]),
    ({'Sv': 229.1404, 'Pp': 47.3834, 'bhp': 137.8181, 'shmin': 33.5615, 'UCS': 92.4709, 'phi': 0.5374, 'mu': 0.6494, 'nu': 0.1661, 'PhiBr': 75.0315}, [100.9626, 229.1404, 165.0515]),
    ({'Sv': 83.1782, 'Pp': 63.9942, 'bhp': 85.7948, 'shmin': 28.6879, 'UCS': 67.0704, 'phi': 0.3811, 'mu': 0.6095, 'nu': 0.3683, 'PhiBr': 63.2858}, [70.0452, 83.1782, 76.6117]),
    ({'Sv': 67.6196, 'Pp': 75.4093, 'bhp': 50.5065, 'shmin': 89.2807, 'UCS': 66.5875, 'phi': 0.4175, 'mu': 0.6776, 'nu': 0.2452, 'PhiBr': 29.5873}, [47.7159, 67.6196, 57.6677]),
])
def test_getSP(inputs, expected_output):
    """Regression test for stresslog.getSP."""
    actual_output = stresslog.getSP(**inputs)
    assert isinstance(actual_output, list)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 35.4274, 's2': 84.4982, 's3': 11.7613, 'alpha': -175.7341, 'beta': 103.0121, 'gamma': 2.5293, 'azim': -165.2040, 'inc': -152.3231, 'theta': -84.6083, 'deltaP': 128.5445, 'Pp': 124.4367, 'nu': 0.2113}, (-417.9725, -145.8275, 9.9480, -145.4643, -418.3357, 2.0907, [87.3791, 176.9312, 1.7979, 90.0441, 93.0685, -88.2045])),
    ({'s1': 144.5612, 's2': 124.7883, 's3': 16.1722, 'alpha': 59.6555, 'beta': -105.6227, 'gamma': 76.8269, 'azim': 155.4072, 'inc': 35.6835, 'theta': -167.5698, 'deltaP': 85.3955, 'Pp': 39.1920, 'nu': 0.4485}, (64.6483, 112.3268, 72.9256, 165.2108, 11.7643, 35.9488, [-91.3744, 91.0406, -1.3409, 91.8435, 177.8829, 30.8055])),
    ({'s1': 178.0231, 's2': 35.3700, 's3': 55.0192, 'alpha': -11.2976, 'beta': -112.0603, 'gamma': -58.5139, 'azim': -65.9976, 'inc': -76.2251, 'theta': -127.3205, 'deltaP': 42.9448, 'Pp': 66.4914, 'nu': 0.2472}, (209.9039, 71.7762, -91.9343, 255.8258, 25.8542, -63.4575, [-1.0280, 89.8029, -91.0213, 91.9558, 178.0343, 83.2207])),
    ({'s1': 178.8526, 's2': 81.7116, 's3': 52.1995, 'alpha': -166.9458, 'beta': -129.8018, 'gamma': 43.7453, 'azim': 17.9182, 'inc': -135.8343, 'theta': 62.5113, 'deltaP': 138.1066, 'Pp': 96.2036, 'nu': 0.2359}, (-246.6555, 68.1851, -23.5977, 69.9439, -248.4143, -4.2627, [89.3518, 87.0885, 0.7635, 92.2655, 176.3098, 51.4023])),
    ({'s1': 63.8884, 's2': 145.9600, 's3': 47.6145, 'alpha': -144.6337, 'beta': 99.2024, 'gamma': 146.7888, 'azim': -18.4375, 'inc': -60.7516, 'theta': 135.8760, 'deltaP': 46.0259, 'Pp': 40.0552, 'nu': 0.1926}, (-10.5400, 74.9180, -48.0646, 96.5006, -32.1225, -24.1816, [43.7513, 176.3500, 1.7331, 92.5608, 92.5992, -88.3832])),
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
    ({'sx': 234.6019, 'sy': 85.3074, 'sz': 15.1485, 'alpha': -57.7293, 'beta': 7.9745, 'gamma': -6.8365}, (np.array([124.9188, -65.4548, -22.9687]), np.array([-65.4548, 189.7917, 20.9937]), np.array([-22.9687, 20.9937, 20.3472]))),
    ({'sx': 124.0286, 'sy': 85.9459, 'sz': 49.3796, 'alpha': -115.4571, 'beta': -121.5740, 'gamma': -93.8594}, (np.array([59.8013, 19.4515, 6.1730]), np.array([19.4515, 86.0103, 15.9582]), np.array([6.1730, 15.9582, 113.5426]))),
    ({'sx': 171.6019, 'sy': 82.5092, 'sz': 36.0928, 'alpha': 100.9619, 'beta': -25.5800, 'gamma': -149.5679}, (np.array([70.1515, -6.4341, -27.1045]), np.array([-6.4341, 149.0085, 43.7811]), np.array([-27.1045, 43.7811, 71.0438]))),
    ({'sx': 67.5786, 'sy': 25.6707, 'sz': 35.9567, 'alpha': -116.3669, 'beta': 176.1793, 'gamma': -58.2862}, (np.array([39.6339, 13.4593, -5.2655]), np.array([13.4593, 60.8855, -0.2888]), np.array([-5.2655, -0.2888, 28.6866]))),
    ({'sx': 55.4064, 'sy': 66.5479, 'sz': 35.3111, 'alpha': -28.3942, 'beta': -97.0576, 'gamma': -64.5650}, (np.array([66.3250, -1.6268, 1.2849]), np.array([-1.6268, 35.4528, 0.9976]), np.array([1.2849, 0.9976, 55.4876]))),
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
    ({'alpha': 11.3971, 'beta': -70.5597, 'gamma': 30.0256}, (42.9007, 73.2522, -137.0993)),
    ({'alpha': 133.9777, 'beta': -162.0495, 'gamma': -2.3641}, (126.3479, 161.8996, -53.6521)),
    ({'alpha': 95.3134, 'beta': -38.7642, 'gamma': -136.3377}, (332.0462, 124.3380, 152.0462)),
    ({'alpha': 43.5578, 'beta': 116.8549, 'gamma': -55.1849}, (281.7419, 104.9459, 101.7419)),
    ({'alpha': -100.4094, 'beta': 151.6645, 'gamma': 57.2310}, (6.5794, 118.4507, -173.4206)),
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
    ({'sx': 190.2979, 'sy': 143.1853, 'sz': 50.9997, 'alpha': 133.3312, 'beta': 112.9357, 'gamma': -79.9628}, (182.71793561024518+0j)),
    ({'sx': 246.3511, 'sy': 27.8920, 'sz': 10.0041, 'alpha': 107.0758, 'beta': 121.7205, 'gamma': -41.4190}, (183.17931198090403+0j)),
    ({'sx': 141.4635, 'sy': 28.1008, 'sz': 58.7364, 'alpha': 23.3794, 'beta': -26.5101, 'gamma': -109.0211}, (53.292231536340175+106.37234356404227j)),
    ({'sx': 60.7344, 'sy': 28.4061, 'sz': 11.3186, 'alpha': 153.1399, 'beta': 49.4389, 'gamma': 108.7091}, (46.32139164347114+0j)),
    ({'sx': 212.6191, 'sy': 45.9967, 'sz': 13.8776, 'alpha': 147.3969, 'beta': -153.5851, 'gamma': 147.8530}, (60.504021575001474+0j)),
])
def test_getVertical(inputs, expected_output):
    """Regression test for stresslog.getVertical."""
    actual_output = stresslog.getVertical(**inputs)
    assert actual_output.real == pytest.approx(expected_output.real, rel=RTOL, abs=ATOL)
    assert actual_output.imag == pytest.approx(expected_output.imag, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ROP': 375.0862, 'RPM': 483.7651, 'WOB': 9.6831, 'BTDIA': 16.2927, 'ECD': 2.2630, 'pn': 1.9481}, 0.3159),
    ({'ROP': 208.1323, 'RPM': 340.6430, 'WOB': 35.5568, 'BTDIA': 5.3086, 'ECD': 1.7458, 'pn': 1.8837}, 0.5249),
    ({'ROP': 78.0356, 'RPM': 216.3021, 'WOB': 271.1692, 'BTDIA': 18.8301, 'ECD': 2.0492, 'pn': 2.3652}, 0.6813),
    ({'ROP': 134.3215, 'RPM': 287.7531, 'WOB': 99.6205, 'BTDIA': 1.0011, 'ECD': 1.0918, 'pn': 2.1094}, 1.3941),
    ({'ROP': 464.4000, 'RPM': 2.0000, 'WOB': 213.6097, 'BTDIA': 9.4791, 'ECD': 2.3823, 'pn': 1.8668}, np.nan),
])
def test_get_Dxc(inputs, expected_output):
    """Regression test for stresslog.get_Dxc."""
    actual_output = stresslog.get_Dxc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ObgTgcc': 2.2354, 'pn': 1.4302, 'd': 0.0027, 'nde': 0.9810, 'tvdbgl': 2220.8903, 'D0': 0.5023, 'Dxc': 0.5903}, 2.2329),
    ({'ObgTgcc': 2.1691, 'pn': 1.3014, 'd': 0.0027, 'nde': 1.0315, 'tvdbgl': 2122.0964, 'D0': 0.8918, 'Dxc': 1.4095}, 2.1653),
    ({'ObgTgcc': 1.6980, 'pn': 1.1167, 'd': 0.0062, 'nde': 0.8308, 'tvdbgl': 4529.4539, 'D0': 1.4916, 'Dxc': 0.9633}, 1.6980),
    ({'ObgTgcc': 1.6197, 'pn': 2.1106, 'd': 0.0026, 'nde': 1.4909, 'tvdbgl': 5734.1434, 'D0': 0.9872, 'Dxc': 0.5374}, 1.6197),
    ({'ObgTgcc': 1.9552, 'pn': 1.1962, 'd': 0.0085, 'nde': 1.4676, 'tvdbgl': 4675.5708, 'D0': 1.0095, 'Dxc': 0.6535}, 1.9552),
])
def test_get_PPgrad_Dxc_gcc(inputs, expected_output):
    """Regression test for stresslog.get_PPgrad_Dxc_gcc."""
    actual_output = stresslog.get_PPgrad_Dxc_gcc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ObgTgcc': 1.3007, 'pn': 1.6285, 'be': 0.0047, 'ne': 0.6949, 'tvdbgl': 5030.6040, 'res0': 0.9555, 'resdeep': 0.7087}, 1.3007),
    ({'ObgTgcc': 1.6478, 'pn': 2.3468, 'be': 0.0021, 'ne': 0.9185, 'tvdbgl': 2378.6242, 'res0': 1.3900, 'resdeep': 0.5184}, 1.6508),
    ({'ObgTgcc': 2.1927, 'pn': 2.0210, 'be': 0.0023, 'ne': 0.9001, 'tvdbgl': 2292.9441, 'res0': 1.2778, 'resdeep': 1.1122}, 2.1912),
    ({'ObgTgcc': 1.4062, 'pn': 2.0492, 'be': 0.0044, 'ne': 1.1140, 'tvdbgl': 2188.9416, 'res0': 1.1055, 'resdeep': 0.7345}, 1.4062),
    ({'ObgTgcc': 2.1521, 'pn': 2.3029, 'be': 0.0032, 'ne': 1.0127, 'tvdbgl': 4627.8896, 'res0': 1.2557, 'resdeep': 0.8201}, 2.1521),
])
def test_get_PPgrad_Eaton_gcc(inputs, expected_output):
    """Regression test for stresslog.get_PPgrad_Eaton_gcc."""
    actual_output = stresslog.get_PPgrad_Eaton_gcc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'ObgTgcc': 2.1429, 'pn': 2.1700, 'b': 0.0084, 'tvdbgl': 5653.6936, 'c': 0.0051, 'mudline': 205.2000, 'matrick': 69.6000, 'deltmu0': 92.5000, 'dalm': 105.5000}, 2.1442),
    ({'ObgTgcc': 2.3150, 'pn': 1.4489, 'b': 0.0017, 'tvdbgl': 3782.9901, 'c': 0.0062, 'mudline': 210.1000, 'matrick': 58.3000, 'deltmu0': 208.8000, 'dalm': 144.3000}, 2.2380),
    ({'ObgTgcc': 1.9080, 'pn': 2.2382, 'b': 0.0040, 'tvdbgl': 4203.3272, 'c': 0.0049, 'mudline': 208.9000, 'matrick': 50.7000, 'deltmu0': 99.2000, 'dalm': 78.2000}, 1.9382),
    ({'ObgTgcc': 2.3243, 'pn': 0.9582, 'b': 0.0053, 'tvdbgl': 5815.6651, 'c': 0.0050, 'mudline': 186.8000, 'matrick': 64.5000, 'deltmu0': 115.9000, 'dalm': 126.3000}, 2.2923),
    ({'ObgTgcc': 1.1722, 'pn': 1.4951, 'b': 0.0048, 'tvdbgl': 4799.5777, 'c': 0.0088, 'mudline': 217.3000, 'matrick': 50.1000, 'deltmu0': 89.5000, 'dalm': 142.5000}, 1.1714),
])
def test_get_PPgrad_Zhang_gcc(inputs, expected_output):
    """Regression test for stresslog.get_PPgrad_Zhang_gcc."""
    actual_output = stresslog.get_PPgrad_Zhang_gcc(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'nu2': 0.1529, 'ObgTppg': 15.2428, 'biot': 1.1152, 'ppgZhang': 19.5393, 'tecB': 0.9876}, 35.6615),
    ({'nu2': 0.3138, 'ObgTppg': 16.6009, 'biot': 1.2544, 'ppgZhang': 16.8619, 'tecB': 0.5380}, 28.0012),
    ({'nu2': 0.3936, 'ObgTppg': 15.7633, 'biot': 0.8463, 'ppgZhang': 18.9132, 'tecB': 1.2742}, 35.9345),
    ({'nu2': 0.2142, 'ObgTppg': 9.7557, 'biot': 0.5323, 'ppgZhang': 17.8960, 'tecB': 0.5895}, 15.3399),
    ({'nu2': 0.2408, 'ObgTppg': 17.8199, 'biot': 1.1153, 'ppgZhang': 17.3306, 'tecB': 0.5319}, 28.3286),
])
def test_get_Shmin_grad_Daine_ppg(inputs, expected_output):
    """Regression test for stresslog.get_Shmin_grad_Daine_ppg."""
    actual_output = stresslog.get_Shmin_grad_Daine_ppg(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 189.5899, 'sy': 136.1640, 'sz': 61.0453, 'txy': 10.8112, 'tyz': 5.0610, 'tzx': 8.6770, 'phi': 0.4707, 'cohesion': 19.8524, 'pp': 84.3505}, 38.0384),
    ({'sx': 138.3870, 'sy': 27.6796, 'sz': 40.3665, 'txy': 12.4766, 'tyz': 8.2898, 'tzx': 3.6703, 'phi': 0.6431, 'cohesion': 13.6029, 'pp': 85.7931}, 53.9512),
    ({'sx': 235.3876, 'sy': 21.7595, 'sz': 63.4737, 'txy': 7.6781, 'tyz': 16.5520, 'tzx': 6.1440, 'phi': 0.7345, 'cohesion': 9.4689, 'pp': 118.5298}, 69.6192),
    ({'sx': 122.6077, 'sy': 79.3375, 'sz': 57.4563, 'txy': 12.8227, 'tyz': 15.7841, 'tzx': 9.1338, 'phi': 0.7254, 'cohesion': 1.5301, 'pp': 29.3920}, 67.7048),
    ({'sx': 171.6848, 'sy': 32.0719, 'sz': 28.9134, 'txy': 16.0208, 'tyz': 3.5579, 'tzx': 4.1155, 'phi': 0.7755, 'cohesion': 19.1604, 'pp': 24.6282}, 79.5620),
])
def test_lade(inputs, expected_output):
    """Regression test for stresslog.lade."""
    actual_output = stresslog.lade(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 173.5369, 'sy': 116.4269, 'sz': 62.5506, 'txy': 4.0243, 'tyz': 12.8952, 'tzx': 11.1549, 'phi': 0.5411, 'cohesion': 4.5151, 'pp': 106.7797}, 72.8165),
    ({'sx': 42.4438, 'sy': 54.6106, 'sz': 42.5401, 'txy': 2.8438, 'tyz': 19.4346, 'tzx': 15.3101, 'phi': 0.7531, 'cohesion': 15.0181, 'pp': 85.4078}, 167.1624),
    ({'sx': 236.8019, 'sy': 139.6075, 'sz': 64.7904, 'txy': 5.3520, 'tyz': 12.1946, 'tzx': 6.2720, 'phi': 0.7783, 'cohesion': 17.9832, 'pp': 93.1230}, 184.6681),
    ({'sx': 213.5164, 'sy': 52.4283, 'sz': 67.2592, 'txy': 11.5800, 'tyz': 17.1464, 'tzx': 3.4107, 'phi': 0.4950, 'cohesion': 9.0160, 'pp': 16.0886}, 12.8383),
    ({'sx': 203.1773, 'sy': 38.1638, 'sz': 29.9540, 'txy': 8.2198, 'tyz': 10.4788, 'tzx': 6.9760, 'phi': 0.3323, 'cohesion': 12.0490, 'pp': 14.5266}, 0.3650),
])
def test_lade_failure(inputs, expected_output):
    """Regression test for stresslog.lade_failure."""
    actual_output = stresslog.lade_failure(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sx': 36.1224, 'sy': 58.7073, 'sz': 55.6651}, 37.4079),
    ({'sx': 202.0892, 'sy': 130.1734, 'sz': 15.2101}, 31.6848),
    ({'sx': 57.8007, 'sy': 25.3217, 'sz': 28.7377}, 26.9887),
    ({'sx': 104.9546, 'sy': 27.1130, 'sz': 65.4915}, 34.2541),
    ({'sx': 204.0363, 'sy': 135.0649, 'sz': 23.1607}, 39.0662),
])
def test_mogi(inputs, expected_output):
    """Regression test for stresslog.mogi."""
    actual_output = stresslog.mogi(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 186.4778, 's2': 78.1192, 's3': 41.6746}, 52.5784),
    ({'s1': 57.8294, 's2': 30.4443, 's3': 58.4120}, 45.0717),
    ({'s1': 149.6615, 's2': 105.7027, 's3': 11.3378}, 22.7930),
    ({'s1': 104.6501, 's2': 38.8659, 's3': 17.6845}, 24.1400),
    ({'s1': 248.8312, 's2': 77.0871, 's3': 22.3944}, 39.1412),
])
def test_mogi_failure(inputs, expected_output):
    """Regression test for stresslog.mogi_failure."""
    actual_output = stresslog.mogi_failure(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'s1': 131.4070, 's3': 13.9574, 'cohesion': 17.3489, 'phi': 0.7481}, 3.4318),
    ({'s1': 114.6650, 's3': 68.1758, 'cohesion': 3.5262, 'phi': 0.7205}, 39.7189),
    ({'s1': 172.2408, 's3': 45.6049, 'cohesion': 13.2775, 'phi': 0.6958}, 16.6941),
    ({'s1': 145.6762, 's3': 36.2744, 'cohesion': 12.9966, 'phi': 0.3800}, -8.8852),
    ({'s1': 144.7370, 's3': 47.1665, 'cohesion': 15.9195, 'phi': 0.3173}, -3.7198),
])
def test_mohr_failure(inputs, expected_output):
    """Regression test for stresslog.mohr_failure."""
    actual_output = stresslog.mohr_failure(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sigmamax': 53.7693, 'sigmamin': 66.2928, 'pp': 43.1219, 'ucs': 14.4320, 'nu': 0.2815}, 39.0551),
    ({'sigmamax': 202.1756, 'sigmamin': 12.5135, 'pp': 34.7841, 'ucs': 61.3024, 'nu': 0.3385}, 341.1530),
    ({'sigmamax': 208.3812, 'sigmamin': 23.1362, 'pp': 16.5017, 'ucs': 91.9914, 'nu': 0.2205}, 388.3140),
    ({'sigmamax': 195.7642, 'sigmamin': 28.4339, 'pp': 87.8551, 'ucs': 41.1514, 'nu': 0.2294}, 351.3981),
    ({'sigmamax': 196.2838, 'sigmamin': 52.1754, 'pp': 135.3575, 'ucs': 95.9351, 'nu': 0.2581}, 261.5014),
])
def test_willson_sanding_cwf(inputs, expected_output):
    """Regression test for stresslog.willson_sanding_cwf."""
    actual_output = stresslog.willson_sanding_cwf(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)


@pytest.mark.parametrize("inputs, expected_output", [
    ({'sigmamax': 112.3202, 'sigmamin': 34.6316, 'pp': 114.7957, 'ucs': 33.0168, 'k0': 1.1921, 'nu': 0.2520}, 172.2686),
    ({'sigmamax': 39.1409, 'sigmamin': 17.9362, 'pp': 24.0638, 'ucs': 96.5631, 'k0': 1.2563, 'nu': 0.3833}, -4.7916),
    ({'sigmamax': 151.1721, 'sigmamin': 34.0938, 'pp': 58.6394, 'ucs': 94.4759, 'k0': 0.9192, 'nu': 0.4047}, 167.5463),
    ({'sigmamax': 39.8046, 'sigmamin': 35.3859, 'pp': 68.4228, 'ucs': 86.0527, 'k0': 0.5075, 'nu': 0.2595}, -17.4624),
    ({'sigmamax': 45.3002, 'sigmamin': 32.9526, 'pp': 78.9885, 'ucs': 48.5492, 'k0': 0.5762, 'nu': 0.3172}, 4.7645),
])
def test_zhang_sanding_cwf(inputs, expected_output):
    """Regression test for stresslog.zhang_sanding_cwf."""
    actual_output = stresslog.zhang_sanding_cwf(**inputs)
    assert actual_output == pytest.approx(expected_output, rel=RTOL, abs=ATOL, nan_ok=True)

# --- Phase 2: Vectorized Function Consistency Tests ---

@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ROP': np.array([28.0477, 98.1337, 64.8416, 92.2720, 284.0000]), 'RPM': np.array([7.6803, 475.4919, 184.6431, 133.0035, 2.4000]), 'WOB': np.array([176.9067, 448.1936, 76.8986, 192.5598, 68.1542]), 'BTDIA': np.array([5.7820, 18.8748, 13.8176, 2.1669, 13.6771]), 'ECD': np.array([1.0987, 1.7098, 1.6622, 2.0332, 2.0101]), 'pn': np.array([1.3911, 1.7150, 2.0006, 1.9073, 2.1370])}, np.array([0.4480, 0.6970, 0.6436, 0.6113, np.nan])),
    ({'ROP': np.array([72.4303, 485.5668, 374.4689, 411.4581, 252.0000]), 'RPM': np.array([255.2801, 418.2472, 61.5763, 188.3092, 2.4000]), 'WOB': np.array([76.0541, 262.7301, 333.8601, 215.7475, 456.6456]), 'BTDIA': np.array([10.6061, 13.0527, 11.5484, 17.0442, 14.7432]), 'ECD': np.array([2.0484, 1.6146, 2.3236, 0.9349, 2.3507]), 'pn': np.array([2.2431, 1.1847, 1.4246, 2.0274, 1.9411])}, np.array([0.6264, 0.3476, 0.1762, 0.8171, np.nan])),
    ({'ROP': np.array([218.2651, 333.2008, 302.2402, 91.1468, 486.2000]), 'RPM': np.array([62.2663, 59.0847, 304.7743, 444.5703, 4.9000]), 'WOB': np.array([115.8307, 302.8003, 100.5752, 57.5106, 379.6978]), 'BTDIA': np.array([11.5396, 6.3609, 11.2143, 9.2949, 3.4834]), 'ECD': np.array([1.0187, 2.0861, 1.2619, 1.8061, 1.4836]), 'pn': np.array([1.7037, 2.1449, 1.9505, 1.6828, 1.9364])}, np.array([0.5263, 0.3256, 0.6940, 0.5565, np.nan])),
    ({'ROP': np.array([44.0366, 155.6915, 86.6300, 244.4607, 303.3000]), 'RPM': np.array([96.7439, 238.0875, 349.6202, 245.3528, 2.8000]), 'WOB': np.array([343.3413, 431.5311, 433.3360, 470.3222, 56.9258]), 'BTDIA': np.array([16.9967, 2.7069, 18.1586, 6.3829, 11.5841]), 'ECD': np.array([1.2358, 2.2429, 0.9538, 2.3111, 2.1986]), 'pn': np.array([1.8004, 2.0814, 1.9480, 1.2094, 2.1319])}, np.array([0.8542, 0.6700, 1.3743, 0.3050, np.nan])),
    ({'ROP': np.array([432.9617, 189.8484, 143.8525, 422.3427, 194.9000]), 'RPM': np.array([389.0977, 322.0145, 48.6019, 122.9000, 1.3000]), 'WOB': np.array([326.2038, 123.1076, 464.6462, 65.2764, 229.7386]), 'BTDIA': np.array([16.6338, 1.5567, 10.6969, 4.4293, 10.7950]), 'ECD': np.array([1.6119, 2.0063, 1.4812, 1.0736, 1.0080]), 'pn': np.array([1.2594, 2.1887, 1.0543, 1.4423, 1.8895])}, np.array([0.3729, 0.7246, 0.2833, 0.4447, np.nan])),
])
def test_get_Dxc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_Dxc_vec."""
    actual_vector_output = stresslog.get_Dxc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ObgTgcc': np.array([2.2652, 1.5430, 2.2396, 1.2056, 2.1681]), 'pn': np.array([1.7188, 1.6798, 1.0651, 2.2368, 1.4564]), 'd': np.array([0.0019, 0.0063, 0.0045, 0.0055, 0.0040]), 'nde': np.array([0.9252, 1.0405, 1.2062, 1.2269, 1.3286]), 'tvdbgl': np.array([4540.9398, 4413.4240, 3085.1338, 3655.7927, 4211.0422]), 'D0': np.array([0.5201, 0.8539, 0.5133, 1.2169, 1.4184]), 'Dxc': np.array([0.5297, 0.8401, 1.0838, 1.1609, 0.7088])}, np.array([2.2651, 1.5430, 2.2396, 1.2056, 2.1681])),
    ({'ObgTgcc': np.array([1.2346, 0.9510, 1.9772, 2.0971, 1.2958]), 'pn': np.array([1.6576, 1.0284, 1.3883, 1.8424, 0.9534]), 'd': np.array([0.0069, 0.0077, 0.0032, 0.0047, 0.0053]), 'nde': np.array([0.8123, 0.6432, 1.0400, 1.3941, 0.9861]), 'tvdbgl': np.array([3107.5812, 4212.0005, 1556.5508, 5885.7915, 2934.1955]), 'D0': np.array([1.1930, 0.5685, 0.8495, 1.0110, 1.4612]), 'Dxc': np.array([1.0329, 0.7169, 0.6641, 1.1845, 0.9499])}, np.array([1.2346, 0.9510, 1.9746, 2.0971, 1.2958])),
    ({'ObgTgcc': np.array([2.0496, 1.7236, 1.7991, 0.9988, 1.2351]), 'pn': np.array([1.0844, 1.2611, 0.9125, 1.9371, 1.8239]), 'd': np.array([0.0016, 0.0089, 0.0017, 0.0024, 0.0037]), 'nde': np.array([1.2346, 0.5057, 0.6743, 0.6914, 1.4043]), 'tvdbgl': np.array([5423.6299, 3925.5819, 2350.2870, 4444.6035, 3698.4821]), 'D0': np.array([0.7211, 0.8674, 0.9593, 0.5403, 0.9376]), 'Dxc': np.array([1.0800, 0.7582, 1.1219, 1.1644, 0.6080])}, np.array([2.0496, 1.7236, 1.7314, 0.9998, 1.2351])),
    ({'ObgTgcc': np.array([1.9203, 2.0516, 1.2242, 1.2185, 1.9969]), 'pn': np.array([2.3717, 1.4391, 1.3816, 1.0214, 1.5215]), 'd': np.array([0.0066, 0.0086, 0.0065, 0.0082, 0.0022]), 'nde': np.array([1.1600, 1.4209, 1.0940, 1.1222, 1.0507]), 'tvdbgl': np.array([2983.8182, 5417.7055, 5591.8443, 1550.1799, 1291.3298]), 'D0': np.array([1.1255, 1.3581, 1.3555, 1.2197, 1.4714]), 'Dxc': np.array([0.9534, 1.0081, 1.1084, 1.1757, 1.0382])}, np.array([1.9203, 2.0516, 1.2242, 1.2185, 1.9811])),
    ({'ObgTgcc': np.array([1.2138, 1.8917, 1.9714, 1.5778, 2.1143]), 'pn': np.array([1.0746, 1.5950, 1.8412, 1.9967, 0.9054]), 'd': np.array([0.0086, 0.0029, 0.0053, 0.0055, 0.0042]), 'nde': np.array([0.5500, 0.8633, 0.5998, 1.2210, 0.5583]), 'tvdbgl': np.array([1486.6766, 5375.4943, 1607.4661, 2721.8422, 4290.5035]), 'D0': np.array([1.1330, 1.0340, 0.8651, 1.3955, 1.0694]), 'Dxc': np.array([1.0801, 1.1835, 0.5264, 0.5197, 1.3559])}, np.array([1.2137, 1.8917, 1.9708, 1.5778, 2.1142])),
])
def test_get_PPgrad_Dxc_gcc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_PPgrad_Dxc_gcc_vec."""
    actual_vector_output = stresslog.get_PPgrad_Dxc_gcc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ObgTgcc': np.array([1.7457, 2.0899, 2.2847, 1.7747, 1.4365]), 'pn': np.array([1.1795, 2.0086, 0.9373, 1.1200, 2.3193]), 'be': np.array([0.0081, 0.0018, 0.0012, 0.0014, 0.0034]), 'ne': np.array([1.4290, 1.2982, 0.9129, 1.1985, 0.9291]), 'tvdbgl': np.array([3467.4168, 3026.6689, 3915.2920, 4885.5885, 5653.6087]), 'res0': np.array([0.7973, 0.9187, 0.6260, 0.7877, 1.3154]), 'resdeep': np.array([1.0346, 1.3922, 0.6198, 1.2896, 1.4678])}, np.array([1.7457, 2.0898, 2.2636, 1.7744, 1.4365])),
    ({'ObgTgcc': np.array([2.1877, 2.3445, 2.2283, 1.8534, 1.8182]), 'pn': np.array([1.1347, 1.7077, 2.2130, 0.9003, 1.5833]), 'be': np.array([0.0059, 0.0049, 0.0022, 0.0083, 0.0079]), 'ne': np.array([1.1584, 0.9380, 1.3938, 1.2232, 0.7518]), 'tvdbgl': np.array([2135.4814, 3839.6469, 5251.1090, 4454.6034, 3823.0734]), 'res0': np.array([1.3293, 0.8865, 0.5098, 1.2572, 0.6974]), 'resdeep': np.array([1.3958, 1.3708, 0.6124, 0.6260, 1.4605])}, np.array([2.1876, 2.3445, 2.2283, 1.8534, 1.8182])),
    ({'ObgTgcc': np.array([2.1644, 0.9540, 1.7307, 1.9452, 1.1906]), 'pn': np.array([1.4578, 1.4064, 1.4616, 1.6109, 1.1598]), 'be': np.array([0.0040, 0.0070, 0.0022, 0.0049, 0.0043]), 'ne': np.array([1.0343, 0.9394, 0.6572, 0.7151, 1.1318]), 'tvdbgl': np.array([2000.2176, 4012.8325, 5192.7302, 1615.0912, 4539.8096]), 'res0': np.array([1.1725, 1.3873, 0.9632, 1.3461, 1.3298]), 'resdeep': np.array([1.4296, 1.2606, 0.8027, 1.2769, 1.3806])}, np.array([2.1642, 0.9540, 1.7306, 1.9441, 1.1906])),
    ({'ObgTgcc': np.array([1.6367, 2.3210, 1.9307, 2.3886, 1.0400]), 'pn': np.array([2.1692, 1.2426, 2.3462, 2.3868, 0.9314]), 'be': np.array([0.0022, 0.0017, 0.0013, 0.0090, 0.0085]), 'ne': np.array([0.8160, 1.2373, 0.6870, 0.8425, 0.9436]), 'tvdbgl': np.array([2056.0020, 3451.6791, 5241.1082, 2778.7336, 4471.8854]), 'res0': np.array([1.4843, 1.1314, 1.2678, 1.4149, 0.7851]), 'resdeep': np.array([0.9204, 1.1547, 0.8712, 0.9806, 0.8802])}, np.array([1.6463, 2.3201, 1.9337, 2.3886, 1.0400])),
    ({'ObgTgcc': np.array([1.2957, 1.9945, 1.3885, 1.7580, 1.2634]), 'pn': np.array([1.8252, 1.5154, 1.2058, 1.0873, 1.2128]), 'be': np.array([0.0042, 0.0046, 0.0036, 0.0013, 0.0073]), 'ne': np.array([0.9750, 0.7537, 1.3974, 0.6002, 1.3522]), 'tvdbgl': np.array([1887.3699, 2055.3723, 3046.1047, 1197.8277, 5074.9865]), 'res0': np.array([0.7776, 1.4756, 1.2718, 1.3136, 1.4177]), 'resdeep': np.array([0.8311, 1.0042, 1.1728, 0.8336, 0.8828])}, np.array([1.2959, 1.9942, 1.3885, 1.5501, 1.2634])),
])
def test_get_PPgrad_Eaton_gcc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_PPgrad_Eaton_gcc_vec."""
    actual_vector_output = stresslog.get_PPgrad_Eaton_gcc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'ObgTgcc': np.array([2.0300, 1.9292, 1.1765, 1.9584, 1.2274]), 'pn': np.array([2.2409, 1.7647, 2.0241, 1.3707, 0.9687]), 'b': np.array([0.0028, 0.0014, 0.0051, 0.0073, 0.0082]), 'tvdbgl': np.array([5267.4192, 3664.1258, 2025.1759, 4954.1096, 3571.9305]), 'c': np.array([0.0030, 0.0031, 0.0046, 0.0052, 0.0024]), 'mudline': np.array([217.3000, 197.5000, 187.0000, 191.4000, 219.6000]), 'matrick': np.array([69.5000, 63.1000, 55.6000, 59.6000, 50.8000]), 'deltmu0': np.array([205.1000, 163.1000, 150.1000, 127.4000, 84.2000]), 'dalm': np.array([210.1000, 122.6000, 99.7000, 71.6000, 79.7000])}, np.array([2.0307, 1.9086, 1.2752, 1.9034, 1.1750])),
    ({'ObgTgcc': np.array([1.9854, 1.1691, 1.7890, 1.3323, 1.3573]), 'pn': np.array([1.9905, 2.2250, 1.3831, 2.1401, 1.2712]), 'b': np.array([0.0083, 0.0085, 0.0045, 0.0071, 0.0039]), 'tvdbgl': np.array([1555.8071, 2975.1247, 4191.5003, 3527.0569, 5418.1166]), 'c': np.array([0.0015, 0.0050, 0.0036, 0.0088, 0.0017]), 'mudline': np.array([218.4000, 186.6000, 200.2000, 202.3000, 195.2000]), 'matrick': np.array([65.5000, 60.6000, 58.5000, 66.6000, 58.4000]), 'deltmu0': np.array([121.4000, 81.9000, 181.0000, 69.5000, 96.6000]), 'dalm': np.array([117.1000, 99.2000, 172.9000, 169.9000, 83.9000])}, np.array([1.9878, 1.2526, 1.7831, 1.3165, 1.3413])),
    ({'ObgTgcc': np.array([1.2745, 1.5057, 2.2017, 1.5958, 1.4304]), 'pn': np.array([2.2751, 1.6209, 0.9791, 1.4685, 1.5470]), 'b': np.array([0.0071, 0.0057, 0.0022, 0.0084, 0.0087]), 'tvdbgl': np.array([3802.9855, 1607.0718, 2653.8474, 5657.4394, 1033.6989]), 'c': np.array([0.0014, 0.0056, 0.0025, 0.0070, 0.0082]), 'mudline': np.array([200.9000, 214.8000, 217.0000, 195.8000, 187.2000]), 'matrick': np.array([64.9000, 53.0000, 63.3000, 58.8000, 51.1000]), 'deltmu0': np.array([162.1000, 192.8000, 125.5000, 163.2000, 102.8000]), 'dalm': np.array([143.9000, 77.9000, 112.3000, 182.7000, 76.5000])}, np.array([1.3731, 1.5297, 1.9828, 1.5955, 1.4536])),
    ({'ObgTgcc': np.array([1.4958, 1.5685, 1.9105, 1.5099, 1.7057]), 'pn': np.array([0.9881, 1.0369, 2.0179, 1.2281, 1.2201]), 'b': np.array([0.0022, 0.0035, 0.0048, 0.0012, 0.0052]), 'tvdbgl': np.array([1913.0196, 3534.0494, 2954.1122, 5942.0201, 1638.9583]), 'c': np.array([0.0025, 0.0078, 0.0074, 0.0018, 0.0019]), 'mudline': np.array([218.1000, 211.9000, 195.9000, 188.9000, 215.1000]), 'matrick': np.array([65.0000, 62.7000, 65.0000, 67.6000, 64.3000]), 'deltmu0': np.array([102.4000, 195.3000, 108.3000, 152.2000, 99.1000]), 'dalm': np.array([83.8000, 113.5000, 132.6000, 112.9000, 173.5000])}, np.array([1.2634, 1.5254, 1.9125, 1.4760, 1.6543])),
    ({'ObgTgcc': np.array([1.9220, 2.2083, 2.2523, 1.0472, 2.2658]), 'pn': np.array([1.3782, 1.0647, 1.3524, 1.4799, 1.5224]), 'b': np.array([0.0076, 0.0016, 0.0016, 0.0031, 0.0086]), 'tvdbgl': np.array([5095.3458, 2928.6717, 5759.5342, 3136.2611, 5409.2011]), 'c': np.array([0.0049, 0.0066, 0.0072, 0.0013, 0.0084]), 'mudline': np.array([207.5000, 211.5000, 192.6000, 217.1000, 214.5000]), 'matrick': np.array([61.1000, 54.7000, 55.4000, 61.3000, 55.8000]), 'deltmu0': np.array([154.3000, 96.3000, 117.9000, 161.2000, 114.1000]), 'dalm': np.array([110.0000, 175.0000, 102.4000, 62.9000, 197.9000])}, np.array([1.8979, 2.3928, 2.2069, 1.5446, 2.2640])),
])
def test_get_PP_grad_Zhang_gcc_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_PP_grad_Zhang_gcc_vec."""
    actual_vector_output = stresslog.get_PP_grad_Zhang_gcc_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("vector_inputs, expected_vector_output", [
    ({'nu2': np.array([0.3867, 0.1586, 0.3358, 0.1850, 0.2767]), 'ObgTppg': np.array([14.7420, 16.3642, 13.1817, 13.1932, 11.5784]), 'biot': np.array([1.0559, 0.6059, 1.3090, 1.0050, 0.8176]), 'ppgZhang': np.array([11.3893, 13.7208, 19.1523, 18.7889, 9.3217]), 'tecB': np.array([0.9206, 0.6251, 0.6763, 0.6094, 1.0115])}, np.array([27.3103, 20.0604, 27.9740, 25.6306, 20.8463])),
    ({'nu2': np.array([0.3981, 0.3513, 0.3134, 0.2227, 0.1717]), 'ObgTppg': np.array([8.7309, 19.6968, 14.6989, 19.9939, 7.7792]), 'biot': np.array([0.5745, 0.6692, 0.9772, 0.6930, 0.5452]), 'ppgZhang': np.array([19.5768, 14.2663, 16.1625, 17.4711, 8.0775]), 'tecB': np.array([0.9777, 1.2555, 0.8734, 0.8056, 1.3520])}, np.array([18.1189, 39.7747, 28.1326, 30.4742, 15.6209])),
    ({'nu2': np.array([0.2051, 0.2530, 0.3512, 0.4018, 0.3806]), 'ObgTppg': np.array([18.7297, 17.2749, 13.9724, 12.9345, 9.2934]), 'biot': np.array([0.6262, 1.4004, 0.6061, 1.4576, 1.3604]), 'ppgZhang': np.array([9.8848, 18.1278, 10.2749, 15.9533, 8.8779]), 'tecB': np.array([0.5537, 1.1154, 1.1677, 0.5738, 1.3068])}, np.array([19.7949, 41.9072, 26.7352, 23.7425, 22.5115])),
    ({'nu2': np.array([0.3877, 0.2681, 0.3594, 0.4417, 0.4418]), 'ObgTppg': np.array([8.7067, 13.4356, 8.0627, 8.4565, 16.8721]), 'biot': np.array([0.6341, 0.9555, 1.2992, 1.2264, 0.6971]), 'ppgZhang': np.array([18.6046, 14.4957, 11.9757, 10.1593, 16.0599]), 'tecB': np.array([0.6795, 1.3474, 0.8357, 1.2991, 1.0494])}, np.array([15.7567, 31.8016, 18.0917, 20.2777, 33.3937])),
    ({'nu2': np.array([0.2647, 0.4062, 0.3394, 0.3293, 0.3718]), 'ObgTppg': np.array([13.6065, 11.3371, 10.7766, 17.3836, 19.1601]), 'biot': np.array([0.8259, 0.8602, 1.4685, 0.9176, 0.8233]), 'ppgZhang': np.array([17.2999, 9.5709, 17.7400, 7.5589, 16.4586]), 'tecB': np.array([1.4444, 0.5539, 0.6678, 1.1475, 1.1714])}, np.array([33.6958, 16.6367, 25.4003, 32.0143, 39.3138])),
])
def test_get_Shmin_grad_Daine_ppg_vec_consistency(vector_inputs, expected_vector_output):
    """Consistency test for vectorized function stresslog.get_Shmin_grad_Daine_ppg_vec."""
    actual_vector_output = stresslog.get_Shmin_grad_Daine_ppg_vec(**vector_inputs)
    np.testing.assert_allclose(actual_vector_output, expected_vector_output, rtol=RTOL, atol=ATOL, equal_nan=True)

from stresslog import (
    convert_rop,
    convert_wob,
    convert_ecd,
    convert_flowrate,
    convert_torque,
    #convert_dataframe_units,
)
#convert_rop, convert_wob, convert_ecd, convert_torque, convert_flowrate

# --- Test convert_rop ---------------------------------------------------------

def test_convert_rop_min_per_m_to_ft_per_hr():
    values = np.array([1.0])  # 1 min/m
    result = convert_rop(values, "minute / meter")
    # Expected: about 196.85 ft/hr
    assert np.isclose(result[0], 196.85, rtol=RTOL)


def test_convert_rop_m_per_hr_to_ft_per_hr():
    values = np.array([10.0])  # 10 m/hr
    result = convert_rop(values, "M/HR")
    # 10 m/hr = 32.808 ft/hr
    assert np.isclose(result[0], 32.808, rtol=RTOL)


# --- Test convert_wob ---------------------------------------------------------

def test_convert_wob_ton_to_lb():
    values = np.array([1.0])  # 1 ton (2000 lb)
    result = convert_wob(values, "TON")
    assert np.isclose(result[0], 2000, rtol=RTOL)


def test_convert_wob_metric_ton_to_lb():
    values = np.array([1.0])  # 1 metric ton = 2204.62 lb
    result = convert_wob(values, "MTON")
    assert np.isclose(result[0], 2204.62, rtol=RTOL)


# --- Test convert_ecd ---------------------------------------------------------

def test_convert_ecd_ppg_to_sg():
    values = np.array([10.0])  # 10 ppg
    result = convert_ecd(values, "ppg", "SG")
    # Known: 10 ppg ≈ 1.197 SG
    assert np.isclose(result[0], 1.197, rtol=RTOL)


def test_convert_ecd_sg_to_ppg():
    values = np.array([1.2])  # 1.2 SG
    result = convert_ecd(values, "SG", "ppg")
    # Reverse: 1.2 SG ≈ 10.02 ppg
    assert np.isclose(result[0], 10.02, rtol=RTOL)


# --- Test convert_flowrate ----------------------------------------------------

def test_convert_flowrate_lpm_to_gpm():
    values = np.array([3.785])  # 3.785 L/min = 1 gpm
    result = convert_flowrate(values, "LPM", "GPM")
    assert np.isclose(result[0], 1.0, rtol=RTOL)


def test_convert_flowrate_gpm_to_lpm():
    values = np.array([1.0])  # 1 gpm
    result = convert_flowrate(values, "GPM", "LPM")
    assert np.isclose(result[0], 3.785, rtol=RTOL)


# --- Test convert_torque ------------------------------------------------------

def test_convert_torque_ftlb_to_nm():
    values = np.array([10.0])  # 10 ft-lb
    result = convert_torque(values, "FTLB", "NM")
    assert np.isclose(result[0], 13.56, rtol=RTOL)


def test_convert_torque_nm_to_ftlb():
    values = np.array([10.0])  # 10 Nm
    result = convert_torque(values, "NM", "FTLB")
    assert np.isclose(result[0], 7.376, rtol=RTOL)

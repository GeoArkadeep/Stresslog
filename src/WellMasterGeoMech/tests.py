import unittest
from DrawSP import getSP  # Adjust this import based on your actual file organization
import numpy as np
import math

class TestGetSPFunction(unittest.TestCase):
    def test_typical_conditions(self):
        """
        Test typical conditions to verify if the function returns expected minSH, maxSH, and midSH.
        """
        Sv = 48
        Pp = 19
        bhp = 22
        shmin = 25
        UCS = 46
        phi = 45
        flag = 0
        mu = 0.6
        expected_output = [25, 48.98, 36.99]  # Example expected values, adjust based on your calculations
        result = getSP(Sv, Pp, bhp, shmin, UCS, phi, flag, mu)
        for expected, actual in zip(expected_output, result):
            self.assertAlmostEqual(expected, actual, places=2)

    def test_edge_case_low_shmin(self):
        """
        Test an edge case where shmin is lower than physically possible based on Sv, Pp, and other parameters.
        """
        Sv = 48
        Pp = 19
        bhp = 22
        shmin = 15  # Unphysically low shmin value
        UCS = 46
        phi = 45
        flag = 0
        mu = 0.6
        expected_output = [np.nan, np.nan, np.nan]  # Assuming the function handles this with NaNs
        result = getSP(Sv, Pp, bhp, shmin, UCS, phi, flag, mu)
        self.assertTrue(all(math.isnan(x) for x in result), "Expected NaNs for unphysical shmin")

    def test_edge_case_high_flag(self):
        """
        Test handling of high flag values indicating specific geological observations.
        """
        Sv = 48
        Pp = 19
        bhp = 22
        shmin = 25
        UCS = 46
        phi = 45
        flag = 4  # Indicates both breakouts and tensile fractures on log
        mu = 0.6
        # Example expected values under these conditions, adjust based on your implementation
        expected_output = [25.5, 49.5, 37.5]
        result = getSP(Sv, Pp, bhp, shmin, UCS, phi, flag, mu)
        for expected, actual in zip(expected_output, result):
            self.assertAlmostEqual(expected, actual, places=2)

# Add more test cases as needed to thoroughly test your function.

if __name__ == '__main__':
    unittest.main()

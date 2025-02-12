"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""
import pytest
import stresslog as lst
import pandas as pd


def test_integration_basic():
# Test case 1: Basic well log generation and plotting
    attrib = [10, 0, 0, 0, 0, 0, 0, 0] #KB, GL, 
    well = lst.create_random_well(kb=35, gl=-200, step=5)
    print(well.df())
    dev = lst.getwelldev(wella=well)
    print(dev.df())
    output = lst.compute_geomech(dev, writeFile=False)

    assert output is not None, "plotPPzhang did not return an output"

def test_integration_doi():
    # Test case 2: Well with attributes and DOI
    attrib = [10, -200, 0, 0, 0, 0, 0, 0]
    well = lst.create_random_well(kb=312, gl=300, step=4)
    dev = lst.getwelldev(wella=well, kickoffpoint=3000, final_angle=90, rateofbuild=0.2, azimuth=270)
    output = lst.compute_geomech(dev, attrib=attrib, doi=2625, writeFile=False)
    assert output is not None, "plotPPzhang with attributes and DOI did not return an output"

def test_integration_advanced():
    # Test case 3: Shallow angle well and custom TECB
    test_m_depth=2000
    attrib = [10, -200, 0, 0, 0, 0, 0, 0]
    well = lst.create_random_well(kb=35, gl=-200, step=4, seed=33)
    dev = lst.getwelldev(wella=well, kickoffpoint=300, final_angle=10, rateofbuild=0.2, azimuth=270)
    output = lst.compute_geomech(dev, attrib=attrib, doi=2625, tecb=0.55, writeFile=False)
    
    index = lst.find_nearest_depth(output[0]['MD'].to_numpy(),test_m_depth)[0]
    
    assert abs(output[0]['OVERBURDEN_PRESSURE'].to_numpy()[index] - 5309.732562366982) < 0.001, f"Overburden check fail: {output[0]['OVERBURDEN_PRESSURE'].to_numpy()[index]}"
    print(f"Overburden check passed: {output[0]['OVERBURDEN_PRESSURE'].to_numpy()[index]}psi")
    
    assert abs(output[0]['FracPressure'].to_numpy()[index] - 9402.524110909559) < 0.001, f"Frac Pressure check fail: {output[0]['FracPressure'].to_numpy()[index]}"
    print(f"Frac Pressure check passed: {output[0]['FracPressure'].to_numpy()[index]}psi")

#if __name__=="__main__":
#    integrationTest()

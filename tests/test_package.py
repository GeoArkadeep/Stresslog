"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""
import pytest
import stresslog as lst
import pandas as pd
import numpy as np

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
    # Test case 3: Shallow angle well and custom TECB. now with Formations, Imagelog and UCS data

    ucs = pd.read_csv('https://raw.githubusercontent.com/GeoArkadeep/supporting-data-for-EOS-Northern-Lights/main/UCSdata.csv')
    formations = pd.read_csv('https://raw.githubusercontent.com/GeoArkadeep/supporting-data-for-EOS-Northern-Lights/main/NorthernLights-31_5-7.csv')
    imagelog = pd.read_csv('https://raw.githubusercontent.com/GeoArkadeep/supporting-data-for-EOS-Northern-Lights/main/31_5-7_Image.csv')
    
    test_m_depth=2000
    attrib = [10, -200, 0, 0, 0, 0, 0, 0]
    well = lst.create_random_well(kb=35, gl=-200, step=4, seed=33)
    dev = lst.getwelldev(wella=well, kickoffpoint=300, final_angle=10, rateofbuild=0.2, azimuth=270)
    output = lst.compute_geomech(dev, attrib=attrib, doi=2625, tecb=0.55, display=True, forms = formations, UCSs = ucs, flags = imagelog, debug = True)
    
    index = lst.find_nearest_depth(output[0]['MD'].to_numpy(),test_m_depth)[0]
    
    assert abs(output[0]['OVERBURDEN_PRESSURE'].to_numpy()[index] - 5309.732562366982) < 0.001, f"Overburden check fail: {output[0]['OVERBURDEN_PRESSURE'].to_numpy()[index]}"
    print(f"Overburden check passed: {output[0]['OVERBURDEN_PRESSURE'].to_numpy()[index]}psi")
    
    assert abs(output[0]['FracPressure'].to_numpy()[index] - 9402.524110909559) < 0.001, f"Frac Pressure check fail: {output[0]['FracPressure'].to_numpy()[index]}"
    print(f"Frac Pressure check passed: {output[0]['FracPressure'].to_numpy()[index]}psi")

def test_analog():
    #Test case 4: Test analog well
    formations = pd.read_csv('https://raw.githubusercontent.com/GeoArkadeep/supporting-data-for-EOS-Northern-Lights/main/NorthernLights-31_5-7.csv')
    well = lst.create_random_well(kb=35, gl=-200, step=4, seed=33)
    dev = lst.getwelldev(wella=well, kickoffpoint=300, final_angle=10, rateofbuild=0.2, azimuth=270)
    
    new_forms = formations.rename(columns={"Top TVD": "TopTVD", "Formation Name": "Name"})[["TopTVD", "Name"]]
    target_forms = formations.rename(columns={"Top TVD": "TopTVD", "Formation Name": "Name"}).assign(TopTVD=lambda df: df["TopTVD"] + 150)[["TopTVD", "Name"]]
    analogwell,analoglas, analogdev = lst.get_analog(dev,new_forms, target_forms, kb=40,gl=-200,dev=None,kop=0,ma=0,rob=0.1,azi=0,debug=False,td=5000)
    
    df_analog = analogwell.df()
    dev_df = dev.df() # 'dev' is the source well passed to lst.get_analog

    idx_orig = lst.find_nearest_depth(dev_df['TVDM'].to_numpy(), 2900.0)[0]
    gr_orig = dev_df['GR'].iloc[idx_orig]
    rhob_orig = dev_df['RHOB'].iloc[idx_orig]

    idx_analog = lst.find_nearest_depth(df_analog['TVDM'].to_numpy(), 3050.0)[0]
    gr_analog = df_analog['GR'].iloc[idx_analog]
    rhob_analog = df_analog['RHOB'].iloc[idx_analog]

    # Seed(33) makes random logs deterministic. Allow small tolerance for interpolations.
    assert np.isclose(gr_orig, gr_analog, rtol=1e-2),f"GR mismatch after depth shift: original GR={gr_orig} at TVDM~2900, analog GR={gr_analog} at TVDM~3050"
    print(f"GR match after depth shift: original GR={gr_orig} at TVDM~2900, analog GR={gr_analog} at TVDM~3050")
    assert np.isclose(rhob_orig, rhob_analog, rtol=1e-2),f"RHOB mismatch after depth shift: original RHOB={rhob_orig} at TVDM~2900, analog RHOB={rhob_analog} at TVDM~3050"
    print(f"RHOB match after depth shift: original RHOB={rhob_orig} at TVDM~2900, analog RHOB={rhob_analog} at TVDM~3050")
    
    
"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

from .syntheticLogs import create_random_well, getwelldev
from .geomechanics import plotPPzhang

#def test_plotPPzhang():
    # Generate a random well log
    #for i in range(5):
attrib=[10,0,0,0,0,0,0,0]
x = create_random_well(kb=35, gl=-200)
y = getwelldev(wella = x)
# Call the function you want to test
output = plotPPzhang(y,attrib=attrib)
attrib=[10,-200,0,0,0,0,0,0]
output = plotPPzhang(x,attrib=attrib)
output = plotPPzhang(x,attrib=attrib, doi=1690)

# Assert statements or output checks can go here
assert output is not None, "Function did not return an output"
print("Test passed: plotPPzhang successfully executed.")

"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""
import pandas as pd
import stresslog as lst
#def test_plotPPzhang():
    # Generate a random well log
    #for i in range(5):
attrib=[10,0,0,0,0,0,0,0]
#form = pd.read_csv('put.csv')
#form2 = pd.read_csv('call.csv')
#flag = pd.read_csv('flags.csv')
#litho = pd.read_csv('lithos.csv')
#print(form)
x = lst.create_random_well(kb=35, gl=-200, step=30)
y = lst.getwelldev(wella = x)#, kickoffpoint=200, final_angle=70, rateofbuild=0.2, azimuth=270)
# Call the function you want to test
output = lst.plotPPzhang(y, writeFile=False)
attrib=[10,-200,0,0,0,0,0,0]
#output = lst.plotPPzhang(x,attrib=attrib)
#output = lst.plotPPzhang(x,attrib=attrib, doi=2624)
z = lst.getwelldev(wella = lst.create_random_well(kb=312, gl=300, step=10),kickoffpoint=3000, final_angle=90, rateofbuild=0.2, azimuth=270)#Can we handle case of angle greater than 90?
output = lst.plotPPzhang(z,attrib=attrib, doi=2625, writeFile=False)
output = lst.plotPPzhang(lst.getwelldev(wella = lst.create_random_well(kb=35, gl=-200, step=10),kickoffpoint=300, final_angle=10, rateofbuild=0.2, azimuth=270),attrib=attrib, doi=2625,tecb=0.55)

#output = lst.plotPPzhang(y,attrib=[210,200,0,0,0,0,0,0], forms=form,writeFile=False)
#output = lst.plotPPzhang(y,attrib=[210,200,0,0,0,0,0,0], forms=form2, flags=flag, lithos=litho, writeFile=True)
#check with Forms

#Thats about it
# Assert statements or output checks can go here
assert output is not None, "Function did not return an output"
print("Test passed: plotPPzhang successfully executed.")

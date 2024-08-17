import numpy as np
def getColumnHeights(tvd,structop,goc,owc):
    #Returns three heights h1,h2,h3,
    #which represent heights of
    #water colum, oil column, and gas column
    #respectively
    h3 = 0
    h2 = 0
    h1 = 0
    
    if tvd<=structop:#invalid input
        return [h1,h2,h3]
    
    if np.isnan(goc) or goc==0 or goc<structop:
        goc = structop
        if np.isnan(owc) or owc==0 or owc<structop:
            owc = structop
    else:
        if np.isnan(owc) or owc==0 or owc<structop:
            owc = goc
            
    if goc>owc:#Cannot ordinarily happen
        goc=owc
    
    #print("Top Structure: ",structop," , Top Oil: ",goc," , Top Water: ",owc," , TVD is: ",tvd)
    
    
    
    if goc>0 and goc>=structop: #goc is valid and non zero
        if tvd<=goc:#given depth is in gas column
            h3 = structop - tvd
        if tvd>goc:#given depth is below gas column
            h3 = structop-goc
    
    if owc>0 and owc>=structop and owc!=goc: #owc is valid and non zero
        if tvd<=owc and tvd>goc:#given depth is in oil column
            h2 = goc - tvd
        if tvd>owc:
            h2 = goc - owc
    
    if owc>=structop and tvd>owc:
        h1 = owc - tvd
    else:
        h1 = 0
    #print(-h1,-h2,-h3)
    return [-h1,-h2,-h3]

def getPPfromTop(sealintegrity, stressratio, overburden, oilgrad, watergrad, structop,goc,owc,tvd):
    g = 9.81
    k = 0.145037737731556
    gasgrad = getGasDensity((stressratio*overburden) - sealintegrity)
    [h1,h2,h3] = getColumnHeights(tvd, structop, goc, owc)
    pp = ((stressratio*overburden) - sealintegrity) + (h1*watergrad*g*k) + (h2*oilgrad*g*k) + (h3*gasgrad*g*k)
    return pp

def getPPfromTopRecursive(sealintegrity, stressratio, overburden, oilgrad, watergrad, structop, goc, owc, tvd):
    g = 9.81
    k = 0.145037737731556
    [h1, h2, h3] = getColumnHeights(tvd, structop, goc, owc)
    # Calculate initial pressure at the top of the compartment
    p_top = (stressratio * overburden) - sealintegrity
    
    # Recursive function to calculate pressure at any point in the gas cap
    def recursivePressure(p, h, step, target_height):
        if h >= target_height:
            # Base case: reached the target height
            return p
        else:
            # Recursive case: update pressure and height, then call recursively
            gas_density = getGasDensity(p)
            gasgrad = gas_density * g
            p_next = p + gasgrad * step * 0.145037737731556  # Converting from Pa/m to psi/m
            return recursivePressure(p_next, h + step, step, target_height)

    # Set step size for recursion
    step_size = 1  # Step size in meters

    # Calculate pressure at the bottom of the gas column recursively
    p_bottom_gas = recursivePressure(p_top, 0, step_size, h3)
    
    # Calculate gas gradient at the bottom of the gas column
    gas_density_bottom = getGasDensity(p_bottom_gas)
    gasgrad = gas_density_bottom * g
    
    # Calculate the pore pressure
    pp = p_top + (h1 * watergrad * g * k) + (h2 * oilgrad * g * k) + (h3 * gasgrad * k)
    return pp

def getPPfromCentroidRecursive(sealintegrity, stressratio, overburden, oilgrad, watergrad, structop, goc, owc, tvd, strucbot,z=0.5):
    g = 9.81
    k = 0.145037737731556
    centroid = structop+((strucbot-structop)*z)
    [h1, h2, h3] = getColumnHeights(tvd, structop, goc, owc)
    # Calculate initial pressure at the top of the compartment
    p_top = (stressratio * overburden) - sealintegrity
    
    # Recursive function to calculate pressure at any point in the gas cap
    def recursivePressure(p, h, step, target_height):
        if h >= target_height:
            # Base case: reached the target height
            return p
        else:
            # Recursive case: update pressure and height, then call recursively
            gas_density = getGasDensity(p)
            gasgrad = gas_density * g
            p_next = p + gasgrad * step * 0.145037737731556  # Converting from Pa/m to psi/m
            return recursivePressure(p_next, h + step, step, target_height)

    # Set step size for recursion
    step_size = 1  # Step size in meters

    # Calculate pressure at the bottom of the gas column recursively
    p_bottom_gas = recursivePressure(p_top, 0, step_size, h3)
    
    # Calculate gas gradient at the bottom of the gas column
    gas_density_bottom = getGasDensity(p_bottom_gas)
    gasgrad = gas_density_bottom * g
    
    # Calculate the pore pressure
    pp = p_top + (h1 * watergrad * g * k) + (h2 * oilgrad * g * k) + (h3 * gasgrad * k)
    return pp


def getGasDensity(p,t = 100):#P is in psi T is in C
    #PV = nRT, P is in Pa, T is in kelvin, V is in M3, R = 8.314
    #Molar mass of methane is 16.04 g/mol
    P = p*6894.76
    T = t+273
    V = (1*T*8.314)/P #in m3
    #print(V)
    D = (0.00001604)/V #in kg/m3
    return D

def getHydrostaticPsi(tvd,gradient): #tvd in metres gradient in g/cc, returns hydrostatic head in psi
    return gradient * 8.3454063545262 * tvd * 3.28084 * 0.052

from scipy.optimize import minimize

def compute_optimal_offset(tvds, porepressures, gradient):
    def objective(offset, tvds, porepressures, gradient):
        hydrostatic = getHydrostaticPsi(tvds + offset, gradient)
        valid_mask = ~np.isnan(porepressures)
        return np.abs(np.nansum(porepressures[valid_mask] - hydrostatic[valid_mask]))
    
    # Mask out corresponding tvds where porepressures are nan
    valid_mask = ~np.isnan(porepressures)
    tvds_valid = tvds[valid_mask]
    porepressures_valid = porepressures[valid_mask]
    
    # Initial guess for the offset, scaled appropriately
    initial_guess = [0]

    result = minimize(objective, x0=initial_guess, args=(tvds_valid, porepressures_valid, gradient))
    return result.x[0]


def compute_optimal_gradient(tvds, porepressures):
    def objective(gradient, tvds, porepressures):
        hydrostatic = getHydrostaticPsi(tvds, gradient)
        valid_mask = ~np.isnan(porepressures)
        return np.abs(np.nansum(porepressures[valid_mask] - hydrostatic[valid_mask]))
    
    valid_mask = ~np.isnan(porepressures)
    tvds_valid = tvds[valid_mask]
    porepressures_valid = porepressures[valid_mask]
    
    initial_guess = [1.0]
    result = minimize(objective, x0=initial_guess, args=(tvds_valid, porepressures_valid))
    return result.x[0]
"""
def getPPfromCentroid(sealintegrity, stressratio, overburden, gasgrad, oilgrad, watergrad, structop,goc,owc,tvd, strucbottom, bottomshalepressure):
    g = 9.81
    k = 0.145037737731556
    [h1,h2,h3] = getColumnHeights(strucbottom, structop, goc, owc)
    ppbottom = (stressratio*overburden) + (h1*watergrad*g*k) + (h2*oilgrad*g*k) + (h3*gasgrad*g*k) - sealintegrity
    if ppbottom>bottomshalepressure:
        x = ppbottom-bottomshalepressure
        if x<0:seal breached unequivocally
            ppbottom = 
            
    return pp
"""

import random
def test_getColumnHeights():
    # Generate random inputs
    tvd = round(random.uniform(1000, 2000),-1)  # True vertical depth
    structop = round(random.uniform(1000, 2000),-1)  # Structural top
    goc = round(random.uniform(1000, 2000),-1)  # Gas-oil contact
    owc = round(random.uniform(1000, 2000),-1)  # Oil-water contact

    # Call the function with generated inputs
    heights = getColumnHeights(tvd, structop, goc, owc)
    
    # Compute the sum of the returned array
    height_sum = sum(heights)
    
    # Check the condition
    if tvd > structop:
        expected_sum = tvd - structop
    else:
        expected_sum = 0

    # Assert the expected condition
    assert np.isclose(height_sum, expected_sum), f"Test failed for inputs: tvd={tvd}, structop={structop}, goc={goc}, owc={owc}, got heights={heights} with sum={height_sum} but expected sum={expected_sum}"
    
    print(f"Test passed for inputs: tvd={tvd}, structop={structop}, goc={goc}, owc={owc}, heights={heights}")

"""
# Run the test function
#test_getColumnHeights()
from matplotlib import pyplot as plt
arr = np.zeros(300)
depths = np.zeros(300)
hydrostatic = np.zeros(300)
for i in range(300): 
    depths[i] = 1500+i
    arr[i] = getPPfromTopRecursive(0, 0.8, 13300,0.85, 1.025, 1500, 1600, 1700, 1500+i)
plt.plot(arr,depths)
grads = np.zeros(299)#
grad = (arr/(depths*3.28))*2.3066587258
for i in range (299):
    grads[i] = ((arr[i+1]-arr[i])/((depths[i+1]-depths[i])*3.28))*2.3066587258
plt.gca().invert_yaxis()
plt.show()
plt.plot(grads,depths[0:299])
plt.plot(grad,depths)
plt.gca().invert_yaxis()
#plt.xlim(0.75,1.25)
plt.show()

of = compute_optimal_gradient(depths, arr)
print(of)
depths2 = depths
hydros = np.zeros(300)
for i in range(300):
    hydros[i] = getHydrostaticPsi(depths2[i],of)

centroid_pressure = hydros[150]

cap_centroid = arr[150]
print(centroid_pressure,cap_centroid)
shift = centroid_pressure-cap_centroid
#arr = arr+ shift

plt.plot(arr,depths)
plt.plot(hydros,depths)
plt.gca().invert_yaxis()
plt.show()
"""
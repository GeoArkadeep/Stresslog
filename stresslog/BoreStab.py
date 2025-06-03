"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
import matplotlib.pyplot as plt2
import math

from scipy.optimize import minimize

def get_optimal_old(sx,sy,sz,alpha=0, beta=0, gamma=0):
    """
    Optimizes sx, sy, sz to achieve a specified SV
    under given angular conditions.
    
    Parameters:
    - sx, sy, sz: The values of the principal stresses
    - alpha, beta, gamma: Euler angles of the principal stresses.
    
    Returns:
    - Optimized values in the order of sx, sy, sz and Euler angles if successful; 
      otherwise, returns original values.
    """
    if min(sx,sy,sz)==sz:#reverse faulting
        optimal = get_optimalRF(sx,sy,sz,alpha, beta, gamma)
    else: #normal and strike slip
        optimal = get_optimalNS(sx,sy,sz,alpha, beta, gamma)
    return optimal[0],optimal[1],optimal[2]


def get_principal_stress(sx, sy, sz, dalpha=0, dbeta=0, dgamma=0):
    """
    Estimate the principal stresses (s1, s2, s3) such that their rotated stress tensor
    matches the given stresses (sx, sy, sz) in the geographic coordinate system.
    
    Parameters:
    - sx, sy, sz: Stresses in the geographic coordinate system
    - alpha, beta, gamma: Rotation angles (in radians)
    
    Returns:
    - s1, s2, s3: Optimized principal stresses
    """
    alpha = 0#np.radians(dalpha)
    beta = np.radians(dbeta)
    gamma = np.radians(dgamma)
    # Rotation matrix Rs
    def rotation_matrix(alpha, beta, gamma):
        return np.array([[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]])
    
    Rs = rotation_matrix(alpha, beta, gamma)
    
    # Objective function for optimization
    def objective(principal_stresses):
        # Extract s1, s2, s3
        s1, s2, s3 = principal_stresses
        # Principal stress tensor
        Sp = np.diag([s1, s2, s3])
        # Geographic stress tensor after "de-rotation"
        Sg = Rs.T @ Sp @ Rs
        
        # Extract diagonal elements of Sg
        geo_diag = np.diag(Sg)
        # Compute the cost as the sum of squared differences
        cost = ((geo_diag[0] - sx) ** 2 +
                (geo_diag[1] - sy) ** 2 +
                (geo_diag[2] - sz) ** 2)
        return cost
    
    # Initial guess for the principal stresses
    initial_guess = [sx, sy, sz]
    bounds = [
        (0, None),  # s1 must be non-negative
        (0, None),  # s2 must be non-negative
        (0, None)   # s3 must be non-negative
    ]
    # Perform optimization
    result = minimize(objective, initial_guess, method="L-BFGS-B", bounds=bounds)
    
    # Extract optimized principal stresses
    s1, s2, s3 = result.x

    return s1, s2, s3


def get_optimalNS(sx, sy, specified_SV, alpha=0, beta=0, gamma=0):
    """
    Optimizes the largest and second largest of sx, sy, sz to achieve a specified SV
    under given angular conditions without modifying the smallest of the three.
    The ordering of variables is maintained when calling the getVertical function.
    
    Parameters:
    - sx, sy: The input values along with specified_SV, one of which will be optimized.
    - specified_SV: The target sigmaV value to achieve through optimization.
    - alpha, beta, gamma: Euler angles for rotation.
    
    Returns:
    - Optimized values in the order of sx, sy, sz and Euler angles if successful; 
      otherwise, returns original values and a message indicating failure.
    """
    # Determine the smallest of the three and sort the values to find initial guesses
    sorted_indices = np.argsort([sx, sy, specified_SV])
    values = [sx, sy, specified_SV]
    sorted_values = np.sort([sx, sy, specified_SV])
    s3 = sorted_values[0]  # Smallest value, kept fixed
    initial_s2, initial_s1 = sorted_values[1:]  # Initial guesses for optimization
    specified_SH = np.max([sx,sy])
    # Define the objective function to minimize: difference from the specified_SV
    def objective(x):
        # Prepare the inputs maintaining the original order
        alpha=0.1
        inputs = [0, 0, 0]
        inputs[sorted_indices[0]] = s3
        inputs[sorted_indices[1]] = x[0]
        inputs[sorted_indices[2]] = x[1]
        calculated_SV = getVertical(*inputs, alpha, beta, gamma)
        if sorted_values[0]==specified_SV:#normal slip
            return np.abs(calculated_SV - complex(specified_SV,specified_SH))
        else:#if sorted_values[1]==specified_SV:#strike slip
            return np.abs(calculated_SV - complex(specified_SV,specified_SH))
        #if sorted_values[2]==specified_SV:#reverse slip handled seperately
        #    return np.abs(calculated_SV - complex(specified_SV,specified_SH/100000))
        
    
    
    # Constraints: s2 and s1 are bound by their relationship to s3 and each other
    constraints = (
        {'type': 'ineq', 'fun': lambda x: x[0] - s3},    # s2 >= s3
        {'type': 'ineq', 'fun': lambda x: x[1] - x[0]},  # s1 >= s2
    )
    
    # Bounds for s2 and s1, ensuring they do not exceed practical limits
    bounds = [(s3+0.1, 3.1*s3), (s3+0.1, 3.1*s3)]
    
    # Initial guess for the optimization
    initial_guess = [initial_s2, initial_s1]
    
    # Perform the optimization
    try:
        result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints,bounds=bounds,tol=0.00001) 
    #if result.success:
        optimized = [0, 0, 0]
        optimized[sorted_indices[0]] = s3
        optimized[sorted_indices[1]] = result.x[0]
        optimized[sorted_indices[2]] = result.x[1]
        return optimized[0], optimized[1], optimized[2]#, alpha, beta, gamma
    except:
        return values[0], values[1], values[2], "Optimization failed to converge. Using initial estimates."
       
def get_optimalRF(sx, sy, specified_SV, alpha=0, beta=0, gamma=0):
    """
    Optimizes stress components for reverse faulting conditions.

    Parameters
    ----------
    sx : float
        Initial horizontal stress component in x-direction
    sy : float
        Initial horizontal stress component in y-direction
    specified_SV : float
        Target vertical stress value
    alpha : float, optional
        Rotation angle alpha in degrees, default is 0
    beta : float, optional
        Rotation angle beta in degrees, default is 0
    gamma : float, optional
        Rotation angle gamma in degrees, default is 0

    Returns
    -------
    tuple or str
        If optimization succeeds:
            optimized_sx : float
                Optimized stress in x-direction
            optimized_sy : float
                Optimized stress in y-direction
            optimized_sz : float
                Optimized stress in z-direction
        If optimization fails:
            str
                Error message indicating optimization failure
    
    Notes
    -----
    Optimizes stress components under the constraint that σx > σy > σz,
    which is characteristic of reverse faulting conditions.
    """
    # Initial setup
    initial_guess = [sx, sy, specified_SV]  # All three can vary

    # Define constraints to ensure osx > osy > osz
    def constraint_osx_osy(vars):
        # osx should be greater than osy
        return vars[0] - vars[1]
    
    def constraint_osy_osz(vars):
        # osy should be greater than osz
        return vars[1] - vars[2]

    constraints = (
        {'type': 'ineq', 'fun': constraint_osx_osy},
        {'type': 'ineq', 'fun': constraint_osy_osz},
    )

    # Objective function for reverse faulting with all variables flexible
    def objective_reverse_faulting(vars):
        sx, sy, sz = vars  # Now optimizing sx, sy, sz directly
        calculated_SV = getVertical(abs(sx), abs(sy), abs(sz), alpha, beta, gamma)
        # Objective: minimize the difference from the specified SV
        # Adjusted to only consider real part of calculated_SV for comparison
        return abs(calculated_SV.real - specified_SV)

    # Perform the optimization without bounds
    result = minimize(objective_reverse_faulting, initial_guess, method='SLSQP', constraints=constraints)

    # Process the optimization result
    if result.success:
        optimized_sx, optimized_sy, optimized_sz = result.x
        return optimized_sx, optimized_sy, optimized_sz
    else:
        return "Optimization failed to converge. Using initial estimates."
    
def getVertical(sx,sy,sz,alpha=0,beta=0,gamma=0):
    """
    Calculate the vertical stress component after stress tensor rotation.

    Parameters
    ----------
    sx : float
        Stress component in x-direction
    sy : float
        Stress component in y-direction
    sz : float
        Stress component in z-direction
    alpha : float, optional
        Rotation angle alpha in degrees, default is 0
    beta : float, optional
        Rotation angle beta in degrees, default is 0
    gamma : float, optional
        Rotation angle gamma in degrees, default is 0

    Returns
    -------
    complex
        Vertical stress component with imaginary part representing maximum
        horizontal stress depending on faulting regime:
        - Normal slip: imaginary part is max(σyy, σxx)
        - Strike slip: imaginary part is max(σyy, σxx)
        - Reverse slip: imaginary part is 0
    """
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    
    sorted_values = np.sort([sx, sy, sz])
    
    Rs = np.array([[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]])
    #print(Rs)
    RsT = np.transpose(Rs)
    #s3 = s3 + (s3 - s3*Rs[2][2])/Rs[2][2]
    Ss = np.array([[sx,0,0],[0,sy,0],[0,0,sz]])
    Sg = RsT@Ss@Rs

    if sorted_values[2]==sz:#normal slip
        return complex(Sg[2][2],max(Sg[1][1],Sg[0][0]))
    if sorted_values[1]==sz:#strike slip
        return complex(Sg[2][2],max(Sg[1][1],Sg[0][0]))
    if sorted_values[0]==sz:#reverse slip
        return complex(Sg[2][2],0)
    #return complex(Sg[2][2],max(Sg[1][1],Sg[0][0]))
    
def getAlignedStress(sx,sy,sz,alpha,beta,gamma,azim,inc):
    """
    Calculate the stress tensor aligned to a given well azimuth and inclination (Borehole coordinate system)

    Parameters
    ----------
    sx : float
        Stress component in x-direction
    sy : float
        Stress component in y-direction
    sz : float
        Stress component in z-direction
    alpha : float
        First rotation angle in degrees
    beta : float
        Second rotation angle in degrees
    gamma : float
        Third rotation angle in degrees
    azim : float
        Azimuth angle in degrees
    inc : float
        Inclination angle in degrees

    Returns
    -------
    ndarray
        3x3 stress tensor in the rotated coordinate system
    """

    Ss = np.array([[sx,0,0],[0,sy,0],[0,0,sz]])
    #print(Ss)

    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    Rs = np.array([[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]])
    #print(Rs)
    sVo = np.array([[0.0],[0.0],[1.0]])
    sNo = np.array([[1.0],[0.0],[0.0]])
    sEo = np.array([[0.0],[1.0],[0.0]])
    
    uvec = getOrit(sx,sy,sz,alpha,beta,gamma)
    sVr = uvec[0]#Rs@sVo
    sNr = uvec[1]#Rs@sNo
    sEr = uvec[2]#Rs@sEo

    sNt1 = np.degrees(np.arctan2(sNr[1],sNr[0]))
    sNt2 =np.degrees(np.arctan2((np.hypot(sNr[0],sNr[1])),sNr[2]))
    sEt1 =np.degrees(np.arctan2(sEr[1],sEr[0]))
    sEt2 =np.degrees(np.arctan2(((np.hypot(sEr[0],sEr[1]))),sEr[2]))
    sVt2 =np.degrees(np.arctan2(((np.hypot(sVr[0],sVr[1]))),sVr[2]))
    sVt1 =np.degrees(np.arctan2(sVr[1],sVr[0]))
    if sVt1>90:
        #print("Hey",sVt2)
        sVt1=180-sVt1
        
    if sNt1>90:
        sNt1=180-sNt1
    if sEt1>90:
        sEt1=180-sEt1
    
    orit = [sNt1,sNt2,sEt1,sEt2,sVt2,sVt1]
    
    delta = math.radians(azim)
    phi   = math.radians(inc)
 
    Rb = np.array([[(-1)*math.cos(delta)*math.cos(phi), (-1)*math.sin(delta)*math.cos(phi), math.sin(phi)],
                   [math.sin(delta), (-1)*math.cos(delta), 0],
                   [math.cos(delta)*math.sin(phi), math.sin(delta)*math.sin(phi), math.cos(phi)]])
    #print(Rb)
    RsT = np.transpose(Rs)
    RbT = np.transpose(Rb)

    Sg = RsT@Ss@Rs
    #print(Sg)
    Sb = Rb@RsT@Ss@Rs@RbT
    return Sb

    

def getRota(alpha,beta,gamma):
    """
    Generate a rotation matrix from Euler angles.

    Parameters
    ----------
    alpha : float
        First rotation angle in degrees
    beta : float
        Second rotation angle in degrees
    gamma : float
        Third rotation angle in degrees

    Returns
    -------
    ndarray
        3x3 rotation matrix
    """
    
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    Rs = np.array([[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]])
    return Rs


def getStens(sx,sy,sz,alpha,beta,gamma,debug=False):
    """
    Calculate the stress tensor in the NED (Geographic) Coordinate System.

    Parameters
    ----------
    sx : float
        Stress component in x-direction
    sy : float
        Stress component in y-direction
    sz : float
        Stress component in z-direction
    alpha : float
        First rotation angle in degrees
    beta : float
        Second rotation angle in degrees
    gamma : float
        Third rotation angle in degrees

    Returns
    -------
    tuple
        Three 1D arrays representing the rows of the rotated stress tensor
        (σxx, σxy, σxz), (σyx, σyy, σyz), (σzx, σzy, σzz)

    Notes
    -----
    Optionaly prints eigenvalues, eigenvectors, vector dip, dip direction, and
    vertical/horizontal stress components.
    """
    #print(Ss)

    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    Rs = np.array([[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]])
    #print(Rs)
    RsT = np.transpose(Rs)
    Ss = np.array([[sx,0,0],[0,sy,0],[0,0,sz]])
    Sg = RsT@Ss@Rs
    print(Sg) if debug else None
    print(np.linalg.eigh(Sg)) if debug else None
    print("Vector Dip is:",np.degrees(np.arccos(Rs[2][2]))) if debug else None
    dip_direction = np.degrees(np.arctan2(Rs[2][1], Rs[2][0]))
    print("Dip Direction is:", dip_direction) if debug else None
    print("Calculated Vertical Component is:", Sg[2][2]) if debug else None
    print("Calculated Max Horizontal Component is:", max(Sg[1][1],Sg[0][0])) if debug else None
    print("Calculated Min Horizontal Component is:", min(Sg[1][1],Sg[0][0])) if debug else None  
    return Sg[0],Sg[1],Sg[2]

def getStrikeDip(alpha,beta,gamma):
    """
    Calculate strike, dip, and dip direction from Euler angles.

    Parameters
    ----------
    alpha : float
        First rotation angle in degrees
    beta : float
        Second rotation angle in degrees
    gamma : float
        Third rotation angle in degrees

    Returns
    -------
    tuple
        strike_direction : float
            Strike direction in degrees
        dip_angle : float
            Dip angle in degrees
        dip_direction : float
            Dip direction in degrees
    """
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    Rs = np.array([[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]])
    dip_direction = np.degrees(np.arctan2(Rs[2][1], Rs[2][0]))
    dip_angle = np.degrees(np.arccos(Rs[2][2]))
    strike_direction = (dip_direction+180)%360
    return strike_direction,dip_angle,dip_direction

def getEuler(alpha,strike, dip):
    """
    Optimize beta and gamma angles to match given strike and dip.

    Parameters
    ----------
    alpha : float
        Fixed rotation angle alpha in degrees
    strike : float
        Target strike angle in degrees
    dip : float
        Target dip angle in degrees

    Returns
    -------
    tuple
        beta_opt : float
            Optimized beta angle in degrees
        gamma_opt : float
            Optimized gamma angle in degrees

    Notes
    -----
    Uses Nelder-Mead optimization to find beta and gamma angles that
    produce the desired strike and dip angles.
    """
    def objective_function(x, strike, dip):
        beta, gamma = x
        estimated_strike, estimated_dip, _ = getStrikeDip(alpha,beta, gamma)
        return (estimated_strike - strike)**2 + (estimated_dip - dip)**2
    bounds = [(0, None), (0, None)]
    initial_guess = [360, 360]  # Initial guess for beta and gamma
    result = minimize(objective_function, initial_guess, args=(strike, dip), method='Nelder-Mead')
    beta_opt, gamma_opt = result.x
    beta_opt=beta_opt%360
    gamma_opt=gamma_opt%360
    if beta_opt>180:
        beta_opt=beta_opt-360
    if gamma_opt>180:
        gamma_opt=gamma_opt-360
    return beta_opt, gamma_opt

def getOrit(s1,s2,s3,alpha,beta,gamma):
    """
    Calculate principal stress directions after stress tensor rotation.

    Parameters
    ----------
    s1 : float
        First principal stress magnitude
    s2 : float
        Second principal stress magnitude
    s3 : float
        Third principal stress magnitude
    alpha : float
        First rotation angle in degrees
    beta : float
        Second rotation angle in degrees
    gamma : float
        Third rotation angle in degrees

    Returns
    -------
    ndarray
        3x3 matrix where each column represents a principal stress direction
        in the rotated coordinate system. The columns correspond to the
        vertical, north, and east directions respectively.
    """

    Ss = np.array([[s1,0,0],[0,s2,0],[0,0,s3]])
    #print(Ss)

    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    Rs = np.array([[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]])
    #print(Rs)
    RsT = np.transpose(Rs)
    Sg = RsT@Ss@Rs
    orit = np.linalg.eigh(Sg)[1]
    return(orit)

def getSigmaTT(s1,s2,s3,alpha,beta,gamma,azim,inc,theta,deltaP,Pp,nu=0.35,bt=0,ym=0,delT=0): #Converts farfield stress tensor to at-wall stress, at a single point on the wall of an inclined borehole
    """
    Calculate stress components at a point on the wall of an inclined borehole.

    Parameters
    ----------
    s1 : float
        First principal stress magnitude
    s2 : float
        Second principal stress magnitude
    s3 : float
        Third principal stress magnitude
    alpha : float
        First rotation angle in degrees
    beta : float
        Second rotation angle in degrees
    gamma : float
        Third rotation angle in degrees
    azim : float
        Borehole azimuth in degrees
    inc : float
        Borehole inclination in degrees
    theta : float
        Angular position on borehole wall in degrees
    deltaP : float
        Difference between mud pressure and pore pressure
    Pp : float
        Pore pressure
    nu : float, optional
        Poisson's ratio, default is 0.35
    bt : float, optional
        Linear thermal expansion coefficient, default is 0
    ym : float, optional
        Young's modulus, default is 0
    delT : float, optional
        Temperature difference, default is 0

    Returns
    -------
    tuple
        Stt : float
            Tangential stress
        Szz : float
            Axial stress
        Ttz : float
            Shear stress
        STMax : float
            Maximum principal stress
        STMin : float
            Minimum principal stress
        omega : float
            Angle between maximum principal stress and borehole axis
        orit : list
            List containing orientation angles [NorthAzimuth, NorthInclination, 
            EastAzimuth, EastInclination, VerticalInclination, VerticalAzimuth]

    Notes
    -----
    This function converts far-field stress tensor to at-wall stress state at a
    single point on the wall of an inclined borehole, accounting for thermal and
    poroelastic effects.
    """
    Ss = np.array([[s1,0,0],[0,s2,0],[0,0,s3]])
    #print(Ss)

    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    Rs = np.array([[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]])
    #print(Rs)
    sVo = np.array([[0.0],[0.0],[1.0]])
    sNo = np.array([[1.0],[0.0],[0.0]])
    sEo = np.array([[0.0],[1.0],[0.0]])
    
    uvec = getOrit(s1,s2,s3,alpha,beta,gamma)
    sVr = uvec[0]#Rs@sVo
    sNr = uvec[1]#Rs@sNo
    sEr = uvec[2]#Rs@sEo

    sNt1 = np.degrees(np.arctan2(sNr[1],sNr[0]))
    sNt2 =np.degrees(np.arctan2((np.hypot(sNr[0],sNr[1])),sNr[2]))
    sEt1 =np.degrees(np.arctan2(sEr[1],sEr[0]))
    sEt2 =np.degrees(np.arctan2(((np.hypot(sEr[0],sEr[1]))),sEr[2]))
    sVt2 =np.degrees(np.arctan2(((np.hypot(sVr[0],sVr[1]))),sVr[2]))
    sVt1 =np.degrees(np.arctan2(sVr[1],sVr[0]))
    if sVt1>90:
        #print("Hey",sVt2)
        sVt1=180-sVt1
        
    if sNt1>90:
        sNt1=180-sNt1
    if sEt1>90:
        sEt1=180-sEt1
    
    orit = [sNt1,sNt2,sEt1,sEt2,sVt2,sVt1]
    
    delta = math.radians(azim)
    phi   = math.radians(inc)
 
    Rb = np.array([[(-1)*math.cos(delta)*math.cos(phi), (-1)*math.sin(delta)*math.cos(phi), math.sin(phi)],
                   [math.sin(delta), (-1)*math.cos(delta), 0],
                   [math.cos(delta)*math.sin(phi), math.sin(delta)*math.sin(phi), math.cos(phi)]])
    #print(Rb)
    RsT = np.transpose(Rs)
    RbT = np.transpose(Rb)

    Sg = RsT@Ss@Rs
    #print(Sg)
    Sb = Rb@RsT@Ss@Rs@RbT
    #print(Sb)
    Sb[0][0] = Sb[0][0] - Pp
    Sb[1][1] = Sb[1][1] - Pp
    Sb[2][2] = Sb[2][2] - Pp
    
    theta = math.radians(theta)
    sigmaT = (ym*bt*delT)/(1-nu)

    Szz = Sb[2][2] - ((2*nu)*(Sb[0][0]-Sb[1][1])*(2*math.cos(2*theta))) - (4*nu*Sb[0][1]*math.sin(2*theta))
    Stt = Sb[0][0] + Sb[1][1] -(2*(Sb[0][0] - Sb[1][1])*math.cos(2*theta)) - (4*Sb[0][1]*math.sin(2*theta)) - deltaP -sigmaT
    Ttz = 2*((Sb[1][2]*math.cos(theta))-(Sb[0][2]*math.sin(theta)))
    Srr = deltaP
    
    #print(Szz,Stt,Ttz,Srr)

    STMax = 0.5*(Szz + Stt + (((Szz-Stt)**2)+(4*(Ttz**2)))**0.5)
    Stmin = 0.5*(Szz + Stt - (((Szz-Stt)**2)+(4*(Ttz**2)))**0.5)
    #omega = np.degrees(np.arctan2(Szz,(((STMax**2)-(Szz**2))**0.5)))
    tens2d = [[Stt,Ttz],[Ttz,Szz]]
    twoomega = np.degrees(np.arctan2(2*Ttz,Szz-Stt))
    omega2 = twoomega/2
    if omega2==0:
        omega2=0.00000000001
    omega = omega2
    if omega==0:
        omega=0.00000000001
    #omega = np.degrees(np.arctan2(np.linalg.eigh(tens2d)[1][0][0],np.linalg.eigh(tens2d)[1][0][1]))
    #if theta>math.radians(180):
        #omega = np.degrees(np.arctan2((((Stt**2)-(Szz**2))**0.5),-Szz))
        #omega = 180-np.degrees(np.arctan2(Stt,Szz))
    #print(STMax-Stmin, np.degrees(theta))
    return Stt,Szz,Ttz,STMax,Stmin,omega,orit


def getHoop(inc,azim,s1,s2,s3,deltaP,Pp, ucs, alpha=0,beta=0,gamma=0,nu=0.35,bt=0,ym=0,delT=0,path=None,ten_fac=10):
    """Calculate and plot hoop stresses around a wellbore circumference.

    This function computes various stress components around the wellbore wall and generates
    a plot showing hoop stresses, stress angles, and identifies regions of potential failure.
    It uses modified Zhang equations for stress calculations.

    Parameters
    ----------
    inc : float
        Wellbore inclination in degrees
    azim : float
        Wellbore azimuth in degrees
    s1 : float
        Maximum principal stress
    s2 : float
        Intermediate principal stress
    s3 : float
        Minimum principal stress
    deltaP : float
        Pressure differential (wellbore pressure - pore pressure)
    Pp : float
        Pore pressure
    ucs : float
        Unconfined compressive strength
    alpha : float, optional
        Principal stress rotation angle alpha in degrees, default 0
    beta : float, optional
        Principal stress rotation angle beta in degrees, default 0
    gamma : float, optional
        Principal stress rotation angle gamma in degrees, default 0
    nu : float, optional
        Poisson's ratio, default 0.35
    bt : float, optional
        Biot's coefficient, default 0
    ym : float, optional
        Young's modulus, default 0
    delT : float, optional
        Temperature difference, default 0
    path : str, optional
        File path to save the plot. If None, returns the plot object

    Returns
    -------
    tuple
        If path is provided:
            - crush : ndarray
                Binary array indicating compressive failure regions (1 for failure)
            - frac : ndarray
                Binary array indicating tensile failure regions (1 for failure)
            - minstress : int
                Index of minimum stress location in first 180 degrees
            - maxstress : int
                Index of maximum stress location in first 180 degrees
            - angle_min : float
                Principal stress angle at minimum stress location
            - angle_min_opposite : float
                Principal stress angle at opposite of minimum stress location
            - angle : ndarray
                Array of principal stress angles around wellbore
        
        If path is None:
            Returns all above plus matplotlib.pyplot object as the last element

    Notes
    -----
    The function calculates:
    - Tangential (hoop) stresses
    - Axial stresses
    - Shear stresses
    - Principal stress angles
    - Potential failure regions (both tensile and compressive)
    
    The plot shows:
    - Principal stress angles
    - Effective hoop stresses (STT - Pp)
    - Effective axial stresses (SZZ - Pp)
    - Shear stresses (TTZ)

    Uses Zhang's equations with internal friction angle calculated from
    Poisson's ratio: phi = arcsin(1-2nu)
    """
    phi = np.arcsin(1-(2*nu)) #unModified Zhang
    mui = (1+np.sin(phi))/(1-np.sin(phi))
    fmui = ((((mui**2)+1)**0.5)+mui)**2
    
    #values = np.zeros((10,37))
    
    pointer= alpha
    line = np.zeros(360)
    line2 = np.zeros(360)
    eline = np.zeros(360)
    eline2 = np.zeros(360)
    line1 = np.zeros(360)
    angle= np.zeros(360)
    width= 0
    frac = np.zeros(360)
    crush = np.zeros(360)
    widthR = np.zeros(360)
    ts = -ucs/ten_fac if ten_fac>0 else 0
    while pointer<360+alpha:
        STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim, inc, pointer, deltaP,Pp,nu,bt,ym,delT)
        line[round(pointer%360)] = stm
        line2[round(pointer%360)] = STM
        eline[round(pointer%360)] = STT - Pp
        eline2[round(pointer%360)] = SZZ - Pp        
        line1[round(pointer%360)] = TTZ
        angle[round(pointer%360)] = omega
        
        if stm<ts:
            width+=1
            frac[round(pointer%360)] = 1
        else:
            frac[round(pointer%360)] = 0
        
        if ucs<((STM)-(fmui*(deltaP))):
            crush[round(pointer%360)] = 1
        else:
            crush[round(pointer%360)] = 0
        #if pointer>180:
            #frac[pointer] = frac[360-pointer]
        widthR[round(pointer%360)] = ((round(pointer%360))/360)*0.67827 #in metres
        pointer+=1
    minstress = np.argmin(line[0:180])
    maxstress = np.argmax(line2[0:180])
    minstress2 = minstress+180
    maxstress2 = maxstress+180
    
    # Plotting
    fig, ax1 = plt2.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()  # Secondary y-axis for angle

    # Primary y-axis: hoop, axial, shear stresses (auto-scaled)
    l1, = ax1.plot(eline, label="Hoop Stress", color="b")
    l2, = ax1.plot(eline2, label="Axial Stress", color="g")
    l3, = ax1.plot(line1, label="Shear Stress", color="r")
    ax1.set_xlabel("Circumferential Angle (degrees)")
    ax1.set_ylabel("Stress (MPa)")
    #ax1.legend(loc="upper left")

    # Secondary y-axis: principal angles (-180 to 180 degrees)
    l4, = ax2.plot(angle, label="Principal Angle", color="purple", linestyle="dashed")
    ax2.set_ylim(-90, 90)
    ax2.set_xlim(0, 360)
    ax2.set_ylabel("Angle w.r.t bore-axis (degrees)")
    #ax2.legend(loc="upper right")

    # Adjust the position of both axes to make space for the legend
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0 + box1.height * 0.1, box1.width, box1.height * 0.9])

    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0 + box2.height * 0.1, box2.width, box2.height * 0.9])

    # Shared legend between the plot and title
    fig.legend(handles=[l1, l2, l3, l4], loc=8, bbox_to_anchor=(0.5125, 0.88), ncol=4)

    # Title with padding to avoid overlap
    size = fig.get_size_inches()*fig.dpi # get fig size in pixels
    plt2.title("Hoop Stresses and Principal Stress Angles", pad=30)

    if path is not None:
        plt2.savefig(path)
        plt2.close()
        return crush,frac,minstress,maxstress,angle[minstress],angle[(minstress+180)%360],angle
    else:
        b64png = plot_to_base64_png(plt2)
        plt2.close()
        return crush,frac,minstress,maxstress,angle[minstress],angle[(minstress+180)%360],angle,b64png

def draw(tvd,s1,s2,s3,deltaP,Pp,UCS = 0,alpha=0,beta=0,gamma=0,offset=0,nu=0.35,  azimuthu=0,inclinationi=0,bt=0,ym=0,delT=0,path=None,ten_fac=10, debug=False, display=False):
    """Generate wellbore stability plots showing mud weight headroom and breakout widths.

    This function creates two polar projection plots:
    1. A contour plot showing mud weight headroom in SG units
    2. A contour plot showing breakout widths in degrees
    
    The plots are generated for various wellbore orientations (inclination and azimuth)
    considering in-situ stresses, rock properties, and wellbore conditions.

    Parameters
    ----------
    tvd : float
        True vertical depth in meters
    s1 : float
        Maximum principal stress
    s2 : float
        Intermediate principal stress
    s3 : float
        Minimum principal stress
    deltaP : float
        Pressure differential (wellbore pressure - pore pressure)
    Pp : float
        Pore pressure
    UCS : float, optional
        Unconfined compressive strength, default 0
    alpha : float, optional
        Principal stress rotation angle alpha in degrees, default 0
    beta : float, optional
        Principal stress rotation angle beta in degrees, default 0
    gamma : float, optional
        Principal stress rotation angle gamma in degrees, default 0
    offset : float, optional
        Azimuthal offset in degrees, default 0
    nu : float, optional
        Poisson's ratio, default 0.35
    azimuthu : float, optional
        Wellbore azimuth in degrees, default 0
    inclinationi : float, optional
        Wellbore inclination in degrees, default 0
    bt : float, optional
        Biot's coefficient, default 0
    ym : float, optional
        Young's modulus, default 0
    delT : float, optional
        Temperature difference, default 0
    path : str, optional
        File path to save the plot. If None, returns the plot object
    debug : bool, optional
        Prints debug statements to console
    display : bool, optional
        Displays the plot in addition to saving

    Returns
    -------
    matplotlib.pyplot
        If path is None, returns the matplotlib pyplot object containing the stability plots
    None
        If path is provided, saves the plot to the specified path and returns None

    Notes
    -----
    The function uses Zhang's equation (phi = np.arcsin(1-(2*nu))) for wellbore stability analysis.
    The first plot shows mud weight headroom in SG units with a jet_r colormap.
    The second plot shows breakout widths in degrees with a jet colormap.
    Both plots use polar projections with inclination (0-90°) and azimuth (0-360°).
    
    The plots include:
    - Contour plots of stability parameters
    - Green marker showing the actual wellbore orientation
    - Horizontal colorbars with appropriate units
    - Title showing TVD and key parameters (UCS, deltaP, deltaT, Nu)
    """
    #phi = 183-(163*nu) ## wayy too high
    #phi = np.arcsin(1-(nu/(1-nu))) #Still too high
    phi = np.arcsin(1-(2*nu)) #unModified Zhang
    mui = (1+np.sin(phi))/(1-np.sin(phi))
    #mui = 1.9
    print("Mu_i = ",mui) if debug else None
    fmui = ((((mui**2)+1)**0.5)+mui)**2
    values = np.zeros((10,37))
    values2 = np.zeros((10,37))
    inclination = np.zeros((10,37))
    azimuth = np.zeros((10,37))
    inc = 0
    TS = -UCS/ten_fac if ten_fac>0 else 0
    #TS = 0
    while inc<10:
        azim = 0
        while azim<37:
            pointer= 0
            line = np.zeros(360)
            line2 = np.zeros(360)
            angle= np.zeros(360)
            width= 0
            width2 = 0
            frac = np.zeros(360)
            widthR = np.zeros(360)
            while pointer<360:
                STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim*10, inc*10, pointer,deltaP,Pp,nu,bt,ym,delT)
                line[pointer] = STT
                angle[pointer] = omega
                if stm<TS:
                    width+=1
                    frac[pointer] = frac[pointer-1]+(1/math.tan(math.radians(omega)))
                else:
                    frac[pointer] = 0
                #if pointer>180:
                    #frac[pointer] = frac[360-pointer]
                widthR[pointer] = (pointer/360)*0.67827 #in metres
                pointer+=1
                
                if UCS<((STM)-(fmui*(deltaP))):
                    width2+=0.5
                    
            #if width>0:
                #print("Width = ",width/2,", omega =",np.max(angle), " at inclination = ",inc*10, " and azimuth= ",azim*10)
                #plt2.scatter(np.array(range(0,360)),frac)
                #plt2.plot(angle)
                #plt2.plot(line)
                #plt2.xlim((0,0.67827))
                #plt2.ylim((1,151))
                #plt2.show()
            values[inc][azim] = np.min(line)
            values2[inc][azim] = width2
            inclination[inc][azim] = inc*10
            azimuth[inc][azim] = math.radians(azim*10+offset)
            azim+=1
        #print(round((inc/10)*100),"%")
        inc+=1

        
    print(orit) if debug else None
    
    fig = plt2.figure()
    ax = fig.add_subplot(121,projection='polar')
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_rmax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    levels = np.linspace(0,np.min([s1,s2,s3]),1000)
    cax = ax.contourf(azimuth, inclination, values, 1000, levels=levels, extend = 'both', cmap = 'jet_r', alpha = 1)
    ax.scatter(math.radians(azimuthu),inclinationi, s=50, color = 'green', edgecolors='black', label='Bore')

    
    #ax.scatter(math.radians(orit[0]),orit[1], s=20, color = 'black', edgecolors='black', label=s3)
    #ax.text(math.radians(orit[0]),orit[1], " "+str(round(s3,1)))
    #if(orit[3]<=90):
    #ax.scatter(math.radians(-orit[2]),orit[3], s=20, color = 'black', edgecolors='black', label=s1)
    #ax.text(math.radians(-orit[2]),orit[3], " "+str(round(s1,1)))
    #else:
    #    ax.scatter(math.radians(-orit[2]),(90-(orit[3]-90)), s=20, color = 'white', edgecolors='black', label=s1)
    #    ax.text(math.radians(-orit[2]),(90-(orit[3]-90)), " "+str(round(s1,1)))
    #if(orit[5]<=90):
    #ax.scatter(math.radians(-orit[4]),orit[5], s=20, color = 'black', edgecolors='black',label=s2)
    #ax.text(math.radians(-orit[4]),orit[5], " "+str(round(s2,1)))
    #else:
    #    ax.scatter(math.radians(-orit[4]),(90-(orit[5]-90)), s=20, color = 'white', edgecolors='black', label=s2)
    #    ax.text(math.radians(-orit[4]),(90-(orit[5]-90)), " "+str(round(s2,1)))
    conversion_constantSG = 0.102/(tvd/1000)  # Change this to your desired conversion constant
    conversion_constantPPG = 0.102*8.345/(tvd/1000)  # Change this to your desired conversion constant
    ticks = np.linspace(0, np.min([s1,s2,s3]), 7)  # 10 evenly spaced ticks from 0 to s3
    ticks = np.round(ticks, 2)  # Round to one decimal place
    cb = fig.colorbar(cax, ticks=ticks,orientation = 'horizontal')
    current_ticks = cb.get_ticks()  # Get current tick locations
    new_labels = [f"{value * conversion_constantSG:.1f}" for value in current_ticks]  # Create custom labels
    cb.set_ticklabels(new_labels)  # Set new labels without changing positions
    #plt2.title( "DeltaP = "+str(round(deltaP,2))+", Nu = "+str(round(nu,2)) , loc="center")
    cb.set_label("Mud Weight Headroom in SG")
    
    aws = fig.add_subplot(122,projection='polar')
    aws.grid(False)
    aws.set_yticklabels([])
    aws.set_xticklabels([])
    aws.set_rmax(90)
    aws.set_theta_zero_location("N")
    aws.set_theta_direction(-1)
    levels = np.linspace(0,120,1300)
    cax2 = aws.contourf(azimuth, inclination, values2, 1300, levels=levels, extend = 'both', cmap = 'jet', alpha = 1)
    print(orit) if debug else None
    aws.scatter(math.radians(azimuthu),inclinationi, s=50, color = 'green', edgecolors='black', label='Bore')
    #aws.text(math.radians(orit[0]),orit[1], " "+str(round(s3,1)))
    #aws.scatter(math.radians(orit[0]),orit[1], s=20, color = 'black', edgecolors='black', label=s3)
    #aws.text(math.radians(orit[0]),orit[1], " "+str(round(s3,1)))
    #if(orit[3]<90):
    #aws.scatter(math.radians(-orit[2]),orit[3], s=20, color = 'black', edgecolors='black', label=s1)
    #aws.text(math.radians(-orit[2]),orit[3], " "+str(round(s1,1)))
    #else:
    #    aws.scatter(math.radians(-orit[2]),(90-(orit[3]-90)), s=20, color = 'white', edgecolors='black', label=s1)
    #    aws.text(math.radians(-orit[2]),(90-(orit[3]-90)), " "+str(round(s1,1)))
    #if(orit[5]<90):
    #aws.scatter(math.radians(-orit[4]),orit[5], s=20, color = 'black', edgecolors='black',label=s2)
    #aws.text(math.radians(-orit[4]),orit[5], " "+str(round(s2,1)))
    #else:
    #    aws.scatter(math.radians(-orit[4]),(90-(orit[5]-90)), s=20, color = 'white', edgecolors='black', label=s2)
    #    ax.text(math.radians(-orit[4]),(90-(orit[5]-90)), " "+str(round(s2,1)))
    cb2 = fig.colorbar(cax2, ticks=[0,20,40,60,80,100,120], orientation = 'horizontal')
    cb2.set_label("Breakout Widths in Degrees")
    fig.suptitle("Stability Plot at "+str(round(tvd,2))+"m TVD")
    fig.text(0.5, 0.87, "UCS = " + str(round(UCS)) + ", DeltaP = " + str(round(deltaP)) + ", DeltaT = " + str(round(delT,2)) + ", Nu = " + str(round(nu,2)), 
         ha='center', fontsize=10)
    if path is not None:
        plt2.savefig(path,dpi=600)
        if display:
            plt2.show()
        plt2.clf()
    else:
        return plt2

def critical_bhp_calculator(bhp, pp, sigmaT, nu, ucs, Sb, theta):
    """
    Calculate the difference between the minimum principal stress (Stmin) and the tensile strength (tensilestrength).
    
    Parameters:
    bhp (float): Bottomhole pressure
    pp (float): Pore pressure
    ym (float): Young's modulus
    bt (float): Biot coefficient
    delT (float): Temperature change
    nu (float): Poisson's ratio
    ucs (float): Uniaxial compressive strength
    Sb (numpy.ndarray): In-situ stress tensor
    
    Returns:
    float: Difference between minimum principal stress (Stmin) and tensile strength (tensilestrength)
    """
    
    deltaP = bhp - pp
    #sigmaT = (ym * bt * delT) / (1 - nu)
    tensilestrength = -(ucs / 10)
    

    Szz = Sb[2][2] - ((2 * nu) * (Sb[0][0] - Sb[1][1]) * (2 * math.cos(2 * theta))) - (4 * nu * Sb[0][1] * math.sin(2 * theta))
    Stt = Sb[0][0] + Sb[1][1] - (2 * (Sb[0][0] - Sb[1][1]) * math.cos(2 * theta)) - (4 * Sb[0][1] * math.sin(2 * theta)) - deltaP - sigmaT
    Ttz = 2 * ((Sb[1][2] * math.cos(theta)) - (Sb[0][2] * math.sin(theta)))
    
    STMax = 0.5 * (Szz + Stt + ((Szz - Stt) ** 2 + 4 * (Ttz ** 2)) ** 0.5)
    Stmin = 0.5 * (Szz + Stt - ((Szz - Stt) ** 2 + 4 * (Ttz ** 2)) ** 0.5)
        
    return abs(Stmin - tensilestrength)

def get_critical_bhp(Sb, pp, ucs, theta, nu=0.25, sigmaT=0 ):
    """
    Find the critical borehole pressure (critical_bhp) by minimizing the difference between the minimum principal stress (Stmin) and the tensile strength (tensilestrength).
    
    Parameters:
    pp (float): Pore pressure
    ym (float): Young's modulus
    bt (float): Biot coefficient
    delT (float): Temperature change
    nu (float): Poisson's ratio
    ucs (float): Uniaxial compressive strength
    Sb (numpy.ndarray): In-situ stress tensor
    
    Returns:
    float: Critical borehole pressure (critical_bhp)
    """
    res = minimize(critical_bhp_calculator, x0=pp, args=(pp, sigmaT, nu, ucs, Sb, theta), method='Nelder-Mead')    
    if not res.success:
        return np.nan
    
    critical_bhp = res.x[0]
    return critical_bhp

def get_bhp_critical(Sb, pp, ucs, theta, nu=0.25, sigmaT=0):
    """
    Calculate the critical bottomhole pressure (BHP) based on a closed-form solution.
    
    Parameters:
    pp (float): Pore pressure
    ucs (float): Uniaxial compressive strength
    Sb (numpy.ndarray): At-wall stress tensor (3x3 matrix)
    theta (float): Circumferential angle in radians corresponding to the minimum principal stress on the hole wall
    nu (float): Poisson's ratio (default is 0.25)
    sigmaT (float): Thermal stress (default is 0)
    
    Returns:
    float: Critical bottomhole pressure (BHP)
    """
    
    # Extract components of the stress tensor for readability
    Sb11, Sb12, Sb13 = Sb[0, 0], Sb[0, 1], Sb[0, 2]
    Sb21, Sb22, Sb23 = Sb[1, 0], Sb[1, 1], Sb[1, 2]
    Sb31, Sb32, Sb33 = Sb[2, 0], Sb[2, 1], Sb[2, 2]
    
    # Numerator of the closed-form expression
    numerator = (
        10 * pp * ucs - 10 * sigmaT * ucs + ucs ** 2 
        + 10 * ucs * Sb11 - 200 * nu * pp * np.cos(2 * theta) * Sb11
        + 200 * nu * sigmaT * np.cos(2 * theta) * Sb11 - 20 * ucs * np.cos(2 * theta) * Sb11
        - 20 * nu * ucs * np.cos(2 * theta) * Sb11 - 200 * nu * np.cos(2 * theta) * Sb11 ** 2
        + 400 * nu * (np.cos(2 * theta) ** 2) * Sb11 ** 2
        + 10 * ucs * Sb22 + 200 * nu * pp * np.cos(2 * theta) * Sb22
        - 200 * nu * sigmaT * np.cos(2 * theta) * Sb22 + 20 * ucs * np.cos(2 * theta) * Sb22
        + 20 * nu * ucs * np.cos(2 * theta) * Sb22 - 800 * nu * (np.cos(2 * theta) ** 2) * Sb11 * Sb22
        + 200 * nu * np.cos(2 * theta) * Sb22 ** 2 + 400 * nu * (np.cos(2 * theta) ** 2) * Sb22 ** 2
        - 400 * (np.cos(theta) ** 2) * Sb23 ** 2 + 100 * pp * Sb33 - 100 * sigmaT * Sb33
        + 10 * ucs * Sb33 + 100 * Sb11 * Sb33 - 200 * np.cos(2 * theta) * Sb11 * Sb33
        + 100 * Sb22 * Sb33 + 200 * np.cos(2 * theta) * Sb22 * Sb33
        + 800 * np.cos(theta) * Sb13 * Sb23 * np.sin(theta) - 400 * Sb13 ** 2 * (np.sin(theta) ** 2)
        - 400 * nu * pp * Sb12 * np.sin(2 * theta) + 400 * nu * sigmaT * Sb12 * np.sin(2 * theta)
        - 40 * ucs * Sb12 * np.sin(2 * theta) - 40 * nu * ucs * Sb12 * np.sin(2 * theta)
        - 400 * nu * Sb11 * Sb12 * np.sin(2 * theta) + 1600 * nu * np.cos(2 * theta) * Sb11 * Sb12 * np.sin(2 * theta)
        - 400 * nu * Sb12 * Sb22 * np.sin(2 * theta) - 1600 * nu * np.cos(2 * theta) * Sb12 * Sb22 * np.sin(2 * theta)
        - 400 * Sb12 * Sb33 * np.sin(2 * theta) + 1600 * nu * Sb12 ** 2 * (np.sin(2 * theta) ** 2)
    )
    
    # Denominator of the closed-form expression
    denominator = (
        10 * ucs - 200 * nu * np.cos(2 * theta) * Sb11 + 200 * nu * np.cos(2 * theta) * Sb22
        + 100 * Sb33 - 400 * nu * Sb12 * np.sin(2 * theta)
    )
    
    # Return the calculated critical BHP
    return numerator / denominator

def get_frac_pressure(Sb, pp, tns, theta, nu=0.25, sigmaT=0):
    """
    Calculate the critical bottomhole pressure (BHP) based on a closed-form solution.
    
    Parameters:
    pp (float): Pore pressure
    ucs (float): Uniaxial compressive strength
    Sb (numpy.ndarray): At-wall stress tensor (3x3 matrix)
    theta (float): Circumferential angle in radians corresponding to the minimum principal stress on the hole wall
    nu (float): Poisson's ratio (default is 0.25)
    sigmaT (float): Thermal stress (default is 0)
    
    Returns:
    float: Critical bottomhole pressure (BHP)
    """
    
    # Extract components of the stress tensor for readability
    Sb00, Sb01, Sb02 = Sb[0, 0], Sb[0, 1], Sb[0, 2]
    Sb10, Sb11, Sb12 = Sb[1, 0], Sb[1, 1], Sb[1, 2]
    Sb20, Sb21, Sb22 = Sb[2, 0], Sb[2, 1], Sb[2, 2]
    
    return (-4.0*Sb00**2*nu*np.cos(2.0*theta)**2 + 2.0*Sb00**2*nu*np.cos(2.0*theta) + 4.0*Sb00*Sb01*nu*np.sin(2.0*theta) - 8.0*Sb00*Sb01*nu*np.sin(4.0*theta) + 8.0*Sb00*Sb11*nu*np.cos(2.0*theta)**2 + 2.0*Sb00*Sb22*np.cos(2.0*theta) - Sb00*Sb22 + 2.0*Sb00*nu*pp*np.cos(2.0*theta) - 2.0*Sb00*nu*sigmaT*np.cos(2.0*theta) - 2.0*Sb00*nu*tns*np.cos(2.0*theta) - 2.0*Sb00*tns*np.cos(2.0*theta) + Sb00*tns - 16.0*Sb01**2*nu*np.sin(2.0*theta)**2 + 4.0*Sb01*Sb11*nu*np.sin(2.0*theta) + 8.0*Sb01*Sb11*nu*np.sin(4.0*theta) + 4.0*Sb01*Sb22*np.sin(2.0*theta) + 4.0*Sb01*nu*pp*np.sin(2.0*theta) - 4.0*Sb01*nu*sigmaT*np.sin(2.0*theta) - 4.0*Sb01*nu*tns*np.sin(2.0*theta) - 4.0*Sb01*tns*np.sin(2.0*theta) + 4.0*Sb02**2*np.sin(theta)**2 - 4.0*Sb02*Sb12*np.sin(2.0*theta) - 4.0*Sb11**2*nu*np.cos(2.0*theta)**2 - 2.0*Sb11**2*nu*np.cos(2.0*theta) - 2.0*Sb11*Sb22*np.cos(2.0*theta) - Sb11*Sb22 - 2.0*Sb11*nu*pp*np.cos(2.0*theta) + 2.0*Sb11*nu*sigmaT*np.cos(2.0*theta) + 2.0*Sb11*nu*tns*np.cos(2.0*theta) + 2.0*Sb11*tns*np.cos(2.0*theta) + Sb11*tns + 4.0*Sb12**2*np.cos(theta)**2 - Sb22*pp + Sb22*sigmaT + Sb22*tns + pp*tns - sigmaT*tns - tns**2)/(2.0*Sb00*nu*np.cos(2.0*theta) + 4.0*Sb01*nu*np.sin(2.0*theta) - 2.0*Sb11*nu*np.cos(2.0*theta) - Sb22 + tns)

from io import BytesIO
import base64
def plot_to_base64_png(matplot, dpi=300) ->str:
    """
    Saves the last plot made using ``matplotlib.pyplot`` to a base64-encoded PNG string.
    
    Returns:
        The corresponding base64 PNG string.
    """
    buf = BytesIO()
    matplot.savefig(buf, format='png')
    buf.seek(0)
    png_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    matplot.close()
    return png_base64
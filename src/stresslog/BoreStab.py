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

def get_optimal(sx,sy,sz,alpha=0, beta=0, gamma=0):
    if min(sx,sy,sz)==sz:#reverse faulting
        optimal = get_optimalRF(sx,sy,sz,alpha, beta, gamma)
    else: #normal and strike slip
        optimal = get_optimalNS(sx,sy,sz,alpha, beta, gamma)
    return optimal[0],optimal[1],optimal[2]

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
    Optimizes sx, sy, sz for reverse faulting, allowing all three to vary,
    with the condition that osx > osy > osz.
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
    
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    Rs = np.array([[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]])
    return Rs


def getStens(sx,sy,sz,alpha,beta,gamma):
    
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
    print(Sg)
    print(np.linalg.eigh(Sg))
    print("Vector Dip is:",np.degrees(np.arccos(Rs[2][2])))
    dip_direction = np.degrees(np.arctan2(Rs[2][1], Rs[2][0]))
    print("Dip Direction is:", dip_direction)
    print("Calculated Vertical Component is:", Sg[2][2])
    print("Calculated Max Horizontal Component is:", max(Sg[1][1],Sg[0][0]))
    print("Calculated Min Horizontal Component is:", min(Sg[1][1],Sg[0][0]))    
    return Sg[0],Sg[1],Sg[2]

def getStrikeDip(alpha,beta,gamma):
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


def drawStab(s1,s2,s3,deltaP,Pp,UCS,alpha=0,beta=0,gamma=0):
    values = np.zeros((10,37))
    inclination = np.zeros((10,37))
    azimuth = np.zeros((10,37))
    inc = 0
    while inc<10:
        azim = 0
        while azim<37:
            pointer= 0
            line = np.zeros(360)
            while pointer<360:
                STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim*10, inc*10, pointer, deltaP,Pp)
                line[pointer] = STM
                pointer+=1
            values[inc][azim] = np.max(line)
            inclination[inc][azim] = inc*10
            azimuth[inc][azim] = math.radians(azim*10)
            azim+=1
        #print(round((inc/10)*100),"%")
        inc+=1


    fig = plt2.figure()
    ax = fig.add_subplot(projection='polar')
    ax.grid(True)
    ax.set_yticklabels([])
    ax.set_rmax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    cax = ax.contourf(azimuth, inclination, values, 1000, cmap = 'jet')
    cb = fig.colorbar(cax, orientation = 'horizontal')
    cb.set_label("Sigma Theta-Theta Max")
    plt2.show()


def drawBreak(s1,s2,s3,deltaP,Pp,UCS,alpha=0,beta=0,gamma=0,nu=0.35):
    values = np.zeros((10,37))
    inclination = np.zeros((10,37))
    azimuth = np.zeros((10,37))
    inc = 0
    while inc<10:
        azim = 0
        while azim<37:
            pointer= 0
            line = np.zeros(360)
            width = 0
            while pointer<360:
                STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim*10, inc*10, pointer, deltaP,Pp,nu)
                if (STT-UCS)>0:
                    width+=1
                pointer+=1
            #print(width)
            values[inc][azim] = width/2
            inclination[inc][azim] = inc*10
            azimuth[inc][azim] = math.radians(azim*10)
            azim+=1
        #print(round((inc/10)*100),"%")
        inc+=1


    fig = plt2.figure()
    ax = fig.add_subplot(projection='polar')
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_rmax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    levels = np.linspace(0,120,13)
    cax = ax.contourf(azimuth, inclination, values, 13, levels=levels, extend = 'max', cmap = 'jet', alpha = 0.8)
    print(orit)
    ax.scatter(math.radians(orit[0]),orit[1], s=20, color = 'black', edgecolors='black', label=s3)
    ax.text(math.radians(orit[0]),orit[1], " "+str(round(s3,1)))
    if(orit[3]<90):
        ax.scatter(math.radians(-orit[2]),orit[3], s=20, color = 'black', edgecolors='black', label=s1)
        ax.text(math.radians(-orit[2]),orit[3], " "+str(round(s1,1)))
    else:
        ax.scatter(math.radians(-orit[2]),(90-(orit[3]-90)), s=20, color = 'white', edgecolors='black', label=s1)
        ax.text(math.radians(-orit[2]),(90-(orit[3]-90)), " "+str(round(s1,1)))
    if(orit[5]<90):
        ax.scatter(math.radians(-orit[4]),orit[5], s=20, color = 'black', edgecolors='black',label=s2)
        ax.text(math.radians(-orit[4]),orit[5], " "+str(round(s2,1)))
    else:
        ax.scatter(math.radians(-orit[4]),(90-(orit[5]-90)), s=20, color = 'white', edgecolors='black', label=s2)
        ax.text(math.radians(-orit[4]),(90-(orit[5]-90)), " "+str(round(s2,1)))
    cb = fig.colorbar(cax, orientation = 'horizontal')
    cb.set_label("Breakout Widths in Degrees")
    plt2.title( "UCS = "+str(UCS)+", DeltaP = "+str(deltaP)+", Nu = "+str(nu) , loc="center")
    plt2.show()
    
def drawDITF(s1,s2,s3,deltaP,Pp,alpha=0,beta=0,gamma=0,offset=0,nu=0.35):
    values = np.zeros((10,37))
    inclination = np.zeros((10,37))
    azimuth = np.zeros((10,37))
    inc = 0
    while inc<10:
        azim = 0
        while azim<37:
            pointer= 0
            line = np.zeros(360)
            angle= np.zeros(360)
            width= 0
            frac = np.zeros(360)
            widthR = np.zeros(360)
            while pointer<360:
                STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim*10, inc*10, pointer, deltaP,Pp,nu)
                line[pointer] = STT
                angle[pointer] = omega
                if STT<0:
                    width+=1
                    frac[pointer] = frac[pointer-1]+(1/math.tan(math.radians(omega)))
                else:
                    frac[pointer] = 0
                #if pointer>180:
                    #frac[pointer] = frac[360-pointer]
                widthR[pointer] = (pointer/360)*0.67827 #in metres
                pointer+=1
            if width>0:
                print("Width = ",width/2,", omega =",np.max(angle), " at inclination = ",inc*10, " and azimuth= ",azim*10)
                #plt2.scatter(np.array(range(0,360)),frac)
                #plt2.plot(angle)
                #plt2.plot(line)
                #plt2.xlim((0,0.67827))
                #plt2.ylim((1,151))
                #plt2.show()
            values[inc][azim] = np.min(line)
            inclination[inc][azim] = inc*10
            azimuth[inc][azim] = math.radians(azim*10+offset)
            azim+=1
        #print(round((inc/10)*100),"%")
        inc+=1


    fig = plt2.figure()
    ax = fig.add_subplot(projection='polar')
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_rmax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    levels = np.linspace(0,s3,2100)
    cax = ax.contourf(azimuth, inclination, values, 2100, levels=levels, extend = 'both', cmap = 'jet_r', alpha = 0.8)
    ax.scatter(math.radians(orit[0]),orit[1], s=20, color = 'black', edgecolors='black', label=s3)
    ax.text(math.radians(orit[0]),orit[1], " "+str(round(s3,1)))
    if(orit[3]<=90):
        ax.scatter(math.radians(-orit[2]),orit[3], s=20, color = 'black', edgecolors='black', label=s1)
        ax.text(math.radians(-orit[2]),orit[3], " "+str(round(s1,1)))
    else:
        ax.scatter(math.radians(-orit[2]),(90-(orit[3]-90)), s=20, color = 'white', edgecolors='black', label=s1)
        ax.text(math.radians(-orit[2]),(90-(orit[3]-90)), " "+str(round(s1,1)))
    if(orit[5]<=90):
        ax.scatter(math.radians(-orit[4]),orit[5], s=20, color = 'black', edgecolors='black',label=s2)
        ax.text(math.radians(-orit[4]),orit[5], " "+str(round(s2,1)))
    else:
        ax.scatter(math.radians(-orit[4]),(90-(orit[5]-90)), s=20, color = 'white', edgecolors='black', label=s2)
        ax.text(math.radians(-orit[4]),(90-(orit[5]-90)), " "+str(round(s2,1)))
    cb = fig.colorbar(cax, orientation = 'horizontal')
    plt2.title( "DeltaP = "+str(round(deltaP,2))+", Nu = "+str(nu) , loc="center")
    cb.set_label("Excess Mud Pressure to TensileFrac")
    plt2.show()

def getHoop(inc,azim,s1,s2,s3,deltaP,Pp, ucs, alpha=0,beta=0,gamma=0,nu=0.35,bt=0,ym=0,delT=0,path=None):
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
    ts = -ucs/10
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
    
    #print("Width = ",width/20,", omega =",np.max(angle), " at inclination = ",inc, " and azimuth= ",azim)
    #plt2.scatter(np.array(range(0,360)),frac)
    plt2.title("Hoop Stresses and Principal Stress Angles")
    plt2.plot(angle)
    plt2.plot(eline)
    plt2.plot(eline2)
    plt2.plot(line1)
    #plt2.plot(frac)
    #plt2.plot(crush)
    #plt2.xlim((0,0.67827))
    #plt2.ylim((1,151))
    if path is not None:
        plt2.savefig(path)
        plt2.close()
        return crush,frac,minstress,maxstress,angle[minstress],angle[(minstress+180)%360],angle
    else:
        return crush,frac,minstress,maxstress,angle[minstress],angle[(minstress+180)%360],angle,plt2

def draw(tvd,s1,s2,s3,deltaP,Pp,UCS = 0,alpha=0,beta=0,gamma=0,offset=0,nu=0.35,  azimuthu=0,inclinationi=0,bt=0,ym=0,delT=0,path=None):
    #phi = 183-(163*nu) ## wayy too high
    #phi = np.arcsin(1-(nu/(1-nu))) #Still too high
    phi = np.arcsin(1-(2*nu)) #unModified Zhang
    mui = (1+np.sin(phi))/(1-np.sin(phi))
    #mui = 1.9
    print("Mu_i = ",mui)
    fmui = ((((mui**2)+1)**0.5)+mui)**2
    values = np.zeros((10,37))
    values2 = np.zeros((10,37))
    inclination = np.zeros((10,37))
    azimuth = np.zeros((10,37))
    inc = 0
    TS = -UCS/10
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

        
    print(orit)
    
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
    print(orit)
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
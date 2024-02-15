import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit

s1 = 45
s2 = 34
s3 = 29
alpha = 5.4
beta = 2.3
gamma = 13.5

azim = 187
inc  = 16

nu = 0.35
theta = 37
deltaP = 3


def getSigmaTT(s1,s2,s3,alpha,beta,gamma,azim,inc,theta,deltaP,nu=0.35):
    Ss = np.array([[s1,0,0],[0,s2,0],[0,0,s3]])
    #print(Ss)

    alpha = math.radians(alpha)
    beta = math.radians(beta)
    gamma = math.radians(gamma)

    Rs = np.array([[math.cos(alpha)*math.cos(beta), math.sin(alpha)*math.cos(beta), (-1)*math.sin(beta)] ,
                   [(math.cos(alpha)*math.sin(beta)*math.sin(gamma))-(math.sin(alpha)*math.cos(gamma)), (math.sin(alpha)*math.sin(beta)*math.sin(gamma))+(math.cos(alpha)*math.cos(gamma)), math.cos(beta)*math.sin(gamma)],
                   [(math.cos(alpha)*math.sin(beta)*math.cos(gamma))+(math.sin(alpha)*math.sin(gamma)), (math.sin(alpha)*math.sin(beta)*math.cos(gamma))-(math.cos(alpha)*math.sin(gamma)), math.cos(beta)*math.cos(gamma)]])
    #print(Rs)
    sVo = np.array([[0.0],[0.0],[1.0]])
    sNo = np.array([[1.0],[0.0],[0.0]])
    sEo = np.array([[0.0],[1.0],[0.0]])

    sVr = Rs@sVo
    sNr = Rs@sNo
    sEr = Rs@sEo

    sNt1 = np.degrees(np.arctan2(sNr[1],sNr[0]))
    sNt2 =np.degrees(np.arctan2(((sNr[0]**2)+(sNr[1]**2))**0.5,sNr[2]))
    sEt1 =np.degrees(np.arctan2(sEr[1],sEr[0]))
    sEt2 =np.degrees(np.arctan2(((sEr[0]**2)+(sEr[1]**2))**0.5,sEr[2]))
    sVt1 =np.degrees(np.arctan2(((sVr[0]**2)+(sVr[1]**2))**0.5,sVr[2]))
    sVt2 =np.degrees(np.arctan2(sVr[1],sVr[0]))
    orit = [sVt2,sVt1,sNt1,sNt2,sEt1,sEt2]
    
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

    theta = math.radians(theta)

    Szz = Sb[2][2] - ((2*nu)*(Sb[0][0]-Sb[1][1])*(2*math.cos(2*theta))) - (4*nu*Sb[0][1]*math.sin(2*theta))
    Stt = Sb[0][0] + Sb[1][1] -(2*(Sb[0][0] - Sb[1][1])*math.cos(2*theta)) - (4*Sb[0][1]*math.sin(2*theta)) - deltaP
    Ttz = 2*((Sb[1][2]*math.cos(theta))-(Sb[0][2]*math.sin(theta)))
    Srr = deltaP
    
    #print(Szz,Stt,Ttz,Srr)

    STMax = 0.5*(Szz + Stt + (((Szz-Stt)**2)+(4*(Ttz**2)))**0.5)
    Stmin = 0.5*(Szz + Stt - (((Szz-Stt)**2)+(4*(Ttz**2)))**0.5)
    #omega = np.degrees(np.arctan2(Szz,(((STMax**2)-(Szz**2))**0.5)))
    omega = np.degrees(np.arctan2(Stt,Szz))
    if theta>math.radians(180):
        #omega = np.degrees(np.arctan2((((Stt**2)-(Szz**2))**0.5),-Szz))
        omega = 180-np.degrees(np.arctan2(Stt,Szz))
    #print(STMax-Stmin, np.degrees(theta))
    return Stt,Szz,Ttz,STMax,Stmin,omega,orit





def drawStability(s1,s2,s3,deltaP,alpha=0,beta=0,gamma=0):
    values = np.zeros((10,37))
    inclination = np.zeros((10,37))
    azimuth = np.zeros((10,37))
    inc = 0
    while inc<10:
        azim = 0
        while azim<37:
            pointer= 0
            line = np.zeros(361)
            while pointer<361:
                STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim*10, inc*10, pointer, deltaP)
                line[pointer] = STM
                pointer+=1
            values[inc][azim] = np.min(line)
            inclination[inc][azim] = inc*10
            azimuth[inc][azim] = math.radians(azim*10)
            azim+=1
        inc+=1


    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_rmax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    cax = ax.contourf(azimuth, inclination, values, 100, cmap = 'jet')
    cb = fig.colorbar(cax, orientation = 'horizontal')
    cb.set_label("Sigma Theta-Theta Max")
    plt.show()


def drawBreak(s1,s2,s3,deltaP,UCS,alpha=0,beta=0,gamma=0,nu=0.35):
    values = np.zeros((10,37))
    inclination = np.zeros((10,37))
    azimuth = np.zeros((10,37))
    inc = 0
    while inc<10:
        azim = 0
        while azim<37:
            pointer= 0
            line = np.zeros(361)
            width = 0
            while pointer<361:
                STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim*10, inc*10, pointer, deltaP,nu)
                if (STT-UCS)>0:
                    width+=1
                pointer+=1
            values[inc][azim] = width/2
            inclination[inc][azim] = inc*10
            azimuth[inc][azim] = math.radians(azim*10)
            azim+=1
        inc+=1


    fig = plt.figure()
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
    ax.text(math.radians(orit[0]),orit[1], " "+str(s3))
    if(orit[3]<90):
        ax.scatter(math.radians(-orit[2]),orit[3], s=20, color = 'black', edgecolors='black', label=s1)
        ax.text(math.radians(-orit[2]),orit[3], " "+str(s1))
    else:
        ax.scatter(math.radians(-orit[2]),(90-(orit[3]-90)), s=20, color = 'white', edgecolors='black', label=s1)
        ax.text(math.radians(-orit[2]),(90-(orit[3]-90)), " "+str(s1))
    if(orit[5]<90):
        ax.scatter(math.radians(-orit[4]),orit[5], s=20, color = 'black', edgecolors='black',label=s2)
        ax.text(math.radians(-orit[4]),orit[5], " "+str(s2))
    else:
        ax.scatter(math.radians(-orit[4]),(90-(orit[5]-90)), s=20, color = 'white', edgecolors='black', label=s2)
        ax.text(math.radians(-orit[4]),(90-(orit[5]-90)), " "+str(s2))
    cb = fig.colorbar(cax, orientation = 'horizontal')
    cb.set_label("Breakout Widths in Degrees")
    plt.title( "UCS = "+str(UCS)+", DeltaP = "+str(deltaP)+", Nu = "+str(nu) , loc="center")
    plt.show()
    
def drawDITF(s1,s2,s3,deltaP,alpha=0,beta=0,gamma=0,nu=0.35):
    values = np.zeros((10,37))
    inclination = np.zeros((10,37))
    azimuth = np.zeros((10,37))
    inc = 0
    while inc<10:
        azim = 0
        while azim<37:
            pointer= 0
            line = np.zeros(361)
            angle= np.zeros(361)
            width= 0
            frac = np.zeros(361)
            widthR = np.zeros(361)
            while pointer<361:
                STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim*10, inc*10, pointer, deltaP)
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
                #plt.scatter(np.array(range(0,361)),frac)
                #plt.plot(angle)
                #plt.plot(line)
                #plt.xlim((0,0.67827))
                #plt.ylim((1,151))
                #plt.show()
            values[inc][azim] = np.min(line)
            inclination[inc][azim] = inc*10
            azimuth[inc][azim] = math.radians(azim*10)
            azim+=1
        inc+=1


    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_rmax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    levels = np.linspace(0,s3,2100)
    cax = ax.contourf(azimuth, inclination, values, 2100, levels=levels, extend = 'both', cmap = 'jet_r', alpha = 0.8)
    ax.scatter(math.radians(orit[0]),orit[1], s=20, color = 'black', edgecolors='black', label=s3)
    ax.text(math.radians(orit[0]),orit[1], " "+str(s3))
    if(orit[3]<=90):
        ax.scatter(math.radians(-orit[2]),orit[3], s=20, color = 'black', edgecolors='black', label=s1)
        ax.text(math.radians(-orit[2]),orit[3], " "+str(s1))
    else:
        ax.scatter(math.radians(-orit[2]),(90-(orit[3]-90)), s=20, color = 'white', edgecolors='black', label=s1)
        ax.text(math.radians(-orit[2]),(90-(orit[3]-90)), " "+str(s1))
    if(orit[5]<=90):
        ax.scatter(math.radians(-orit[4]),orit[5], s=20, color = 'black', edgecolors='black',label=s2)
        ax.text(math.radians(-orit[4]),orit[5], " "+str(s2))
    else:
        ax.scatter(math.radians(-orit[4]),(90-(orit[5]-90)), s=20, color = 'white', edgecolors='black', label=s2)
        ax.text(math.radians(-orit[4]),(90-(orit[5]-90)), " "+str(s2))
    cb = fig.colorbar(cax, orientation = 'horizontal')
    plt.title( "DeltaP = "+str(deltaP)+", Nu = "+str(nu) , loc="center")
    cb.set_label("Excess Mud Pressure to TensileFrac")
    plt.show()

def getHoopMin(inc,azim,s1,s2,s3,deltaP,alpha=0,beta=0,gamma=0,nu=0.35):
    values = np.zeros((10,37))

    pointer= 0
    line = np.zeros(3610)
    angle= np.zeros(3610)
    width= 0
    frac = np.zeros(3610)
    widthR = np.zeros(3610)
    while pointer<3610:
        STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim, inc, pointer/10, deltaP)
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
        print("Width = ",width/20,", omega =",np.max(angle), " at inclination = ",inc, " and azimuth= ",azim)
        #plt.scatter(np.array(range(0,3610)),frac)
        #plt.plot(angle)
        plt.plot(line)
        #plt.xlim((0,0.67827))
        #plt.ylim((1,151))
        plt.show()
    return np.min(line)
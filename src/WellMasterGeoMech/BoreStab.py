import numpy as np
import matplotlib.pyplot as plt2
import math
#from numba import jit

#s1 = 45
#s2 = 34
#s3 = 29
#alpha = 5.4
#beta = 2.3
#gamma = 13.5

#azim = 187
#inc  = 16

#nu = 0.35
#theta = 37
#deltaP = 3

def getStens(s1,s2,s3,alpha,beta,gamma):
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
    return Sg[0],Sg[1],Sg[2]

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

def getSigmaTT(s1,s2,s3,alpha,beta,gamma,azim,inc,theta,deltaP,Pp,nu=0.35):
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

    Szz = Sb[2][2] - ((2*nu)*(Sb[0][0]-Sb[1][1])*(2*math.cos(2*theta))) - (4*nu*Sb[0][1]*math.sin(2*theta))
    Stt = Sb[0][0] + Sb[1][1] -(2*(Sb[0][0] - Sb[1][1])*math.cos(2*theta)) - (4*Sb[0][1]*math.sin(2*theta)) - deltaP
    Ttz = 2*((Sb[1][2]*math.cos(theta))-(Sb[0][2]*math.sin(theta)))
    Srr = deltaP
    
    #print(Szz,Stt,Ttz,Srr)

    STMax = 0.5*(Szz + Stt + (((Szz-Stt)**2)+(4*(Ttz**2)))**0.5)
    Stmin = 0.5*(Szz + Stt - (((Szz-Stt)**2)+(4*(Ttz**2)))**0.5)
    #omega = np.degrees(np.arctan2(Szz,(((STMax**2)-(Szz**2))**0.5)))
    omega = np.degrees(np.arctan2(Stt,Szz))
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
            line = np.zeros(361)
            while pointer<361:
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
            line = np.zeros(361)
            width = 0
            while pointer<361:
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
            line = np.zeros(361)
            angle= np.zeros(361)
            width= 0
            frac = np.zeros(361)
            widthR = np.zeros(361)
            while pointer<361:
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
                #plt2.scatter(np.array(range(0,361)),frac)
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

def getHoop(inc,azim,s1,s2,s3,deltaP,Pp, ucs, alpha=0,beta=0,gamma=0,nu=0.35):
    phi = np.arcsin(1-(2*nu)) #unModified Zhang
    mui = (1+np.sin(phi))/(1-np.sin(phi))
    fmui = ((((mui**2)+1)**0.5)+mui)**2
    
    #values = np.zeros((10,37))
    
    pointer= 0
    line = np.zeros(361)
    line2 = np.zeros(361)
    angle= np.zeros(361)
    width= 0
    frac = np.zeros(361)
    crush = np.zeros(361)
    widthR = np.zeros(361)
    ts = -ucs/10
    while pointer<361:
        STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim, inc, pointer, deltaP,Pp,nu)
        line[pointer] = stm
        line2[pointer] = STM
        angle[pointer] = omega
        if stm<ts:
            width+=1
            frac[pointer] = 1
        else:
            frac[pointer] = 0
        
        if ucs<((STM)-(fmui*(deltaP))):
            crush[pointer] = 1
        else:
            crush[pointer] = 0
        #if pointer>180:
            #frac[pointer] = frac[360-pointer]
        widthR[pointer] = (pointer/360)*0.67827 #in metres
        pointer+=1
        
    #print("Width = ",width/20,", omega =",np.max(angle), " at inclination = ",inc, " and azimuth= ",azim)
    #plt2.scatter(np.array(range(0,3610)),frac)
    #plt2.plot(angle)
    #plt2.plot(line)
    #plt2.plot(line2)
    #plt2.plot(frac)
    #plt2.plot(crush)
    #plt2.xlim((0,0.67827))
    #plt2.ylim((1,151))
    #plt2.show()
    return crush,frac

def draw(path,tvd,s1,s2,s3,deltaP,Pp,UCS = 0,alpha=0,beta=0,gamma=0,offset=0,nu=0.35,  azimuthu=0,inclinationi=0):
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
    TS = 0
    while inc<10:
        azim = 0
        while azim<37:
            pointer= 0
            line = np.zeros(361)
            line2 = np.zeros(361)
            angle= np.zeros(361)
            width= 0
            width2 = 0
            frac = np.zeros(361)
            widthR = np.zeros(361)
            while pointer<361:
                STT,SZZ,TTZ,STM,stm,omega,orit = getSigmaTT(s1,s2,s3, alpha,beta,gamma, azim*10, inc*10, pointer, deltaP,Pp,nu)
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
                #plt2.scatter(np.array(range(0,361)),frac)
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
    fig.text(0.5, 0.87, "UCS = " + str(round(UCS)) + ", DeltaP = " + str(round(deltaP)) + ", Nu = " + str(round(nu,2)), 
         ha='center', fontsize=10)
    
    plt2.savefig(path,dpi=600)
    plt2.clf()
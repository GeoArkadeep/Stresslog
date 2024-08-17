import matplotlib.pyplot as plt3
from matplotlib.patches import Polygon
import numpy as np
import math

Sv = 48
Pp = 19
bhp = 22
UCS = 46
PhiB = 45
mu = 0.6

def drawSP(path,Sv,Pp,bhp,shmin,UCS = 0,phi = 0, flag = 0,mu = 0.65,nu=0,bt=0,ym=0,delT=0):
    
    #mu = (1-(2*nu))/(2*((nu*(1-nu))**0.5))
    
    PhiBr = 15
    biot = 1
    maxSH = 0
    minSH = 0
    midSH = 0
    sigmaV = Sv-Pp
    sigmahmin = shmin-Pp
    ufac = ((((mu**2)+1)**0.5)+mu)**2
    print("Mu factor: ",ufac)

    ShmP = ((Sv-Pp)/ufac)+Pp
    SHMP = ((Sv-Pp)*ufac)+Pp
    print("Corners: ",ShmP,SHMP)


    #maxSt = 1.02*SHMP
    #minSt = 0.98*ShmP
    
    maxSt = 200
    minSt = 0

    fig,ax = plt3.subplots()
    ax.axis([minSt,maxSt,minSt,maxSt])
    limit = np.array([(0,0),(maxSt,maxSt), (maxSt,0)])
    LM =  Polygon(limit, fill=False ,hatch='\\')
    ax.add_patch(LM)

    X = ShmP
    Y = SHMP

    NNcorners = np.array([(Sv,Sv),(X,Sv),(X,X)])
    SScorners = np.array([(Sv,Sv),(X,Sv),(Sv,Y)])
    RRcorners = np.array([(Sv,Sv),(Sv,Y),(Y,Y)])

    StrikeSlipE = Polygon(SScorners, fill=False)
    NormalE =  Polygon(NNcorners, fill=False)
    ReverseE =  Polygon(RRcorners, fill=False)

    StrikeSlip = Polygon(SScorners, color='blue', alpha=0.05)
    Normal =  Polygon(NNcorners, color='green', alpha = 0.05)
    Reverse =  Polygon(RRcorners, color='red', alpha = 0.05)

    ax.add_patch(StrikeSlip)
    ax.add_patch(Normal)
    ax.add_patch(Reverse)
    ax.add_patch(StrikeSlipE)
    ax.add_patch(NormalE)
    ax.add_patch(ReverseE)

    #UCS = 46
    UCShigh = UCS + (0.2*UCS)
    UCSlow = UCS - (0.2*UCS)
    Shm = ShmP
    Shm2 = SHMP
    #PhiB = 0.1 #degrees
    PhiBr = math.radians(PhiBr)
    TwoCosPhiB = 2*(math.cos((math.pi)-(PhiBr)))
    print("TwoCosPhiB: ",TwoCosPhiB)
    #ShmP = UCS
    print("Phi = ",math.degrees(phi))
    twocos2Beta = 2 * (math.cos(PhiBr))
    q = (1+np.sin(phi))/(1-np.sin(phi)) #Zhang 6.70
    
    SHM1 = ((UCS - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm))/(1+twocos2Beta)
    SHM2 = ((UCS - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm2))/(1+twocos2Beta)
    SHM1H = ((UCShigh - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm))/(1+twocos2Beta)
    SHM2H = ((UCShigh - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm2))/(1+twocos2Beta)
    SHM1L = ((UCSlow - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm))/(1+twocos2Beta)
    SHM2L = ((UCSlow - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm2))/(1+twocos2Beta)
    
    """
    SHM1 = ((UCS + (2*Pp) + (bhp-Pp)) - ((Shm)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2 = ((UCS + (2*Pp) + (bhp-Pp)) - ((Shm2)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    
    SHM1H = ((UCShigh + (2*Pp) + (bhp-Pp)) - ((Shm)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2H = ((UCShigh + (2*Pp) + (bhp-Pp)) - ((Shm2)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM1L = ((UCSlow + (2*Pp) + (bhp-Pp)) - ((Shm)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2L = ((UCSlow + (2*Pp) + (bhp-Pp)) - ((Shm2)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    """
    #print(SHM1)
    br1 = np.array([(Shm,SHM1),(Shm2,SHM2)])
    br2 = np.array([(Shm,SHM1H),(Shm2,SHM2H)])
    br3 = np.array([(Shm,SHM1L),(Shm2,SHM2L)])
    
    lowerulow = [Shm,SHM1H]
    upperulow = [Shm2,SHM2H]
    loweruhigh = [Shm,SHM1L]
    upperuhigh = [Shm2,SHM2L]
    lowerucs = [Shm,SHM1]
    upperucs = [Shm2,SHM2]
    
    #print(br1)
    Breakout2 =  Polygon(br2, color='red', label = "UCS- "+str(round(UCShigh))+"MPa")
    Breakout1 =  Polygon(br1, color='green', label = "UCS- "+str(round(UCS))+"MPa")
    Breakout3 =  Polygon(br3, color='blue', label = "UCS- "+str(round(UCSlow))+"MPa")
    ax.add_patch(Breakout2)
    ax.add_patch(Breakout1)
    ax.add_patch(Breakout3)
    
    Shm3 = ShmP
    Shm4 = Y
    
    sigmaT = (ym*bt*delT)/(1-nu)
        
    #DITFshmax3 = 3*Shm3 - 2*Pp
    #DITFshmax4 = 3*Shm4 - 2*Pp
    DITFshmax3 = 3*Shm3-bhp-Pp-sigmaT
    DITFshmax4 = 3*Shm4-bhp-Pp-sigmaT
    #DITFshmax3 = (ufac*Shm3) - ((ufac-1)*Pp) - (bhp-Pp) - sigmaT
    #DITFshmax4 = (ufac*Shm4) - ((ufac-1)*Pp) - (bhp-Pp) - sigmaT
    ditf = np.array([(Shm3,DITFshmax3),(Shm4,DITFshmax4)])
    DITF =  Polygon(ditf, color='aqua', label = 'DITF')
    lowerd = [Shm3,DITFshmax3]
    upperd = [Shm4,DITFshmax4]
    
    if(shmin>Sv):
        minSH = Sv
        maxSH = SHMP
        #return [Sv,shmin,(shmin+Sv)/2]
    if shmin > Sv:
        minSH = shmin
        maxSH = SHMP
    else:
        #lower = np.array([ShmP, Sv])
        #upper = np.array([Sv, SHMP])
        y = np.array([Sv,SHMP])
        x = np.array([ShmP,Sv])
        I1 = np.interp(shmin, x, y)
        #print(shmin,Sv,I1)
        minSH = shmin
        maxSH = I1
    
    
        if flag>0.5:
            UCShigh = UCS + (0.2 * UCS)
            UCSlow = UCS - (0.2 * UCS)
            maxSt = 1.1*SHMP
            minSt = 0.90*ShmP
            
            xulow = np.array([Shm,Shm2])
            yulow = np.array([SHM1H,SHM2H])
            xuhigh = np.array([Shm,Shm2])
            yuhigh = np.array([SHM1L,SHM2L])
            xucs = np.array([Shm,Shm2])
            yucs = np.array([SHM1,SHM2])
            xd = np.array([Shm3,Shm4])
            yd = np.array([DITFshmax3,DITFshmax4])
            
            if flag > 0.5 and flag < 1.5: #no breakouts or tensile fractures seen on existing image log
                
                minSH = shmin
                maxSH = np.interp(shmin, xucs, yucs)
            if flag > 1.5 and flag <2.5: #breakout observed on image log
                minSH = np.interp(shmin, xulow, yulow)
                maxSH = np.interp(shmin, xuhigh, yuhigh)
            if flag > 2.5 and flag < 3.5: #tensile fractures observed on image log
                minSH = np.interp(shmin, xucs, yucs)
                maxSH = np.interp(shmin, xd, yd)
            if flag>3.5:
                maxSH = np.interp(shmin, xd, yd)
                minSH = np.interp(shmin, xucs, yucs)
    midSH = (minSH + maxSH) / 2
    print([maxSH,minSH,shmin,Sv])
    print("DITF :",ditf)
    ax.add_patch(DITF)
    # Draw a vertical purple line for Shmin
    ax.plot([shmin, shmin], [minSH, maxSH], color='purple', linewidth=1)
    ax.hlines(y=minSH, xmin=0, xmax=shmin, colors='black', linestyles='dotted', linewidth=0.5)
    ax.hlines(y=maxSH, xmin=0, xmax=shmin, colors='black', linestyles='dotted', linewidth=0.5)
    ax.hlines(y=midSH, xmin=0, xmax=shmin, colors='black', linestyles='dotted', linewidth=1)

    # Adjustments to move the annotations further away from the axes to avoid collision with the main axis labels
    # Determine offsets
    x_offset = 0.019 * (ax.get_xlim()[1] - ax.get_xlim()[0])  # 2% of the x-axis range
    y_offset = 0.019 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # 2% of the y-axis range

    # For the x-axis
    #ax.text(round(shmin), ax.get_ylim()[0] - 4 * y_offset, '{:.0f}'.format(shmin), ha='center', va='top', rotation=0)
    ax.text(round(Sv), ax.get_ylim()[0] - 4 * y_offset, '{:.0f}'.format(Sv), ha='center', va='top', rotation=0)

    # For the y-axis
    # Adjust the x coordinate for y-axis labels to move them further from the axis
    if maxSH-minSH>25:
        ax.text(ax.get_xlim()[0] - 5 * x_offset, round(minSH), '{:.0f}'.format(minSH), ha='right', va='center', rotation=90)
        ax.text(ax.get_xlim()[0] - 5 * x_offset, round(midSH), '{:.0f}'.format(midSH), ha='right', va='center', rotation=90)
        ax.text(ax.get_xlim()[0] - 5 * x_offset, round(maxSH), '{:.0f}'.format(maxSH), ha='right', va='center', rotation=90)
    else:
        ax.text(ax.get_xlim()[0] - 5 * x_offset, round(midSH), '{:.0f}'.format(midSH), ha='right', va='center', rotation=90)
    ax.legend(loc='lower right')
    plt3.gca().set_aspect('equal', adjustable='box')
    plt3.title("Stress Polygon")
    plt3.xlabel("Shmin", labelpad=10)  # Increase labelpad as needed
    plt3.ylabel("SHmax", labelpad=10)  # Increase labelpad as needed
    plt3.savefig(path, dpi=600)
    plt3.clf()
    plt3.close()

def getSP(Sv,Pp,bhp,shmin,UCS = 0,phi = 0, flag = 0,mu = 0.6,nu=0,bt=0,ym=0,delT=0):
    
    PhiBr = 15
    biot = 1
    maxSH = 0
    minSH = 0
    midSH = 0
    sigmaV = Sv-Pp
    sigmahmin = shmin-Pp
    ufac = ((((mu**2)+1)**0.5)+mu)**2
    #print("Mu factor: ",ufac)

    ShmP = ((Sv-Pp)/ufac)+Pp
    SHMP = ((Sv-Pp)*ufac)+Pp
    #print("Corners: ",ShmP,SHMP)
    
    if shmin<ShmP:
        shmin=ShmP
    if shmin>SHMP:
        shmin=SHMP
    #maxSt = 1.02*SHMP
    #minSt = 0.98*ShmP
    
    maxSt = 200
    minSt = 0

    #fig,ax = plt3.subplots()
    #ax.axis([minSt,maxSt,minSt,maxSt])
    limit = np.array([(0,0),(maxSt,maxSt), (maxSt,0)])
    #LM =  Polygon(limit, fill=False ,hatch='\\')
    #ax.add_patch(LM)

    X = ShmP
    Y = SHMP

    NNcorners = np.array([(Sv,Sv),(X,Sv),(X,X)])
    SScorners = np.array([(Sv,Sv),(X,Sv),(Sv,Y)])
    RRcorners = np.array([(Sv,Sv),(Sv,Y),(Y,Y)])

    #StrikeSlipE = Polygon(SScorners, fill=False)
    #NormalE =  Polygon(NNcorners, fill=False)
    #ReverseE =  Polygon(RRcorners, fill=False)

    #StrikeSlip = Polygon(SScorners, color='blue', alpha=0.05)
    #Normal =  Polygon(NNcorners, color='green', alpha = 0.05)
    #Reverse =  Polygon(RRcorners, color='red', alpha = 0.05)

    #ax.add_patch(StrikeSlip)
    #ax.add_patch(Normal)
    #ax.add_patch(Reverse)
    #ax.add_patch(StrikeSlipE)
    #ax.add_patch(NormalE)
    #ax.add_patch(ReverseE)

    #UCS = 46
    UCShigh = UCS + (0.2*UCS)
    UCSlow = UCS - (0.2*UCS)
    Shm = ShmP
    Shm2 = SHMP
    #PhiB = 0.1 #degrees
    PhiBr = math.radians(PhiBr)
    TwoCosPhiB = 2*(math.cos((math.pi)-(PhiBr)))
    #print("TwoCosPhiB: ",TwoCosPhiB)
    #ShmP = UCS
    #print("Phi = ",math.degrees(phi))
    twocos2Beta = 2 * (np.cos(PhiBr))
    q = (1+np.sin(phi))/(1-np.sin(phi)) # zhang 6.70
    
    SHM1 = ((UCS - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm))/(1+twocos2Beta)
    SHM2 = ((UCS - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm2))/(1+twocos2Beta)
    SHM1H = ((UCShigh - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm))/(1+twocos2Beta)
    SHM2H = ((UCShigh - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm2))/(1+twocos2Beta)
    SHM1L = ((UCSlow - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm))/(1+twocos2Beta)
    SHM2L = ((UCSlow - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm2))/(1+twocos2Beta)
    
    """
    SHM1 = ((UCS + (2*Pp) + (bhp-Pp)) - ((Shm)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2 = ((UCS + (2*Pp) + (bhp-Pp)) - ((Shm2)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    
    SHM1H = ((UCShigh + (2*Pp) + (bhp-Pp)) - ((Shm)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2H = ((UCShigh + (2*Pp) + (bhp-Pp)) - ((Shm2)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM1L = ((UCSlow + (2*Pp) + (bhp-Pp)) - ((Shm)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2L = ((UCSlow + (2*Pp) + (bhp-Pp)) - ((Shm2)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    """
    #print(SHM1)
    br1 = np.array([(Shm,SHM1),(Shm2,SHM2)])
    br2 = np.array([(Shm,SHM1H),(Shm2,SHM2H)])
    br3 = np.array([(Shm,SHM1L),(Shm2,SHM2L)])
    
    lowerulow = [Shm,SHM1H]
    upperulow = [Shm2,SHM2H]
    loweruhigh = [Shm,SHM1L]
    upperuhigh = [Shm2,SHM2L]
    lowerucs = [Shm,SHM1]
    upperucs = [Shm2,SHM2]
    
    #print(br1)
    #Breakout2 =  Polygon(br2, color='red', label = "UCS- "+str(round(UCShigh))+"MPa")
    #Breakout1 =  Polygon(br1, color='green', label = "UCS- "+str(round(UCS))+"MPa")
    #Breakout3 =  Polygon(br3, color='blue', label = "UCS- "+str(round(UCSlow))+"MPa")
    #ax.add_patch(Breakout2)
    #ax.add_patch(Breakout1)
    #ax.add_patch(Breakout3)
    
    Shm3 = ShmP
    Shm4 = Y
    
    sigmaT = (ym*bt*delT)/(1-nu)
    
    #DITFshmax3 = 3*Shm3 - 2*Pp
    #DITFshmax4 = 3*Shm4 - 2*Pp
    DITFshmax3 = (ufac*Shm3) - ((ufac-1)*Pp) - (bhp-Pp) - sigmaT
    DITFshmax4 = (ufac*Shm4) - ((ufac-1)*Pp) - (bhp-Pp) - sigmaT
    ditf = np.array([(Shm3,DITFshmax3),(Shm4,DITFshmax4)])
    DITF =  Polygon(ditf, color='aqua', label = 'DITF')
    lowerd = [Shm3,DITFshmax3]
    upperd = [Shm4,DITFshmax4]
    

    if shmin > Sv:
        minSH = shmin
        maxSH = SHMP
    else:
        #lower = np.array([ShmP, Sv])
        #upper = np.array([Sv, SHMP])
        y = np.array([Sv,SHMP])
        x = np.array([ShmP,Sv])
        I1 = np.interp(shmin, x, y)
        #print(shmin,Sv,I1)
        minSH = shmin
        maxSH = I1
    
    
        if flag>0.5:
            UCShigh = UCS + (0.2 * UCS)
            UCSlow = UCS - (0.2 * UCS)
            maxSt = 1.1*SHMP
            minSt = 0.90*ShmP
            
            xulow = np.array([Shm,Shm2])
            yulow = np.array([SHM1H,SHM2H])
            xuhigh = np.array([Shm,Shm2])
            yuhigh = np.array([SHM1L,SHM2L])
            xucs = np.array([Shm,Shm2])
            yucs = np.array([SHM1,SHM2])
            xd = np.array([Shm3,Shm4])
            yd = np.array([DITFshmax3,DITFshmax4])
            
            if flag > 0.5 and flag < 1.5: #no breakouts or tensile fractures seen on existing image log
                
                minSH = shmin
                maxSH = np.interp(shmin, xucs, yucs)
            if flag > 1.5 and flag <2.5: #breakout observed on image log
                minSH = np.interp(shmin, xulow, yulow)
                maxSH = np.interp(shmin, xuhigh, yuhigh)
            if flag > 2.5 and flag < 3.5: #tensile fractures observed on image log
                minSH = np.interp(shmin, xucs, yucs)
                maxSH = np.interp(shmin, xd, yd)
            if flag>3.5:
                maxSH = np.interp(shmin, xd, yd)
                minSH = np.interp(shmin, xucs, yucs)
        if maxSH<minSH:
            maxSH=minSH
    midSH = (minSH + maxSH) / 2

    return [minSH, maxSH, midSH]
    #print([maxSH,minSH,shmin,Sv])
    #print("DITF :",ditf)
    #ax.add_patch(DITF)
    #minmin = np.array([(shmin,minSH),(shmin,maxSH)])
    #maxmaxsh = np.array([(shmin,minSH),(shmin,maxSH),(0,maxSH),(0,minSH)])
    #Shmin =  Polygon(maxmaxsh, color='purple', label = 'Sh min',alpha=0.2)
    #ax.add_patch(Shmin)
    #ax.legend(loc='lower right')
    #plt3.gca().set_aspect('equal')
    #plt3.title("Stress Polygon")
    #plt3.xlabel("Shmin")
    #plt3.ylabel("SHmax")
    #plt3.savefig(path,dpi=600)
    #plt3.clf()


from BoreStab import draw
from BoreStab import drawStab
import numpy as np
import math

def getSHMax_optimized(Sv, Pp, bhp, shmin, UCS=0, phi=30 , flag=0, mu=0.6, biot=1):
    
    ThetaB=15
    #biot=1
    
    ufac = ((((mu**2) + 1)**0.5) + mu)**2

    ShmP = ((Sv - Pp) / ufac) + Pp
    SHMP = ((Sv - Pp) * ufac) + Pp

    

    if shmin<ShmP or shmin>SHMP:
        return [np.nan, np.nan, np.nan]
    if shmin > Sv:
        minSH = shmin
        maxSH = SHMP
    else:
        #lower = np.array([ShmP, Sv])
        #upper = np.array([Sv, SHMP])
        y = np.array([Sv,SHMP])
        x = np.array([ShmP,Sv])
        I1 = np.interp(shmin, x, y)
        #print(shmin,Sv,I1)
        minSH = shmin
        maxSH = I1
    
    
        if flag>0.5:
            UCShigh = UCS + (0.2 * UCS)
            UCSlow = UCS - (0.2 * UCS)
            maxSt = 1.1*SHMP
            minSt = 0.90*ShmP
            Shm = ShmP
            Shm2 = SHMP

            PhiBr = math.radians(ThetaB)
            TwoCosPhiB = 2 * (math.cos((math.pi) - (PhiBr)))
            twocos2Beta = 2 * (math.cos((math.pi) - (2*(PhiBr))))
            q = (1+math.sin(phi))/(1-math.sin(phi))
            
            SHM1 = ((UCS - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm))/(1+twocos2Beta)
            SHM2 = ((UCS - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm2))/(1+twocos2Beta)
            SHM1H = ((UCShigh - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm))/(1+twocos2Beta)
            SHM2H = ((UCShigh - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm2))/(1+twocos2Beta)
            SHM1L = ((UCSlow - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm))/(1+twocos2Beta)
            SHM2L = ((UCSlow - (biot*(q-1)*Pp)) + ((q+1)*bhp) - ((1-twocos2Beta)*Shm2))/(1+twocos2Beta)          
            
            """
            SHM1 = ((UCS + 2 * Pp + (bhp - Pp)) - Shm * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            SHM2 = ((UCS + 2 * Pp + (bhp - Pp)) - Shm2 * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            SHM1H = ((UCShigh + 2 * Pp + (bhp - Pp)) - Shm * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            SHM2H = ((UCShigh + 2 * Pp + (bhp - Pp)) - Shm2 * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            SHM1L = ((UCSlow + 2 * Pp + (bhp - Pp)) - Shm * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            SHM2L = ((UCSlow + 2 * Pp + (bhp - Pp)) - Shm2 * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            """
            xulow = np.array([Shm,Shm2])
            yulow = np.array([SHM1H,SHM2H])
            xuhigh = np.array([Shm,Shm2])
            yuhigh = np.array([SHM1L,SHM2L])
            xucs = np.array([Shm,Shm2])
            yucs = np.array([SHM1,SHM2])


            Shm3 = 1
            Shm4 = maxSt

            DITFshmax3 = ufac * Shm3 - (ufac - 1) * Pp - (bhp - Pp)
            DITFshmax4 = ufac * Shm4 - (ufac - 1) * Pp - (bhp - Pp)
            ditf = np.array([(Shm3, DITFshmax3), (Shm4, DITFshmax4)])
            
            xd = np.array([Shm3,Shm4])
            yd = np.array([DITFshmax3,DITFshmax4])
            if flag > 0.5 and flag < 1.5: #no breakouts or tensile fractures seen on existing image log
                minSH = shmin
                maxSH = np.interp(shmin, xucs, yucs)
            if flag > 1.5 and flag <2.5: #breakout observed on image log
                minSH = np.interp(shmin, xulow, yulow)
                maxSH = np.interp(shmin, xuhigh, yuhigh)
            if flag > 2.5 and flag < 3.5: #tensile fractures observed on image log
                minSH = np.interp(shmin, xucs, yucs)
                maxSH = np.interp(shmin, xd, yd)
            if flag>3.5:# Both breakouts and tensile fractures on log
                maxSH = np.interp(shmin, xd, yd)
                minSH = np.interp(shmin, xucs, yucs)
    #if maxSH<minSH:        
    #    maxSH=minSH
    midSH = (minSH + maxSH) / 2
    return [minSH, maxSH, midSH]

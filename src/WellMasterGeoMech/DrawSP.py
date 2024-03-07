import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import math

Sv = 48
Pp = 19
bhp = 22
UCS = 46
PhiB = 45
mu = 0.6

def drawSP(path,Sv,Pp,bhp,shmin,UCS = 0,PhiB = 0, flag = 0,mu = 0.6):
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

    fig,ax = plt.subplots()
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
    PhiBr = math.radians(PhiB)
    TwoCosPhiB = 2*(math.cos((math.pi)-(PhiBr)))
    print("TwoCosPhiB: ",TwoCosPhiB)
    #ShmP = UCS
    SHM1 = ((UCS + (2*Pp) + (bhp-Pp)) - ((Shm)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2 = ((UCS + (2*Pp) + (bhp-Pp)) - ((Shm2)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    
    SHM1H = ((UCShigh + (2*Pp) + (bhp-Pp)) - ((Shm)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2H = ((UCShigh + (2*Pp) + (bhp-Pp)) - ((Shm2)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM1L = ((UCSlow + (2*Pp) + (bhp-Pp)) - ((Shm)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2L = ((UCSlow + (2*Pp) + (bhp-Pp)) - ((Shm2)*(1+TwoCosPhiB)))/(1-TwoCosPhiB)

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
    Breakout2 =  Polygon(br2, color='red', label = "UCS- "+str(UCShigh)+"MPa")
    Breakout1 =  Polygon(br1, color='green', label = "UCS- "+str(UCS)+"MPa")
    Breakout3 =  Polygon(br3, color='blue', label = "UCS- "+str(UCSlow)+"MPa")
    ax.add_patch(Breakout2)
    ax.add_patch(Breakout1)
    ax.add_patch(Breakout3)
    
    Shm3 = ShmP
    Shm4 = Y
    
    #DITFshmax3 = 3*Shm3 - 2*Pp
    #DITFshmax4 = 3*Shm4 - 2*Pp
    DITFshmax3 = (ufac*Shm3) - ((ufac-1)*Pp) - (bhp-Pp)
    DITFshmax4 = (ufac*Shm4) - ((ufac-1)*Pp) - (bhp-Pp)
    ditf = np.array([(Shm3,DITFshmax3),(Shm4,DITFshmax4)])
    DITF =  Polygon(ditf, color='aqua', label = 'DITF')
    lowerd = [Shm3,DITFshmax3]
    upperd = [Shm4,DITFshmax4]
    
    if(shmin>Sv):
        minSH = Sv
        maxSH = SHMP
        #return [Sv,shmin,(shmin+Sv)/2]
    if shmin > Sv:
        minSH = Sv
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
                
                minSH = np.interp(shmin, xucs, yucs)
                maxSH = np.interp(shmin, xd, yd)
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
    minmin = np.array([(shmin,minSH),(shmin,maxSH)])
    Shmin =  Polygon(minmin, color='purple', label = 'Sh min')
    ax.add_patch(Shmin)
    ax.legend()
    plt.gca().set_aspect('equal')
    plt.title("Stress Polygon")
    plt.xlabel("Shmin")
    plt.ylabel("SHmax")
    plt.savefig(path)
    plt.clf()


def getSHMax(Sv,Pp,bhp,shmin, UCS = 0, ThetaB = 45, breakouts = 0, ditflag = 0, mu=0.6):
    maxSH = 0
    minSH = 0
    midSH = 0
    
    ufac = ((((mu**2)+1)**0.5)+mu)**2
    #print("Mu factor: ",ufac)

    ShmP = ((Sv-Pp)/ufac)+Pp
    SHMP = ((Sv-Pp)*ufac)+Pp
    #print("Corners: ",ShmP,SHMP)

    maxSt = 1.05*SHMP
    minSt = 0.90*ShmP

    limit = np.array([(0,0),(maxSt,maxSt), (maxSt,0)])

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

    #UCS = 46
    UCShigh = UCS + (0.2*UCS)
    UCSlow = UCS - (0.2*UCS)
    Shm = 1
    Shm2 = maxSt
    #PhiB = 0.1 #degrees
    PhiBr = math.radians(ThetaB)
    TwoCosPhiB = 2*(math.cos((math.pi)-(PhiBr)))


    SHM1 = ((UCS + (2*Pp) + (bhp-Pp)) - (Shm*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2 = ((UCS + (2*Pp) + (bhp-Pp)) - (Shm2*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM1H = ((UCShigh + (2*Pp) + (bhp-Pp)) - (Shm*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2H = ((UCShigh + (2*Pp) + (bhp-Pp)) - (Shm2*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM1L = ((UCSlow + (2*Pp) + (bhp-Pp)) - (Shm*(1+TwoCosPhiB)))/(1-TwoCosPhiB)
    SHM2L = ((UCSlow + (2*Pp) + (bhp-Pp)) - (Shm2*(1+TwoCosPhiB)))/(1-TwoCosPhiB)

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


    Shm3 = 1
    Shm4 = maxSt

    #DITFshmax3 = 3*Shm3 - 2*Pp
    #DITFshmax4 = 3*Shm4 - 2*Pp
    DITFshmax3 = (ufac*Shm3) - ((ufac-1)*Pp) - (bhp-Pp)
    DITFshmax4 = (ufac*Shm4) - ((ufac-1)*Pp) - (bhp-Pp)
    ditf = np.array([(Shm3,DITFshmax3),(Shm4,DITFshmax4)])
    lowerd = [Shm3,DITFshmax3]
    upperd = [Shm4,DITFshmax4]
    
    if(shmin<ShmP):
        return[ShmP,ShmP,ShmP]
    if(shmin>Sv):
        minSH = Sv
        maxSH = SHMP
        #return [Sv,shmin,(shmin+Sv)/2]
    else:
        lower = [ShmP,Sv]
        upper = [Sv,SHMP]
        I1 = np.interp(shmin,lower,upper)
        #print("Intercept is: ",I1)
        minSH = shmin
        maxSH = I1
        #return [shmin,I1,(shmin+I1)/2]
    
    if ditflag>0 and breakouts>0:
        minSH = loweruhigh[1]+(shmin-loweruhigh[0])*(upperuhigh[1]-loweruhigh[1])/(upperuhigh[0]-loweruhigh[0])
        maxSH = lowerd[1]+(shmin-lowerd[0])*(upperd[1]-lowerd[1])/(upperd[0]-lowerd[0])
    else:
        if ditflag>0:
            minSH = lowerd[1]+(shmin-lowerd[0])*(upperd[1]-lowerd[1])/(upperd[0]-lowerd[0])#np.interp(shmin,lowerd,upperd)
        else:
            maxSH = lowerd[1]+(shmin-lowerd[0])*(upperd[1]-lowerd[1])/(upperd[0]-lowerd[0])#np.interp(shmin,lowerd,upperd)
        if breakouts>0:
            maxSH = loweruhigh[1]+(shmin-loweruhigh[0])*(upperuhigh[1]-loweruhigh[1])/(upperuhigh[0]-loweruhigh[0])
            minSH = lowerulow[1]+(shmin-lowerulow[0])*(upperulow[1]-lowerulow[1])/(upperulow[0]-lowerulow[0])
        else:
            maxSH = lowerucs[1]+(shmin-lowerucs[0])*(upperucs[1]-lowerucs[1])/(upperucs[0]-lowerucs[0])#np.interp(shmin,lowerucs,upperucs)
            #print("nobreak")
            #print(lowerucs,upperucs,maxSH)
    midSH=(minSH+maxSH)/2    
    return [minSH,maxSH,midSH]
from BoreStab import draw
from BoreStab import drawStab
import numpy as np
import math

def getSHMax_optimized(Sv, Pp, bhp, shmin, UCS=0, ThetaB=45, flag=0, mu=0.6):
    ufac = ((((mu**2) + 1)**0.5) + mu)**2

    ShmP = ((Sv - Pp) / ufac) + Pp
    SHMP = ((Sv - Pp) * ufac) + Pp



    if shmin<ShmP or shmin>SHMP:
        return [np.nan, np.nan, np.nan]
    if shmin > Sv:
        minSH = Sv
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
            Shm = 1
            Shm2 = maxSt

            PhiBr = math.radians(ThetaB)
            TwoCosPhiB = 2 * (math.cos((math.pi) - (PhiBr)))

            SHM1 = ((UCS + 2 * Pp + (bhp - Pp)) - Shm * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            SHM2 = ((UCS + 2 * Pp + (bhp - Pp)) - Shm2 * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            SHM1H = ((UCShigh + 2 * Pp + (bhp - Pp)) - Shm * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            SHM2H = ((UCShigh + 2 * Pp + (bhp - Pp)) - Shm2 * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            SHM1L = ((UCSlow + 2 * Pp + (bhp - Pp)) - Shm * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            SHM2L = ((UCSlow + 2 * Pp + (bhp - Pp)) - Shm2 * (1 + TwoCosPhiB)) / (1 - TwoCosPhiB)
            
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
                minSH = np.interp(shmin, xucs, yucs)
                maxSH = np.interp(shmin, xd, yd)
            if flag > 1.5 and flag <2.5: #breakout observed on image log
                minSH = np.interp(shmin, xulow, yulow)
                maxSH = np.interp(shmin, xuhigh, yuhigh)
            if flag > 2.5 and flag < 3.5: #tensile fractures observed on image log
                minSH = np.interp(shmin, xucs, yucs)
                maxSH = np.interp(shmin, xd, yd)
            if flag>3.5:# Both breakouts and tensile fractures on log
                maxSH = np.interp(shmin, xd, yd)
                minSH = np.interp(shmin, xucs, yucs)
    midSH = (minSH + maxSH) / 2
    return [minSH, maxSH, midSH]

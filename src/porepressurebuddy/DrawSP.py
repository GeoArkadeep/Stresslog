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

def drawSP(Sv,Pp,bhp,UCS = 0,PhiB = 0,mu = 0.6):

    ufac = ((((mu**2)+1)**0.5)+mu)**2
    print("Mu factor: ",ufac)

    ShmP = ((Sv-Pp)/ufac)+Pp
    SHMP = ((Sv-Pp)*ufac)+Pp
    print("Corners: ",ShmP,SHMP)


    maxSt = 1.05*SHMP
    minSt = 0.90*ShmP
    
    

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
    Shm = 1
    Shm2 = maxSt
    #PhiB = 0.1 #degrees
    PhiBr = math.radians(PhiB)
    TwoCosPhiB = 2*(math.cos((math.pi)-(PhiBr)))
    print("TwoCosPhiB: ",TwoCosPhiB)
    #ShmP = UCS
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
    #print(br1)
    Breakout2 =  Polygon(br2, color='red', label = "UCS- "+str(UCShigh)+"MPa")
    Breakout1 =  Polygon(br1, color='green', label = "UCS- "+str(UCS)+"MPa")
    Breakout3 =  Polygon(br3, color='blue', label = "UCS- "+str(UCSlow)+"MPa")
    ax.add_patch(Breakout2)
    ax.add_patch(Breakout1)
    ax.add_patch(Breakout3)
    
    Shm3 = 1
    Shm4 = maxSt
    
    DITFshmax3 = (ufac*Shm3) - ((ufac-1)*Pp) - (bhp-Pp)
    DITFshmax4 = (ufac*Shm4) - ((ufac-1)*Pp) - (bhp-Pp)
    ditf = np.array([(Shm3,DITFshmax3),(Shm4,DITFshmax4)])
    DITF =  Polygon(ditf, color='aqua', label = 'DITF')
    ax.add_patch(DITF)
    
    ax.legend()
    plt.gca().set_aspect('equal')
    plt.title("Stress Polygon")
    plt.xlabel("Shmin")
    plt.ylabel("SHmax")
    plt.show()


def getSP(Sv,Pp,bhp,mu = 0.6, UCS = 0, PhiB = 0):

    ufac = ((((mu**2)+1)**0.5)+mu)**2
    print("Mu factor: ",ufac)

    ShmP = ((Sv-Pp)/ufac)+Pp
    SHMP = ((Sv-Pp)*ufac)+Pp
    print("Corners: ",ShmP,SHMP)


    maxSt = 1.05*SHMP
    minSt = 0.90*ShmP



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
    Shm = 1
    Shm2 = maxSt
    #PhiB = 0.1 #degrees
    PhiBr = math.radians(PhiB)
    TwoCosPhiB = 2*(math.cos((math.pi)-(PhiBr)))
    print("TwoCosPhiB: ",TwoCosPhiB)
    #ShmP = UCS
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
    #print(br1)
    Breakout2 =  Polygon(br2, color='red', label = "UCS- "+str(UCShigh)+"MPa")
    Breakout1 =  Polygon(br1, color='green', label = "UCS- "+str(UCS)+"MPa")
    Breakout3 =  Polygon(br3, color='blue', label = "UCS- "+str(UCSlow)+"MPa")
    ax.add_patch(Breakout2)
    ax.add_patch(Breakout1)
    ax.add_patch(Breakout3)

    Shm3 = 1
    Shm4 = maxSt

    DITFshmax3 = (ufac*Shm3) - ((ufac-1)*Pp) - (bhp-Pp)
    DITFshmax4 = (ufac*Shm4) - ((ufac-1)*Pp) - (bhp-Pp)
    ditf = np.array([(Shm3,DITFshmax3),(Shm4,DITFshmax4)])
    DITF =  Polygon(ditf, color='aqua', label = 'DITF')
    ax.add_patch(DITF)

    ax.legend()
    plt.gca().set_aspect('equal')
    plt.title("Stress Polygon")
    plt.xlabel("Shmin")
    plt.ylabel("SHmax")
    plt.show()

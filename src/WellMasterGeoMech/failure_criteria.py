import numpy as np

def mod_lad_cmw(sxx,syy,szz,txy,tyz,tzx,theta,pp,phi,cohesion):#Stress tensor rotated to borehole frame of reference
    ps1 = (cohesion/np.tan(np.radians(phi)))
    pfac= (ps1-pp)
    I1 = (sxx+pfac)+(syy+pfac)+(szz+pfac)
    I3 = (sxx+pfac)*(syy+pfac)*(szz+pfac)+(2*txy*tyz*tzx)-((sxx+pfac)*(tyz**2))-((syy+pfac)*(tzx**2))-((szz+pfac)*(txy**2))
    eta = ((I1**3)/I3)-27
    #eta = (4*(np.tan(np.radians(phi))**2)*(9-(7*np.sin(np.radians(phi)))))/(1-np.sin(np.radians(phi)))

    A = szz+pfac
    Stn = sxx+syy-(2*(sxx-syy)*(np.cos(2*np.radians(theta))))-(4*txy*(np.sin(2*np.radians(theta))))
    Ttz = 2*((tyz*(np.cos(np.radians(theta))))-(tzx*(np.sin(np.radians(theta)))))
    B = (A*Stn)-(Ttz**2)
    D = ((Stn+szz+(3*ps1)-(3*pp))**3)/(27+eta)
    C = (B**2)-(4*A*(D-((ps1-pp)*((A*(Stn+ps1-pp))-(Ttz**2)))))
    Pw = (B-(C**0.5))/(2*A)
    
    return Pw

def mogi_failure(s1,s2,s3):
    F = (0.5*(s1+s3))-((1/3)*((((s1-s2)**2)+((s2-s3)**2)+((s3-s1)**2))**0.5))
    return F

def mohr_failure(s1,s3,cohesion,phi):
    sm2 = (s1+s3)/2
    tmax = (s1-s3)/2
    F = (cohesion*np.cos(np.radians(phi)))+(np.sin(np.radians(phi))*sm2) - tmax
    return F

def lade_failure(sx,sy,sz,txy,tyz,tzx,phi,cohesion,pp):
    s3,s2,s1 = np.sort([sx,sy,sz])
    ps1 = (cohesion/np.tan(np.radians(phi)))
    pfac= (ps1-pp)
    #I1 = (s1+pfac)+(s2+pfac)+(s3+pfac)
    #I3 = (s1+pfac)*(s2+pfac)*(s3+pfac)
    I1 = (sx+pfac)+(sy+pfac)+(sz+pfac)
    I3 = (sx+pfac)*(sy+pfac)*(sz+pfac)+(2*txy*tyz*tzx)-((sx+pfac)*(tyz**2))-((sy+pfac)*(tzx**2))-((sz+pfac)*(txy**2))
    eta = ((I1**3)/I3)-27
    eta2 = (4*(np.tan(np.radians(phi))**2)*(9-(7*np.sin(np.radians(phi)))))/(1-np.sin(np.radians(phi)))
    
    F = 27 + eta2 - I1/I3
    F2 = 27 + eta2 - eta
    return F2

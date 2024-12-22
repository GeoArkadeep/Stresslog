"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np

def mod_lad_cmw(sxx,syy,szz,txy,tyz,tzx,theta,phi,cohesion,pp):#Stress tensor rotated to borehole frame of reference
    ps1 = (cohesion/np.tan(phi))
    pfac= (ps1-pp)
    I1 = (sxx+pfac)+(syy+pfac)+(szz+pfac)
    I3 = (sxx+pfac)*(syy+pfac)*(szz+pfac)+(2*txy*tyz*tzx)-((sxx+pfac)*(tyz**2))-((syy+pfac)*(tzx**2))-((szz+pfac)*(txy**2))
    eta2 = ((I1**3)/I3)-27
    eta = (4*((np.tan(phi)**2))*(9-(7*np.sin(phi))))/(1-np.sin(phi))

    A = szz+pfac
    Stn = sxx+syy-(2*(sxx-syy)*(np.cos(2*np.radians(theta))))-(4*txy*(np.sin(2*np.radians(theta))))
    Ttz = 2*((tyz*(np.cos(np.radians(theta))))-(tzx*(np.sin(np.radians(theta)))))
    B = (A*Stn)-(Ttz**2)
    D = ((Stn+szz+(3*ps1)-(3*pp))**3)/(27+eta)
    C = (B**2)-(4*A*(D-((ps1-pp)*((A*(Stn+ps1-pp))-(Ttz**2)))))
    Pw = (B-(C**0.5))/(2*A)
    
    return Pw

def mod_lad_cmw2(sxx, syy, szz, txy, tyz, tzx, theta, phi, cohesion, pp, nu):
    # Convert theta to radians
    theta_rad = np.radians(theta)

    # Calculate S1 (cohesion term)
    S1 = cohesion / np.tan(phi)
    pfac = S1 - pp

    # Eq. 10.18: Calculate Stn (sigma_theta_theta)
    Stn = sxx + syy - 2*(sxx - syy)*np.cos(2*theta_rad) - 4*txy*np.sin(2*theta_rad)

    # Eq. 10.6: Calculate Ttz (tau_theta_z)
    Ttz = 2*(tyz*np.cos(theta_rad) - tzx*np.sin(theta_rad))

    # Eq. 10.5: Calculate szz
    szz = szz - 2*nu*((sxx - syy)*np.cos(2*theta_rad) + 2*txy*np.sin(2*theta_rad))

    # Continue with the rest of the calculations
    A = szz + pfac
    B = (A * Stn) - (Ttz**2)

    eta = (4 * (np.tan(phi)**2) * (9 - 7*np.sin(phi))) / (1 - np.sin(phi))
    D = ((Stn + szz + (3*S1) - (3*pp))**3) / (27 + eta)

    C = (B**2) - (4 * A * (D - ((S1 - pp) * ((A * (Stn + S1 - pp)) - (Ttz**2)))))

    Pw = (B - (C**0.5)) / (2 * A)
    
    return Pw

def mogi_failure(s1,s2,s3):
    F = (0.5*(s1+s3))-((1/3)*((((s1-s2)**2)+((s2-s3)**2)+((s3-s1)**2))**0.5))
    return F

def mohr_failure(s1,s3,cohesion,phi):
    sm2 = (s1+s3)/2
    tmax = (s1-s3)/2
    F = (cohesion*np.cos(phi))+(np.sin(phi)*sm2) - tmax
    return F

def lade_failure(sx,sy,sz,txy,tyz,tzx,phi,cohesion,pp):
    s3,s2,s1 = np.sort([sx,sy,sz])
    ps1 = (cohesion/np.tan(phi))
    pfac= (ps1-pp)
    #I1 = (s1+pfac)+(s2+pfac)+(s3+pfac)
    #I3 = (s1+pfac)*(s2+pfac)*(s3+pfac)
    I1 = (sx+pfac)+(sy+pfac)+(sz+pfac)
    I3 = (sx+pfac)*(sy+pfac)*(sz+pfac)+(2*txy*tyz*tzx)-((sx+pfac)*(tyz**2))-((sy+pfac)*(tzx**2))-((sz+pfac)*(txy**2))
    eta = ((I1**3)/I3)-27
    eta2 = (4*(np.tan(phi)**2)*(9-(7*np.sin(phi))))/(1-np.sin(phi))
    
    F = 27 + eta2 - I1/I3
    F2 = 27 + eta2 - eta
    return F2

def mogi(sx, sy, sz):
    # Ensure all inputs are numpy arrays
    sx, sy, sz = map(np.asarray, (sx, sy, sz))
    
    # Stack the arrays and sort along the last axis
    stresses = np.stack((sx, sy, sz))
    s3, s2, s1 = np.sort(stresses, axis=0)
    
    # Calculate the Mogi failure criterion
    F = (0.5 * (s1 + s3)) - ((1/3) * np.sqrt((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))
    
    return F

def lade(sx, sy, sz, txy, tyz, tzx, phi, cohesion, pp):
    # Ensure all inputs are numpy arrays
    sx, sy, sz, txy, tyz, tzx, phi, cohesion, pp = map(np.asarray, (sx, sy, sz, txy, tyz, tzx, phi, cohesion, pp))
    
    # Calculate principal stresses
    s3 = np.minimum(np.minimum(sx, sy), sz)
    s1 = np.maximum(np.maximum(sx, sy), sz)
    s2 = sx + sy + sz - s1 - s3
    
    ps1 = (cohesion / np.tan(phi))
    pfac = (ps1 - pp)
    
    I1 = (sx + pfac) + (sy + pfac) + (sz + pfac)
    I3 = ((sx + pfac) * (sy + pfac) * (sz + pfac) + 
          (2 * txy * tyz * tzx) - 
          ((sx + pfac) * (tyz**2)) - 
          ((sy + pfac) * (tzx**2)) - 
          ((sz + pfac) * (txy**2)))
    
    eta = ((I1**3) / I3) - 27
    eta2 = (4 * (np.tan(phi)**2) * (9 - (7 * np.sin(phi)))) / (1 - np.sin(phi))
    
    F = 27 + eta2 - I1/I3
    F2 = 27 + eta2 - eta
    
    return F

def zhang_sanding_cwf(sigmamax,sigmamin,pp,ucs,k0,nu,biot=1):
    return (k0*(1-nu))*(3*sigmamax-sigmamin-(biot*((1-(2*nu))/(1-nu))*pp)-ucs)

def willson_sanding_cwf(sigmamax,sigmamin,pp,ucs,nu,biot=1):
    A = biot*(1-(2*nu))/(1-nu)
    return (((3*sigmamax)-sigmamin-ucs)/(2-A)) - (pp*(A/(2-A)))

import matplotlib.pyplot as plt


def plot_sanding(sigmamax, sigmamin,sigma_axial, pp, ucs, k0, nu, biot=1, path=None):
    pparray = np.linspace(pp+50, 0, num=1000)
    ppx = np.linspace(pp, 0, num=100)
    ppy = np.full(len(ppx),pp)
    cwfarray = zhang_sanding_cwf(max(sigmamax,sigma_axial), min(sigmamax,sigma_axial), pparray, ucs, k0, nu, biot)
    cwfarray2 = zhang_sanding_cwf(max(sigmamin,sigma_axial), min(sigmamin,sigma_axial), pparray, ucs, k0, nu, biot)
    cwfarray3 = willson_sanding_cwf(max(sigmamax,sigma_axial), min(sigmamax,sigma_axial), pparray, ucs, nu, biot)
    cwfarray4 = willson_sanding_cwf(max(sigmamin,sigma_axial), min(sigmamin,sigma_axial), pparray, ucs, nu, biot)
    cwf_at_pp = zhang_sanding_cwf(sigmamax, sigmamin, pp, ucs, k0, nu, biot)
    # Limit cwfarray to be below or equal to pparray
    cwfarray = np.minimum(cwfarray, pparray)
    cwfarray2 = np.minimum(cwfarray2, pparray)
    cwfarray3 = np.minimum(cwfarray3, pparray)
    cwfarray4 = np.minimum(cwfarray4, pparray)
    
    plt.figure(figsize=(6, 6))
    
    
    # Plot the cwfarray line
    plt.plot(pparray, cwfarray, 'c:', label='CWF Perf oriented @ SHmin, Zhang, k0='+str(k0))
    plt.plot(pparray, cwfarray2, 'm:', label='CWF Perf oriented @ SHMax, Zhang, k0='+str(k0))
    plt.plot(pparray, cwfarray3, 'c--', label='CWF Perf oriented @ SHmin, Willson')
    plt.plot(pparray, cwfarray4, 'm--', label='CWF Perf oriented @ SHMax, Willson')
    plt.plot(ppy, ppx,'k-.', label='Current Reservoir Pressure')
    # Shade the area below the diagonal
    plt.fill_between(pparray, 0, pparray, alpha=0.3, color='green')
    
    # Shade the area below cwfarray (but above 0)
    plt.fill_between(pparray, 0, cwfarray, where=(cwfarray > 0), alpha=1, color='orange')
    plt.fill_between(pparray, 0, cwfarray2, where=(cwfarray > 0), alpha=1, color='red')
    
    # Plot the diagonal line
    plt.plot(pparray, pparray, 'k-', label=None)
    
    plt.xlabel('Initial Reservoir Pressure')
    plt.ylabel('Flowing Bottom Hole Pressure')
    plt.title('Sanding Analysis')
    plt.xlim(0,pp+50)
    plt.ylim(0,pp+50)
    plt.legend()
    plt.grid(False)
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        return plt

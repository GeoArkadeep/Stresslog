"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np

def mod_lad_cmw(sxx,syy,szz,txy,tyz,tzx,theta,phi,cohesion,pp):#Stress tensor rotated to borehole frame of reference
    """
    Calculate the modified Lade critical mud weight with pore pressure correction in the wellbore coordinate system.
    The stress tensor should already have been rotated to be in the borehole coordinate system.

    Parameters
    ----------
    sxx : float or array_like
        Normal stress component in x-direction
    syy : float or array_like
        Normal stress component in y-direction
    szz : float or array_like
        Normal stress component in z-direction
    txy : float or array_like
        Shear stress component in xy-plane
    tyz : float or array_like
        Shear stress component in yz-plane
    tzx : float or array_like
        Shear stress component in zx-plane
    theta : float
        Wellbore angle in degrees
    phi : float
        Internal friction angle in radians
    cohesion : float
        Rock cohesion strength
    pp : float
        Pore pressure

    Returns
    -------
    float or array_like
        Critical wellbore pressure (Pw)
    """

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
    """
    Calculate modified Lade critical mud weight with pore pressure and Poisson's ratio corrections.
    The stress tensor should already have been rotated to be in the borehole coordinate system

    Parameters
    ----------
    sxx : float or array_like
        Normal stress component in x-direction
    syy : float or array_like
        Normal stress component in y-direction
    szz : float or array_like
        Normal stress component in z-direction
    txy : float or array_like
        Shear stress component in xy-plane
    tyz : float or array_like
        Shear stress component in yz-plane
    tzx : float or array_like
        Shear stress component in zx-plane
    theta : float
        Wellbore angle in degrees
    phi : float
        Internal friction angle in radians
    cohesion : float
        Rock cohesion strength
    pp : float
        Pore pressure
    nu : float
        Poisson's ratio

    Returns
    -------
    float or array_like
        Critical wellbore pressure (Pw) with Poisson's ratio correction
    """
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
    """
    Calculate the Mogi failure criterion for principal stresses.

    Parameters
    ----------
    s1 : float or array_like
        Maximum principal stress
    s2 : float or array_like
        Intermediate principal stress
    s3 : float or array_like
        Minimum principal stress

    Returns
    -------
    float or array_like
        Mogi failure criterion value (F)
    """
    F = (0.5*(s1+s3))-((1/3)*((((s1-s2)**2)+((s2-s3)**2)+((s3-s1)**2))**0.5))
    return F

def mohr_failure(s1,s3,cohesion,phi):
    """
    Calculate the Mohr-Coulomb failure criterion.

    Parameters
    ----------
    s1 : float or array_like
        Maximum principal stress
    s3 : float or array_like
        Minimum principal stress
    cohesion : float
        Rock cohesion strength
    phi : float
        Internal friction angle in radians

    Returns
    -------
    float or array_like
        Mohr-Coulomb failure criterion value (F)
    """
    sm2 = (s1+s3)/2
    tmax = (s1-s3)/2
    F = (cohesion*np.cos(phi))+(np.sin(phi)*sm2) - tmax
    return F

def lade_failure(sx,sy,sz,txy,tyz,tzx,phi,cohesion,pp):
    """
    Calculate the Modified Lade failure criterion with pore pressure correction.

    Parameters
    ----------
    sx : float or array_like
        Normal stress in x-direction
    sy : float or array_like
        Normal stress in y-direction
    sz : float or array_like
        Normal stress in z-direction
    txy : float or array_like
        Shear stress in xy-plane
    tyz : float or array_like
        Shear stress in yz-plane
    tzx : float or array_like
        Shear stress in zx-plane
    phi : float
        Internal friction angle in radians
    cohesion : float
        Rock cohesion strength
    pp : float
        Pore pressure

    Returns
    -------
    float or array_like
        Lade failure criterion value (F2)
    """

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
    """
    Calculate the Mogi failure criterion for stress components with numpy array support.

    Parameters
    ----------
    sx : array_like
        Normal stress in x-direction
    sy : array_like
        Normal stress in y-direction
    sz : array_like
        Normal stress in z-direction

    Returns
    -------
    array_like
        Mogi failure criterion value (F)
    """
    # Ensure all inputs are numpy arrays
    sx, sy, sz = map(np.asarray, (sx, sy, sz))
    
    # Stack the arrays and sort along the last axis
    stresses = np.stack((sx, sy, sz))
    s3, s2, s1 = np.sort(stresses, axis=0)
    
    # Calculate the Mogi failure criterion
    F = (0.5 * (s1 + s3)) - ((1/3) * np.sqrt((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))
    
    return F

def lade(sx, sy, sz, txy, tyz, tzx, phi, cohesion, pp):
    """
    Calculate the Lade failure criterion with numpy array support.

    Parameters
    ----------
    sx : array_like
        Normal stress in x-direction
    sy : array_like
        Normal stress in y-direction
    sz : array_like
        Normal stress in z-direction
    txy : array_like
        Shear stress in xy-plane
    tyz : array_like
        Shear stress in yz-plane
    tzx : array_like
        Shear stress in zx-plane
    phi : array_like
        Internal friction angle in radians
    cohesion : array_like
        Rock cohesion strength
    pp : array_like
        Pore pressure

    Returns
    -------
    array_like
        Lade failure criterion value (F)
    """
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
    """
    Calculate the Zhang critical wellbore flowing pressure for sanding prediction.

    Parameters
    ----------
    sigmamax : float or array_like
        Maximum principal stress
    sigmamin : float or array_like
        Minimum principal stress
    pp : float or array_like
        Pore pressure
    ucs : float
        Unconfined compressive strength
    k0 : float
        Earth stress ratio
    nu : float
        Poisson's ratio
    biot : float, optional
        Biot's coefficient (default is 1)

    Returns
    -------
    float or array_like
        Critical wellbore flowing pressure
    """
    return (k0*(1-nu))*(3*sigmamax-sigmamin-(biot*((1-(2*nu))/(1-nu))*pp)-ucs)

def willson_sanding_cwf(sigmamax,sigmamin,pp,ucs,nu,biot=1):
    """
    Calculate the Willson critical wellbore flowing pressure for sanding prediction.

    Parameters
    ----------
    sigmamax : float or array_like
        Maximum principal stress
    sigmamin : float or array_like
        Minimum principal stress
    pp : float or array_like
        Pore pressure
    ucs : float
        Unconfined compressive strength
    nu : float
        Poisson's ratio
    biot : float, optional
        Biot's coefficient (default is 1)

    Returns
    -------
    float or array_like
        Critical wellbore flowing pressure
    """
    A = biot*(1-(2*nu))/(1-nu)
    return (((3*sigmamax)-sigmamin-ucs)/(2-A)) - (pp*(A/(2-A)))

import matplotlib.pyplot as plt


def plot_sanding(sigmamax, sigmamin,sigma_axial, pp, ucs, k0, nu, biot=1, path=None, display=False):
    """
    Create a sanding analysis plot comparing Zhang and Willson criteria.

    Parameters
    ----------
    sigmamax : float
        Maximum principal stress
    sigmamin : float
        Minimum principal stress
    sigma_axial : float
        Axial stress
    pp : float
        Pore pressure
    ucs : float
        Unconfined compressive strength
    k0 : float
        Earth stress ratio
    nu : float
        Poisson's ratio
    biot : float, optional
        Biot's coefficient (default is 1)
    path : str, optional
        Path to save the plot (default is None)

    Returns
    -------
    matplotlib.pyplot
        Plot object if path is None, otherwise saves plot to specified path
    """
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
    plt.legend(frameon=False, facecolor='none')
    plt.grid(False)
    if path is not None:
        plt.savefig(path)
        if display:
            plt.show()
        plt.close()
    else:
        return plt

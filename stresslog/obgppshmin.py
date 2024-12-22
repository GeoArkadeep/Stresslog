"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import math
import numpy as np

def get_OBG_pascals_vec(tvd, tvdbgl, tvdmsl, rhogcc, water, wdf, glwd):
    """
    Vectorized calculation of comprehensive overburden stress considering offshore and onshore scenarios.
    
    Parameters:
    tvd (array-like): True Vertical Depth
    tvdbgl (array-like): True Vertical Depth Below Ground Level
    tvdmsl (array-like): True Vertical Depth from Mean Sea Level
    rhogcc (array-like): Density in g/cc
    water (float): Water density
    wdf (float): Water depth factor
    glwd (float): Ground Level Water Depth (negative for offshore)
    
    Returns:
    tuple: Arrays of integrho (Pa), integrhopsift, and ObgTppg
    """
    tvd = np.asarray(tvd)
    tvdbgl = np.asarray(tvdbgl)
    tvdmsl = np.asarray(tvdmsl)
    rhogcc = np.asarray(rhogcc)
    
    n = len(tvd)
    integrho = np.zeros(n)
    integrhopsift = np.zeros(n)
    ObgTppg = np.zeros(n)
    
    maxwaterppg = wdf * 8.34540426515252 * water
    
    # Calculate depth differences
    depth_diff = np.diff(tvdbgl, prepend=tvdbgl[0])
    
    # Offshore case
    if glwd < 0:
        # Mudline mask
        mudline_mask = tvdbgl > 0
        
        # Calculate integrho
        water_pressure = np.cumsum(water * 9806.65 * depth_diff * (~mudline_mask))
        rock_pressure = np.cumsum(rhogcc * 9806.65 * depth_diff * mudline_mask)
        integrho = water_pressure + rock_pressure
        
        # Calculate integrhopsift
        integrhopsift = (integrho * 0.000145038) / tvd
        
        # Calculate ObgTppg
        ObgTppg = np.where(tvdmsl > 0,
                           np.where(mudline_mask,
                                    (maxwaterppg + (np.cumsum(rhogcc * 8.34540426515252 * depth_diff) / tvdbgl)) / tvdmsl,
                                    8.34540426515252 * water),
                           0)  # Set to 0 where tvdmsl <= 0
    else:
        # Onshore case
        onshore_mask = tvdbgl > 0
        
        # Calculate integrho
        integrho = np.cumsum(rhogcc * 9806.65 * depth_diff * onshore_mask)
        
        # Calculate integrhopsift
        integrhopsift = (integrho * 0.000145038) / tvd
        
        # Calculate ObgTppg (Curved Top Obg Gradient)
        ObgTppg = np.where(onshore_mask,
                           (np.cumsum(rhogcc * 8.34540426515252 * depth_diff) / tvdbgl) / tvd,
                           0)  # Set to 0 where tvdbgl <= 0
    
    return integrho, integrhopsift, ObgTppg

def get_PPgrad_Zhang_gcc(ObgTgcc, pn, b, tvdbgl, c, mudline, matrick, deltmu0, dalm, biot=1):
    if b>=c:
        numerator = ObgTgcc - ((ObgTgcc-pn)*((math.log((mudline-matrick))-(math.log(dalm-matrick)))/(c*tvdbgl)))
    else:
        numerator = ObgTgcc - ((ObgTgcc - pn) / (b * tvdbgl)) * ((((b - c) / c) * (math.log((mudline - matrick) / (deltmu0 - matrick)))) + (math.log((mudline - matrick) / (dalm - matrick))))
    
    return numerator / biot

def get_PP_grad_Zhang_gcc_vec(ObgTgcc, pn, b, tvdbgl, c, mudline, matrick, deltmu0, dalm, biot=1):
    # Ensure all inputs are numpy arrays
    ObgTgcc = np.asarray(ObgTgcc)
    pn = np.asarray(pn)
    b = np.asarray(b)
    tvdbgl = np.asarray(tvdbgl)
    c = np.asarray(c)
    mudline = np.asarray(mudline)
    matrick = np.asarray(matrick)
    deltmu0 = np.asarray(deltmu0)
    dalm = np.asarray(dalm)
    biot = np.asarray(biot)

    # Broadcast arrays to a common shape
    ObgTgcc, pn, b, tvdbgl, c, mudline, matrick, deltmu0, dalm, biot = np.broadcast_arrays(
        ObgTgcc, pn, b, tvdbgl, c, mudline, matrick, deltmu0, dalm, biot
    )

    # Apply the condition element-wise
    numerator = np.where(
        b >= c,
        ObgTgcc - ((ObgTgcc - pn) * (np.log((mudline - matrick)) - np.log(dalm - matrick)) / (c * tvdbgl)),
        ObgTgcc - ((ObgTgcc - pn) / (b * tvdbgl)) * (
            (((b - c) / c) * np.log((mudline - matrick) / (deltmu0 - matrick))) +
            np.log((mudline - matrick) / (dalm - matrick))
        )
    )

    return numerator / biot


def get_PPgrad_Eaton_gcc(ObgTgcc, pn, be, ne, tvdbgl, res0, resdeep, biot=1):
    numerator = ObgTgcc - ((ObgTgcc - pn)*((resdeep/(res0*np.exp(be*tvdbgl)))**ne))
    return numerator / biot

def get_PPgrad_Eaton_gcc_vec(ObgTgcc, pn, be, ne, tvdbgl, res0, resdeep, biot=1):
    # Ensure all inputs are numpy arrays
    ObgTgcc = np.asarray(ObgTgcc)
    pn = np.asarray(pn)
    be = np.asarray(be)
    ne = np.asarray(ne)
    tvdbgl = np.asarray(tvdbgl)
    res0 = np.asarray(res0)
    resdeep = np.asarray(resdeep)
    biot = np.asarray(biot)

    # Broadcast scalar values to match the shape of the largest array
    ObgTgcc, pn, be, ne, tvdbgl, res0, resdeep, biot = np.broadcast_arrays(
        ObgTgcc, pn, be, ne, tvdbgl, res0, resdeep, biot
    )

    # Calculate numerator
    numerator = ObgTgcc - ((ObgTgcc - pn) * ((resdeep / (res0 * np.exp(be * tvdbgl))) ** ne))
    return numerator / biot


def get_PPgrad_Dxc_gcc(ObgTgcc, pn, d, nde, tvdbgl, D0, Dxc, biot=1):
    numerator = ObgTgcc - ((ObgTgcc - pn)*((Dxc/(D0*np.exp(d*tvdbgl)))**nde))
    return numerator / biot

def get_PPgrad_Dxc_gcc_vec(ObgTgcc, pn, d, nde, tvdbgl, D0, Dxc, biot=1):
    # Ensure all inputs are numpy arrays
    ObgTgcc = np.asarray(ObgTgcc)
    pn = np.asarray(pn)
    d = np.asarray(d)
    nde = np.asarray(nde)
    tvdbgl = np.asarray(tvdbgl)
    D0 = np.asarray(D0)
    Dxc = np.asarray(Dxc)
    biot = np.asarray(biot)

    # Broadcast scalar values to match the shape of the largest array
    ObgTgcc, pn, d, nde, tvdbgl, D0, Dxc, biot = np.broadcast_arrays(
        ObgTgcc, pn, d, nde, tvdbgl, D0, Dxc, biot
    )

    # Calculate numerator
    numerator = ObgTgcc - ((ObgTgcc - pn) * ((Dxc / (D0 * np.exp(d * tvdbgl))) ** nde))
    return numerator / biot

def get_Dxc(ROP,RPM,WOB,BTDIA,ECD,pn):
    #units= ROP:ft/hr, WOB:lbs, BTDIA:in, 
    Dxc = (np.log10(ROP/(60*RPM))*pn)/(np.log10((12*WOB)/((10**6)*BTDIA))*ECD)
    return Dxc if Dxc>0.1 else np.nan

def get_Dxc_vec(ROP, RPM, WOB, BTDIA, ECD, pn):
    # Ensure all inputs are numpy arrays
    ROP = np.asarray(ROP)
    RPM = np.asarray(RPM)
    WOB = np.asarray(WOB)
    BTDIA = np.asarray(BTDIA)
    ECD = np.asarray(ECD)
    pn = np.asarray(pn)

    # Broadcast scalar values to match the shape of the largest array
    ROP, RPM, WOB, BTDIA, ECD, pn = np.broadcast_arrays(ROP, RPM, WOB, BTDIA, ECD, pn)

    # Calculate Dxc
    Dxc = (np.log10(ROP / (60 * RPM)) * pn) / (np.log10((12 * WOB) / (10**6 * BTDIA)) * ECD)
    return Dxc


def get_Shmin_grad_Daine_ppg(nu2, ObgTppg, biot, ppgZhang, tecB):
    """
    Scalar calculation of fracture gradient pressure for a single sample.
    
    Parameters:
    nu2 (float): Poisson's ratio
    ObgTppg (float): Overburden gradient in ppg
    biot (float): Biot's coefficient
    ppgZhang (float): Zhang's pore pressure in ppg
    tecB (float): Tectonic factor
    
    Returns:
    float: Fracture gradient pressure in ppg
    """
    return (nu2 / (1 - nu2)) * (ObgTppg - (biot * ppgZhang)) + (biot * ppgZhang) + (tecB * ObgTppg)

def get_Shmin_grad_Daine_ppg_vec(nu2, ObgTppg, biot, ppgZhang, tecB):
    """
    Vectorized calculation of fracture gradient pressure.
    
    Parameters:
    nu2 (array-like): Poisson's ratio
    ObgTppg (array-like): Overburden gradient in ppg
    biot (array-like): Biot's coefficient
    ppgZhang (array-like): Zhang's pore pressure in ppg
    tecB (array-like): Tectonic factor
    
    Returns:
    numpy.ndarray: Fracture gradient pressure in ppg
    """
    # Convert all inputs to numpy arrays
    nu2 = np.asarray(nu2)
    ObgTppg = np.asarray(ObgTppg)
    biot = np.asarray(biot)
    ppgZhang = np.asarray(ppgZhang)
    tecB = np.asarray(tecB)
    
    return (nu2 / (1 - nu2)) * (ObgTppg - (biot * ppgZhang)) + (biot * ppgZhang) + (tecB * ObgTppg)

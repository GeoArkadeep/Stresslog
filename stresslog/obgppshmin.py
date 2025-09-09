"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import math
import numpy as np

def get_OBG_pascals_vec(tvd, tvdbgl, tvdmsl, rhogcc, water, glwd):
    """
    Vectorized calculation of comprehensive overburden stress considering offshore and onshore scenarios.
    
    Parameters
    ----------
    tvd : array-like
        True Vertical Depth
    tvdbgl : array-like
        True Vertical Depth Below Ground Level
    tvdmsl : array-like
        True Vertical Depth from Mean Sea Level
    rhogcc: array-like
        Density in g/cc
    water :float
        Water density in g/cc
    glwd :float
        Ground Level/Water Depth in meters (negative for offshore)
    
    Returns
    -------
    tuple:
        Arrays of integrho (Rho_a), integrhopsift, and ObgTppg
    
    Notes
    -----
    - Inputs may be scalars (floats) or 1D arrays. Scalars are broadcast
      across arrays automatically.
    - All array inputs must have the same length; mismatched lengths
      will raise a ValueError due to NumPy broadcasting rules.
    """
    tvd = np.asarray(tvd)
    tvdbgl = np.asarray(tvdbgl)
    tvdmsl = np.asarray(tvdmsl)
    rhogcc = np.asarray(rhogcc)
    
    n = len(tvd)
    integrho = np.zeros(n)
    integrhopsift = np.zeros(n)
    ObgTppg = np.zeros(n)
    
    
    
    # Calculate depth differences
    depth_diff = np.diff(tvdbgl, prepend=tvdbgl[0])
    
    # Offshore case
    if glwd < 0:
        maxwaterppg = (abs(glwd) * 3.28084) * 8.34540426515252 * water
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
    """Calculate pore pressure gradient using Zhang's method for a single point.

    Parameters
    ----------
    ObgTgcc : float
        Overburden gradient in gcc units
    pn : float
        Normal (Hydrostatic) pressure gradient in g/cc
    b : float
        Compaction coefficient for unloading case
    tvdbgl : float
        True vertical depth below ground level
    c : float
        Compaction coefficient for loading case
    mudline : float
        interval travel time at mudline (uspf)
    matrick : float
        matrix interval travel time (uspf)
    deltmu0 : float
        interval travel time as recorded on P-sonic log at
        the depth considered to be the top of unloading condition
    dalm : float
        travel time as recorded on P-sonic log (uspf)
        for the given depth
    biot : float, optional
        Biot's coefficient, by default 1

    Returns
    -------
    float
        Calculated pore pressure gradient in gcc units

    Notes
    -----
    Implements Zhang's method for calculating pore pressure gradient.
    The calculation varies based on the relationship between b and c coefficients.
    """
    biot = np.clip(biot, 0.0001, 1.0)
    if b>=c:
        numerator = ObgTgcc - ((ObgTgcc-pn)*((math.log((mudline-matrick))-(math.log(dalm-matrick)))/(c*tvdbgl)))
    else:
        numerator = ObgTgcc - ((ObgTgcc - pn) / (b * tvdbgl)) * ((((b - c) / c) * (math.log((mudline - matrick) / (deltmu0 - matrick)))) + (math.log((mudline - matrick) / (dalm - matrick))))
    
    return numerator / biot

def get_PP_grad_Zhang_gcc_vec(ObgTgcc, pn, b, tvdbgl, c, mudline, matrick, deltmu0, dalm, biot=1):
    """
    Calculate pressure gradient using Zhang's method with vectorized inputs.

    Parameters
    ----------
    ObgTgcc : array_like
        Overburden gradient in g/cc
    pn : array_like
        Normal (Hydrostatic) pressure gradient in g/cc
    b : float/array_like
        Compaction coefficient for unloading case
    tvdbgl : array_like
        True vertical depth below ground level in metres
    c : float/array_like
        Compaction coefficient for loading case
    mudline : float/array_like
        interval travel time at mudline (uspf)
    matrick : float/array_like
        interval travel time in matrix (0 porosity case) (uspf)
    deltmu0 : array_like
        Interval travel time at top of unloading condition in uspf
    dalm : array_like
        P-sonic log array in uspf
    biot : array_like, optional
        Biot coefficient, defaults to 1

    Returns
    -------
    ndarray
        Pressure gradient calculated using Zhang's method

    Notes
    -----
    This function applies Zhang's method for pressure gradient calculation
    with support for vectorized operations. All inputs are broadcast to
    compatible shapes before calculation. Arrays must be of the same length if used.
    """
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
    biot = np.clip(biot, 0.0001, 1.0)
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
    """
    Calculate pressure gradient using Eaton's method.

    Parameters
    ----------
    ObgTgcc : float
        Overburden gradient in g/cc
    pn : float
        Normal (Hydrostatic) pressure gradient in g/cc
    be : float
        Eaton's parameter b
    ne : float
        Eaton's exponent
    tvdbgl : float
        True vertical depth below ground level
    res0 : float
        Resistivity at mudline
    resdeep : float
        Deep resistivity measurement of the given sample
    biot : float, optional
        Biot coefficient, defaults to 1

    Returns
    -------
    float
        Pressure gradient calculated using Eaton's method

    Notes
    -----
    This is the single-point version of the Eaton pressure gradient calculation.
    For vectorized operations, use get_PPgrad_Eaton_gcc_vec.
    """
    biot = np.clip(biot, 0.0001, 1.0)
    if be*tvdbgl>708:
        return np.nan
    numerator = ObgTgcc - ((ObgTgcc - pn)*((resdeep/(res0*np.exp(be*tvdbgl)))**ne))
    return numerator / biot

def get_PPgrad_Eaton_gcc_vec(ObgTgcc, pn, be, ne, tvdbgl, res0, resdeep, biot=1):
    """
    Calculate pressure gradient using Eaton's method with vectorized inputs.

    Parameters
    ----------
    ObgTgcc : array_like
        Overburden gradient in g/cc
    pn : array_like
        Normal (Hydrostatic) pressure gradient in g/cc
    be : float/array_like
        Eaton's parameter b
    ne : float/array_like
        Eaton's exponent
    tvdbgl : array_like
        True vertical depth below ground level in metres
    res0 : float/array_like
        Resistivity at mudline in ohm.m
    resdeep : array_like
        Deep resistivity measurements log in ohm.m
    biot : float/array_like, optional
        Biot coefficient, defaults to 1

    Returns
    -------
    ndarray
        Pressure gradient calculated using Eaton's method

    Notes
    -----
    Vectorized version of the Eaton pressure gradient calculation.
    All inputs are broadcast to compatible shapes before calculation. Arrays must be of the same length if used.
    """
    # Ensure all inputs are numpy arrays
    ObgTgcc = np.asarray(ObgTgcc)
    pn = np.asarray(pn)
    be = np.asarray(be)
    ne = np.asarray(ne)
    tvdbgl = np.asarray(tvdbgl)
    res0 = np.asarray(res0)
    resdeep = np.asarray(resdeep)
    biot = np.asarray(biot)
    biot = np.clip(biot, 0.0001, 1.0)
    # Broadcast scalar values to match the shape of the largest array
    ObgTgcc, pn, be, ne, tvdbgl, res0, resdeep, biot = np.broadcast_arrays(
        ObgTgcc, pn, be, ne, tvdbgl, res0, resdeep, biot
    )

    # Calculate numerator
    mask = be * tvdbgl > 708
    numerator = ObgTgcc - ((ObgTgcc - pn) * ((resdeep / (res0 * np.exp(be * tvdbgl))) ** ne))
    numerator[mask] = np.nan
    return numerator / biot


def get_PPgrad_Dxc_gcc(ObgTgcc, pn, d, nde, tvdbgl, D0, Dxc, biot=1):
    """
    Calculate pressure gradient using d-exponent method.

    Parameters
    ----------
    ObgTgcc : float
        Overburden gradient in g/cc
    pn : float
        Normal pressure in g/cc
    d : float
        d-exponent parameter
    nde : float
        d-exponent power
    tvdbgl : float
        True vertical depth below ground level
    D0 : float
        d-exponent value at surface
    Dxc : float
        Corrected d-exponent
    biot : float, optional
        Biot coefficient, defaults to 1

    Returns
    -------
    float
        Pressure gradient calculated using d-exponent method

    Notes
    -----
    This is the scalar version of the d-exponent pressure gradient calculation.
    For vectorized operations, use get_PPgrad_Dxc_gcc_vec.
    """
    biot = np.clip(biot, 0.0001, 1.0)
    if d*tvdbgl> 708:
        return np.nan
    numerator = ObgTgcc - ((ObgTgcc - pn)*((Dxc/(D0*np.exp(d*tvdbgl)))**nde))
    return numerator / biot

def get_PPgrad_Dxc_gcc_vec(ObgTgcc, pn, d, nde, tvdbgl, D0, Dxc, biot=1):
    """
    Calculate pressure gradient using d-exponent method with vectorized inputs.

    Parameters
    ----------
    ObgTgcc : array_like
        Overburden gradient in g/cc
    pn : array_like
        Normal pressure in g/cc
    d : float/array_like
        d-exponent parameter
    nde : float/array_like
        d-exponent power
    tvdbgl : array_like
        True vertical depth below ground level
    D0 : float/array_like
        Reference d-exponent value
    Dxc : array_like
        Corrected d-exponent
    biot : float/array_like, optional
        Biot coefficient, defaults to 1

    Returns
    -------
    ndarray
        Pressure gradient calculated using d-exponent method

    Notes
    -----
    Vectorized version of the d-exponent pressure gradient calculation.
    All inputs are broadcast to compatible shapes before calculation. Arrays must be of the same length if used.
    """
    # Ensure all inputs are numpy arrays
    ObgTgcc = np.asarray(ObgTgcc)
    pn = np.asarray(pn)
    d = np.asarray(d)
    nde = np.asarray(nde)
    tvdbgl = np.asarray(tvdbgl)
    D0 = np.asarray(D0)
    Dxc = np.asarray(Dxc)
    biot = np.asarray(biot)
    biot = np.clip(biot, 0.0001, 1.0)
    # Broadcast scalar values to match the shape of the largest array
    ObgTgcc, pn, d, nde, tvdbgl, D0, Dxc, biot = np.broadcast_arrays(
        ObgTgcc, pn, d, nde, tvdbgl, D0, Dxc, biot
    )

    # Calculate numerator
    mask = d * tvdbgl > 708
    numerator = ObgTgcc - ((ObgTgcc - pn) * ((Dxc / (D0 * np.exp(d * tvdbgl))) ** nde))
    numerator[mask] = np.nan
    return numerator / biot

def get_Dxc(ROP,RPM,WOB,BTDIA,ECD,pn):
    """
    Calculate corrected d-exponent.

    Parameters
    ----------
    ROP : float
        Rate of penetration in ft/hr
    RPM : float
        Rotations per minute
    WOB : float
        Weight on bit in lbs
    BTDIA : float
        Bit diameter in inches
    ECD : float
        Equivalent circulating density
    pn : float
        Hydrostatic pressure gradient, in same units as ECD

    Returns
    -------
    float
        Corrected d-exponent if greater than 0.1, otherwise NaN

    Notes
    -----
    This is the scalar version of the corrected d-exponent calculation.
    For vectorized operations, use get_Dxc_vec.
    """
    #units= ROP:ft/hr, WOB:lbs, BTDIA:in, 
    Dxc = (np.log10(ROP/(60*RPM))*pn)/(np.log10((12*WOB)/((10**6)*BTDIA))*ECD)
    return Dxc if Dxc>0.1 else np.nan

def get_Dxc_vec(ROP, RPM, WOB, BTDIA, ECD, pn):
    """
    Calculate corrected d-exponent with vectorized inputs.

    Parameters
    ----------
    ROP : array_like
        Rate of penetration in ft/hr
    RPM : array_like
        Rotations per minute
    WOB : array_like
        Weight on bit in lbs
    BTDIA : array_like
        Bit diameter in inches
    ECD : array_like
        Equivalent circulating density
    pn : array_like
        Hydrostatic pressure gradient, in same units as ECD

    Returns
    -------
    ndarray
        Array of corrected d-exponent values

    Notes
    -----
    Vectorized version of the corrected d-exponent calculation.
    All inputs are broadcast to compatible shapes before calculation. Arrays must be of the same length if used.
    """
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
    return np.where(Dxc > 0.1, Dxc, np.nan)


def get_Shmin_grad_Daine_ppg(nu2, ObgTppg, biot, ppgZhang, tecB):
    """
    Scalar calculation of fracture gradient pressure for a single sample.
    
    Parameters
    ----------
    nu2 : float
        Poisson's ratio
    ObgTppg : float
        Overburden gradient in ppg
    biot : float
        Biot's coefficient
    ppgZhang : float
        Zhang's pore pressure in ppg
    tecB : float
        Tectonic factor
    
    Returns
    -------
    float
        Fracture gradient pressure in ppg
    """
    if nu2>0.5 or nu2<0:
        return np.nan
    return (nu2 / (1 - nu2)) * (ObgTppg - (biot * ppgZhang)) + (biot * ppgZhang) + (tecB * ObgTppg)

def get_Shmin_grad_Daine_ppg_vec(nu2, ObgTppg, biot, ppgZhang, tecB):
    """
    Vectorized calculation of fracture gradient pressure.
    
    Parameters
    ----------
    nu2 : array-like
        Poisson's ratio
    ObgTppg : array-like
        Overburden gradient in ppg
    biot : array-like
        Biot's coefficient
    ppgZhang : array-like
        Zhang's pore pressure in ppg
    tecB : array-like
        Tectonic factor
    
    Returns
    -------
    numpy.ndarray
        Fracture gradient pressure in ppg
    
    Notes
    -----
    - Inputs may be scalars (floats) or 1D arrays. Scalars are broadcast
      across arrays automatically.
    - All array inputs must have the same length; mismatched lengths
      will raise a ValueError due to NumPy broadcasting rules.
    """
    # Convert all inputs to numpy arrays
    nu2 = np.asarray(nu2)
    ObgTppg = np.asarray(ObgTppg)
    biot = np.asarray(biot)
    ppgZhang = np.asarray(ppgZhang)
    tecB = np.asarray(tecB)
    mask = (nu2 >= 0) & (nu2 <= 0.5)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Perform the calculation for all elements. Elements where nu2 is invalid
        # might result in inf or nan, but we will handle that next.
        calculation = (nu2 / (1 - nu2)) * (ObgTppg - (biot * ppgZhang)) + (biot * ppgZhang) + (tecB * ObgTppg)
    result = np.where(mask, calculation, np.nan)
    return result #(nu2 / (1 - nu2)) * (ObgTppg - (biot * ppgZhang)) + (biot * ppgZhang) + (tecB * ObgTppg)

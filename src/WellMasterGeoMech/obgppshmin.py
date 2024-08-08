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
    numerator = ObgTgcc - ((ObgTgcc - pn) / (b * tvdbgl)) * (
        (((b - c) / c) * (math.log((mudline - matrick) / (deltmu0 - matrick)))) +
        (math.log((mudline - matrick) / (dalm - matrick)))
    )
    
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

    # Broadcast scalar values to match the shape of the largest array
    max_shape = max(arr.shape for arr in [ObgTgcc, pn, b, tvdbgl, c, mudline, matrick, deltmu0, dalm, biot])
    ObgTgcc, pn, b, tvdbgl, c, mudline, matrick, deltmu0, dalm, biot = np.broadcast_arrays(
        ObgTgcc, pn, b, tvdbgl, c, mudline, matrick, deltmu0, dalm, biot
    )

    numerator = ObgTgcc - ((ObgTgcc - pn) / (b * tvdbgl)) * (
        (((b - c) / c) * (np.log((mudline - matrick) / (deltmu0 - matrick)))) +
        (np.log((mudline - matrick) / (dalm - matrick)))
    )
    
    return numerator / biot

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
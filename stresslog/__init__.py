"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

# BoreStab.py
from .BoreStab import (
    get_optimal,
    getVertical,
    getAlignedStress,
    getRota,
    getStens,
    getStrikeDip,
    getEuler,
    getOrit,
    getSigmaTT,
    getHoop,
    draw
)

from .thirdparty import datasets_to_las

from .syntheticLogs import create_random_las, getwelldev, create_random_well

# DrawSP.py
from .DrawSP import drawSP, getSP

# failure_criteria.py
from .failure_criteria import (
    mod_lad_cmw,
    mod_lad_cmw2,
    mogi_failure,
    mohr_failure,
    lade_failure,
    mogi,
    lade,
    zhang_sanding_cwf,
    willson_sanding_cwf,
    plot_sanding
)


# geomechanics.py
from .geomechanics import plotPPzhang

# hydraulics.py
from .hydraulics import (
    getColumnHeights,
    getPPfromTop,
    getPPfromTopRecursive,
    getGasDensity,
    getHydrostaticPsi,
    compute_optimal_offset,
    compute_optimal_gradient
)

# obgppshmin.py
from .obgppshmin import (
    get_OBG_pascals_vec,
    get_PPgrad_Zhang_gcc,
    get_PP_grad_Zhang_gcc_vec,
    get_PPgrad_Eaton_gcc,
    get_PPgrad_Eaton_gcc_vec,
    get_PPgrad_Dxc_gcc,
    get_PPgrad_Dxc_gcc_vec,
    get_Dxc,
    get_Dxc_vec,
    get_Shmin_grad_Daine_ppg,
    get_Shmin_grad_Daine_ppg_vec
)

# plotangle.py
from .plotangle import plotfrac

from .unit_converter import convert_rop, convert_wob, convert_ecd, convert_torque, convert_flowrate

# Plotter.py
from .Plotter import plot_logs_labels

# We want to define __all__ to control what gets imported with "from package import *"
__all__ = [
    # BoreStab
    "get_optimal", "getVertical", "getAlignedStress", "getRota", "getStens",
    "getStrikeDip", "getEuler", "getOrit", "getSigmaTT", "getHoop", "draw",
    # DrawSP
    "drawSP", "getSP",
    # failure_criteria
    "mod_lad_cmw", "mod_lad_cmw2", "mogi_failure", "mohr_failure", "lade_failure",
    "mogi", "lade", "zhang_sanding_cwf", "willson_sanding_cwf", "plot_sanding",

    # geomechanics
    "plotPPzhang",

    # hydraulics
    "getColumnHeights", "getPPfromTop", "getPPfromTopRecursive", "getPPfromCentroidRecursive",
    "getGasDensity", "getHydrostaticPsi", "compute_optimal_offset", "compute_optimal_gradient",
    # obgppshmin
    "get_OBG_pascals_vec", "get_PPgrad_Zhang_gcc", "get_PP_grad_Zhang_gcc_vec",
    "get_PPgrad_Eaton_gcc", "get_PPgrad_Eaton_gcc_vec", "get_PPgrad_Dxc_gcc",
    "get_PPgrad_Dxc_gcc_vec", "get_Dxc", "get_Dxc_vec", "get_Shmin_grad_Daine_ppg",
    "get_Shmin_grad_Daine_ppg_vec",
    # plotangle
    "plotfrac",
    # Plotter
    "plot_logs_labels"
]
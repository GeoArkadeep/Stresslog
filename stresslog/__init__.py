"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

# BoreStab.py
from .BoreStab import (
    get_principal_stress,
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

from .syntheticLogs import create_random_las, getwelldev, create_random_well, create_header, get_las_from_dlis, get_well_from_dlis, get_dlis_data, get_dlis_header

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

from .forecast import convert_df_tvd, get_analog

# geomechanics.py
from .geomechanics import compute_geomech, remove_curves, add_curves, find_TVD, find_nearest_depth

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

# forecast.py
from .forecast import get_analog, convert_df_tvd, create_blank_well

# plotangle.py
from .plotangle import plotfrac

from .unit_converter import convert_rop, convert_wob, convert_ecd, convert_torque, convert_flowrate

# Plotter.py
from .Plotter import plot_logs_labels

# We want to define __all__ to control what gets imported with "from package import *"
__all__ = [
    # BoreStab
    "get_principal_stress", "getVertical", "getAlignedStress", "getRota", "getStens",
    "getStrikeDip", "getEuler", "getOrit", "getSigmaTT", "getHoop", "draw",
    # DrawSP
    "drawSP", "getSP",
    # failure_criteria
    "mod_lad_cmw", "mod_lad_cmw2", "mogi_failure", "mohr_failure", "lade_failure",
    "mogi", "lade", "zhang_sanding_cwf", "willson_sanding_cwf", "plot_sanding",

    # geomechanics
    "compute_geomech",

    # hydraulics
    "getColumnHeights", "getPPfromTop", "getPPfromTopRecursive",
    "getGasDensity", "getHydrostaticPsi", "compute_optimal_offset", "compute_optimal_gradient",
    # obgppshmin
    "get_OBG_pascals_vec", "get_PPgrad_Zhang_gcc", "get_PP_grad_Zhang_gcc_vec",
    "get_PPgrad_Eaton_gcc", "get_PPgrad_Eaton_gcc_vec", "get_PPgrad_Dxc_gcc",
    "get_PPgrad_Dxc_gcc_vec", "get_Dxc", "get_Dxc_vec", "get_Shmin_grad_Daine_ppg",
    "get_Shmin_grad_Daine_ppg_vec",
    # plotangle
    "plotfrac",
    # Misc.
    "getwelldev","create_random_well","find_TVD","add_curves","remove_curves","create_header","datasets_to_las","get_las_from_dlis","get_dlis_data", "get_dlis_header",
    # forecast
    "get_analog", "convert_df_tvd",
    # Plotter
    "plot_logs_labels"
]
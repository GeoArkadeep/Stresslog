"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""
import warnings
warnings.filterwarnings("ignore")

import pint
import os
import numpy as np
import scipy
import matplotlib

from matplotlib import pyplot as plt
import pandas as pd
import lasio as laua
import json
from welly import Curve
from welly import Well
from collections import defaultdict
import io

import warnings
warnings.filterwarnings("ignore")

user_home = os.path.expanduser('~/Documents')
app_data = os.getenv('APPDATA')
up = ['psi', 'Ksc', 'Bar', 'Atm', 'MPa']
us = ['MPa', 'psi', 'Ksc', 'Bar', 'Atm']
ug = ['gcc', 'sg', 'ppg', 'psi/foot']
ul = ['metres', 'feet']
ut = ['degC', 'degF', 'degR', 'degK']
uregdef = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
uregdef.define('ppg = 0.051948 psi/foot')
uregdef.define('sg = 0.4335 psi/foot = gcc = SG = GCC')
uregdef.define(
    'ksc = 1.0000005979/0.0703069999987293 psi = KSC = KSc = KsC = ksC = Ksc')
uregdef.define('HR = hour')
uregdef.define('M = meter')
uregdef.define('mpa = MPa = Mpa')
unitchoicedef = [0, 0, 0, 0, 0]

def get_path_dict(user_home):
    path_dict = {}
    path_dict['output_dir'] = os.path.join(user_home, 'Stresslog_Plots')
    path_dict['output_dir1'] = os.path.join(user_home, 'Stresslog_Data')
    path_dict['input_dir'] = os.path.join(user_home, 'Stresslog_Config')
    path_dict['motor_dir'] = os.path.join(user_home, 'Mud_Motor')
    path_dict.update({'plot_figure': os.path.join(path_dict['output_dir'],
        'PlotFigure.png'), 'plot_stability': os.path.join(path_dict[
        'output_dir'], 'PlotStability.png'), 'plot_polygon': os.path.join(
        path_dict['output_dir'], 'PlotPolygon.png'), 'plot_vec': os.path.join(
        path_dict['output_dir'], 'PlotVec.png'), 'plot_bhi': os.path.join(
        path_dict['output_dir'], 'PlotBHI.png'), 'plot_hoop': os.path.join(
        path_dict['output_dir'], 'PlotHoop.png'), 'plot_frac': os.path.join(
        path_dict['output_dir'], 'PlotFrac.png'), 'plot_all': os.path.join(
        path_dict['output_dir'], 'PlotAll.png'), 'output_csv': os.path.join(
        path_dict['output_dir1'], 'output.csv'), 'output_forms': os.path.join(
        path_dict['output_dir1'], 'tempForms.csv'), 'output_ucs': os.path.join(
        path_dict['output_dir1'], 'tempUCS.csv'), 'output_las': os.path.join(
        path_dict['output_dir1'], 'output.las'), 'model_path': os.path.join(
        path_dict['input_dir'], 'model.csv'), 'alias_path': os.path.join(
        path_dict['input_dir'], 'alias.txt'), 'unit_path': os.path.join(
        path_dict['input_dir'], 'units.txt'), 'styles_path': os.path.join(
        path_dict['input_dir'], 'styles.txt'), 'pstyles_path': os.path.join(
        path_dict['input_dir'], 'pstyles.txt'), 'motor_db_path': os.path.join(
        path_dict['motor_dir'], 'motor_db.json')})
    return path_dict

from io import StringIO


def safe_block(curve, cutoffs=None, values=(0, 1)):
    """
    A safer wrapper around the welly Curve.block() function that handles edge cases.
    
    Args:
        curve: welly Curve object
        cutoffs: the values at which to create the blocks
        values: the values to map to (must be one more than cutoffs)
        
    Returns:
        numpy array of blocked values
    """
    import numpy as np
    from welly import utils
    data = np.digitize(curve.df.values, [cutoffs] if np.isscalar(cutoffs) else
        cutoffs)
    data = data.astype(float)
    tops, vals = utils.find_edges(data)
    if len(tops) == 0:
        return np.full_like(data, values[0])
    if len(tops) == 1:
        val_idx = int(vals[0])
        return np.full_like(data, values[val_idx])
    result = np.copy(data)
    for i in range(len(tops)):
        val_idx = int(vals[i])
        if i == len(tops) - 1:
            result[tops[i]:] = values[val_idx]
        else:
            result[tops[i]:tops[i + 1]] = values[val_idx]
    return result


def plot_to_svg(matplot) ->str:
    """
    Saves the last plot made using ``matplotlib.pyplot`` to a SVG string.
    
    Returns:
        The corresponding SVG string.
    """
    s = StringIO()
    matplot.savefig(s, format='svg')
    matplot.close()
    return s.getvalue()


def polynomial(x, *coeffs):
    return sum(c * x ** i for i, c in enumerate(coeffs))


def interpolate_curves(coeffs_list, flow_rate, flow_rates):
    if not isinstance(coeffs_list[0], (list, tuple, np.ndarray)):
        coeffs_list = [coeffs_list]
    num_coeffs = len(coeffs_list[0])
    interpolated_coeffs = []
    for i in range(num_coeffs):
        coeffs = [coeffs[i] for coeffs in coeffs_list]
        interpolator = interp1d(flow_rates, coeffs, kind='linear',
            fill_value='extrapolate')
        interpolated_coeffs.append(interpolator(flow_rate))
    return interpolated_coeffs


def calculate_curve(coeffs, x_values):
    return polynomial(np.array(x_values).astype(float), *coeffs)


def interpolate_values(flow_rate, coeffs_list, flow_rates, x_values):
    interpolated_coeffs = interpolate_curves(coeffs_list, float(flow_rate),
        flow_rates)
    return calculate_curve(interpolated_coeffs, np.array(x_values, dtype=float)
        )


def get_interpolated_value(x, flow_rate, coeffs_list, flow_rates):
    print(f'x: {x}')
    print(f'flow_rate: {flow_rate}')
    print(f'coeffs_list type: {type(coeffs_list)}')
    print(f'coeffs_list: {coeffs_list}')
    print(f'flow_rates: {flow_rates}')
    interpolated = interpolate_curves(coeffs_list, flow_rate, flow_rates)
    print(f'interpolated: {interpolated}')
    return calculate_curve(interpolated, x)


from scipy.interpolate import interp1d


def calculate_motor_rpms(torque_array, flow_array, motor_id_array, dbpath):
    try:
        with open(dbpath, 'r') as f:
            json_data = json.load(f)
        rpm_array = np.zeros_like(torque_array)
        for i, (torque, flow, motor_id) in enumerate(zip(torque_array,
            flow_array, motor_id_array)):
            try:
                motor_data = json_data[motor_id]
                coeffs_list = [[float(coeff) for coeff in coeffs] for
                    coeffs in motor_data['coeffs_list']]
                flow_list = [float(flow) for flow in motor_data['flow_list']]
                em = float(motor_data['em'])
                diff_psi = em * torque
                interp_funcs = []
                for j in range(len(coeffs_list[0])):
                    coeffs = [coeffs[j] for coeffs in coeffs_list]
                    interp_funcs.append(interp1d(flow_list, coeffs, kind=
                        'linear', fill_value='extrapolate'))
                interpolated_coeffs = [func(flow) for func in interp_funcs]
                rpm_array[i] = polynomial(diff_psi, *interpolated_coeffs)
            except:
                pass
        return np.round(rpm_array, 2)
    except Exception as e:
        #print(f'An error occurred: {str(e)}')
        return np.zeros(len(torque_array))


def assign_motor_ids(md, motor_list):
    dtype = [('id', 'U20'), ('threshold', float)]
    motor_array = np.array([(str(m[0]), float(m[1])) for m in motor_list],
        dtype=dtype)
    motor_array.sort(order='threshold')
    motorids = np.full_like(md, motor_array['id'][0], dtype='U20')
    for i in range(1, len(motor_array)):
        motorids[md >= motor_array['threshold'][i - 1]] = motor_array['id'][i]
    return motorids


def getNu(well, nun, alias):
    import math
    header = well._get_curve_mnemonics()
    alias['gr'] = [elem for elem in header if elem in set(alias['gr'])]
    alias['sonic'] = [elem for elem in header if elem in set(alias['sonic'])]
    alias['shearsonic'] = [elem for elem in header if elem in set(alias[
        'shearsonic'])]
    alias['resdeep'] = [elem for elem in header if elem in set(alias[
        'resdeep'])]
    alias['resshal'] = [elem for elem in header if elem in set(alias[
        'resshal'])]
    alias['density'] = [elem for elem in header if elem in set(alias[
        'density'])]
    alias['neutron'] = [elem for elem in header if elem in set(alias[
        'neutron'])]
    vp = 1 / well.data[alias['sonic'][0]].values
    vs = 1 / well.data[alias['shearsonic'][0]].values
    vpvs = vp / vs
    nu = (vpvs ** 2 - 2) / (2 * vpvs ** 2 - 2)
    nu = [(x if not math.isnan(x) else nun) for x in nu]
    nu = [(x if not math.isnan(x) else nun) for x in nu]
    nu = [(x if not (x == float('inf') or x == float('-inf')) else nun) for
        x in nu]
    #print('nu: ', nu)
    return nu


def read_aliases_from_file(file_path, writeConfig=True):
    import json
    try:
        with open(file_path, 'r') as file:
            aliases = json.load(file)
        return aliases
    except:
        aliases = {'sonic': ['none', 'DTC', 'DT24', 'DTCO', 'DT', 'AC',
            'AAC', 'DTHM'], 'shearsonic': ['none', 'DTSM', 'DTS'], 'gr': [
            'none', 'GR', 'GRD', 'CGR', 'GRR', 'GRCFM'], 'resdeep': ['none',
            'HDRS', 'LLD', 'M2RX', 'MLR4C', 'RD', 'RT90', 'RLA1', 'RDEP',
            'RLLD', 'RILD', 'ILD', 'RT_HRLT', 'RACELM'], 'resshal': ['none',
            'LLS', 'HMRS', 'M2R1', 'RS', 'RFOC', 'ILM', 'RSFL', 'RMED',
            'RACEHM'], 'density': ['none', 'ZDEN', 'RHOB', 'RHOZ', 'RHO',
            'DEN', 'RHO8', 'BDCFM'], 'neutron': ['none', 'CNCF', 'NPHI',
            'NEU'], 'pe': ['none', 'PE'], 'ROP': ['none', 'ROPAVG'], 'RPM':
            ['none', 'SURFRPM'], 'WOB': ['none', 'WOBAVG'], 'ECD': ['none',
            'ACTECDM'], 'BIT': ['none', 'BIT'], 'TORQUE': ['none', 'TORQUE',
            'TORQUEAVG'], 'FLOWRATE': ['none', 'FLOWRATE', 'FLOWIN']}
        if not writeConfig:
            return aliases
        try:
            with open(file_path, 'w') as file:
                json.dump(aliases, file, indent=4)
        except:
            pass
        return aliases


def read_styles_from_file(minpressure, maxchartpressure, pressure_units,
    strength_units, gradient_units, ureg, file_path, debug=False, writeConfig=True):

    def convert_value(value, from_unit, to_unit, ureg=ureg):
        return (value * ureg(from_unit.lower())).to(to_unit.lower()).magnitude
    try:
        with open(file_path, 'r') as file:
            styles = json.load(file)
        print('Reading Styles file from ', file_path) if debug else None
    except:
        print('Using default Styles, file read failed.') if debug else None
        styles = {'lresnormal': {'color': 'red', 'linewidth': 1.5, 'style':
            '--', 'track': 1, 'left': -3, 'right': 1, 'type': 'linear',
            'unit': 'ohm.m'}, 'lresdeep': {'color': 'black', 'linewidth': 
            0.5, 'style': '-', 'track': 1, 'left': -3, 'right': 1, 'type':
            'linear', 'unit': 'ohm.m'}, 'Dexp': {'color': 'brown',
            'linewidth': 0.5, 'style': '-', 'track': 1, 'left': 0, 'right':
            2, 'type': 'linear', 'unit': ''}, 'dexnormal': {'color':
            'brown', 'linewidth': 1.5, 'style': '-.', 'track': 1, 'left': 0,
            'right': 2, 'type': 'linear', 'unit': ''}, 'dalm': {'color':
            'green', 'linewidth': 1, 'style': '-', 'track': 1, 'left': 
            300, 'right': 50, 'type': 'linear', 'unit': 'us/ft'},
            'dtNormal': {'color': 'green', 'linewidth': 1.5, 'style': '--',
            'track': 1, 'left': 300, 'right': 50, 'type': 'linear', 'unit':
            'us/ft'}, 'mudweight': {'color': 'brown', 'linewidth': 1.5,
            'style': '-', 'track': 2, 'left': 0, 'right': 3, 'type':
            'linear', 'unit': 'gcc'},'shmin_grad': {'color': 'dodgerblue', 'linewidth': 
            1, 'style': '-', 'track': 2, 'left': 0, 'right': 3, 'type':
            'linear', 'unit': 'gcc'}, 'fg': {'color': 'blue', 'linewidth': 
            1.5, 'style': '-', 'track': 2, 'left': 0, 'right': 3, 'type':
            'linear', 'unit': 'gcc'}, 'pp': {'color': 'red', 'linewidth': 
            1.5, 'style': '-', 'track': 2, 'left': 0, 'right': 3, 'type':
            'linear', 'unit': 'gcc'}, 'sfg': {'color': 'olive', 'linewidth':
            1.5, 'style': '-', 'track': 2, 'left': 0, 'right': 3, 'type':
            'linear', 'unit': 'gcc'}, 'obgcc': {'color': 'lime',
            'linewidth': 1.5, 'style': '-', 'track': 2, 'left': 0, 'right':
            3, 'type': 'linear', 'unit': 'gcc'}, 'fgpsi': {'color': 'blue',
            'linewidth': 1.5, 'style': '-', 'track': 3, 'left': minpressure,
            'right': maxchartpressure, 'type': 'linear', 'unit': 'psi'},
            'ssgHMpsi': {'color': 'pink', 'linewidth': 1.5, 'style': '-',
            'track': 3, 'left': minpressure, 'right': maxchartpressure,
            'type': 'linear', 'unit': 'psi'}, 'obgpsi': {'color': 'green',
            'linewidth': 1.5, 'style': '-', 'track': 3, 'left': minpressure,
            'right': maxchartpressure, 'type': 'linear', 'unit': 'psi'},
            'hydropsi': {'color': 'aqua', 'linewidth': 1.5, 'style': '-',
            'track': 3, 'left': minpressure, 'right': maxchartpressure,
            'type': 'linear', 'unit': 'psi'}, 'pppsi': {'color': 'red',
            'linewidth': 1.5, 'style': '-', 'track': 3, 'left': minpressure,
            'right': maxchartpressure, 'type': 'linear', 'unit': 'psi'},
            'mudpsi': {'color': 'brown', 'linewidth': 1.5, 'style': '-',
            'track': 3, 'left': minpressure, 'right': maxchartpressure,
            'type': 'linear', 'unit': 'psi'}, 'sgHMpsiL': {'color': 'lime',
            'linewidth': 0.25, 'style': '-', 'track': 3, 'left':
            minpressure, 'right': maxchartpressure, 'type': 'linear',
            'unit': 'psi'}, 'sgHMpsiU': {'color': 'orange', 'linewidth': 
            0.25, 'style': '-', 'track': 3, 'left': minpressure, 'right':
            maxchartpressure, 'type': 'linear', 'unit': 'psi'}, 'slal': {
            'color': 'blue', 'linewidth': 1.5, 'style': '-', 'track': 4,
            'left': 0, 'right': 100, 'type': 'linear', 'unit': 'MPa'},
            'ucs_horsrud': {'color': 'red', 'linewidth': 1.5, 'style': '-',
            'track': 4, 'left': 0, 'right': 100, 'type': 'linear', 'unit':
            'MPa'}, 'GR': {'color': 'green', 'linewidth': 0.25, 'style':
            '-', 'track': 0, 'left': 0, 'right': 150, 'type': 'linear',
            'unit': 'gAPI', 'fill': 'none', 'fill_between': {'reference':
            'GR_CUTOFF', 'colors': ['green', 'yellow'], 'colorlog': 'obgcc',
            'cutoffs': [1.8, 2.67, 2.75], 'cmap': 'Set1_r'}}, 'GR_CUTOFF':
            {'color': 'black', 'linewidth': 0, 'style': '-', 'track': 0,
            'left': 0, 'right': 150, 'type': 'linear', 'unit': 'gAPI'}}
    for key, value in styles.items():
        if value['track'] == 2:
            if value['unit'] != gradient_units:
                value['left'] = round(convert_value(value['left'], value[
                    'unit'], gradient_units), 1)
                value['right'] = round(convert_value(value['right'], value[
                    'unit'], gradient_units), 1)
                value['unit'] = gradient_units
        elif value['track'] == 3:
            if value['unit'] != pressure_units:
                value['unit'] = pressure_units
            value['left'] = round(convert_value(minpressure, 'psi',
                pressure_units))
            value['right'] = round(convert_value(maxchartpressure, 'psi',
                pressure_units))
        elif value['track'] == 4:
            if value['unit'] != strength_units:
                value['left'] = round(convert_value(value['left'], value[
                    'unit'], strength_units))
                value['right'] = round(convert_value(value['right'], value[
                    'unit'], strength_units))
                value['unit'] = strength_units
    if not writeConfig:
        return styles
    with open(file_path, 'w') as file:
        json.dump(styles, file, indent=4)
    return styles


def read_pstyles_from_file(minpressure, maxchartpressure, pressure_units,
    strength_units, gradient_units, ureg, file_path, writeConfig=True):

    def convert_value(value, from_unit, to_unit, ureg=ureg):
        return (value * ureg(from_unit.lower())).to(to_unit.lower()).magnitude
    try:
        with open(file_path, 'r') as file:
            pstyles = json.load(file)
    except:
        pstyles = {'frac_grad': {'color': 'black', 'pointsize': 100,
            'symbol': 4, 'track': 2, 'left': 0, 'right': 3, 'type':
            'linear', 'unit': 'gcc'}, 'flow_grad': {'color': 'orange',
            'pointsize': 100, 'symbol': 5, 'track': 2, 'left': 0, 'right': 
            3, 'type': 'linear', 'unit': 'gcc'}, 'frac_psi': {'color':
            'dodgerblue', 'pointsize': 100, 'symbol': 4, 'track': 3, 'left':
            minpressure, 'right': maxchartpressure, 'type': 'linear',
            'unit': 'psi'}, 'flow_psi': {'color': 'orange', 'pointsize': 
            100, 'symbol': 5, 'track': 3, 'left': minpressure, 'right':
            maxchartpressure, 'type': 'linear', 'unit': 'psi'}, 'ucs': {
            'color': 'lime', 'pointsize': 30, 'symbol': 'o', 'track': 4,
            'left': 0, 'right': 100, 'type': 'linear', 'unit': 'MPa'}}
    for key, value in pstyles.items():
        if value['track'] == 2:
            if value['unit'] != gradient_units:
                value['left'] = round(convert_value(value['left'], value[
                    'unit'], gradient_units), 1)
                value['right'] = round(convert_value(value['right'], value[
                    'unit'], gradient_units), 1)
                value['unit'] = gradient_units
        elif value['track'] == 3:
            if value['unit'] != pressure_units:
                value['unit'] = pressure_units
            value['left'] = round(convert_value(minpressure, 'psi',
                pressure_units))
            value['right'] = round(convert_value(maxchartpressure, 'psi',
                pressure_units))
        elif value['track'] == 4:
            if value['unit'] != strength_units:
                value['left'] = round(convert_value(value['left'], value[
                    'unit'], strength_units))
                value['right'] = round(convert_value(value['right'], value[
                    'unit'], strength_units))
                value['unit'] = strength_units
    if not writeConfig:
        return pstyles
    with open(file_path, 'w') as file:
        json.dump(pstyles, file, indent=4)
    return pstyles


def pad_val(array_like, value):
    array = array_like.copy()
    nans = np.isnan(array)

    def get_x(a):
        return a.nonzero()[0]
    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])
    return array


def find_nearest_depth(array, value):
    import math
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) <
        math.fabs(value - array[idx])):
        return [idx - 1, array[idx - 1]]
    else:
        return [idx, array[idx]]


def interpolate_nan(array_like):
    array = array_like.copy()
    nans = np.isnan(array)

    def get_x(a):
        return a.nonzero()[0]
    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])
    return array


from scipy.signal import medfilt


def median_filter_downsampler(curve, window_size=21):
    if window_size == 1:
        return curve
    if len(curve.values) < window_size:
        window_size = len(curve.values)
    shape = curve.values.size - window_size + 1, window_size
    strides = curve.values.strides[0], curve.values.strides[0]
    sliding_windows = np.lib.stride_tricks.as_strided(curve.values, shape=
        shape, strides=strides)
    filtered_data = np.nanmedian(sliding_windows, axis=1)
    filtered_curve = Curve(mnemonic=curve.mnemonic, units=curve.units, data
        =filtered_data, basis=curve.basis[window_size // 2:-(window_size // 2)]
        )
    resampled_curve = filtered_curve.to_basis(step=curve.step * window_size)
    return resampled_curve


def average_downsampler(curve, window_size=21):
    if window_size == 1:
        return curve
    if len(curve.values) < window_size:
        window_size = len(curve.values)
    shape = curve.values.size - window_size + 1, window_size
    strides = curve.values.strides[0], curve.values.strides[0]
    sliding_windows = np.lib.stride_tricks.as_strided(curve.values, shape=
        shape, strides=strides)
    filtered_data = np.nanmean(sliding_windows, axis=1)
    filtered_curve = Curve(mnemonic=curve.mnemonic, units=curve.units, data
        =filtered_data, basis=curve.basis[window_size // 2:-(window_size // 2)]
        )
    resampled_curve = filtered_curve.to_basis(step=curve.step * window_size)
    return resampled_curve


def generate_weights(window_size, window_type='v_shape'):
    if window_type == 'v_shape':
        weights = np.abs(np.arange(window_size) - (window_size - 1) / 2)
        weights = 1 - weights / weights.max()
    elif window_type == 'hanning':
        weights = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / (
            window_size - 1))
    elif window_type == 'hamming':
        weights = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(window_size) /
            (window_size - 1))
    else:
        raise ValueError(
            "Invalid window_type. Choose from 'v_shape', 'hanning', 'hamming'."
            )
    return weights


def weighted_average_downsampler(curve, window_size=21, window_type='v_shape'):
    weights = generate_weights(window_size, window_type)
    if len(curve.values) < window_size:
        window_size = len(curve.values)
        weights = weights[:window_size]
    shape = curve.values.size - window_size + 1, window_size
    strides = curve.values.strides[0], curve.values.strides[0]
    sliding_windows = np.lib.stride_tricks.as_strided(curve.values, shape=
        shape, strides=strides)
    normalized_weights = weights / np.nansum(weights)
    filtered_data = np.nansum(sliding_windows * normalized_weights, axis=1)
    filtered_curve = Curve(mnemonic=curve.mnemonic, units=curve.units, data
        =filtered_data, basis=curve.basis[window_size // 2:-(window_size // 2)]
        )
    resampled_curve = filtered_curve.to_basis(step=curve.step * window_size)
    return resampled_curve


from scipy.interpolate import CubicSpline


def find_TVD(well, md):
    df = well.df()
    md_values = df['MD'].values
    tvdm_values = df['TVDM'].values
    spline = CubicSpline(md_values, tvdm_values, extrapolate=True)
    tvd = spline(md)
    return float(tvd)


def add_curves(well, df, clear=False):
    """
    Adds all columns from a DataFrame as curves to the Well object while preserving start, stop, and step values.
    
    Parameters:
    well (Well): The welly Well object.
    df (pd.DataFrame): The DataFrame containing the data to be added as curves.
    clear (bool): If True, all existing columns are cleared before adding new ones. Defaults to False
    """
    indexcurve = df['DEPT']
    if clear:
        well.data.clear()
        print('All existing curves have been removed.')
    for column in df.columns:
        if column in well.data:
            print(
                f"Curve with mnemonic '{column}' already exists in the well. Skipping."
                )
        else:
            curve = Curve(mnemonic=column, data=df[column].values, index=
                indexcurve, null=np.nan)
            well.data[column] = curve
            print(f'Added curve: {column}')
    return well


def remove_curves(well, mnemonics_to_remove, debug=False):
    """
    Removes curves with the specified mnemonics from the well object.
    
    Parameters:
    well (Well): The welly Well object.
    mnemonics_to_remove (list): A list of mnemonics to be removed.
    """
    for mnemonic in mnemonics_to_remove:
        keys_to_delete = [key for key, curve in well.data.items() if curve.
            mnemonic == mnemonic]
        for key in keys_to_delete:
            del well.data[key]
            print(f'Removed curve: {mnemonic}') if debug else None
    return well


mnemonics_to_remove = ['ResD', 'ShaleFlag', 'RHO', 'OBG_AMOCO', 'DTCT',
    'PP_GRADIENT', 'SHmin_DAINES', 'SHmin_ZOBACK', 'FracGrad',
    'FracPressure', 'GEOPRESSURE', 'SHmin_PRESSURE', 'SHmax_PRESSURE',
    'MUD_PRESSURE', 'OVERBURDEN_PRESSURE', 'HYDROSTATIC_PRESSURE',
    'MUD_GRADIENT', 'S0_Lal', 'S0_Lal_Phi', 'UCS_horsrud', 'UCS_Lal',
    'Poisson_Ratio', 'ML90', 'Youngs_Modulus', 'Shear_Modulus', 'Bulk_Modulus']
unitdictdef = {'pressure': 'psi', 'strength': 'MPa', 'gradient': 'gcc',
    'length': 'm'}


def compute_geomech(well, rhoappg=16.33, lamb=0.0008, ul_exp=0.0008,
    ul_depth=0, a=0.63, nu=0.25, mu = 0.65, sfs=1.0, window=1, plotstart=0, plotend=6000,
    dtml=210, dtmt=60, water=1.0, underbalancereject=1, tecb=0, doi=0,
    offset=0, dip_dir=0, dip=0, mudtemp=0, res0=0.98, be=0.00014, ne=0.6,
    dex0=0.5, de=0.00017, nde=0.5, lala=-1.0, lalb=1.0, lalm=5, lale=0.5,
    lall=5, horsruda=0.77, horsrude=2.93, mabw=90, unitchoice=unitchoicedef,
    ureg=uregdef, mwvalues=[[1.0, 0.0, 0.0, 0.0, 0.0, 0]], flowgradvals=[[0,
    0]], fracgradvals=[[0, 0]], flowpsivals=[[0, 0]], fracpsivals=[[0, 0]],
    attrib=[1, 0, 0, 0, 0, 0, 0, 0], flags=None, UCSs=None, forms=None,
    lithos=None, user_home=user_home, program_option=[300,
    4, 0, 0, 0], writeFile=False, aliasdict=None, unitdict=unitdictdef,
    debug=False, penetration=False, ten_fac = 10, ehmin = None, ehmax= None, writeConfig=True, display=False):
    """
    Performs geomechanical calculations, data processing, and pore pressure estimation based on 
    well log data and additional user inputs.

    Parameters
    ----------
    well : welly.Well
        The well data containing curves for various parameters. It is essential that the curves 
        extend all the way to 0 depth and contain deviation data (even if the well is vertical).
    rhoappg : float, optional
        Density at mudline in g/cc (default is 16.33).
    lamb : float, optional
        Compaction exponent for regions without unloading (default is 0.0008).
    ul_exp : float, optional
        Compaction exponent for regions with unloading (default is 0.0008).
    ul_depth : float, optional
        Depth where unloading starts, in metres (default is 0).
    a : float, optional
        Density compaction exponent for overburden calculations (default is 0.630).
    nu : float, optional
        Poisson's ratio, used in stress calculations (default is 0.25).
    mu : float, optional
        Coefficient of sliding friction, used in shmin (zoback) calculation (default is 0.65).
    sfs : float, optional
        Shale flag resistivity or GR cutoff, representing the difference between deep and shallow 
        resistivity in ohm.m (default is 1.0).
    window : int, optional
        The window size for down-sampling well data (default is 1).
    plotstart : int, optional
        Starting depth for plotted image (if any), in metres (default is 0).
    plotend : int, optional
        End depth for plotted image (if any), in metres (default is 2000).
    dtml : int, optional
        Delta T at mudline, in microseconds per foot (default is 210).
    dtmt : int, optional
        Delta T of matrix, in microseconds per foot (default is 60).
    res0 : float, optional
        Resistivity at mudline (default is 0.98).
    be : float, optional
        Base exponential coefficient for resistivity gradient calculation (default is 0.00014).
    ne : float, optional
        Exponent for normal resistivity calculations (default is 0.6).
    dex0 : float, optional
        D-exp at mudline (default is 0.5).
    de : float, optional
        Coefficient for drilling exponent gradient calculation (default is 0.00014).
    nde : float, optional
        Exponent for normal drilling exponent calculations (default is 0.5).
    paths : dict, optional
        Dictionary containing paths for saving output files, including plots, CSVs, and models. Keys 
        typically include `output_dir`, `plot_figure`, and others for structured saving (default is None).
    water : float, optional
        Water density in g/cc (default is 1.0).
    underbalancereject : int, optional
        Minimum pore pressure gradient (in g/cc) below which underbalanced data is rejected (default is 1).
    tecb : int, optional
        Daines' parameter related to tectonic stress, used to calculate shmin (default is 0).
    doi : int, optional
        Depth of interest for detailed calculations, in metres (default is 0).
    offset : int, optional
        Azimuth of the maximum horizontal stress (SHMax) in degrees (default is 0).
    dip_dir : int, optional
        Dip direction of the stress tensor in degrees (default is 0).
    dip : int, optional
        Dip angle of the stress tensor in degrees (default is 0).
    mudtemp : int, optional
        Mud temperature, in degrees Celsius (default is 0).
    lala : float, optional
        Parameter for Lal's cohesion method (default is -1.0).
    lalb : float, optional
        Parameter for Lal's cohesion method (default is 1.0).
    lalm : int, optional
        Parameter for Lal's cohesion method (default is 5).
    lale : float, optional
        Parameter for Lal's cohesion method (default is 0.5).
    lall : int, optional
        Parameter for Lal's cohesion method (default is 5).
    mabw : float, optional
        Maximum Allowable Breakout Width in degrees (default is 90).
    horsruda : float, optional
        Parameter for horsrud's stress method (default is 0.77).
    horsrude : float, optional
        Parameter for horsrud's stress method (default is 2.93).
    unitchoice : list, optional
        <DEPRECATED, will be removed in a future version> List specifying the unit system for file output plots (default is [0, 0, 0, 0, 0]).
    unitdict : dict, optional
        Unit dictionary to be used for unit conversion, default is {'pressure':'psi', 'strength':'MPa', 'gradient':'gcc', 'length':'m'}.
    ureg : pint.UnitRegistry, optional
        Unit registry for unit conversions (default is a pint.UnitRegistry with 
        `autoconvert_offset_to_baseunit=True`).
    mwvalues : list of lists, optional
        Section attributes, including parameters like maximum ECD, casing shoe depth, casing diameter, 
        bit diameter, mud salinity, and bottom-hole temperature (BHT) at the shoe
    flowgradvals : list of lists, optional
        Flow gradient values for different depths (i.e. [emw in g/cc at which kick taken/RFT/DST/whathaveyou, MD]) (default is [[0, 0]]).
    fracgradvals : list of lists, optional
        Fracture gradient values for different depths (i.e. [emw in g/cc at which mud lost/(x)LOT/minifrac/whathaveyou, MD]) (default is [[0, 0]]).
    flowpsivals : list of lists, optional
        Flow pressure values for different depths (i.e. [bhp in psi at which kick taken/RFT/DST/whathaveyou, MD]) (default is [[0, 0]]).
    fracpsivals : list of lists, optional
        Fracture pressure values for different depths (i.e. [bhp in psi at which mud lost/(x)LOT/minifrac/whathaveyou, MD]) (default is [[0, 0]]).
    attrib : list, optional
        Well attributes list. The fields correspond to KB, GL/WD, WL, Latitude, Longitude, BHT, Mud Resistance, Mud filtrate Resistance (Note: the water level parameter will be implemented fully in the future)
        (default is [1, 0, 0, 0, 0, 0, 0, 0]).
    flags : pandas.DataFrame, optional
        Dataframe containing depths and conditions identified from image logs, such as 
        breakouts or drilling-induced fractures (default is None).
    UCSs : pandas.DataFrame, optional
        Dataframe containing measured depth (MD) and unconfined compressive strength (UCS) values 
        (default is None).
    forms : pandas.DataFrame, optional
        Dataframe containing formation tops and associated formation-specific parameters (default is None).
    lithos : pandas.DataFrame, optional
        Dataframe containing interpreted lithology data and lithology-specific parameters 
        (default is None).
    user_home : str, optional
        Path to the root of output directories (default is `~/Documents`).
    program_option : list, optional
        List controlling algorithm behavior. The entries are: dpi (of the saved plots), choice of pore pressure algorithm (0 is sonic, 1 is resistivity, 2 is dexp, 4-9 are best available with the priorities changing), choice of shmin algorithm (0 is daines, 1 is zoback), the last two parameters are reserved for future use.
        (default is [300, 4, 0, 0, 0]).
    writeFile : bool, optional
        Whether to write results to files in the specified paths (default is True).
    aliasdict : dict, optional
        Dictionary mapping curve mnemonics to standardized aliases (default is None).
    ten_fac : float, optional
        Parameter defining scaling factor from compressive to tensile strength (default 10) can be over-ridden from the lithology input.
    ehmin : float, optional
        Strain in direction of minimum horizontal stress, in absolute or microstrains (default None) if provided overrides the tecb parameter in shmin calculation.
    ehmax : float, optional
        Strain in direction of maximum horizontal stress, in absolute or microstrains (default None) if provided overrides the tecb parameter in shmin calculation.
    writeConfig: bool, optional
        Whether to write config files in the specified paths (default is True)
    display: bool, optional
        Whether to interactively show output during processing (default is False)

    Returns
    -------
    tuple
        The tuple contains the following:
        0 : Well log DataFrame (original and computed values) with mnemonics as headers
        1 : LAS file as StringIO object containing original and computed values
        2 : Base64 encoded plot strings for properties calculated at depth of interest (or None if written to files or not calculated at doi=0)
        3 : Depth of Interest as specified (in meters)
        4 : Welly object containing all data

    Notes
    -----
    - If `writeFile` is True, all generated plots and data will be saved to disk.
    - If `display` is True, the stress polygon, directional stability plot, synthetic borehole image, sanding risk plot and well plot will be shown in addition to being saved to disk. writeFile is assumed to be True internally in this case.
    - The following dataframes must have fixed formats for their respective columns:
      
      - **forms**: Must contain columns in this order:
         'Formation Top Measured Depth', 'Formation Number', 'Formation Name', 'GR Cutoff', 'Structural Top', 'Structural Bottom', 'Centroid Ratio', 'OWC Depth', 'GOC Depth',  'Coefficient of thermal expansion bulk', 'Alpha', 'Beta', 'Gamma', 'Tectonic Factor', 'SH/SV Ratio',  'Biot Coefficient', 'DT Normal', 'Resistivity Normal', 'DEX Normal'
      
      - **lithos**: Must contain columns in this order:
        'Measured Depth', 'Lithology Code', 'Poisson Ratio', 'Friction Coefficient', 'UCS'
      
      - **flags**: Must contain columns in this order:
        ['Measured Depth', 'Condition Code']
        - 'Condition Code' integer, allowed values:
            0 : No Image log exists
            1 : Image log exists, No observations
            2 : DITF observed on image log
            3 : Breakouts observed on image log
            4 : Both DITF and Breakouts observed on image log
        
      - **UCSs**: Must contain columns in this order:
        ['Measured Depth', 'UCS in MPa']
      
      Any deviation in column order, or missing values will result in errors during processing.
    """
    print('Starting Geomech Calculation...') if debug else None
    if display:
        writeFile = True
    
    numodel = [0.35, 0.26, 0.23, 0.25]
    try:
        if ehmin>1:
            ehmin *= 1/1000000
    except:
        pass
    try:
        if ehmax>1:
            ehmax *= 1/1000000
    except:
        pass
    
    paths = get_path_dict(user_home)
    output_dir = paths['output_dir']
    output_dir1 = paths['output_dir1']
    input_dir = paths['input_dir']
    motor_dir = paths['motor_dir']
    if writeFile:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir1, exist_ok=True)
    if writeConfig:
        os.makedirs(input_dir, exist_ok=True)
    output_file = paths['plot_figure']
    output_fileS = paths['plot_stability']
    output_fileSP = paths['plot_polygon']
    output_fileVec = paths['plot_vec']
    output_fileBHI = paths['plot_bhi']
    output_fileHoop = paths['plot_hoop']
    output_fileFrac = paths['plot_frac']
    output_fileAll = paths['plot_all']
    output_file2 = paths['output_csv']
    output_forms = paths['output_forms']
    output_ucs = paths['output_ucs']
    output_file3 = paths['output_las']
    modelpath = paths['model_path']
    aliaspath = paths['alias_path']
    unitpath = paths['unit_path']
    stylespath = paths['styles_path']
    pstylespath = paths['pstyles_path']
    motordbpath = paths['motor_db_path']
    rv1 = rv2 = rv3 = rv4 = rv5 = None
    well = remove_curves(well, mnemonics_to_remove)
    mabw = max(0, min(180, mabw))
    devdata = well.location.deviation
    print('DEVDATA :', devdata) if debug else None
    incdata = devdata[:, 1]
    azmdata = devdata[:, 2]
    md = well.data['MD'].values
    incdata, azmdata = (np.resize(data, len(md)) if len(data) < len(md) else
        data[:len(md)] for data in (incdata, azmdata))
    print(
        f"{'Extended' if len(incdata) < len(md) else 'Truncated'} by {abs(len(md) - len(incdata))} points."
        ) if debug else None
    inclinationi = Curve(incdata, mnemonic='INCL', units='degrees', index=
        md, null=0)
    well.data['INCL'] = inclinationi
    azimuthu = Curve(azmdata, mnemonic='AZIM', units='degrees', index=md,
        null=0)
    well.data['AZIM'] = azimuthu
    for curve_name, curve in well.data.items():
        print(f'Curve: {curve_name}') if debug else None
        if program_option[4] == 0:
            curve = average_downsampler(curve, window)
        else:
            curve = median_filter_downsampler(curve, window)
        well.data[curve_name] = curve
    incdata = well.data['INCL'].values
    azmdata = well.data['AZIM'].values
    md = well.data['MD'].values
    devdata = np.column_stack((md, incdata, azmdata))
    finaldepth = well.df().index[-1]
    if aliasdict is None:
        alias = read_aliases_from_file(aliaspath,writeConfig=writeConfig)
    else:
        alias = aliasdict
    print(alias) if debug else None
    print(well.uwi, well.name) if debug else None
    start_depth = well.df().index[0]
    final_depth = well.df().index[-1]
    plt.clf()
    from .BoreStab import getEuler
    if dip_dir != 0 or dip != 0:
        tilt, tiltgamma = getEuler(offset, dip_dir, dip)
        print('Alpha :', offset, ', Beta: ', tilt, ', Gamma :', tiltgamma
            ) if debug else None
    else:
        tilt = 0
        tiltgamma = 0
    header = well._get_curve_mnemonics()
    print(header) if debug else None
    alias['gr'] = [elem for elem in header if elem in set(alias['gr'])]
    alias['sonic'] = [elem for elem in header if elem in set(alias['sonic'])]
    alias['shearsonic'] = [elem for elem in header if elem in set(alias[
        'shearsonic'])]
    alias['resdeep'] = [elem for elem in header if elem in set(alias[
        'resdeep'])]
    alias['resshal'] = [elem for elem in header if elem in set(alias[
        'resshal'])]
    alias['density'] = [elem for elem in header if elem in set(alias[
        'density'])]
    alias['neutron'] = [elem for elem in header if elem in set(alias[
        'neutron'])]
    alias['ROP'] = [elem for elem in header if elem in set(alias['ROP'])]
    alias['WOB'] = [elem for elem in header if elem in set(alias['WOB'])]
    alias['ECD'] = [elem for elem in header if elem in set(alias['ECD'])]
    alias['RPM'] = [elem for elem in header if elem in set(alias['RPM'])]
    alias['BIT'] = [elem for elem in header if elem in set(alias['BIT'])]
    alias['TORQUE'] = [elem for elem in header if elem in set(alias['TORQUE'])]
    alias['FLOWRATE'] = [elem for elem in header if elem in set(alias[
        'FLOWRATE'])]
    detail = mwvalues
    print(detail) if debug else None
    i = 0
    mud_weight = []
    bht_point = []
    casing_dia = []
    bit_dia = []
    motor_list = []
    while i < len(detail):
        mud_weight.append([detail[i][0], detail[i][1]])
        bht_point.append([detail[i][-1], detail[i][1]])
        casing_dia.append([detail[i][3], detail[i][1]])
        bit_dia.append([detail[i][2], detail[i][1]])
        motor_list.append([detail[i][4], detail[i][1]])
        i += 1
    print(mud_weight) if debug else None
    print('Hole size array:') if debug else None
    bit_dia[-1][1] = final_depth
    print(bit_dia) if debug else None
    print('Mud Motor array:') if debug else None
    motor_list[-1][1] = final_depth
    print(motor_list) if debug else None
    first = [mud_weight[0][0], 0]
    last = [mud_weight[-1][0], final_depth]
    top_bht = [15, 0]
    bottom_bht = [float(attrib[5]), final_depth]
    frac_grad_data = fracgradvals
    flow_grad_data = flowgradvals
    frac_psi_data = fracpsivals
    flow_psi_data = flowpsivals
    mud_weight.insert(0, first)
    mud_weight.append(last)
    bht_point.insert(0, top_bht)
    bht_point.append(bottom_bht)
    print('MudWeights: ', mud_weight) if debug else None
    print('BHTs: ', bht_point) if debug else None
    print(len(bht_point)) if debug else None
    print(alias['sonic']) if debug else None
    if program_option[1] == 0:
        if alias['sonic'][0] == 'none':
            print('No p-sonic velocity log found, no pore pressure calculable'
                ) if debug else None
            if writeFile:
                return None, None
            return None, None, None, None, None
    if program_option[1] == 1:
        if alias['resdeep'][0] == 'none':
            print('No deep resistivity log found, no pore pressure calculable'
                ) if debug else None
            if writeFile:
                return None, None
            return None, None, None, None, None
    vp = 0
    vs = 0
    vpvs = 0
    nu2 = []
    md = well.data['MD'].values
    try:
        dt = well.data[alias['sonic'][0]]
    except:
        dt = Curve(np.full(len(md), np.nan), mnemonic='DTCO', units='uspf',
            index=md, null=-999.25)
    try:
        rdeep = well.data[alias['resdeep'][0]]
    except:
        rdeep = Curve(np.full(len(md), np.nan), mnemonic='ILD', units=
            'ohm.m', index=md, null=-999.25)
    from .unit_converter import convert_rop, convert_wob, convert_ecd, convert_torque, convert_flowrate
    try:
        rop = convert_rop(well.data[alias['ROP'][0]].values, well.data[
            alias['ROP'][0]].units)
        print('ROP units as specified:') if debug else None
        print(well.data[alias['ROP'][0]].units) if debug else None
    except:
        rop = np.full(len(md), np.nan)
    try:
        wob = convert_wob(well.data[alias['WOB'][0]].values, well.data[
            alias['WOB'][0]].units)
        print('WOB units as specified:') if debug else None
        print(well.data[alias['WOB'][0]].units) if debug else None
    except:
        wob = np.full(len(md), np.nan)
    try:
        rpm = well.data[alias['RPM'][0]].values
        print(rpm) if debug else None
        rpm = np.where((rpm < 0) | (rpm > 300), np.nan, rpm)
        print(rpm) if debug else None
        print('RPM units as specified:') if debug else None
        print(well.data[alias['RPM'][0]].units) if debug else None
    except:
        rpm = np.full(len(md), np.nan)
    try:
        ecd = convert_ecd(well.data[alias['ECD'][0]].values, well.data[
            alias['ECD'][0]].units)
        print('ECD units as specified:') if debug else None
        print(well.data[alias['ECD'][0]].units) if debug else None
    except:
        ecd = np.full(len(md), 0)
    try:
        bit = well.data[alias['BIT'][0]]
        print('BIT units as specified:') if debug else None
        print(well.data[alias['BIT'][0]].units) if debug else None
    except:
        bit_dia = np.array(bit_dia)
        indices = np.clip(np.searchsorted(bit_dia[:, 1], md, side='right'),
            0, len(bit_dia) - 1)
        bit = np.take(bit_dia[:, 0], indices)
    try:
        flow = convert_flowrate(well.data[alias['FLOWRATE'][0]].values,
            well.data[alias['FLOWRATE'][0]].units)
        print('Flowrate units as specified:') if debug else None
        print(well.data[alias['FLOWRATE'][0]].units) if debug else None
        flow = np.where((flow < 0) | (flow > 2000), np.nan, flow)
    except:
        flow = np.full(len(md), np.nan)
    try:
        torque = convert_torque(well.data[alias['TORQUE'][0]].values, well.
            data[alias['TORQUE'][0]].units)
        print('Torque units as specified:') if debug else None
        print(well.data[alias['TORQUE'][0]].units) if debug else None
    except:
        torque = np.full(len(md), np.nan)
    motorids = assign_motor_ids(md, motor_list)
    print(motorids) if debug else None
    Mrpm = calculate_motor_rpms(torque, flow, motorids, motordbpath)
    rpm = rpm + Mrpm
    try:
        nu2 = getNu(well, nu, alias)
    except:
        nu2 = [nu] * len(md)
    try:
        zden2 = well.data[alias['density'][0]].values
    except:
        zden2 = np.full(len(md), np.nan)
    try:
        gr = well.data[alias['gr'][0]].values
    except:
        gr = np.full(len(md), np.nan)
    try:
        cald = well.data[alias['cald'][0]].values
    except:
        cald = np.full(len(md), np.nan)
    try:
        cal2 = well.data[alias['cal2'][0]].values
    except:
        cal2 = np.full(len(md), np.nan)
    lradiff = np.full(len(md), np.nan)
    if alias['resshal'] != [] and alias['resdeep'] != []:
        rS = well.data[alias['resshal'][0]].values
        rD = well.data[alias['resdeep'][0]].values
        print(rD) if debug else None
        rdiff = rD[:] - rS[:]
        rdiff[np.isnan(rdiff)] = 0
        radiff = (rdiff[:] * rdiff[:]) ** 0.5
        lradiff = radiff
        rediff = Curve(lradiff, mnemonic='ResD', units='ohm/m', index=md,
            null=0)
        well.data['ResDif'] = rediff
        print('sfs = :', sfs) if debug else None
        shaleflag = safe_block(rediff, cutoffs=sfs, values=(0, 1))
        zoneflag = safe_block(rediff, cutoffs=sfs, values=(0, 1))
        print(shaleflag) if debug else None
    else:
        shaleflag = np.zeros(len(md))
        zoneflag = np.zeros(len(md))
    shaleflagN = (np.max(shaleflag) - shaleflag[:]) / np.max(shaleflag)
    flag = Curve(shaleflag, mnemonic='ShaleFlag', units='ohm/m', index=md,
        null=0)
    zone = Curve(zoneflag, mnemonic='ShaleFlag', units='ohm/m', index=md,
        null=0)
    well.data['Flag'] = flag
    array = well.data_as_matrix(return_meta=False)
    dfa = well.df()
    dfa = dfa.dropna()
    print(dfa) if debug else None
    header = well._get_curve_mnemonics()
    csvdf = pd.DataFrame(array, columns=header)
    tvd = well.data['TVDM'].values
    tvdf = tvd * 3.28084
    tvdm = tvdf / 3.28084
    tvdm[-1] = tvd[-1]
    print('length tvdf:', len(tvdf), 'length tvd:', len(tvd)
        ) if debug else None
    print('tvdf:', tvdf) if debug else None
    glwd = float(attrib[1])
    glf = glwd * 3.28084
    wdf = glwd * -3.28084
    if wdf < 0:
        wdf = 0
    print(attrib[1]) if debug else None
    well.location.gl = float(attrib[1])
    well.location.kb = float(attrib[0])
    mudweight = np.zeros(len(tvd))
    bt = np.zeros(len(tvd))
    delTempC = np.zeros(len(tvd))
    tempC = np.zeros(len(tvd))
    tempC[:] = np.nan
    tempG = np.zeros(len(tvd))
    try:
        agf = (float(well.location.kb) - float(well.location.gl)) * 3.28084
    except AttributeError:
        agf = (float(attrib[0]) - float(attrib[1])) * 3.28084
    if glwd >= 0:
        if np.abs(well.location.kb) < np.abs(well.location.gl):
            agf = well.location.kb * 3.28084
            well.location.kb = well.location.gl + well.location.kb
    if glwd < 0:
        agf = well.location.kb * 3.28084
    print('Rig floor is ', well.location.kb * 3.28084, 'feet above MSL'
        ) if debug else None
    tvdmsloffset = well.location.kb
    groundoffset = well.location.gl
    tvdmsl = tvd[:] - tvdmsloffset
    tvdbgl = 0
    tvdbgl = tvdmsl[:] + groundoffset
    tvdbglf = np.zeros(len(tvdbgl))
    tvdmslf = np.zeros(len(tvdmsl))
    wdfi = np.zeros(len(tvdmsl))
    lithotype = np.zeros(len(tvdbgl))
    nulitho = np.zeros(len(tvdbgl))
    dtlitho = np.zeros(len(tvdbgl))
    ilog = np.zeros(len(tvdbgl))
    formtvd = np.full(len(tvd), np.nan)
    hydrotvd = np.full(len(tvd), np.nan)
    hydroheight = np.full(len(tvd), np.nan)
    structop = np.full(len(tvd), np.nan)
    strucbot = np.full(len(tvd), np.nan)
    Owc = np.full(len(tvd), np.nan)
    Goc = np.full(len(tvd), np.nan)
    ttvd = np.full(len(tvd), np.nan)
    btvd = np.full(len(tvd), np.nan)
    btvd2 = np.full(len(tvd), np.nan)
    grcut = np.full(len(tvd), np.nan)
    alphas = np.full(len(tvd), offset)
    betas = np.full(len(tvd), tilt)
    gammas = np.full(len(tvd), tiltgamma)
    tecB = np.full(len(tvd), tecb)
    SHsh = np.full(len(tvd), np.nan)
    biot = np.full(len(tvd), 1)
    dt_zeros = np.full(len(tvd), dtml)
    dt_ncts = np.full(len(tvd), lamb)
    res_ncts = np.full(len(tvd), be)
    res_zeros = np.full(len(tvd), res0)
    res_exp = np.full(len(tvd), ne)
    dex_ncts = np.full(len(tvd), de)
    dex_zeros = np.full(len(tvd), dex0)
    dex_exp = np.full(len(tvd), nde)
    arr_ten_fac = np.full(len(tvd), ten_fac)
    if float(attrib[5]) == 0:
        bht_point[-1][0] = tvdbgl[-1] / 1000 * 30
    i = 0
    while i < len(bht_point):
        if bht_point[i][0] == 0:
            bht_point.pop(i)
        i += 1
    tgrads = np.array(bht_point)
    i = 0
    while i < len(tgrads):
        tgrads[i] = [(bht_point[i][0] - 15) / (tvdbgl[find_nearest_depth(md,
            tgrads[i][1])[0]] / 1000), tgrads[i][1]]
        i += 1
    tgrads[0][0] = tgrads[1][0]
    print('BHTs: ', bht_point) if debug else None
    print('TGs: ', tgrads) if debug else None
    if lithos is not None:
        lithot = lithos.values.tolist()
        firstlith = [0, 0, 0, 0, 0]
        lastlith = [final_depth, 0, 0, 0, 0]
        lithot.insert(0, firstlith)
        lithot.append(lastlith)
    if flags is not None:
        imagelog = flags.values.tolist()
        firstimg = [0, 0]
        lastimg = [final_depth, 0]
        imagelog.insert(0, firstimg)
        imagelog.append(lastimg)
    if forms is not None:
        formlist = forms.values.tolist()
        print('Formation Data: ', formlist) if debug else None
        ttvdlist = np.transpose(formlist)[4]
        ftvdlist = np.transpose(formlist)[0]
        logbotlist = ftvdlist
        ttvdlist = np.append(0, ttvdlist)
        logbotlist = np.append(logbotlist, tvd[-1])
        fttvdlist = np.append(0, ftvdlist)
        difftvd = np.zeros(len(fttvdlist))
        hydrldiff = np.zeros(len(fttvdlist))
        htvdlist = np.zeros(len(fttvdlist))
        owclist = np.transpose(formlist)[7]
        owclist = np.append(0, owclist)
        goclist = np.transpose(formlist)[8]
        goclist = np.append(0, goclist)
        btlist = np.transpose(formlist)[9]
        btlist = np.append(0,btlist)
        #btlist = np.append(btlist, btlist[-1])
        strucbotlist = np.transpose(formlist)[5]
        strucbotlist = np.append(ttvdlist[1], strucbotlist)
        logtoplist = np.append(0, ftvdlist)
        ftvdlist = np.append(ftvdlist, tvd[-1])
        centroid_ratio_list = np.transpose(formlist)[6]
        centroid_ratio_list = np.append([0.5], centroid_ratio_list)
        grlist = np.transpose(formlist)[3]
        grlist = np.append(grlist[0], grlist)
        print(ftvdlist, btlist) if debug else None
        centroid_ratio_list = centroid_ratio_list.astype(float)
        grlist = grlist.astype(float)
        alphalist = np.transpose(formlist)[10]
        alphalist = np.append(offset, alphalist)
        betalist = np.transpose(formlist)[11]
        betalist = np.append(dip_dir, betalist)
        gammalist = np.transpose(formlist)[12]
        gammalist = np.append(dip, gammalist)
        tecBlist = np.transpose(formlist)[13]
        tecBlist = np.append(tecb, tecBlist)
        SHshlist = np.transpose(formlist)[14]
        SHshlist = np.append(1, SHshlist)
        biotlist = np.transpose(formlist)[15]
        biotlist = np.append(1, biotlist)
        dt_nct_list = np.transpose(formlist)[16]
        dt_nct_list = np.append(lamb, dt_nct_list)
        dt_zero_list = np.transpose(formlist)[17]
        dt_zero_list = np.append(dtml, dt_zero_list)
        res_nct_list = np.transpose(formlist)[18]
        res_nct_list = np.append(be, res_nct_list)
        res_exp_list = np.transpose(formlist)[19]
        res_exp_list = np.append(ne, res_exp_list)
        res_zero_list = np.transpose(formlist)[20]
        res_zero_list = np.append(res0, res_zero_list)
        dex_nct_list = np.transpose(formlist)[21]
        dex_nct_list = np.append(de, dex_nct_list)
        dex_exp_list = np.transpose(formlist)[22]
        dex_exp_list = np.append(nde, dex_exp_list)
        dex_zero_list = np.transpose(formlist)[23]
        dex_zero_list = np.append(dex0, dex_zero_list)
        alphalist = alphalist.astype(float)
        betalist = betalist.astype(float)
        gammalist = gammalist.astype(float)
        tecBlist = tecBlist.astype(float)
        SHshlist = SHshlist.astype(float)
        biotlist = biotlist.astype(float)
        dt_nct_list = dt_nct_list.astype(float)
        dt_zero_list = dt_zero_list.astype(float)
        res_nct_list = res_nct_list.astype(float)
        res_exp_list = res_exp_list.astype(float)
        res_zero_list = res_zero_list.astype(float)
        dex_nct_list = dex_nct_list.astype(float)
        dex_exp_list = dex_exp_list.astype(float)
        dex_zero_list = dex_zero_list.astype(float)
        print('TecFacList: ', tecBlist) if debug else None
        print(alphalist, ftvdlist) if debug else None
        i = 0
        while i < len(fttvdlist):
            betalist[i], gammalist[i] = getEuler(float(alphalist[i]), float
                (betalist[i]), float(gammalist[i])) if np.isfinite(float(
                alphalist[i])) and np.isfinite(float(betalist[i])
                ) and np.isfinite(float(gammalist[i])) else (betalist[i],
                gammalist[i])
            print(betalist[i], gammalist[i]) if debug else None
            difftvd[i] = float(fttvdlist[i]) - float(ttvdlist[i])
            fttvdlist[i] = float(fttvdlist[i])
            ttvdlist[i] = float(ttvdlist[i])
            hydrldiff[i] = float(difftvd[i])
            if hydrldiff[i] < 0:
                hydrldiff[i] = 0
            htvdlist[i] = float(fttvdlist[i]) + float(hydrldiff[i])
            i += 1
        difftvd = np.append(difftvd, difftvd[-1])
        hydrldiff = np.append(hydrldiff, hydrldiff[-1])
        fttvdlist = np.append(fttvdlist, final_depth)
        ttvdl = float(fttvdlist[-1]) - float(difftvd[-1])
        htvdl = float(fttvdlist[-1]) - float(hydrldiff[-1])
        ttvdlist = np.append(ttvdlist, ttvdl)
        htvdlist = np.append(htvdlist, htvdl)
        fttvdlist = fttvdlist.astype(float)
        logtoplist = logtoplist.astype(float)
        strucbotlist = strucbotlist.astype(float)
        ttvdlist = ttvdlist.astype(float)
        logbotlist = logbotlist.astype(float)
        htvdlist = htvdlist.astype(float)
        print('Differential TVD list:') if debug else None
        print([difftvd, hydrldiff, fttvdlist, ttvdlist]) if debug else None
        structoplist = np.delete(ttvdlist, -1)
        goclist = np.where(goclist.astype(float) == 0, np.nan, goclist.
            astype(float))
        owclist = np.where(owclist.astype(float) == 0, np.nan, owclist.
            astype(float))
        cdtvdlist = structoplist + (strucbotlist - structoplist
            ) * centroid_ratio_list
        print('Structural tops list', structoplist) if debug else None
        print('Structural bottoms list', strucbotlist) if debug else None
        print('Structural centroids ratios', centroid_ratio_list
            ) if debug else None
        print('Structural centroids list', cdtvdlist) if debug else None
        print('tops list', logtoplist) if debug else None
        print('bottoms list', logbotlist) if debug else None
        print('OWCs', owclist) if debug else None
        print('GOCs', goclist) if debug else None
        print('GR Cutoffs', grlist) if debug else None
    j = 0
    k = 0
    m = 0
    n = 0
    o = 0
    p = 0
    i = 0
    nu3 = [nu] * len(tvd)
    mu2 = [0.6] * len(tvd)
    ucs2 = [np.nan] * len(tvd)
    try:
        print(lithot) if debug else None
    except:
        pass
    while i < len(tvd):
        if tvdbgl[i] < 0:
            tvdbglf[i] = 0
            if tvdmsl[i] > 0:
                wdfi[i] = tvdmsl[i] * 3.2804
        else:
            tvdbglf[i] = tvdbgl[i] * 3.28084
        if tvdmsl[i] < 0:
            tvdmslf[i] = 0
        else:
            tvdmslf[i] = tvdmsl[i] * 3.28084
        if md[i] < mud_weight[j][1]:
            mudweight[i] = mud_weight[j][0]
        else:
            mudweight[i] = mud_weight[j][0]
            j += 1
        if md[i] < tgrads[o][1]:
            y = [bht_point[o - 1][0], bht_point[o][0]]
            x = [tvdbgl[find_nearest_depth(md, bht_point[o - 1][1])[0]],
                tvdbgl[find_nearest_depth(md, bht_point[o][1])[0]]]
            tempC[i] = np.interp(tvdbgl[i], x, y)
            y2 = [tgrads[o - 1][0], tgrads[o][0]]
            x2 = [tvdbgl[find_nearest_depth(md, tgrads[o - 1][1])[0]],
                tvdbgl[find_nearest_depth(md, tgrads[o][1])[0]]]
            tempG[i] = np.interp(tvdbgl[i], x2, y2) / 1000
            delTempC[i] = tempC[i] - mudtemp
        else:
            tempG[i] = tgrads[o][0] / 1000
            tempC[i] = bht_point[o][0]
            delTempC[i] = mudtemp - tempC[i]
            o += 1
        if lithos is not None:
            if md[i] < lithot[k][0]:
                lithotype[i] = int(lithot[k - 1][1])
                if len(lithot[k]) > 2 and lithot[k - 1][2] > 0:
                    try:
                        nu2[i] = lithot[k - 1][2]
                    except:
                        pass
                    try:
                        if lithot[k - 1][3] > 0:
                            mu2[i] = lithot[k - 1][3]
                    except:
                        pass
                    try:
                        ucs2[i] = lithot[k - 1][4]
                    except:
                        pass
                    try:
                        arr_ten_fac[i] = lithot[k - 1][5]
                    except:
                        pass
                else:
                    nu2[i] = float(numodel[int(lithotype[i] - 1)])
            else:
                lithotype[i] = int(lithot[k][1])
                try:
                    nu2[i] = float(lithot[k][2])
                except:
                    pass
                try:
                    if lithot[k][3] > 0:
                        mu2[i] = float(lithot[k][3])
                except:
                    pass
                try:
                    ucs2[i] = float(lithot[k][4])
                except:
                    pass
                try:
                    arr_ten_fac[i] = float(lithot[k][5])
                except:
                    pass
                k += 1
        if flags is not None:
            if md[i] < imagelog[m][0]:
                ilog[i] = int(imagelog[m - 1][1])
            else:
                ilog[i] = imagelog[m][1]
                m += 1
        if forms is not None:
            formtvd[i] = np.interp(tvd[i], fttvdlist, ttvdlist)
            btvd2[i] = np.interp(tvd[i], logtoplist, logbotlist)
            hydrotvd[i] = np.interp(tvd[i], fttvdlist, htvdlist)
            if tvd[i] <= float(ftvdlist[p]):
                alphas[i] = alphalist[p] if np.isfinite(alphalist[p]
                    ) else offset
                betas[i] = betalist[p] if np.isfinite(betalist[p]) else tilt
                gammas[i] = gammalist[p] if np.isfinite(gammalist[p]
                    ) else tiltgamma
                tecB[i] = tecBlist[p] if np.isfinite(tecBlist[p]) else tecb
                dt_ncts[i] = dt_nct_list[p] if np.isfinite(dt_nct_list[p]
                    ) else lamb
                dt_zeros[i] = dt_zero_list[p] if np.isfinite(dt_zero_list[p]
                    ) else dtml
                res_ncts[i] = res_nct_list[p] if np.isfinite(res_nct_list[p]
                    ) else be
                res_exp[i] = res_exp_list[p] if np.isfinite(res_exp_list[p]
                    ) else ne
                res_zeros[i] = res_zero_list[p] if np.isfinite(res_zero_list[p]
                    ) else res0
                dex_ncts[i] = dex_nct_list[p] if np.isfinite(dex_nct_list[p]
                    ) else de
                dex_exp[i] = dex_exp_list[p] if np.isfinite(dex_exp_list[p]
                    ) else nde
                dex_zeros[i] = dex_zero_list[p] if np.isfinite(dex_zero_list[p]
                    ) else dex0
                SHsh[i] = SHshlist[p]
                biot[i] = biotlist[p] if np.isfinite(biotlist[p]) else 1
                grcut[i] = grlist[p]
                ttvd[i] = logtoplist[p]
                btvd[i] = logbotlist[p]
                Owc[i] = owclist[p]
                Goc[i] = goclist[p]
                structop[i] = ttvdlist[p]
                strucbot[i] = strucbotlist[p]
                hydroheight[i] = tvd[i] + hydrldiff[p]
                if np.isfinite(float(btlist[p])):
                    bt[i] = float(btlist[p])
                else:
                    bt[i] = 0
            else:
                dt_ncts[i] = dt_nct_list[p] if np.isfinite(dt_nct_list[p]
                    ) else lamb
                dt_zeros[i] = dt_zero_list[p] if np.isfinite(dt_zero_list[p]
                    ) else dtml
                res_ncts[i] = res_nct_list[p] if np.isfinite(res_nct_list[p]
                    ) else be
                res_exp[i] = res_exp_list[p] if np.isfinite(res_exp_list[p]
                    ) else ne
                res_zeros[i] = res_zero_list[p] if np.isfinite(res_zero_list[p]
                    ) else res0
                dex_ncts[i] = dex_nct_list[p] if np.isfinite(dex_nct_list[p]
                    ) else de
                dex_exp[i] = dex_exp_list[p] if np.isfinite(dex_exp_list[p]
                    ) else nde
                dex_zeros[i] = dex_zero_list[p] if np.isfinite(dex_zero_list[p]
                    ) else dex0
                grcut[i] = grlist[p]
                ttvd[i] = logtoplist[p]
                btvd[i] = logbotlist[p]
                Owc[i] = np.nan
                Goc[i] = np.nan
                structop[i] = structoplist[p]
                strucbot[i] = strucbotlist[p]
                hydroheight[i] = tvd[i] + hydrldiff[p]
                if np.isfinite(float(btlist[p])):
                    bt[i] = float(btlist[p])
                else:
                    bt[i] = 0
                p += 1
        else:
            grcut[i] = np.nanmean(gr)
        i += 1
    """plt.plot(lithotype,md)
    plt.plot(nu3,md)
    plt.plot(nu2,md)
    plt.plot(mu2,md)
    plt.plot(ucs2,md)
    plt.plot(ilog,md)
    plt.show()
    plt.clf()"""
    if forms is not None:
        if writeFile:
            try:
                plt.plot(structop, tvd, label='Structural Tops', linestyle=':')
                plt.plot(strucbot, tvd, label='Structural Bottoms',
                    linestyle=':')
                plt.plot(Owc, tvd, label='OWC', linestyle='-')
                plt.plot(Goc, tvd, label='GOC', linestyle='-')
                plt.plot(btvd, tvd, label='Log Bottom', linestyle='-')
                plt.plot(ttvd, tvd, label='Log Top', linestyle='-')
                plt.gca().invert_yaxis()
                plt.title(well.name + well.uwi + ' Structure Diagram ')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(output_dir, 'Structure.png'))
                plt.close()
            except:
                pass
    lradiff = np.full(len(md), np.nan)
    if alias['resshal'] != [] and alias['resdeep'] != []:
        rS = well.data[alias['resshal'][0]].values
        rD = well.data[alias['resdeep'][0]].values
        print(rD) if debug else None
        rdiff = rD[:] - rS[:]
        rdiff[np.isnan(rdiff)] = 0
        radiff = (rdiff[:] * rdiff[:]) ** 0.5
        lradiff = radiff
        rediff = Curve(lradiff, mnemonic='ResD', units='ohm/m', index=md,
            null=0)
        well.data['ResDif'] = rediff
        print('sfs = :', sfs) if debug else None
        if forms is not None:
            shaleflag = np.where(gr < grcut, 1, 0)
            zoneflag = np.where(gr < grcut, 0, 1)
        else:
            shaleflag = safe_block(rediff, cutoffs=sfs, values=(0, 1))
            zoneflag = safe_block(rediff, cutoffs=sfs, values=(0, 1))
        print(shaleflag) if debug else None
    else:
        shaleflag = np.zeros(len(md))
        zoneflag = np.zeros(len(md))
    shaleflagN = (np.max(shaleflag) - shaleflag[:]) / np.max(shaleflag)
    flag = Curve(shaleflag, mnemonic='ShaleFlag', units='ohm/m', index=md,
        null=0)
    zone = Curve(zoneflag, mnemonic='ShaleFlag', units='ohm/m', index=md,
        null=0)
    well.data['Flag'] = flag
    print('air gap is ', agf, 'feet') if debug else None
    if glwd >= 0:
        print('Ground Level is ', glf, 'feet above MSL') if debug else None
    if glwd < 0:
        print('Seafloor is ', wdf, 'feet below MSL') if debug else None
    rhoppg = np.full(len(tvdf), np.nan)
    rhogcc = np.full(len(tvdf), np.nan)
    ObgTppg = np.full(len(tvdf), np.nan)
    hydrostatic = np.full(len(tvd), np.nan)
    mudhydrostatic = np.full(len(tvd), np.nan)
    lithostatic = np.full(len(tvd), np.nan)
    i = 0
    while i < len(tvdf - 1):
        if glwd < 0:
            if tvdbgl[i] >= 0:
                rhoppg[i] = rhoappg + ((tvdf[i] - agf - wdf) / 3125) ** a
                rhogcc[i] = 0.11982642731 * rhoppg[i]
                if np.isfinite(zden2[i]):
                    if zden2[i] < 4:
                        rhoppg[i] = zden2[i] / 0.11982642731
                        rhogcc[i] = zden2[i]
                hydrostatic[i] = water
                mudhydrostatic[i] = 1.0 * mudweight[i]
            elif tvdmsl[i] < 0:
                rhoppg[i] = 8.34540426515252 * water
                rhogcc[i] = 0.11982642731 * rhoppg[i]
                hydrostatic[i] = 0
                mudhydrostatic[i] = 0
            else:
                rhoppg[i] = 0
                rhogcc[i] = 0
                hydrostatic[i] = water
                mudhydrostatic[i] = 1.0 * mudweight[i]
        elif tvdbgl[i] >= 0:
            rhoppg[i] = rhoappg + (tvdbglf[i] / 3125) ** a
            rhogcc[i] = 0.11982642731 * rhoppg[i]
            if np.isfinite(zden2[i]):
                if zden2[i] < 4:
                    rhoppg[i] = zden2[i] / 0.11982642731
                    rhogcc[i] = zden2[i]
            hydrostatic[i] = water
            mudhydrostatic[i] = 1.0 * mudweight[i]
        else:
            rhoppg[i] = 0
            rhogcc[i] = 0
            hydrostatic[i] = 0
            mudhydrostatic[i] = 0
        i += 1
    hydroppf = 0.4335275040012 * hydrostatic
    mudppf = 0.4335275040012 * mudhydrostatic
    lithostatic = 2.6 * 9.80665 / 6.89476 * tvd
    gradient = lithostatic / tvdf * 1.48816
    rhoppg[np.nanargmin(abs(tvdbgl))] = rhoappg
    if np.nanargmin(abs(tvdbgl)) > 0:
        rhoppg[0:np.nanargmin(abs(tvdbgl))] = np.nan
    rhogcc[np.nanargmin(abs(tvdbgl))] = rhoappg * 0.11982642731
    if np.nanargmin(abs(tvdbgl)) > 0:
        rhogcc[0:np.nanargmin(abs(tvdbgl))] = np.nan
    try:
        rhogcc = [(rhogcc[i] if math.isnan(zden2[i]) else zden2[i]) for i in
            range(len(zden2))]
    except:
        pass
    integrho = np.zeros(len(tvd))
    integrhopsift = np.zeros(len(tvd))
    i = 1
    maxwaterppg = wdf * 8.34540426515252 * water
    while i < len(tvd - 1):
        if glwd < 0.0:
            if tvdbgl[i] >= 0.0:
                integrho[i] = integrho[i - 1] + rhogcc[i] * 9806.65 * (tvdbgl
                    [i] - tvdbgl[i - 1])
                integrhopsift[i] = integrho[i] * 0.000145038 / tvdf[i]
                ObgTppg[i] = (maxwaterppg + np.mean(rhoppg[i]) * tvdbglf[i]
                    ) / tvdmslf[i]
            elif tvdmsl[i] >= 0.0:
                integrho[i] = integrho[i - 1] + water * 9806.65 * (tvdbgl[i
                    ] - tvdbgl[i - 1])
                integrhopsift[i] = integrho[i] * 0.000145038 / tvdf[i]
                ObgTppg[i] = 8.34540426515252 * water
        elif tvdbgl[i] >= 0.0:
            integrho[i] = integrho[i - 1] + rhogcc[i] * 9806.65 * (tvdbgl[i
                ] - tvdbgl[i - 1])
            integrhopsift[i] = integrho[i] * 0.000145038 / tvdf[i]
            ObgTppg[i] = rhoppg[i]
        i += 1
    if glwd >= 0.0:
        ObgTppg = rhoppg
    ObgTgcc = 0.11982642731 * ObgTppg
    ObgTppf = 0.4335275040012 * ObgTgcc
    print('Obg: ', ObgTgcc) if debug else None
    print('len of Obg: ', len(ObgTgcc)) if debug else None
    print('Zden: ', zden2) if debug else None
    print('len of zden: ', len(zden2)) if debug else None
    import math
    coalflag = np.zeros(len(tvd))
    lithoflag = np.zeros(len(tvd))
    try:
        ObgTgcc = [(ObgTgcc[i] if math.isnan(zden2[i]) else zden2[i]) for i in
            range(len(zden2))]
        coalflag = [(0 if math.isnan(zden2[i]) else 1 if zden2[i] < 1.6 else
            0) for i in range(len(zden2))]
        lithoflag = [(0 if shaleflag[i] < 1 else 1 if zden2[i] < 1.6 else 2
            ) for i in range(len(zden2))]
    except:
        pass
    coal = Curve(coalflag, mnemonic='CoalFlag', units='coal', index=tvd, null=0
        )
    litho = Curve(lithoflag, mnemonic='LithoFlag', units='lith', index=tvd,
        null=0)
    ct = 0
    ct = ct * 0.1524
    ct = lamb
    pn = water
    matrick = dtmt
    mudline = dtml
    print('HUZZAH') if debug else None
    dalm = dt.as_numpy() * 1
    resdeep = rdeep.as_numpy() * 1
    tvdm = well.data['TVDM'].as_numpy() * 1
    tvdm[0] = 0.1
    print('TVDM', tvdm) if debug else None
    matrix = np.zeros(len(dalm))
    resnormal = res0 * np.exp(res_ncts * tvdbgl)
    dexnormal = dex0 * np.exp(dex_ncts * tvdbgl)
    lresdeep = np.log10(resdeep)
    lresnormal = np.log10(resnormal)
    dalmflag = np.full(len(tvdf), 1.0)
    resdflag = np.full(len(tvdf), 1.0)
    dtNormal = np.full(len(tvdf),910.0)

    i = 0
    while i < len(dalm):
        matrix[i] = matrick + dt_ncts[i] * i
        if lithotype[i] > 1.5:
            matrix[i] = 65
        if tvdbgl[i] > 0:
            if dalm[i] < matrick:
                dalm[i] = matrick + (dt_zeros[i] - matrick) * math.exp(-
                    dt_ncts[i] * tvdbgl[i])
            if np.isnan(dalm[i]):
                dalm[i] = matrick + (dt_zeros[i] - matrick) * math.exp(-
                    dt_ncts[i] * tvdbgl[i])
                dalmflag[i] = np.nan
            if np.isnan(resdeep[i]):
                resdeep[i] = res0 * np.exp(res_ncts[i] * tvdbgl[i])
                resdflag[i] = np.nan
        else:
            dtNormal[i] = 210.0 if glwd<0 and tvdmsl[i]>0 else 910.0 #Seawater and free air sonic speeds
        i += 1
    import math
    print(dalm) if debug else None
    vpinv = dalm * 10 ** -6 * 3280.84
    vp = 1 / vpinv
    print(vp) if debug else None
    if glwd < 0:
        hydropsi = hydroppf[:] * (tvd[:] * 3.28084)
        obgpsi = integrho * 0.000145038
    else:
        hydropsi = hydroppf[:] * (tvd[:] * 3.28084)
        obgpsi = integrho * 0.000145038
    hydrostaticpsi = (0.4335275040012 * tvdmslf if glwd < 0 else 
        0.4335275040012 * tvdbglf)
    mudpsi = mudppf[:] * tvdf[:]
    i = 0
    ppgZhang = np.zeros(len(tvdf))
    gccZhang = np.zeros(len(tvdf))
    gccEaton = np.zeros(len(tvdf))
    Dexp = np.full(len(tvdf), np.nan)
    gccDexp = np.full(len(tvdf), np.nan)
    psiZhang = np.zeros(len(tvdf))
    psiEaton = np.zeros(len(tvdf))
    psiDexp = np.zeros(len(tvdf))
    psiZhang2 = np.zeros(len(tvdf))
    psiftZhang = np.zeros(len(tvdf))
    psiftZhang2 = np.zeros(len(tvdf))
    gccZhang2 = np.zeros(len(tvdf))
    pnpsi = np.zeros(len(tvdf))
    psipp = np.full(len(tvdf), np.nan)
    psiftpp = np.zeros(len(tvdf))
    horsrud = np.zeros(len(tvdf))
    lal = np.zeros(len(tvdf))
    ym = np.zeros(len(tvdf))
    sm = np.zeros(len(tvdf))
    bm = np.zeros(len(tvdf))
    cm_sip = np.zeros(len(tvdf))
    lal3 = np.zeros(len(tvdf))
    phi = np.zeros(len(tvdf))
    philang = np.zeros(len(tvdf))
    H = np.zeros(len(tvdf))
    K = np.zeros(len(tvdf))

    print('ObgTppg:', ObgTppg) if debug else None
    print('Reject Subhydrostatic below ', underbalancereject
        ) if debug else None
    if UCSs is not None:
        ucss = UCSs.to_numpy()
    print('Lithos: ', lithos) if debug else None
    print('UCS: ', UCSs) if debug else None
    print('IMAGE: ', flags) if debug else None
    maxveldepth = ul_depth
    if ul_depth == 0:
        mvindex = np.nan
    else:
        mvindex = find_nearest_depth(tvd, maxveldepth)[0]
    deltmu0 = np.nanmean(dalm[find_nearest_depth(tvd, maxveldepth)[0] - 5:
        find_nearest_depth(tvd, maxveldepth)[0] + 5])
    c = ct
    b = ct
    print('Max velocity is ', deltmu0, 'uspf') if debug else None
    from .obgppshmin import get_PPgrad_Zhang_gcc, get_PPgrad_Eaton_gcc, get_PPgrad_Dxc_gcc
    from .obgppshmin import get_Dxc
    while i < len(md) - 1:
        if glwd >= 0:
            if tvd[i] > ul_depth:
                b = ul_exp
            if tvdbgl[i] > 0:
                if shaleflag[i] < 0.5:
                    gccZhang[i] = get_PPgrad_Zhang_gcc(ObgTgcc[i], pn, b,
                        tvdbgl[i], dt_ncts[i], dt_zeros[i], matrick,
                        deltmu0, dalm[i], biot[i])
                    gccEaton[i] = get_PPgrad_Eaton_gcc(ObgTgcc[i], pn,
                        res_ncts[i], res_exp[i], tvdbgl[i], res_zeros[i],
                        resdeep[i], biot[i])
                    try:
                        Dexp[i] = get_Dxc(rop[i], rpm[i], wob[i], bit[i],
                            ecd[i], pn)
                    except:
                        Dexp[i] = np.nan
                    gccDexp[i] = get_PPgrad_Dxc_gcc(ObgTgcc[i], pn,
                        dex_ncts[i], dex_exp[i], tvdbgl[i], dex_zeros[i],
                        Dexp[i], biot[i])
                else:
                    gccZhang[i] = np.nan
                    gccEaton[i] = np.nan
                    gccDexp[i] = np.nan
                    try:
                        Dexp[i] = get_Dxc(rop[i], rpm[i], wob[i], bit[i],
                            ecd[i], pn)
                    except:
                        Dexp[i] = np.nan
                if gccZhang[i] < underbalancereject:
                    gccZhang[i] = underbalancereject
                if gccEaton[i] < underbalancereject:
                    gccEaton[i] = underbalancereject
                if gccDexp[i] < underbalancereject:
                    gccDexp[i] = underbalancereject
                ppgZhang[i] = gccZhang[i] * 8.33
                dtNormal[i] = matrick + (dt_zeros[i] - matrick) * math.exp(
                    -dt_ncts[i] * tvdbgl[i])
                lal3[i] = lall * (304.8 / (dalm[i] - 1))
                lal[i] = lalm * (vp[i] + lala) / vp[i] ** lale
                horsrud[i] = horsruda * vp[i] ** horsrude
                if np.isnan(ucs2[i]) or ucs2[i] == 0:
                    ucs2[i] = horsrud[i]
                phi[i] = np.arcsin(1 - 2 * nu2[i])
                philang[i] = np.arcsin((vp[i] - 1) / (vp[i] + 1))
                H[i] = 4 * np.tan(phi[i]) ** 2 * (9 - 7 * np.sin(phi[i])) / (
                    27 * (1 - np.sin(phi[i])))
                K[i] = 4 * lal[i] * np.tan(phi[i]) * (9 - 7 * np.sin(phi[i])
                    ) / (27 * (1 - np.sin(phi[i])))
                ym[i] = 0.076 * vp[i] ** 3.73 * 1000
                sm[i] = 0.03 * vp[i] ** 3.3 * 1000
                bm[i] = ym[i] / (3 * (1 - 2 * nu2[i]))
                psiftpp[i] = 0.4335275040012 * (gccZhang[i] if 
                    program_option[1] == 0 else gccEaton[i] if 
                    program_option[1] == 1 else gccDexp[i] if 
                    program_option[1] == 2 else np.nanmean([gccZhang[i],
                    gccEaton[i], gccDexp[i]]) if program_option[1] == 3 else
                    next((v for v, f in zip([gccZhang[i], gccEaton[i],
                    gccDexp[i]], [dalmflag[i], resdflag[i], Dexp[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 4 else
                    next((v for v, f in zip([gccZhang[i], gccDexp[i],
                    gccEaton[i]], [dalmflag[i], Dexp[i], resdflag[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 5 else
                    next((v for v, f in zip([gccEaton[i], gccZhang[i],
                    gccDexp[i]], [resdflag[i], dalmflag[i], Dexp[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 6 else
                    next((v for v, f in zip([gccEaton[i], gccDexp[i],
                    gccZhang[i]], [resdflag[i], Dexp[i], dalmflag[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 7 else
                    next((v for v, f in zip([gccDexp[i], gccZhang[i],
                    gccEaton[i]], [Dexp[i], dalmflag[i], resdflag[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 8 else
                    next((v for v, f in zip([gccDexp[i], gccEaton[i],
                    gccZhang[i]], [Dexp[i], resdflag[i], dalmflag[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 9 else
                    np.nan)
                psipp[i] = psiftpp[i] * tvdbglf[i] if tvdbglf[i
                    ] > 0 else np.nan
        else:
            if tvd[i] > ul_depth:
                b = ul_exp
            if tvdbgl[i] > 0:
                if shaleflag[i] < 0.5:
                    gccZhang[i] = get_PPgrad_Zhang_gcc(ObgTgcc[i], pn, b,
                        tvdbgl[i], dt_ncts[i], dt_zeros[i], matrick,
                        deltmu0, dalm[i], biot[i])
                    gccEaton[i] = get_PPgrad_Eaton_gcc(ObgTgcc[i], pn,
                        res_ncts[i], res_exp[i], tvdbgl[i], res_zeros[i],
                        resdeep[i], biot[i])
                    try:
                        Dexp[i] = get_Dxc(rop[i], rpm[i], wob[i], bit[i],
                            ecd[i], pn)
                    except:
                        Dexp[i] = np.nan
                    gccDexp[i] = get_PPgrad_Dxc_gcc(ObgTgcc[i], pn,
                        dex_ncts[i], dex_exp[i], tvdbgl[i], dex_zeros[i],
                        Dexp[i], biot[i])
                else:
                    gccZhang[i] = np.nan
                    gccEaton[i] = np.nan
                    gccZhang2[i] = np.nan
                    try:
                        Dexp[i] = get_Dxc(rop[i], rpm[i], wob[i], bit[i],
                            ecd[i], pn)
                    except:
                        Dexp[i] = np.nan
                if gccZhang[i] < underbalancereject:
                    gccZhang[i] = underbalancereject
                if gccEaton[i] < underbalancereject:
                    gccEaton[i] = underbalancereject
                ppgZhang[i] = gccZhang[i] * 8.33
                dtNormal[i] = matrick + (dt_zeros[i] - matrick) * math.exp(
                    -ct * tvdbgl[i])
                lal3[i] = lall * (304.8 / (dalm[i] - 1))
                lal[i] = lalm * (vp[i] + lala) / vp[i] ** lale
                horsrud[i] = horsruda * vp[i] ** horsrude
                if np.isnan(ucs2[i]) or ucs2[i] == 0:
                    ucs2[i] = horsrud[i]
                phi[i] = np.arcsin(1 - 2 * nu2[i])
                philang[i] = np.arcsin((vp[i] - 1) / (vp[i] + 1))
                H[i] = 4 * np.tan(phi[i]) ** 2 * (9 - 7 * np.sin(phi[i])) / (
                    27 * (1 - np.sin(phi[i])))
                K[i] = 4 * lal[i] * np.tan(phi[i]) * (9 - 7 * np.sin(phi[i])
                    ) / (27 * (1 - np.sin(phi[i])))
                ym[i] = 0.076 * vp[i] ** 3.73 * 1000
                sm[i] = 0.03 * vp[i] ** 3.3
                bm[i] = ym[i] / (3 * (1 - 2 * nu2[i]))
                psiftpp[i] = 0.4335275040012 * (gccZhang[i] if 
                    program_option[1] == 0 else gccEaton[i] if 
                    program_option[1] == 1 else gccDexp[i] if 
                    program_option[1] == 2 else np.nanmean([gccZhang[i],
                    gccEaton[i], gccDexp[i]]) if program_option[1] == 3 else
                    next((v for v, f in zip([gccZhang[i], gccEaton[i],
                    gccDexp[i]], [dalmflag[i], resdflag[i], Dexp[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 4 else
                    next((v for v, f in zip([gccZhang[i], gccDexp[i],
                    gccEaton[i]], [dalmflag[i], Dexp[i], resdflag[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 5 else
                    next((v for v, f in zip([gccEaton[i], gccZhang[i],
                    gccDexp[i]], [resdflag[i], dalmflag[i], Dexp[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 6 else
                    next((v for v, f in zip([gccEaton[i], gccDexp[i],
                    gccZhang[i]], [resdflag[i], Dexp[i], dalmflag[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 7 else
                    next((v for v, f in zip([gccDexp[i], gccZhang[i],
                    gccEaton[i]], [Dexp[i], dalmflag[i], resdflag[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 8 else
                    next((v for v, f in zip([gccDexp[i], gccEaton[i],
                    gccZhang[i]], [Dexp[i], resdflag[i], dalmflag[i]]) if 
                    not np.isnan(f)), np.nan) if program_option[1] == 9 else
                    np.nan)
                psipp[i] = psiftpp[i] * tvdbglf[i] if tvdbglf[i
                    ] > 0 else np.nan
        i += 1
    """plt.plot(phi,tvd, label='nu')
    plt.plot(philang,tvd, label='lang')
    plt.plot(philang-phi,tvd, label='delta')
    plt.legend()
    plt.show()
    plt.close()
    
    plt.plot(lresdeep,tvd, label='Resistivity')
    plt.plot(lresnormal,tvd)
    plt.plot(Dexp,tvd, label = 'Dxc')
    plt.plot(dexnormal,tvd)
    #plt.xscale("log")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
    plt.close()
    """
    psiftpp = psipp / tvdbglf if glwd > 0 else psipp / tvdmslf
    gccpp = psiftpp / 0.4335275040012
    dphi = np.degrees(phi[:])
    gccpp[-1] = hydrostatic[-1]
    gccpp[np.nanargmin(abs(tvdbgl))] = water
    gccpp = interpolate_nan(gccpp)
    gccpp = np.where(gccpp < underbalancereject, underbalancereject, gccpp)
    ppgpp = gccpp * 8.33
    psipp[np.nanargmin(abs(tvdbgl))] = hydrostaticpsi[np.nanargmin(abs(tvdbgl))
        ]
    psiftpp = interpolate_nan(psiftpp)
    psipp = interpolate_nan(psipp)
    psipp = np.where(psipp < hydrostaticpsi * underbalancereject, 
        hydrostaticpsi * underbalancereject, psipp)
    if np.nanargmin(abs(tvdbgl)) != 0:
        gccpp[0:np.nanargmin(abs(tvdbgl))] = np.nan
        psipp[0:np.nanargmin(abs(tvdbgl))] = np.nan
        psiftpp[0:np.nanargmin(abs(tvdbgl))] = np.nan
    print('GCC_PP: ', gccpp) if debug else None
    if writeFile:
        plt.plot(gccZhang, tvd, label='zhang', alpha=0.33)
        plt.plot(gccEaton, tvd, label='eaton', alpha=0.33)
        plt.plot(gccDexp, tvd, label='d_exp', alpha=0.33)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'PP_Components.png'))
        plt.close()
    """
    #Check Plot
    plt.plot(gccZhang,tvd, label='Unloading')
    plt.plot(gccZhang2,tvd, label='Loading',alpha=0.5, linestyle='-')
    plt.legend(loc="upper left")
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    """
    """plt.close()
    plt.plot(ym,tvd)
    plt.show()
    plt.clf()
    plt.close()"""
    
    #Daines Shmin
    
    print('Tectonic factor input = ', tecb) if debug else None
    i = 0
    
    if b > 10.0:
        b = 0
    fgppg = np.full(len(ppgZhang), np.nan)
    fgcc = np.full(len(ppgZhang), np.nan)
    dynmu = np.full(len(ppgZhang), np.nan)
    mufgppg = np.full(len(ppgZhang), np.nan)
    dmufgppg = np.full(len(ppgZhang), np.nan)
    mufgcc = np.full(len(ppgZhang), np.nan)
    while i < len(ObgTppg) - 1:
        if tvdbgl[i] > 0:
            if shaleflag[i] < 0.5:
                #fgppg[i] = nu2[i] / (1 - nu2[i]) * (ObgTppg[i] - biot[i] * ppgpp[i]) + biot[i] * ppgpp[i] + tecB[i] * ObgTppg[i]
                dynmu[i] = 0.65 # change with the actual coefficient of internal friction calculated from logs please.
                mufgppg[i] = 1 / ((mu ** 2 + 1) ** 0.5 + mu) ** 2 * (ObgTppg[i] - ppgpp[i]) + ppgpp[i]
                dmufgppg[i] = 1 / ((dynmu[i] ** 2 + 1) ** 0.5 + dynmu[i]) ** 2 * (ObgTppg[i] - ppgpp[i]) + ppgpp[i]
                daines = nu2[i] / (1 - nu2[i]) * (ObgTppg[i] - biot[i] * ppgpp[i]) + biot[i] * ppgpp[i] + (ym[i]/(1-(nu2[i]**2)))*(ehmin + (nu2[i]*ehmax)) if ehmin is not None and ehmax is not None else (nu2[i] / (1 - nu2[i]) * (ObgTppg[i] - biot[i] * ppgpp[i]) + biot[i] * ppgpp[i] + tecB[i] * ObgTppg[i])
                fgppg[i] =  daines if program_option[2] == 0 else mufgppg[i] if program_option[2] == 1 else dmufgppg[i] if program_option[2] == 2 else np.nan
            else:
                fgppg[i] = np.nan
                mufgppg[i] = np.nan
                mufgcc[i] = np.nan
        fgcc[i] = 0.11982642731 * fgppg[i]
        mufgcc[i] = 0.11982642731 * mufgppg[i]
        i += 1
    ppgpp = interpolate_nan(ppgpp)
    if np.nanargmin(abs(tvdbgl)) != 0:
        fgppg[0:np.nanargmin(abs(tvdbgl))] = np.nan
        fgcc[0:np.nanargmin(abs(tvdbgl))] = np.nan
        mufgcc[0:np.nanargmin(abs(tvdbgl))] = np.nan
        mufgppg[0:np.nanargmin(abs(tvdbgl))] = np.nan
    psiftfg = 0.4335275040012 * fgcc
    if glwd < 0:
        psifg = np.where(tvdmslf > 0, psiftfg * tvdmslf, np.nan)
    else:
        psifg = np.where(tvdbglf > 0, psiftfg * tvdbglf, np.nan)
    psimes = (psifg + obgpsi) / 2 + psipp
    shsvratio = psifg / obgpsi
    fgcc[np.nanargmin(abs(tvdbgl))] = rhogcc[np.nanargmin(abs(tvdbgl))
        ] * np.nanmedian(shsvratio)
    mufgcc[np.nanargmin(abs(tvdbgl))] = rhogcc[np.nanargmin(abs(tvdbgl))
        ] * np.nanmedian(shsvratio)
    fgppg = interpolate_nan(fgppg)
    fgcc = interpolate_nan(fgcc)
    psifg = interpolate_nan(psifg)
    psimes = interpolate_nan(psimes)
    mufgcc = interpolate_nan(mufgcc)
    psiftfg = 0.4335275040012 * fgcc
    if glwd < 0:
        psifg = np.where(tvdmslf > 0, psiftfg * tvdmslf, np.nan)
    else:
        psifg = np.where(tvdbglf > 0, psiftfg * tvdbglf, np.nan)
    psifg = np.where(psifg < psipp, psipp, psifg)
    psimes = (psifg + obgpsi) / 2 + psipp
    if np.nanargmin(abs(tvdbgl)) > 0:
        fgcc[0:np.nanargmin(abs(tvdbgl))] = np.nan
        psifg[0:np.nanargmin(abs(tvdbgl))] = np.nan
    if forms is not None:
        psippsand = np.zeros(len(md))
        from .hydraulics import getPPfromTopRecursive
        from .hydraulics import compute_optimal_gradient
        from .hydraulics import getHydrostaticPsi
        gradients = np.zeros(len(md))
        gradlist = np.zeros(len(cdtvdlist))
        eqlithostat = np.zeros(len(md))
        eqlithostat2 = np.zeros(len(md))
        j = 0
        for i in range(len(md)):
            if structop[i] != structop[i - 1]:
                gradients[i] = compute_optimal_gradient(tvd[
                    find_nearest_depth(tvd, ttvd[i])[0]:find_nearest_depth(
                    tvd, btvd[i])[0]], psipp[find_nearest_depth(tvd, ttvd[i
                    ])[0]:find_nearest_depth(tvd, btvd[i])[0]])
                gradlist[j] = gradients[i]
                j += 1
            else:
                gradients[i] = gradients[i - 1]
            eqlithostat[i] = getHydrostaticPsi(tvd[i], gradients[i])
            eqlithostat2[i] = getHydrostaticPsi(tvd[i], gradients[i])
            psippsand[i] = getPPfromTopRecursive(0, shsvratio[
                find_nearest_depth(tvd, structop[i])[0]], obgpsi[
                find_nearest_depth(tvd, structop[i])[0]], 0.85, water,
                structop[i], Goc[i], Owc[i], tvd[i])
        shalepressures = np.zeros((len(cdtvdlist), len(md)))
        shifts = np.zeros((len(cdtvdlist), len(md)))
        for i, depth in enumerate(cdtvdlist):
            shalepressures[i] = getHydrostaticPsi(tvd, gradlist[i])
        centroid_pressures_sand = np.zeros(len(cdtvdlist))
        for i, depth in enumerate(cdtvdlist):
            nearest_idx = find_nearest_depth(tvd, depth)[0]
            centroid_pressures_sand[i] = psippsand[nearest_idx]
        centroid_pressures_shale = np.zeros(len(cdtvdlist))
        for i, depth in enumerate(cdtvdlist):
            nearest_idx = find_nearest_depth(tvd, depth)[0]
            centroid_pressures_shale[i] = shalepressures[i][nearest_idx]
        shifts = centroid_pressures_sand - centroid_pressures_shale
        print('centroid pressure hydrostatic: ', centroid_pressures_sand
            ) if debug else None
        print('centroid pressure in shale: ', centroid_pressures_shale
            ) if debug else None
        print('Max seal integrity pressure: ', shifts) if debug else None
        j = 0
        for i in range(len(md)):
            try:
                if tvd[i] < logbotlist[j]:
                    psippsand[i] = psippsand[i] - shifts[j]
                else:
                    j += 1
                    psippsand[i] = np.nan
            except:
                pass
        for i in range(len(md)):
            if shaleflag[i] > 0.5:
                psipp[i] = psippsand[i]
        psipp = interpolate_nan(psipp)
        shsvratio2 = psifg / obgpsi
        """
        plt.plot(psipp,tvd)
        plt.plot(eqlithostat,tvd)
        plt.plot(psippsand,tvd)
        plt.plot(obgpsi,tvd)
        plt.plot(psifg,tvd)
        plt.gca().invert_yaxis()
        plt.show()
        plt.close()
        
        plt.plot(psipp/tvdf,tvd)
        plt.plot(psippsand/tvdf,tvd)
        plt.plot(obgpsi/tvdf,tvd)
        plt.plot(psifg/tvdf,tvd)
        #plt.plot(gradients,tvd)
        plt.gca().invert_yaxis()
        plt.xlim(0.4,1)
        plt.show()
        plt.close()
        """
    from .DrawSP import getSP
    from .DrawSP import drawSP
    i = 0
    sgHMpsi = np.zeros(len(tvd))
    sgHMpsiL = np.zeros(len(tvd))
    sgHMpsiU = np.zeros(len(tvd))
    psisfl = np.zeros(len(tvd))
    while i < len(tvd) - 1:
        try:
            stresshratio = SHsh[i]
        except:
            stresshratio = np.nan
        if np.isfinite(stresshratio):
            sgHMpsi[i] = psifg[i] * stresshratio
            sgHMpsiL[i] = sgHMpsi[i] * 0.9
            sgHMpsiU[i] = sgHMpsi[i] * 1.1
        else:
            result = getSP(obgpsi[i] / 145.038, psipp[i] / 145.038, mudpsi[
                i] / 145.038, psifg[i] / 145.038, UCS=ucs2[i], phi=phi[i], flag=ilog[i],
                mu=mu2[i], nu=nu2[i], bt=bt[i], ym=ym[i], delT=delTempC[i], biot=biot[i])
            sgHMpsi[i] = result[2] * 145.038
            sgHMpsiL[i] = result[0] * 145.038
            sgHMpsiU[i] = result[1] * 145.038
        if psifg[i] < obgpsi[i]:
            psifg[i] = np.nanmin([psifg[i], sgHMpsiL[i]])
        i += 1
    sgHMpsi = interpolate_nan(sgHMpsi)
    if np.nanargmin(abs(tvdbgl)) > 0:
        sgHMpsi[0:np.nanargmin(abs(tvdbgl))] = np.nan
    from .BoreStab import get_principal_stress
    from .BoreStab import draw, plot_to_base64_png
    i = window
    sfg = fgcc
    spp = gccpp
    spsipp = psipp
    ucs_horsrud = horsrud
    slal = lal
    slal2 = ym
    slal3 = sm
    spsifp = psifg
    ssgHMpsi = sgHMpsi
    ssgHMpsiL = sgHMpsiL
    ssgHMpsiU = sgHMpsiU
    finaldepth = find_nearest_depth(tvdm, finaldepth)[1]
    doi = min(doi, finaldepth - 1)
    if doi > 0:
        doiactual = find_nearest_depth(tvdm, doi)
        print(doiactual) if debug else None
        doiA = doiactual[1]
        doiX = doiactual[0]
        print('Depth of interest :', doiA, ' with index of ', doiX
            ) if debug else None
        devdoi = devdata[doiX]
        incdoi = devdoi[1]
        azmdoi = devdoi[2]
        print('Inclination is :', incdoi, ' towards azimuth of ', azmdoi
            ) if debug else None
        sigmaVmpa = obgpsi[doiX] / 145.038
        sigmahminmpa = spsifp[doiX] / 145.038
        ppmpa = spsipp[doiX] / 145.038
        bhpmpa = mudpsi[doiX] / 145.038
        ucsmpa = ucs_horsrud[doiX]
        ilog_flag = ilog[doiX]
        print('nu is ', nu2[doiX]) if debug else None
        print('phi is ', np.degrees(phi[doiX])) if debug else None
        if writeFile:
            drawSP(sigmaVmpa, ppmpa, bhpmpa, sigmahminmpa, UCS=ucsmpa, phi=phi[doiX
                ], flag=ilog_flag, mu=mu2[doiX], nu=nu2[doiX], bt=bt[doiX], ym=ym[doiX],
                delT=delTempC[doiX], path=output_fileSP, biot=biot[doiX], display=display)
        else:
            rv4 = plot_to_base64_png(drawSP(sigmaVmpa, ppmpa, bhpmpa,
                sigmahminmpa, UCS=ucsmpa, phi=phi[doiX], flag=ilog_flag, mu=mu2[doiX], nu=nu2[
                doiX], bt=bt[doiX], ym=ym[doiX], delT=delTempC[doiX], biot=biot[doiX]))
        sigmaHMaxmpa = sgHMpsi[doiX] / 145.038
        print('SigmaHM = ', sigmaHMaxmpa) if debug else None
        sigmas = [sigmaHMaxmpa, sigmahminmpa, sigmaVmpa]
        print(sigmas) if debug else None
        """
        if sigmas[2]>sigmas[0]:
            alpha = 0
            beta = 90 #normal faulting regime
            gamma = 0
            print("normal")
        else:
            if(sigmas[2]<sigmas[1]):
                alpha = 0
                beta = 0 #reverse faulting regime
                gamma = 0
                print("reverse")                  
            else:
                alpha = 0 #strike slip faulting regime
                beta = 0
                gamma = 90
                print("Strike slip")
        
        """
        alpha = offset
        beta = tilt
        gamma = tiltgamma
        from .BoreStab import getRota
        Rmat = getRota(alphas[doiX], betas[doiX], gammas[doiX])
        print(sigmas) if debug else None
        sigmas.append(bhpmpa - ppmpa)
        sigmas.append(ppmpa)
        from .PlotVec import savevec
        from .PlotVec import showvec
        from .BoreStab import getStens
        print('Alpha :', alphas[doiX], ', Beta: ', betas[doiX], ', Gamma :',
            gammas[doiX]) if debug else None
        print('Actual Sv is ', sigmas[2], 'Mpa') if debug else None
        m = np.min([sigmas[0], sigmas[1], sigmas[2]])
        osx, osy, osz = get_principal_stress(sigmas[0], sigmas[1], sigmas[2], alphas
            [doiX], betas[doiX], gammas[doiX])
        sten = getStens(osx, osy, osz, alphas[doiX], betas[doiX], gammas[doiX], debug=debug)
        sn, se, sd = np.linalg.eigh(sten)[0]
        on, oe, od = np.linalg.eigh(sten)[1]
        if writeFile:
            savevec(on, oe, od, 2, sn, se, sd, output_fileVec)
        if writeFile:
            try:
                draw(tvd[doiX], osx, osy, osz, sigmas[3], sigmas[4], ucsmpa,
                    alphas[doiX], betas[doiX], gammas[doiX], 0, nu2[doiX],
                    azmdoi, incdoi, bt[doiX], ym[doiX], delTempC[doiX],
                    path=output_fileS,ten_fac=arr_ten_fac[i], debug=debug, display=display)
            except:
                pass
        else:
            try:
                rv3 = plot_to_base64_png(draw(tvd[doiX], osx, osy, osz,
                    sigmas[3], sigmas[4], ucsmpa, alphas[doiX], betas[doiX],
                    gammas[doiX], 0, nu2[doiX], azmdoi, incdoi, bt[doiX],
                    ym[doiX], delTempC[doiX],ten_fac=arr_ten_fac[i]))
            except:
                rv3 = None
    from .BoreStab import getHoop, getAlignedStress
    from .failure_criteria import plot_sanding

    def drawBHimage(doi, writeFile=True):
        hfl = 2.5
        doiactual = find_nearest_depth(tvdm, doi - hfl)
        doiS = doiactual[0]
        doiactual2 = find_nearest_depth(tvdm, doi + hfl)
        doiF = doiactual2[0]
        frac = np.zeros([doiF - doiS, 360])
        crush = np.zeros([doiF - doiS, 360])
        data = np.zeros([doiF - doiS, 4])
        i = doiS
        j = 0
        while i < doiF:
            sigmaVmpa = obgpsi[i] / 145.038
            sigmahminmpa = psifg[i] / 145.038
            sigmaHMaxmpa = sgHMpsi[i] / 145.038
            ppmpa = psipp[i] / 145.038
            bhpmpa = mudpsi[i] / 145.038
            ucsmpa = horsrud[i]
            deltaP = bhpmpa - ppmpa
            sigmas = [sigmaHMaxmpa, sigmahminmpa, sigmaVmpa]
            osx, osy, osz = get_principal_stress(sigmas[0], sigmas[1], sigmas[2],
                alphas[i], betas[i], gammas[i])
            sigmas = [osx, osy, osz]
            devdoi = devdata[i]
            incdoi = devdoi[1]
            azmdoi = devdoi[2]
            """
            if sigmas[2]>sigmas[0]:
                alpha = 0
                beta = 90 #normal faulting regime
                gamma = 0
                #print("normal")
            else:
                if(sigmas[2]<sigmas[1]):
                    alpha = 0
                    beta = 0 #reverse faulting regime
                    gamma = 0
                    #print("reverse")                  
                else:
                    alpha = 0 #strike slip faulting regime
                    beta = 0
                    gamma = 90
                    #print("Strike slip")
            sigmas.sort(reverse=True)
            alpha = alpha + offset
            beta= beta+tilt
            """
            cr, fr, minazi, maxazi, minangle, maxangle, angles, noplot = (
                getHoop(incdoi, azmdoi, sigmas[0], sigmas[1], sigmas[2],
                deltaP, ppmpa, ucsmpa, alphas[i], betas[i], gammas[i], nu2[
                i], bt[i], ym[i], delTempC[i],ten_fac=arr_ten_fac[i]))
            crush[j] = cr
            frac[j] = fr
            if np.max(frac[j]) > 0:
                data[j] = [tvd[i], minazi, minangle, maxangle]
            i += 1
            j += 1
        from .plotangle import plotfracs, plotfrac
        i = find_nearest_depth(tvdm, doi)[0]
        j = find_nearest_depth(tvdm, doi)[1]
        sigmaVmpa = obgpsi[i] / 145.038
        sigmahminmpa = psifg[i] / 145.038
        sigmaHMaxmpa = sgHMpsi[i] / 145.038
        ppmpa = psipp[i] / 145.038
        bhpmpa = mudpsi[i] / 145.038
        ucsmpa = horsrud[i]
        deltaP = bhpmpa - ppmpa
        sigmas = [sigmaHMaxmpa, sigmahminmpa, sigmaVmpa]
        osx, osy, osz = get_principal_stress(sigmas[0], sigmas[1], sigmas[2], alphas
            [i], betas[i], gammas[i])
        sigmas = [osx, osy, osz]
        devdoi = devdata[i]
        incdoi = devdoi[1]
        azmdoi = devdoi[2]
        cr, fr, minazi, maxazi, minangle, maxangle, angles, noplot = getHoop(
            incdoi, azmdoi, sigmas[0], sigmas[1], sigmas[2], deltaP, ppmpa,
            ucsmpa, alphas[i], betas[i], gammas[i], nu2[i], bt[i], ym[i],
            delTempC[i],ten_fac=arr_ten_fac[i])
        fr = np.array(fr)
        angles = np.array(angles)
        data2 = j, fr, angles, minazi, maxazi
        if writeFile:
            d, f = plotfrac(data2, output_fileFrac,debug=debug)
        else:
            d, f = plotfrac(data2,debug=debug)
        plotfracs(data)
        plt.imshow(frac, cmap='Reds', alpha=0.5, extent=[0, 360, tvd[doiF],
            tvd[doiS]], aspect=10)
        plt.imshow(crush, cmap='Blues', alpha=0.5, extent=[0, 360, tvd[doiF
            ], tvd[doiS]], aspect=10)
        plt.plot(d, 'k-')
        plt.plot(f, 'k-', alpha=0.1)
        plt.ylim(j + hfl, j - hfl)
        plt.gca().set_aspect(360 / (6.67 * hfl * 2 * 0.1))
        plt.tick_params(axis='x', which='both', bottom=True, top=True,
            labelbottom=True, labeltop=True)
        plt.tick_params(axis='y', which='both', left=True, right=True,
            labelleft=True, labelright=True)
        plt.xticks([0, 90, 180, 270, 360])
        plt.title('Synthetic Borehole Image')
        if writeFile:
            try:
                plt.savefig(output_fileBHI, dpi=1200)
                if display:
                    plt.show()
            except:
                pass
        else:
            print('SBI plot svg will be returned and not saved') if debug else None
            try:
                return plot_to_base64_png(plt)
            except:
                return None
        plt.clf()
        plt.close()

    def plotHoop(doi, writeFile=True):
        doiactual = find_nearest_depth(tvdm, doi)
        doiS = doiactual[0]
        i = doiS
        j = 0
        sigmaVmpa = obgpsi[i] / 145.038
        sigmahminmpa = psifg[i] / 145.038
        sigmaHMaxmpa = sgHMpsi[i] / 145.038
        ppmpa = psipp[i] / 145.038
        bhpmpa = mudpsi[i] / 145.038
        ucsmpa = horsrud[i]
        deltaP = bhpmpa - ppmpa
        sigmas = [sigmaHMaxmpa, sigmahminmpa, sigmaVmpa]
        osx, osy, osz = get_principal_stress(sigmas[0], sigmas[1], sigmas[2], alphas
            [i], betas[i], gammas[i])
        sigmas = [osx, osy, osz]
        devdoi = devdata[i]
        incdoi = devdoi[1]
        azmdoi = devdoi[2]
        if writeFile:
            getHoop(incdoi, azmdoi, sigmas[0], sigmas[1], sigmas[2], deltaP,
                ppmpa, ucsmpa, alphas[i], betas[i], gammas[i], nu2[i], bt[i
                ], ym[i], delTempC[i], output_fileHoop,ten_fac=arr_ten_fac[i])
        else:
            return getHoop(incdoi, azmdoi, sigmas[0],
                sigmas[1], sigmas[2], deltaP, ppmpa, ucsmpa, alphas[i],
                betas[i], gammas[i], nu2[i], bt[i], ym[i], delTempC[i],ten_fac=arr_ten_fac[i])[-1]

    def drawSand(doi, writeFile=True):
        doiactual = find_nearest_depth(tvdm, doi)
        doiS = doiactual[0]
        i = doiS
        j = 0
        sigmaVmpa = obgpsi[i] / 145.038
        sigmahminmpa = psifg[i] / 145.038
        sigmaHMaxmpa = sgHMpsi[i] / 145.038
        ppmpa = psipp[i] / 145.038
        bhpmpa = mudpsi[i] / 145.038
        ucsmpa = horsrud[i]
        deltaP = bhpmpa - ppmpa
        sigmas = [sigmaHMaxmpa, sigmahminmpa, sigmaVmpa]
        osx, osy, osz = get_principal_stress(sigmas[0], sigmas[1], sigmas[2], alphas
            [i], betas[i], gammas[i])
        devdoi = devdata[i]
        incdoi = devdoi[1]
        azmdoi = devdoi[2]
        Sl = getAlignedStress(osx, osy, osz, alphas[i], betas[i], gammas[i],
            azmdoi, incdoi)
        sigmamax = max(Sl[0][0], Sl[1][1])
        sigmamin = min(Sl[0][0], Sl[1][1])
        sigma_axial = Sl[2][2]
        k0 = 1
        if writeFile:
            plot_sanding(sigmamax, sigmamin, sigma_axial, ppmpa, ucsmpa, k0,
                nu2[i], biot[i], os.path.join(output_dir, 'Sanding.png'), display=display)
        else:
            return plot_to_base64_png(plot_sanding(sigmamax, sigmamin,
                sigma_axial, ppmpa, ucsmpa, k0, nu2[i], biot[i]))

    def combineHarvest():
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        image1 = mpimg.imread(output_fileSP)
        image2 = mpimg.imread(output_fileS)
        image3 = mpimg.imread(os.path.join(output_dir, 'Sanding.png'))
        image4 = mpimg.imread(output_fileBHI)
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs[0, 0].imshow(image1)
        axs[0, 1].imshow(image2)
        axs[1, 0].imshow(image3)
        axs[1, 1].imshow(image4)
        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_fileAll)
        plt.close()
    if doi > 0:
        rv5 = plotHoop(doi, writeFile)
        rv1 = drawBHimage(doi, writeFile)
        rv2 = drawSand(doi, writeFile)
        if writeFile:
            combineHarvest()
    from matplotlib.ticker import MultipleLocator
    from .Plotter import plot_logs_labels
    plotend = min(plotend, finaldepth)
    if plotstart > finaldepth or plotstart > plotend:
        plotstart = 0
    mogu1 = np.nanmax(ssgHMpsi[:find_nearest_depth(tvd, plotend)[0]])
    mogu2 = np.nanmax(obgpsi[:find_nearest_depth(tvd, plotend)[0]])
    mogu3 = np.nanmin(hydropsi[find_nearest_depth(tvd, plotstart)[0]:
        find_nearest_depth(tvd, plotend)[0]])
    maxchartpressure = 1000 * math.ceil(max(mogu1, mogu2) / 1000)
    minpressure = round(mogu3)
    
    Sb = np.full((len(tvd), 3, 3), np.nan)
    SbFF = np.full((len(tvd), 3, 3), np.nan)
    hoopmax = np.full(len(tvd), np.nan)
    hoopmin = np.full(len(tvd), np.nan)
    lademax = np.full(len(tvd), np.nan)
    lademin = np.full(len(tvd), np.nan)
    inca = np.full(len(tvd), np.nan)
    minstresspsi = np.nanmin(np.stack((obgpsi, psifg, sgHMpsi)), axis=0)
    trufracmpa = 0.006894 * minstresspsi
    from .failure_criteria import mod_lad_cmw, mogi
    print('calculating aligned far field stresses') if debug else None
    print('Total depth-points to be calculated: ', len(tvd)) if debug else None
    from .BoreStab import get_frac_pressure
    skip = 21 if 2.0 <= window < 21 else window
    mtol = 1 - 2 * mabw / 360
    for i in range(0, len(tvd), 1):
        sigmaVmpa = np.nanmean(obgpsi[i]) / 145.038
        sigmahminmpa = np.nanmean(psifg[i]) / 145.038
        sigmaHMaxmpa = np.nanmean(sgHMpsi[i]) / 145.038
        ppmpa = np.nanmean(psipp[i]) / 145.038
        bhpmpa = np.nanmean(mudpsi[i]) / 145.038
        ucsmpa = np.nanmean(horsrud[i])
        deltaP = bhpmpa - ppmpa
        sigmas = [sigmaHMaxmpa, sigmahminmpa, sigmaVmpa]
        try:
            devdoi = devdata[i]
            incdoi = devdoi[1]
            inca[i] = incdoi
            azmdoi = devdoi[2]
            osx, osy, osz = get_principal_stress(sigmas[0], sigmas[1], sigmas[2],
                alphas[i], betas[i], gammas[i])
            Sb[i] = getAlignedStress(osx, osy, osz, alphas[i], betas[i],
                gammas[i], azmdoi, incdoi)
            SbFF[i] = Sb[i]
            Sb[i][0][0] = Sb[i][0][0] - ppmpa
            Sb[i][1][1] = Sb[i][1][1] - ppmpa
            Sb[i][2][2] = Sb[i][2][2] - ppmpa
            sigmaT = ym[i] * bt[i] * delTempC[i] / (1 - nu2[i])
            Szz = np.full(360, np.nan)
            Stt = np.full(360, np.nan)
            Ttz = np.full(360, np.nan)
            Srr = np.full(360, np.nan)
            STMax = np.full(360, np.nan)
            Stmin = np.full(360, np.nan)
            ladempa = np.full(360, np.nan)
            for j in range(0, 360, 10):
                theta = np.radians(j)
                Szz[j] = Sb[i][2][2] - 2 * nu2[i] * (Sb[i][0][0] - Sb[i][1][1]
                    ) * (2 * math.cos(2 * theta)) - 4 * nu2[i] * Sb[i][0][1
                    ] * math.sin(2 * theta)
                Stt[j] = Sb[i][0][0] + Sb[i][1][1] - 2 * (Sb[i][0][0] - Sb[
                    i][1][1]) * math.cos(2 * theta) - 4 * Sb[i][0][1
                    ] * math.sin(2 * theta) - deltaP - sigmaT
                Ttz[j] = 2 * (Sb[i][1][2] * math.cos(theta) - Sb[i][0][2] *
                    math.sin(theta))
                Srr[j] = deltaP
                STMax[j] = 0.5 * (Szz[j] + Stt[j] + ((Szz[j] - Stt[j]) ** 2 +
                    4 * Ttz[j] ** 2) ** 0.5)
                Stmin[j] = 0.5 * (Szz[j] + Stt[j] - ((Szz[j] - Stt[j]) ** 2 +
                    4 * Ttz[j] ** 2) ** 0.5)
                ladempa[j] = mod_lad_cmw(SbFF[i][0][0], SbFF[i][1][1], SbFF
                    [i][2][2], SbFF[i][0][1], SbFF[i][0][2], SbFF[i][1][2],
                    j, phi[i], lal[i], psipp[i] / 145.038)
            hoopmax[i] = np.nanmax(STMax)
            hoopmin[i] = np.nanmin(Stmin)
            lademax[i] = np.nanpercentile(ladempa, mtol * 100)
            minthetarad = np.radians(np.nanargmin(Stmin))
            lademin[i] = np.nanmin(ladempa)
            if not penetration:
                trufracmpa[i] = get_frac_pressure(Sb[i], ppmpa, -horsrud[i]/arr_ten_fac[i], minthetarad, nu2[i], sigmaT) if arr_ten_fac[i]>0 else get_frac_pressure(Sb[i], ppmpa, 0, minthetarad, nu2[i], sigmaT)
            else:
                trufracmpa[i] = (3*sigmahminmpa - sigmaHMaxmpa - (biot[i]*ppmpa*((1-2*nu2[i])/(1-nu2[i]))) + sigmaT - (horsrud[i]/arr_ten_fac[i]))/(2-(biot[i]*((1-2*nu2[i])/(1-nu2[i])))) if arr_ten_fac[i]>0 else (3*sigmahminmpa - sigmaHMaxmpa - (biot[i]*ppmpa*((1-2*nu2[i])/(1-nu2[i]))) + sigmaT -0)/(2-(biot[i]*((1-2*nu2[i])/(1-nu2[i]))))
        except:
            Sb[i] = np.full((3, 3), np.nan)
            SbFF[i] = np.full((3, 3), np.nan)
            hoopmax[i] = np.nan
            hoopmin[i] = np.nan
            lademax[i] = np.nan
            lademin[i] = np.nan
            trufracmpa[i] = np.nan
    Sby = interpolate_nan(SbFF[:, 1, 1])
    Sbx = interpolate_nan(SbFF[:, 0, 0])
    hoopmin = interpolate_nan(hoopmin)
    trufracmpa = interpolate_nan(trufracmpa)
    hoopmax = interpolate_nan(hoopmax)
    Sbminmpa = np.minimum(Sby, Sbx)
    Sbmaxmpa = np.maximum(Sby, Sbx)
    Sbmingcc = Sbminmpa * 145.038 / tvdf / 0.4335275040012
    Sbmaxgcc = Sbmaxmpa * 145.038 / tvdf / 0.4335275040012
    tensilefracpsi = trufracmpa * 145.038
    if np.nanargmin(abs(tvdbgl)) > 0:
        tensilefracpsi[0:np.nanargmin(abs(tvdbgl))] = np.nan
    tensilefracpsi = np.where(tensilefracpsi < spsipp, spsipp, tensilefracpsi)
    referencepressure = tensilefracpsi[np.nanargmin(abs(tvdbgl))]
    referencedepth = tvdbglf[np.nanargmin(abs(tvdbgl))]
    deltaPressure = tensilefracpsi - referencepressure
    deltaDepth = tvdbglf - referencedepth
    if glwd < 0:
        tensilefracgcc = tensilefracpsi / hydrostaticpsi
    else:
        tensilefracgcc = deltaPressure / deltaDepth / 0.4335275040012
    tensilefracgcc = np.where(tensilefracgcc < spp, spp, tensilefracgcc)
    mogimpa = mogi(psifg / 145.038, sgHMpsi / 145.038, obgpsi / 145.038)
    ladegcc = lademax * 145.038 / tvdf / 0.4335275040012
    mogigcc = mogimpa * 145.038 / tvdf / 0.4335275040012
    ladegcc = interpolate_nan(ladegcc)
    ladegcc = np.where(ladegcc < spp, spp, ladegcc)
    if writeFile:
        plt.close()
        plt.plot(interpolate_nan(SbFF[:, 0, 0]), tvd, label='aligned sx')
        plt.plot(interpolate_nan(SbFF[:, 1, 1]), tvd, label='aligned sy')
        plt.plot(interpolate_nan(SbFF[:, 2, 2]), tvd, label='aligned sz')
        plt.plot(psifg / 145.038, tvd, alpha=0.5, label='initial shm')
        plt.plot(sgHMpsi / 145.038, tvd, alpha=0.5, label='initial sHM')
        plt.plot(obgpsi / 145.038, tvd, alpha=0.5, label='initial sV')
        plt.plot(hoopmin, tvd, alpha=0.5, label='hoopmin')
        plt.plot(hoopmax, tvd, alpha=0.5, label='hoopmax')
        plt.plot(interpolate_nan(inca), tvd, alpha=0.5, label='inclination')
        plt.plot(-horsrud / 10, tvd, alpha=0.5, label='tensile strength')
        plt.plot(tensilefracpsi / 145.038, tvd, label='fracgrad')
        #plt.plot(trufracmpa, tvd, label='trufracgrad')
        plt.plot(np.zeros(len(tvd)), tvd, alpha=0.1)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'StressPlot.png'))
        plt.close()
    print('calculation complete') if debug else None
    TVDF = Curve(tvdf, mnemonic='TVDF', units='m', index=md, null=0)
    TVDMSL = Curve(tvdmsl, mnemonic='TVDMSL', units='m', index=md, null=0)
    TVDBGL = Curve(tvdbgl, mnemonic='TVDMSL', units='m', index=md, null=0)
    TVDM = Curve(tvdm, mnemonic='TVDM', units='m', index=md, null=0)
    amoco2 = Curve(rhogcc, mnemonic='RHO', units='G/C3', index=md, null=0)
    well.data['RHOA'] = amoco2
    obgcc = Curve(ObgTgcc, mnemonic='OBG_AMOCO', units='G/C3', index=md, null=0
        )
    well.data['OBG'] = obgcc
    dtct = Curve(dtNormal, mnemonic='DTCT', units='us/ft', index=md, null=0)
    well.data['DTCT'] = dtct
    pp = Curve(spp, mnemonic='PP_GRADIENT', units='G/C3', index=md, null=0)
    well.data['PP'] = pp
    fg = Curve(sfg, mnemonic='SHmin_DAINES', units='G/C3', index=md, null=0)
    well.data['FG'] = fg
    fg2 = Curve(mufgcc, mnemonic='SHmin_ZOBACK', units='G/C3', index=md, null=0
        )
    well.data['FG2'] = fg2
    fg3 = Curve(tensilefracgcc, mnemonic='FracGrad', units='G/C3', index=md,
        null=0)
    well.data['FG3'] = fg3
    fg4 = Curve(tensilefracpsi, mnemonic='FracPressure', units='psi', index
        =md, null=0)
    well.data['FG4'] = fg4
    pppsi = Curve(spsipp, mnemonic='GEOPRESSURE', units='psi', index=md,
        null=0, index_name='DEPT')
    well.data['PPpsi'] = pppsi
    fgpsi = Curve(spsifp, mnemonic='SHmin_PRESSURE', units='psi', index=md,
        null=0)
    well.data['FGpsi'] = fgpsi
    sHMpsi = Curve(ssgHMpsi, mnemonic='SHmax_PRESSURE', units='psi', index=
        md, null=0)
    well.data['SHMpsi'] = sHMpsi
    mwpsi = Curve(mudpsi, mnemonic='MUD_PRESSURE', units='psi', index=md,
        null=0)
    well.data['mwpsi'] = mwpsi
    overburdenpsi = Curve(obgpsi, mnemonic='OVERBURDEN_PRESSURE', units=
        'psi', index=md, null=0)
    well.data['obgpsi'] = overburdenpsi
    hydrostatpsi = Curve(hydrostaticpsi, mnemonic='HYDROSTATIC_PRESSURE',
        units='psi', index=md, null=0)
    well.data['hydropsi'] = hydrostatpsi
    mhpsi = Curve(mudweight, mnemonic='MUD_GRADIENT', units='g/cc', index=
        md, null=0)
    well.data['mhpsi'] = mhpsi
    c0lalmpa = Curve(slal, mnemonic='S0_Lal', units='MPa', index=md, null=0)
    well.data['C0LAL'] = c0lalmpa
    c0lal2mpa = Curve(slal2, mnemonic='S0_Lal_Phi', units='MPa', index=md,
        null=0)
    well.data['C0LAL2'] = c0lal2mpa
    ucs_horsrudmpa = Curve(ucs_horsrud, mnemonic='UCS_horsrud', units='MPa', index
        =md, null=0)
    well.data['UCS_horsrud'] = ucs_horsrudmpa
    slalmpa = Curve(slal3, mnemonic='S0 Lal', units='MPa', index=md, null=0)
    well.data['S0_LAL'] = slalmpa
    philal = Curve(phi, mnemonic='phi_lal', units='radians', index=md, null=0)
    well.data['PHI_LAL'] = philal
    phi_lang = Curve(philang, mnemonic='phi_lang', units='radians', index=md, null=0)
    well.data['PHI_LANG'] = phi_lang
    poison = Curve(nu2, mnemonic='Poisson_Ratio', units='', index=md, null=nu)
    well.data['NU'] = poison
    modlade = Curve(ladegcc, mnemonic=f'ML{mabw}', units='g/cc', index=md,
        null=0)
    well.data['LADE'] = modlade
    youngsmod = Curve(ym, mnemonic='Youngs_Modulus', units='GPa', index=md,
        null=0)
    well.data['YM'] = youngsmod
    shearmod = Curve(sm, mnemonic='Shear_Modulus', units='GPa', index=md,
        null=0)
    well.data['SM'] = shearmod
    bulkmod = Curve(bm, mnemonic='Bulk_Modulus', units='GPa', index=md, null=0)
    well.data['BM'] = bulkmod
    output_file4 = os.path.join(output_dir1, 'GMech.las')
    output_fileCSV = os.path.join(output_dir1, 'GMech.csv')
    df3 = well.df()
    df3.index.name = 'DEPT'
    if writeFile:
        df3.to_csv(output_fileCSV)
    if 'DEPT' in df3.columns:
        df3 = df3.drop('DEPT', axis=1)
    df3 = df3.reset_index()
    header = well._get_curve_mnemonics()
    lasheader = well.header
    c_units = {'DEPT': 'm', 'MD': 'm', 'TVD': 'm', 'TVDM': 'm', 'RHO':
        'gcc', 'OBG_AMOCO': 'gcc', 'DTCT': 'US/F', 'FracGrad': 'gcc',
        'PP_GRADIENT': 'gcc', 'SHmin_DAINES': 'gcc', 'SHmin_ZOBACK': 'gcc',
        'OVERBURDEN_PRESSURE': 'psi', 'HYDROSTATIC_PRESSURE': 'psi',
        'GEOPRESSURE': 'psi', 'FracPressure': 'psi', 'SHmin_PRESSURE':
        'psi', 'SHmax_PRESSURE': 'psi', 'MUD_PRESSURE': 'psi',
        'MUD_GRADIENT': 'gcc', f'ML{mabw}': 'gcc', 'S0_Lal': 'mpa', 'phi_lal':'radians', 'phi_lang':'radians',
        'S0_Lal_Phi': 'mpa', 'UCS_horsrud': 'mpa', 'S0 Lal': 'mpa'}
    from .thirdparty import datasets_to_las
    category_columns = {'pressure': ['FracPressure', 'GEOPRESSURE',
        'SHmin_PRESSURE', 'SHmax_PRESSURE', 'MUD_PRESSURE',
        'OVERBURDEN_PRESSURE', 'HYDROSTATIC_PRESSURE'], 'strength': [
        'UCS_horsrud', 'S0 Lal'], 'gradient': ['PP_GRADIENT',
        'SHmin_DAINES', 'SHmin_ZOBACK', 'FracGrad', 'MUD_GRADIENT',
        'OBG_AMOCO', 'RHO', f'ML{mabw}'], 'length': ['DEPT', 'MD', 'TVDM']}
    from .unit_converter import convert_dataframe_units
    cdf3, cc_units = convert_dataframe_units(df3, c_units, unitdict,
        category_columns)
    print(lasheader) if debug else None
    lasheader = lasheader.drop(index=0).reset_index(drop=True)
    filestring = datasets_to_las(None, {'Header': lasheader, 'Curves':
        cdf3}, cc_units)
    if not writeFile and not display:
        return cdf3, filestring, rv1, rv2, rv3, rv4, rv5, doi, well
    """
    plt.plot(ladempa)
    plt.plot(mogimpa)
    plt.show()
    plt.close()"""
    """print(gr)
    print(dalm)
    print(dtNormal)
    print(mudweight)
    print(fg.as_numpy())
    print(pp.as_numpy())
    print(obgcc.as_numpy())
    print(fgpsi.as_numpy())
    print(ssgHMpsi)
    print(obgpsi)
    print(hydropsi)
    print(pppsi.as_numpy())
    print(mudpsi)
    print(sgHMpsiL)
    print(sgHMpsiU)
    print(slal)
    print(ucs_horsrud)"""
    results = pd.DataFrame({'dalm': dalm, 'dtNormal': dtNormal,
        'lresnormal': lresnormal, 'lresdeep': lresdeep, 'Dexp': Dexp,
        'dexnormal': dexnormal, 'mudweight': mudweight * ureg.gcc, 'shmin_grad':fg.as_numpy() * ureg.gcc,'fg': 
        fg3.as_numpy() * ureg.gcc, 'pp': pp.as_numpy() * ureg.gcc, 'sfg': 
        ladegcc * ureg.gcc, 'obgcc': obgcc.as_numpy() * ureg.gcc, 'fgpsi': 
        fg4.as_numpy() * ureg.psi, 'ssgHMpsi': ssgHMpsi * ureg.psi,
        'obgpsi': obgpsi * ureg.psi, 'hydropsi': hydropsi * ureg.psi,
        'pppsi': pppsi.as_numpy() * ureg.psi, 'mudpsi': mudpsi * ureg.psi,
        'sgHMpsiL': sgHMpsiL * ureg.psi, 'sgHMpsiU': sgHMpsiU * ureg.psi,
        'slal': slal3 * ureg.MPa, 'ucs_horsrud': ucs_horsrud * ureg.MPa, 'GR': gr,
        'GR_CUTOFF': grcut}, index=tvdm * ureg.m)

    def convert_units(data, pressure_unit, gradient_unit, strength_unit,
        ureg=ureg):
        converted_data = data.copy()
        pressure_unit = pressure_unit.lower()
        gradient_unit = gradient_unit.lower()
        strength_unit = strength_unit.lower()
        unit_mappings = {'pressure': {'psi': ureg.psi, 'ksc': ureg.ksc,
            'bar': ureg.bar, 'atm': ureg.atm, 'mpa': ureg.MPa}, 'gradient':
            {'gcc': ureg.gcc, 'sg': ureg.sg, 'ppg': ureg.ppg, 'psi/foot': 
            ureg.psi / ureg.foot, 'ksc/m': ureg.ksc / ureg.m}, 'strength':
            {'mpa': ureg.MPa, 'psi': ureg.psi, 'ksc': ureg.ksc, 'bar': ureg
            .bar, 'atm': ureg.atm}, 'depth': {'m': ureg.m, 'f': ureg.foot,
            'km': ureg.km, 'mile': ureg.mile, 'nm': ureg.nautical_mile,
            'in': ureg.inch, 'cm': ureg.cm, 'fathom': ureg.fathom}}
        pressure_columns = ['fgpsi', 'ssgHMpsi', 'obgpsi', 'hydropsi',
            'pppsi', 'mudpsi', 'sgHMpsiL', 'sgHMpsiU']
        for col in pressure_columns:
            if col in converted_data.columns:
                converted_data[col] = (converted_data[col].values * ureg.psi
                    ).to(unit_mappings['pressure'][pressure_unit])
        gradient_columns = ['mudweight', 'fg', 'pp', 'sfg', 'obgcc', 'shmin_grad']
        for col in gradient_columns:
            if col in converted_data.columns:
                converted_data[col] = (converted_data[col].values * ureg.gcc
                    ).to(unit_mappings['gradient'][gradient_unit])
        strength_columns = ['slal', 'ucs_horsrud']
        for col in strength_columns:
            if col in converted_data.columns:
                converted_data[col] = (converted_data[col].values * ureg.MPa
                    ).to(unit_mappings['strength'][strength_unit])
        converted_data.index = (converted_data.index.values * ureg.m).to(ul
            [unitchoice[0]]).magnitude
        return converted_data

    def convert_points_data(points_data, pressure_unit, gradient_unit,
        strength_unit, ureg=ureg):
        converted_points = {}
        for key, (x_vals, y_vals) in points_data.items():
            if key in ['frac_grad', 'flow_grad']:
                x_vals = (np.array(x_vals) * ureg.gcc).to(ureg(gradient_unit)
                    ).magnitude
            elif key in ['frac_psi', 'flow_psi']:
                x_vals = (np.array(x_vals) * ureg.psi).to(ureg(pressure_unit)
                    ).magnitude
            elif key == 'ucs':
                x_vals = (np.array(x_vals) * ureg.MPa).to(ureg(strength_unit)
                    ).magnitude
            converted_points[key] = x_vals, y_vals
        return converted_points
    print('unitchoice is: ', unitchoice) if debug else None
    print('unitdict is: ', ureg) if debug else None
    pressure_unit = up[unitchoice[1]]
    gradient_unit = ug[unitchoice[2]]
    print(gradient_unit) if debug else None
    strength_unit = us[unitchoice[3]]
    depth_unit = ul[unitchoice[0]]
    data = convert_units(results, pressure_unit, gradient_unit, strength_unit)
    """pd.DataFrame({
        'dalm': dalm,
        'dtNormal': dtNormal,
        'mudweight': mudweight,
        'fg': fg.as_numpy(),
        'pp': pp.as_numpy(),
        'sfg':ladegcc,
        'obgcc': obgcc.as_numpy(),
        'fgpsi': fgpsi.as_numpy(),
        'ssgHMpsi': ssgHMpsi,
        'obgpsi': obgpsi,
        'hydropsi': hydropsi,
        'pppsi': pppsi.as_numpy(),
        'mudpsi': mudpsi,
        'sgHMpsiL': sgHMpsiL,
        'sgHMpsiU': sgHMpsiU,
        'slal': slal,
        'ucs_horsrud': ucs_horsrud,
        'GR': gr,
        'GR_CUTOFF': grcut
    }, index=tvdm)"""
    styles = read_styles_from_file(minpressure, maxchartpressure,
        pressure_unit, strength_unit, gradient_unit, ureg, stylespath, writeConfig=writeConfig)
    print('max pressure is ', maxchartpressure) if debug else None

    def convert_to_tvd(y_values):
        if unitchoice[0] == 0:
            return [tvd[find_nearest_depth(md, y)[0]] for y in y_values]
        else:
            return [tvdf[find_nearest_depth(md, y)[0]] for y in y_values]

    def create_points_dataframe(points_data):
        aggregated_points = defaultdict(lambda : defaultdict(list))
        for key, (x_vals, y_vals) in points_data.items():
            y_vals_tvd = convert_to_tvd(y_vals)
            for x, y_tvd in zip(x_vals, y_vals_tvd):
                aggregated_points[y_tvd][key].append(x)
        aggregated_means = {index: {key: np.nanmean(values) for key, values in
            data.items()} for index, data in aggregated_points.items()}
        points_df = pd.DataFrame.from_dict(aggregated_means, orient='index')
        points_df = points_df.replace(0, np.nan)
        if 'ucs' not in points_df.columns:
            points_df['ucs'] = np.nan
        return points_df
    points_data = {'frac_grad': zip(*frac_grad_data), 'flow_grad': zip(*
        flow_grad_data), 'frac_psi': zip(*frac_psi_data), 'flow_psi': zip(*
        flow_psi_data)}
    print('casing points', casing_dia) if debug else None
    print('Points:', flow_grad_data) if debug else None
    if UCSs is not None:
        ucss = np.array([[depth, ucs] for ucs, depth in ucss])
        points_data['ucs'] = zip(*ucss)
    pointstyles = read_pstyles_from_file(minpressure, maxchartpressure,
        pressure_unit, strength_unit, gradient_unit, ureg, pstylespath,writeConfig=writeConfig)
    if np.any(~np.isnan(cald)) > 0 or len(casing_dia) > 1:
        print('Track 5 added') if debug else None
        styles.update({'CALIPER1': {'color': 'brown', 'linewidth': 0.5,
            'style': '-', 'track': 5, 'left': -15, 'right': 15, 'type':
            'linear', 'unit': 'in'}, 'CALIPER3': {'color': 'brown',
            'linewidth': 0.5, 'style': '-', 'track': 5, 'left': -15,
            'right': 15, 'type': 'linear', 'unit': 'in'}})
        data['CALIPER1'] = cald / 2
        data['CALIPER3'] = cald / -2
        pointstyles.update({'casingshoe': {'color': 'black', 'pointsize': 
            30, 'symbol': 1, 'track': 5, 'left': -15, 'right': 15, 'type':
            'linear', 'unit': 'in', 'uptosurface': True}, 'casingshoe2': {
            'color': 'black', 'pointsize': 30, 'symbol': 0, 'track': 5,
            'left': -15, 'right': 15, 'type': 'linear', 'unit': 'in',
            'uptosurface': True}})
        casing_dia2 = [[-x / 2, y] for x, y in casing_dia]
        casing_dia3 = [[x / 2, y] for x, y in casing_dia]
        points_data['casingshoe'] = zip(*casing_dia3)
        points_data['casingshoe2'] = zip(*casing_dia2)
    converted_points = convert_points_data(points_data, pressure_unit,
        gradient_unit, strength_unit)
    print(converted_points) if debug else None
    points_df = create_points_dataframe(converted_points)
    points_df = points_df.apply(lambda col: col.dropna())
    print(points_df) if debug else None
    dpif = 300
    if dpif < 100:
        dpif = 100
    if dpif > 900:
        dpif = 900
    figname = well.uwi if well.uwi != '' and well.uwi != None else well.name
    details={"unit":depth_unit,"type":"TVD","reference":"KB/DF","KB":float(attrib[0]),"GL":float(attrib[1])}
    fig, axes = plot_logs_labels(data, styles, y_min=(float(plotend) * ureg(
        'metre').to(ul[unitchoice[0]])).magnitude, y_max=(float(plotstart) *
        ureg('metre').to(ul[unitchoice[0]])).magnitude, width=15, height=10,
        points=points_df, pointstyles=pointstyles, dpi=dpif, output_dir=paths['output_dir'],title=figname, details=details, display=display)
    plt.close()
    #return df3, well
    return cdf3, filestring, rv1, rv2, rv3, rv4, rv5, doi, well

"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
from pint import UnitRegistry

def convert_rop(values, input_unit, target_unit='ft / hour'):
    ureg = UnitRegistry()
    ureg.define('HR = hour')
    ureg.define('M = meter')
    
    # Create a Quantity object with the input values and units
    q = ureg.Quantity(values, input_unit)
    
    # Check if the input is in reciprocal units (e.g., min/m)
    if '/' in input_unit and input_unit.split('/')[0].strip() in ['min', 'minute', 'hour', 'hr', 'h']:
        # If so, invert the quantity
        q = 1 / q
        
        # Construct the correct unit for the inverted quantity
        inverted_unit = f"{input_unit.split('/')[-1].strip()} / {input_unit.split('/')[0].strip()}"
        q = q.to(inverted_unit)
    
    # Convert to the target unit
    result = q.to(target_unit)
    
    return result.magnitude
"""
# Example usage
rop_values = np.array([27, 30, 45, 53])  # Example ROP values
input_unit = 'minute / meter'  # This could also be 'meter / hour'

converted_values = convert_rop(rop_values, input_unit)

print(f"Input ROP values: {rop_values} {input_unit}")
print(f"Converted ROP values: {converted_values} meter / hour")

# Test with the other unit
rop_values_2 = np.array([2.22, 2.5, 2.0, 1.8])
input_unit_2 = 'M/HR'

converted_values_2 = convert_rop(rop_values_2, input_unit_2)

print(f"\nInput ROP values: {rop_values_2} {input_unit_2}")
print(f"Converted ROP values: {converted_values_2} meter / hour")
"""

def convert_wob(values, input_unit, target_unit='lb'):
    ureg = UnitRegistry()
    
    # Define custom unit aliases if needed
    ureg.define('MTON = metric_ton')
    ureg.define('TON = ton')
    
    # Create a Quantity object with the input values and units
    q = ureg.Quantity(values, input_unit)
    
    # Convert to the target unit
    result = q.to(target_unit)
    
    return result.magnitude
"""
# Example usage
wob_values = np.array([10, 15, 20, 25])  # Example WOB values in tons
input_unit = 'TON'  # This could also be 'MTON'

converted_wob = convert_wob(wob_values, input_unit)

print(f"Input WOB values: {wob_values} {input_unit}")
print(f"Converted WOB values: {converted_wob} lb")

# Test with metric tons (Mtons)
wob_values_2 = np.array([1, 1.5, 2, 2.5])  # Example WOB values in metric tons
input_unit_2 = 'MTON'

converted_wob_2 = convert_wob(wob_values_2, input_unit_2)

print(f"\nInput WOB values: {wob_values_2} {input_unit_2}")
print(f"Converted WOB values: {converted_wob_2} lb")
"""

import numpy as np
from pint import UnitRegistry

def convert_ecd(values, input_unit, target_unit='SG'):
    ureg = UnitRegistry()
    
    # Define custom unit aliases
    ureg.define('ppg = 0.051948 psi/foot')
    ureg.define('sg = 0.4335 psi/foot')
    ureg.define('gcc = SG')
    ureg.define('SG = sg')
    ureg.define('GCC = sg')
    
    # Create a Quantity object with the input values and units
    q = ureg.Quantity(values, input_unit)
    
    # Convert to the target unit
    result = q.to(target_unit)
    
    return result.magnitude
"""
# Example usage
mudweight_values = np.array([10, 12, 14, 16])  # Example mudweight values in ppg
input_unit = 'ppg'  # This could also be 'psi/ft'

converted_mudweight = convert_mudweight(mudweight_values, input_unit)

print(f"Input mudweight values: {mudweight_values} {input_unit}")
print(f"Converted mudweight values: {converted_mudweight} SG")

# Test with psi/ft
mudweight_values_2 = np.array([0.4335, 0.5, 0.6, 0.7])  # Example mudweight values in psi/ft
input_unit_2 = 'psi/ft'

converted_mudweight_2 = convert_mudweight(mudweight_values_2, input_unit_2)

print(f"\nInput mudweight values: {mudweight_values_2} {input_unit_2}")
print(f"Converted mudweight values: {converted_mudweight_2} SG")
"""

def convert_flowrate(values, input_unit, target_unit='gallon/minute'):
    ureg = UnitRegistry()
    
    # Define custom unit aliases if needed
    ureg.define('GPM = gallon/minute')
    ureg.define('LPM = liter/minute')
    
    # Create a Quantity object with the input values and units
    q = ureg.Quantity(values, input_unit)
    
    # Convert to the target unit
    result = q.to(target_unit)
    
    return result.magnitude
"""
# Example usage
flowrates_lpm = [100, 200, 300]  # Example values
flowrates_gpm = convert_flowrate(flowrates_lpm, 'LPM')
print(flowrates_gpm)
"""


def convert_torque(values, input_unit, target_unit='g_0 *foot*pound'):
    ureg = UnitRegistry()
    
    # Define custom unit aliases if needed
    ureg.define('FTLB = g_0 *foot*pound')
    ureg.define('NM = newton*meter')
    ureg.define('KJ = kilojoule')
    
    # Create a Quantity object with the input values and units
    q = ureg.Quantity(values, input_unit)
    
    # Convert to the target unit
    result = q.to(target_unit)
    
    return result.magnitude

#Example usage
torque_KJ = [100, 200, 300]  # Example values in KJ
torque_ftlbs = convert_torque(torque_KJ, 'KJ')
print(torque_ftlbs)

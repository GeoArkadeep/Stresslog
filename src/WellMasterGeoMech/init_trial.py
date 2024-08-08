from .app import main as run_gui_app
from .geomechanics import plotPPzhang

# You can also import everything from a module using '*'
# from .module2 import *

# If you want to rename something as you import it:
from .module1 import internal_function as public_function

# Define what should be available when someone does `from your_library import *`
__all__ = ['function1', 'Class1', 'function2', 'function3', 'Class3', 'public_function']

# You can also define or modify functions here
def convenience_function():
    """A convenience function that uses multiple modules"""
    result = function1()
    return function2(result)

# Version information
__version__ = "1.1.2"
from .geomechanics import plotPPzhang
from .app import main as run_gui_app

def calc_geomechanics(well: welly.Well):
    """
    Calculate geomechanics and plot the results using the plotPPzhang function.
    
    Parameters:
    well (welly.Well): The well object containing the well log data.
    """
    plotPPzhang(well)

def rungui():
    """
    Launch the GUI application.
    """
    run_gui_app().main_loop()


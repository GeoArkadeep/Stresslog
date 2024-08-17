__version__ = "1.0.4"
__author__ = "Your Name"

from .BoreStab import *
from .DrawSP import *
from .failure_criteria import *
from .obgppshmin import *
from .hydraulics import *
from .plotter import *
from .geomechanics import plotPPzhang

# Optionally, include a function to show docs for key functions
def show_docs():
    """
    Show documentation for the key functions in the library.
    """
    from .BoreStab import *
    from .DrawSP import *
    from .failure_criteria import *
    from .obgppshmin import *
    from .hydraulics import *
    from .plotter import *
    from .geomechanics import plotPPzhang
    
    modules = [BoreStab, DrawSP, failure_criteria, obgppshmin, hydraulics, plotter]
    for module in modules:
        functions = [func for func in dir(module) if callable(getattr(module, func))]
        for func in functions:
            print(f"{func}:")
            print(getattr(module, func).__doc__)
            print()

    print("plotPPzhang:")
    print(plotPPzhang.__doc__)

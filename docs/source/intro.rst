Stresslog
=========

**Stresslog** is a Python package designed to facilitate geomechanical analysis from well logs. It offers a comprehensive suite of tools to compute various geomechanical parameters, aiding in the assessment and modeling of subsurface stress conditions.

Key Features
------------

- **Comprehensive Calculations**: Computes a wide range of geomechanical parameters, including stress tensors, pore pressure, and fracture gradients.

- **Customization**: Allows users to input specific well attributes and deviation data to tailor the analysis to their datasets.

- **Integration with Welly**: Utilizes the `welly` library for seamless handling of LAS files and well data.

Installation
------------

To install **Stresslog**, ensure you have Python 3.10 or 3.11 installed. You can install the package using `pip`:

.. code-block:: bash

    pip install stresslog

For detailed installation instructions and dependencies, please refer to the [Stresslog documentation](https://stresslog.readthedocs.io).

Basic Usage
-----------

Below is a basic example demonstrating how to use **Stresslog** to perform geomechanical analysis on a well log:

.. code-block:: python

    import stresslog as lst
    from welly import Well

    # Load your well data from a LAS file
    well = Well.from_las('path/to/your/well.las')

    # If deviation data is available as a DataFrame
    import pandas as pd
    deviation_data = pd.read_csv('path/to/deviation_data.csv')  # Ensure columns are MD, inclination, azimuth
    well.data['deviation'] = deviation_data

    # Define well attributes (e.g., KB, GL, etc.)
    attrib = [10, 0, 0, 0, 0, 0, 0, 0]  # Customize as needed

    # Perform geomechanical analysis
    output = lst.compute_geomech(well, attrib=attrib, writeFile=False)

    # The 'output' contains the computed geomechanical parameters

In this example:

- We load a well from a LAS file using the `welly` library.
- Deviation data is added to the well object. The deviation data should be provided as a DataFrame with columns: Measured Depth (MD), inclination, and azimuth, in that order.
- Well attributes are defined in the `attrib` list. These attributes can include parameters like Kelly Bushing (KB) height, Ground Level (GL), and others as required.
- The `compute_geomech` function is called to perform the geomechanical analysis. The `writeFile` parameter is set to `False` to prevent writing output to a file.

For a comprehensive list of parameters and detailed explanations, please refer to the [Stresslog API documentation](https://stresslog.readthedocs.io).

Footnote
--------

*Note: While **Stresslog** includes functions to generate synthetic well logs for testing purposes, it is intended for users to analyze their own well data in practice.*


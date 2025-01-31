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

For developers and pythonistas, you can download the repository from github and install from source as follows:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/GeoArkadeep/Stresslog.git

   # Change to the project directory
   cd Stresslog

   # Install in editable mode
   pip install -e .

Basic Usage
-----------

Below is a basic example demonstrating how to use **Stresslog** to perform geomechanical analysis on a well log:

.. code-block:: python

    import stresslog as lst
    from welly import Well

    # Load your well data from a LAS file

    well = Well.from_las('path/to/your/well.las')

    # Simplest use case: Vertical well with no deviation data at full resolution (will take a fairly long time, enough for a coffee break)
    # This will also write output files to the default directories. If you don't want that, set writeFile=False
    # This will assume all the optional parameters are at their default settings

    output = lst.compute_geomech(well)

    # The 'output' contains the following:
    # - output[0]: Well log DataFrame (original and computed values) with mnemonics as headers
    # - output[1]: LAS file as StringIO object containing original and computed values
    # - output[2] to output[6]: Base64 encoded plot strings for properties calculated at depth of interest (or None if written to files or not calculated at doi=0)
    # - output[7]: Depth of Interest as specified (in meters)
    # - output[8]: Welly object containing all data

.. code-block:: python

    # If deviation data is available as a DataFrame

    import pandas as pd
    # Ensure first three columns are MD, inclination, azimuth
    deviation_data = pd.read_csv('path/to/deviation_data.csv')

    # Next we create a new welly object by combining the original welly object and the deviation data
    # Instead of the welly object, we can also pass the las file directly as a stringIO object using the parameter string_las)
    # We can also resample the welly object at this step. Here we resample it to every 10 metres

    wellwithdeviation = lst.getwelldev(wella=well,deva=deviation_data,step=10)

    # Define well attributes (e.g., KB, GL, etc.)
    attrib = [30, -120, 0, 0, 0, 0, 0, 0]  # Customize as needed

    # Perform geomechanical analysis. This time it will much faster, few seconds maybe
    output = lst.compute_geomech(wellwithdeviation, attrib=attrib, writeFile=False)


In this example:

- We load a well from a LAS file using the `welly` library.
- Deviation data is added to the well object. The deviation data should be provided as a DataFrame with columns: measured depth, inclination, and azimuth, in that order.
- Well attributes are defined in the `attrib` list. These attributes can include parameters like Kelly Bushing (KB) height, Ground Level (GL), and others as required.
- The `compute_geomech` function is called to perform the geomechanical analysis. The `writeFile` parameter is set to `False` to prevent writing output to files.

For a comprehensive list of parameters and detailed explanations, please refer to the api documentation.



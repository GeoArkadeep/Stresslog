Real World Example: Eos Well
----------------------------

This case study demonstrates the iterative process of geomechanical analysis using the Northern Lights dataset (courtesy of Equinor). We'll explore how different modeling assumptions affect our results and show the importance of calibrating models with observed data.

#Initial Setup

First, let's explore the necessary environment setup and data structures:

.. code-block:: python

    import stresslog as lst
    from welly import Well
    import pandas as pd

#Loading Well Data

Here's how we load our well data and supporting datasets:

.. code-block:: python

    alias = {
        "sonic": ["none", "DTC", "DT24", "DTCO", "DT", "AC", "AAC", "DTHM"],
        "ssonic": ["none", "DTSM","DTSH_FINAL"],
        "gr": ["none", "GR", "GRD", "CGR", "GRR", "GRCFM","GR_EDTC"],
        "resdeep": ["none", "HDRS", "LLD", "M2RX", "MLR4C", "RD", "RT90", "RLA1", "RDEP", "RLLD", "RILD", "ILD", "RT_HRLT", "RACELM"],
        "resshal": ["none", "LLS", "HMRS", "M2R1", "RS", "RFOC", "ILM", "RSFL", "RMED", "RACEHM", "RXO_HRLT"],
        "density": ["none", "ZDEN", "RHOB", "RHOZ", "RHO", "DEN", "RHO8", "BDCFM"],
        "neutron": ["none", "CNCF", "NPHI", "NEU", "TNPH", "NPHI_LIM"],
        "pe": ["none", "PEFLA", "PEF8", "PE"]
    }

    # Load well log data
    string_las1 = lst.get_las_from_dlis('WL_RAW_AAC-ARLL-CAL-DEN-GR-NEU_RUN6_EWL_2.DLIS', aliases=alias, step=0.147)
    # we could have used aliases=None (which is the default) but that would have returned ALL the channels in the dlis creating a huge las file which slows the analysis somewhat.
    vertwell = Well.from_las(string_las1)
    # Load supporting data
    survey = pd.read_csv('Deviation.csv')
    formations = pd.read_csv('NorthernLights-31_5-7.csv')
    ucs = pd.read_csv('UCSdata.csv')

Analysis Iteration 1: Perfect Vertical Well
-----------------------------------------

Our first analysis assumes a perfectly vertical well:

.. code-block:: python

    # Set up mud KB, GL, BHT and LOT values
    attrib = [50, -307, 0, 0, 0, 100, 0, 0]
    xlot = [[1.43, 2582.9]]
    # Create vertical well model
    wellwithoutdeviation = lst.getwelldev(wella=vertwell, deva=None)
    # Run initial analysis
    output = lst.compute_geomech(
        wellwithoutdeviation, 
        attrib=attrib,
        rhoappg=17.33,
        a=0.8,
        lamb=0.00075,
        forms=formations,
        UCSs=ucs,
        writeFile=True,
        user_home="./output",
        offset=91,
        dip_dir=180,
        dip=2,
        doi=2627.5,
        mwvalues=[[1.26, 0.0, 0.0, 0.0, 0.0, 0]],
        plotstart=2560,
        plotend=2660,
        mudtemp=50,
        fracgradvals=xlot,
        ten_fac=2000
    )

In this first run, we've made several key assumptions:

- The well is perfectly vertical
- The SHmax azimuth is 91 degrees
-The stress tensor is tilted 2 degrees to the south

The results can be found in the ./output/Stresslog_Plots directory, where PlotAll.png shows the Zobackogram, stability plot, sanding risk plot, and synthetic borehole image.

Analysis Iteration 2: Incorporating Well Deviation
-----------------------------------------------

Looking at the survey data, we notice that the well isn't perfectly vertical. At 2621.97m, there's a slight deviation with an inclination of 0.60° at an azimuth of 40.11°. Could this slight departure from verticality explain the en-echelon fractures we observe?

.. code-block:: python

    # Create deviated well model
    wellwithdeviation = lst.getwelldev(wella=Well.from_las(string_las1), deva=survey)
    # Run analysis with deviation but no stress tensor tilt
    output = lst.compute_geomech(
        wellwithdeviation,
        attrib=attrib,
        rhoappg=17.33,
        lamb=0.00075,
        forms=formations,
        UCSs=ucs,
        writeFile=True,
        user_home="./output0",
        offset=91,
        dip_dir=180,
        dip=0,
        doi=2627.5,
        mwvalues=[[1.26, 0.0, 0.0, 0.0, 0.0, 0]],
        plotstart=2560,
        plotend=2660,
        mudtemp=35,
        fracgradvals=xlot
    )

We observe that this model produces fractures with closure directions opposite to what we see in the actual image logs. This suggests our assumption about well deviation being the primary factor might be incorrect.

Analysis Iteration 3: Reintroducing Stress Tensor Tilt
------------------------------------------------------

Let's try reintroducing the stress tensor tilt while keeping the well deviation:

.. code-block:: python

    output = lst.compute_geomech(
        wellwithdeviation,
        attrib=attrib,
        rhoappg=17.33,
        lamb=0.00075,
        forms=formations,
        UCSs=ucs,
        writeFile=True,
        user_home="./output1",
        offset=91,
        dip_dir=180,
        dip=2,
        doi=2627.5,
        mwvalues=[[1.26, 0.0, 0.0, 0.0, 0.0, 0]],
        plotstart=2560,
        plotend=2660,
        mudtemp=35,
        fracgradvals=xlot
    )

This corrects the closure direction, but now the fracture alignment is incorrect. The results suggest we need an SHmax azimuth above 100°, closer to 120°.

Analysis Iteration 4: Using Log-Derived SHmax Azimuth
-----------------------------------------------------

Digging deeper into the log data, we discover there's actually a proxy for SHmax azimuth in the log itself:

.. code-block:: python

    # Extract SHmax azimuth from log data
    y = lst.get_dlis_data('WL_RAW_AAC-ARLL-CAL-DEN-GR-NEU_RUN6_EWL_2.DLIS')
    z = y[0]["FSH_AZIM_OVERALL"]
    unwrapped_z = z.where(z >= 0, z + 180)

    # Plot the azimuth values
    from matplotlib import pyplot as plt
    plt.plot(unwrapped_z)
    plt.savefig('SHmax_Azim.png')

The log data suggests values around 114°. Let's incorporate this into our model:

.. code-block:: python
    # Final analysis with updated parameters
    output = lst.compute_geomech(
        wellwithdeviation,
        attrib=attrib,
        rhoappg=17.33,
        lamb=0.00075,
        forms=formations,
        UCSs=ucs,
        writeFile=True,
        user_home="./output2",
        offset=114,
        dip_dir=180,
        dip=2,
        doi=2627.5,
        mwvalues=[[1.26, 0.0, 0.0, 0.0, 0.0, 0]],
        plotstart=2560,
        plotend=2660,
        mudtemp=35,
        fracgradvals=xlot,
        ten_fac=0
    )

Discussion and Limitations
--------------------------

This final model provides a much better match with the recorded data. However, there are some important caveats to consider:

The SHmax_Azim values in the log actually range from 90° to 125° in the interval containing the fractures.
If these varying azimuths were accurate, we would expect to see considerable variation in fracture position, which is not observed in the data.

This case study illustrates the complexity of real-world geomechanical analysis. Different models can provide reasonable fits to the data, and choosing between them often requires careful consideration of geological context and the relative importance of different observations.
The final choice of model parameters should be based on:

- Match to observed fracture patterns
- Consistency with regional stress patterns
- Geological reasonableness
- Understanding of measurement uncertainties

Remember that in real-world applications, the "correct" model may not be immediately obvious, and multiple interpretations might be equally valid given the available data.
```
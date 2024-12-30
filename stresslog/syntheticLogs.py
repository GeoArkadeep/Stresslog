"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import random
from io import StringIO

from .thirdparty import datasets_to_las

# Option 1: Using set_printoptions to control decimal places
np.set_printoptions(precision=3, suppress=True)  # suppress will prevent scientific notation



def random_curves_dataframe(
    start_depth, stop_depth, step,
    matrick=60, mudline=210, dt_ncts=0.0008,
    rhoappg=17, a=0.08, water=8.6, mudweight=9.8, glwd=-1, seed=None, drop=[]
):
    """Generate synthetic well log data with random variations.

    Parameters
    ----------
    start_depth : float
        Starting depth for the well log in meters
    stop_depth : float
        Ending depth for the well log in meters
    step : float
        Depth increment between measurements in meters
    matrick : float, optional
        Matrix sonic transit time in us/ft, by default 60
    mudline : float, optional
        Mudline sonic transit time in us/ft, by default 210
    dt_ncts : float, optional
        Compaction trend coefficient, by default 0.0008
    rhoappg : float, optional
        Apparent density gradient in ppg, by default 17
    a : float, optional
        Density compaction coefficient, by default 0.08
    water : float, optional
        Water density in ppg, by default 8.6
    mudweight : float, optional
        Mud weight in ppg, by default 9.8
    glwd : float, optional
        Ground level water depth in meters, by default -1
    seed : int, optional
        Random seed for reproducibility, by default None
    drop : list, optional
        List of curve names to exclude from output, by default []

    Returns
    -------
    pandas.DataFrame
        DataFrame containing synthetic well log curves including:
        - DEPT: Depth
        - DTCO: Compressional sonic
        - RHOB: Bulk density
        - GR: Gamma ray
        - NU: Poisson's ratio
        - DTS: Shear sonic
        - NPHI: Neutron porosity
        - ILD: Induction log deep resistivity
    """

    if seed is None:
        seed = random.randint(0, 1000000)
        print("Seed: ", seed)
    dtco_seed = gr_seed = nu_seed = rhob_seed = nphi_seed = ild_seed = seed

    # Set up random generators
    dtco_rng = np.random.default_rng(dtco_seed)
    gr_rng = np.random.default_rng(gr_seed)
    nu_rng = np.random.default_rng(nu_seed)
    rhob_rng = np.random.default_rng(rhob_seed)
    nphi_rng = np.random.default_rng(nphi_seed)
    ild_rng = np.random.default_rng(ild_seed)

    # Generate DEPT column
    DEPT = np.arange(start_depth, stop_depth, step)

    # Calculate DTCO with controlled randomness
    DTCO = (
        matrick
        + (mudline - matrick) * np.exp(-dt_ncts * DEPT)
        + np.clip(np.cumsum(dtco_rng.normal(0, 5, len(DEPT))) * 0.1, -50, 50)
    )

    # Calculate RHOB
    rhoppg = np.zeros_like(DEPT)
    rhogcc = np.zeros_like(DEPT)

    for i in range(len(DEPT)):
        tvdbgl = DEPT[i] - glwd
        tvdmsl = DEPT[i]

        if glwd < 0:
            if tvdbgl >= 0:
                rhoppg[i] = rhoappg + ((DEPT[i] / 3125) ** a)
                rhogcc[i] = (0.11982642731 * rhoppg[i]) #+ np.clip(np.cumsum(rhob_rng.normal(0, 5, 1)) * 0.005, -0.09, 0.09)
            else:
                if tvdmsl < 0:
                    rhoppg[i] = 8.34540426515252 * water
                    rhogcc[i] = 0.11982642731 * rhoppg[i] 
                else:
                    rhoppg[i] = 0
                    rhogcc[i] = 0
        else:
            if tvdbgl >= 0:
                rhoppg[i] = rhoappg + ((tvdbgl / 3125) ** a)
                rhogcc[i] = (0.11982642731 * rhoppg[i]) #+ np.clip(np.cumsum(rhob_rng.normal(0, 5, 1)) * 0.005, -0.09, 0.09)
            else:
                rhoppg[i] = 0
                rhogcc[i] = 0

    RHOB = rhogcc + np.clip(np.cumsum(rhob_rng.normal(0, 5, len(DEPT))) * 0.000009, -0.009, 0.009)

    # Generate GR with controlled randomness
    GR = np.clip(np.cumsum(gr_rng.normal(0, 5, len(DEPT))) * 0.1, 0, 150)

    # Generate NU with controlled randomness
    NU = np.clip(nphi_rng.normal(0.25, 0.075, len(DEPT)), 0.15, 0.45)#np.clip(np.cumsum(nu_rng.normal(0, 5, len(DEPT))) * 0.00085, 0.2, 0.40)

    # Derived columns: Vp, Vs, and DTS
    Vp = 1 / DTCO
    Vs = np.sqrt(((Vp ** 2) * (2 * NU - 1)) / (2 * (NU - 1)))
    DTS = 1 / Vs
    
    # Generate NPHI and ILD with controlled randomness
    NPHI = np.clip(0.5 - nphi_rng.normal(0, 0.05, len(DEPT)), 0, 1)  # Normalized porosity
    ILD = np.clip(10 ** ild_rng.normal(1, 0.2, len(DEPT)), 0.02, 200)  # Log-normal resistivity

    # Create DataFrame
    data = {
        "DEPT": DEPT,
        "DTCO": DTCO,
        "RHOB": RHOB,
        "GR": GR,
        "NU": NU,
        "DTS": DTS,
        "NPHI": NPHI,
        "ILD": ILD
    }
    # Drop specified curves
    data = {key: value for key, value in data.items() if key not in drop}
    df = pd.DataFrame(data)
    return df



def create_header(name, uwi, strt, stop, step, 
                     comp=None, fld=None, loc=None, cnty=None, stat=None, ctry=None, 
                     api=None, date=None, srvc=None, lati=None, long=None, gdat=None, kb = "1", gl = "0", nully="-999.25"):
    """Create a LAS file header with well information.

    Parameters
    ----------
    name : str
        Well name
    uwi : str
        Unique Well Identifier
    strt : str
        Start depth
    stop : str
        Stop depth
    step : str
        Depth step size
    comp : str, optional
        Company name
    fld : str, optional
        Field name
    loc : str, optional
        Location description
    cnty : str, optional
        County name
    stat : str, optional
        State name
    ctry : str, optional
        Country name
    api : str, optional
        API number
    date : str, optional
        Log date (DD-MMM-YYYY format)
    srvc : str, optional
        Service company name
    lati : str, optional
        Latitude (in degrees)
    long : str, optional
        Longitude (in degrees)
    gdat : str, optional
        Geodetic datum
    kb : str, optional
        Kelly bushing elevation in meters, by default "1"
    gl : str, optional
        Ground level elevation in meters, by default "0"
    nully : str, optional
        Null value indicator, by default "-999.25"

    Returns
    -------
    pandas.DataFrame
        DataFrame containing formatted LAS file header information
    """
    # Core mandatory content
    csv_parts = [
        """\
original_mnemonic,mnemonic,unit,value,descr,section
WRAP,WRAP,,NO,One Line per depth step,Version
PROD,PROD,,RockLab,LAS Producer,Version
PROG,PROG,,DLIS to ASCII 2.3,LAS Program name and version,Version
CREA,CREA,,2023/07/13 12:42                          :LAS Creation date {{YYYY/MM/DD hh,mm}},Version
"""
    ]

    # Conditional content based on optional parameters
    
    if strt:
        csv_parts.append(f"STRT,STRT,M,{strt},START DEPTH,Well")
    if stop:
        csv_parts.append(f"STOP,STOP,M,{stop},STOP DEPTH,Well")
    if step:
        csv_parts.append(f"STEP,STEP,M,{step},STEP,Well")
    if nully:
        csv_parts.append(f"NULL ,NULL , ,{nully},NULL VALUE,Well")
    if comp:
        csv_parts.append(f"COMP,COMP, ,{comp},COMPANY,Well")
    if name:
        csv_parts.append(f"WELL,WELL, ,{name},WELL NAME,Well")
    if uwi:
        csv_parts.append(f"UWI,UWI, ,{uwi},WELL UWI,Well")
    if fld:
        csv_parts.append(f"FLD,FLD, ,{fld},FIELD,Well")
    if loc:
        csv_parts.append(f"LOC,LOC, ,{loc},LOCATION,Well")
    if cnty:
        csv_parts.append(f"CNTY,CNTY, ,{cnty},COUNTY,Well")
    if stat:
        csv_parts.append(f"STAT,STAT, ,{stat},STATE,Well")
    if ctry:
        csv_parts.append(f"CTRY,CTRY, ,{ctry},COUNTRY,Well")
    if api:
        csv_parts.append(f"API,API, , ,API NUMBER,Well")
    if date:
        csv_parts.append(f"DATE,DATE, ,{date},LOG DATE {{DD-MMM-YYYY}},Well")
    if srvc:
        csv_parts.append(f"SRVC,SRVC, ,{srvc},SERVICE COMPANY,Well")
    if lati:
        csv_parts.append(f"LATI,LATI,DEG,\"{lati}\",LATITUDE,Well")
    if long:
        csv_parts.append(f"LONG,LONG,DEG,\"{long}\",LONGITUDE,Well")
    if gdat:
        csv_parts.append(f"GDAT,GDAT, ,{gdat},GeoDetic Datum,Well")
    if kb:
        csv_parts.append(f"EKB,GDAT,m ,{kb},Elevation of Kelly Bushing,Well")
        csv_parts.append(f"KB,GDAT,m ,{kb},Elevation of Kelly Bushing,Well")
    if gl:
        csv_parts.append(f"EGL,GDAT,m ,{gl},Elevation of Ground Level,Well")
        csv_parts.append(f"GL,GDAT,m ,{gl},Elevation of Ground Level,Well")
    
    
    # Static curve information
    csv_parts.append("""\
DEPT,DEPT,M,,DEPTH (BOREHOLE),Curves
DTCO,DTCO,US/F,,Delta-T Compressional,Curves
RHOB,RHOB,G/CC,,Bulk Density,Curves
GR,GR,GAPI,,Gamma Ray,Curves
NU,NU, ,,POISSON RATIO,Curves
DTS,DTS,US/F,,Delta-T Shear,Curves""")

    # Join all parts into one CSV string
    csv_content = "\n".join(csv_parts)
    #print(csv_content)
    # Create DataFrame from the CSV string
    df = pd.read_csv(StringIO(csv_content))
    return df



def create_random_las(
    starter=0, 
    stopper=5800, 
    stepper=0.15, 
    key=None, 
    well_name="WonderWell", 
    company="Wonder", 
    field="Brown", 
    location="Morrdor", 
    county="Shire", 
    latitude="60° 34' 35.15\" N", 
    longitude="3° 26' 36.15\" E", 
    geodetic_datum="WGS84",
    writeFile=False,
    debug=False,
    kb=2,
    gl=0,
    drop=[],
):
    """
    Process well log data by generating a DataFrame, creating a header, plotting, and optionally writing a LAS file.
    
    Parameters:
    -----------
    starter : float, optional
        Starting depth of the well log. Default is 0.
    stopper : float, optional
        Stopping depth of the well log. Default is 2800.
    stepper : float, optional
        Step size for depth intervals. Default is 0.15.
    key : int, optional
        Random seed for data generation. If None, a random seed will be generated.
    well_name : str, optional
        Name of the well. Default is "WonderWell".
    company : str, optional
        Company name. Default is "Wonder".
    field : str, optional
        Field name. Default is "Brown".
    location : str, optional
        Location name. Default is "Morrdor".
    county : str, optional
        County name. Default is "Shire".
    latitude : str, optional
        Latitude of the well. Default is "60° 34' 35.15\" N".
    longitude : str, optional
        Longitude of the well. Default is "3° 26' 36.15\" E".
    geodetic_datum : str, optional
        Geodetic datum. Default is "WGS84".
    writeFile : bool, optional
        If True, writes LAS file to disk. If False, returns string buffer. Default is True.
    
    Returns:
    --------
    If writeFile is True:
        tuple: (DataFrame with well log data, Header DataFrame, Filename of LAS file)
    If writeFile is False:
        tuple: (DataFrame with well log data, Header DataFrame, StringIO buffer of LAS file)
    """
    # Set random seed if not provided
    if key is None:
        key = random.randint(0, 1000)
    
    # Generate DataFrame (assuming generate_dataframe is a predefined function)
    df = random_curves_dataframe(starter, stopper, stepper, seed=key, drop=drop)
    
    # Create header DataFrame
    hdf = create_header(
        name=f"{well_name}-{key}", 
        uwi=f"WELL#{key}", 
        strt=str(starter), 
        stop=str(stopper), 
        step=str(stepper), 
        comp=company, 
        fld=field, 
        loc=location, 
        cnty=county,
        lati=latitude, 
        long=longitude, 
        gdat=geodetic_datum,
        kb=kb,
        gl=gl,
    )
    if writeFile or debug:
        # Determine columns to plot (excluding DEPT)
        columns_to_plot = [col for col in df.columns if col != 'DEPT']
        
        # Columns to reverse x-axis for
        reverse_x_columns = ['DTS', 'DTCO', 'NPHI']
        
        # Calculate subplot layout
        num_cols = len(columns_to_plot)
        
        with plt.xkcd():
            # Create side-by-side subplots
            plt.figure(figsize=(2 * num_cols, 6))
            
            for i, col in enumerate(columns_to_plot, 1):
                plt.subplot(1, num_cols, i)
                
                # Plot the data
                plt.plot(df[col], df['DEPT'])
                
                # Invert y-axis
                plt.gca().invert_yaxis()
                
                # Conditionally reverse x-axis for specific columns
                if col in reverse_x_columns:
                    plt.gca().invert_xaxis()
                
                plt.xlabel(col)
                plt.ylabel(None)
                plt.title(f"{col}")
            
            plt.tight_layout()
            
            plt.show()
    
    # Convert to LAS file
    las_string = datasets_to_las(None, {'Curves': df, 'Header': hdf}, 
                                 {'DEPT': 'm', 'DTCO': 'us/f', 'RHOB': 'g/cc', 'GR': 'gAPI', 'DTS': 'us/f'})
    
    if writeFile:
        # Write LAS file to disk
        filename = f"randomwonder{key}.las"
        with open(filename, 'w') as file:
            file.write(las_string)
        print(f"LAS file has been written to {filename}")
        return df, hdf, filename
    else:
        # Return string buffer instead of writing to file
        string_buffer = StringIO(las_string)
        return df, hdf, string_buffer
    
from welly import Well

def getwelldev(string_las=None, wella=None, deva=None, kickoffpoint=None, final_angle=None, rateofbuild=None, azimuth=None):
    """Calculate well deviation data and update well object with TVD information.

    Parameters
    ----------
    string_las : str, optional
        LAS file content as string
    wella : welly.Well, optional
        Well object to process
    deva : pandas.DataFrame, optional
        Existing deviation data with columns 'MD', 'INC', 'AZIM'
    kickoffpoint : float, optional
        Depth at which well deviation begins
    final_angle : float, optional
        Target inclination angle in degrees
    rateofbuild : float, optional
        Rate of angle build in degrees per meter
    azimuth : float, optional
        Well azimuth in degrees

    Returns
    -------
    welly.Well
        Updated well object with deviation data and TVD calculations

    Notes
    -----
    Either provide string_las or wella, and either deva or kickoffpoint+final_angle+rateofbuild+azimuth
    """
    if wella is None:
        wella = Well.from_las(string_las, index="m")
    
    depth_track = wella.df().index

    if deva is None and kickoffpoint is not None and final_angle is not None and rateofbuild is not None and azimuth is not None:
        start_depth = depth_track[0]
        final_depth = depth_track[-1]
        spacing = (final_depth - start_depth) / len(depth_track)

        # Create a depth array with the same spacing
        md = np.arange(start_depth, final_depth + spacing, spacing)
        inc = np.zeros_like(md)
        azm = np.full_like(md, azimuth, dtype=float)

        # Calculate inclination from kickoff point onwards
        for i, depth in enumerate(md):
            if depth >= kickoffpoint:
                increment = (depth - kickoffpoint) * rateofbuild
                inc[i] = min(increment, final_angle)  # Cap inclination at final_angle

        deva = pd.DataFrame({"MD": md, "INC": inc, "AZIM": azm})

    if deva is not None:
        start_depth = depth_track[0]
        spacing = (depth_track[-1] - depth_track[0]) / len(depth_track)
        padlength = int(start_depth / spacing)
        padval = np.zeros(padlength)

        for i in range(1, padlength):
            padval[-i] = start_depth - (spacing * i)

        md = np.append(padval, depth_track)
        mda = pd.to_numeric(deva["MD"], errors="coerce")
        inca = pd.to_numeric(deva["INC"], errors="coerce")
        azma = pd.to_numeric(deva["AZIM"], errors="coerce")
        inc = np.interp(md, mda, inca)
        azm = np.interp(md, mda, azma)
    else:
        start_depth = depth_track[0]
        spacing = (depth_track[-1] - depth_track[0]) / len(depth_track)
        padlength = int(start_depth / spacing)
        padval = np.zeros(padlength)

        for i in range(1, padlength):
            padval[-i] = start_depth - (spacing * i)

        md = np.append(padval, depth_track)
        inc = np.zeros(len(depth_track) + padlength)
        azm = np.zeros(len(depth_track) + padlength)

    dz = np.transpose([md, inc, azm])
    dz = pd.DataFrame(dz).dropna()

    wella.location.add_deviation(dz, wella.location.td)
    tvdg = wella.location.tvd
    md = wella.location.md

    from welly import curve
    MD = curve.Curve(md, mnemonic='MD', units='m', index=md)
    wella.data['MD'] = MD
    TVDM = curve.Curve(tvdg, mnemonic='TVDM', units='m', index=md)
    wella.data['TVDM'] = TVDM

    wella.unify_basis(keys=None, alias=None, step=spacing)

    print("Great Success!! :D")
    return wella

def create_random_well(kb,gl,kop=0, maxangle=0, step=0.15, drop=[]):
    """Create a random well object with specified parameters.

    Parameters
    ----------
    kb : float
        Kelly bushing elevation in meters
    gl : float
        Ground level elevation in meters
    kop : float, optional
        Kickoff point depth in meters, by default 0
    maxangle : float, optional
        Maximum deviation angle in degrees, by default 0
    step : float, optional
        Depth step size in meters, by default 0.15
    drop : list, optional
        List of curves to exclude from output, by default []

    Returns
    -------
    welly.Well
        Well object containing randomly generated log data
    """
    data_frame_2, header_frame_2, string_buffer = create_random_las(kb=kb,gl=gl, stepper=step,drop=drop)
    string_data = string_buffer.getvalue()
    return Well.from_las(string_data)
    
"""
# Example usage
if __name__ == "__main__":
    # Demonstrate the function with file writing
    #data_frame, header_frame, filename = process_well_data(writeFile=True)
    
    # Demonstrate the function with string buffer
    data_frame_2, header_frame_2, string_buffer = create_random_las(kb=50,gl=-30, debug=True)
    
    # Optional: read string buffer content
    if not isinstance(string_buffer, str):
        print("\nLAS File Content (first 1500 chars):")
        string_buffer.seek(0)
        print(string_buffer.read(1500))
    x = create_random_well(kb=35,gl=20)
    x = getwelldev(wella = x)#,kickoffpoint=1000,final_angle=85, azimuth=89, rateofbuild=0.2)
    print(x)
    y=x.location.trajectory()
    z = x.location.position
    # Assuming z is your array
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D line
    ax.plot3D(z[:, 0], z[:, 1], -z[:, 2])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth (Z)')

    plt.show()
"""

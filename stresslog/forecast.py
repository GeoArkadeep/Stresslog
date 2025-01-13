import io
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from difflib import SequenceMatcher

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from .syntheticLogs import getwelldev, create_random_well, create_header
from .geomechanics import remove_curves, add_curves, find_TVD, compute_geomech
from .thirdparty import datasets_to_las


#from stresslog import *


def create_blank_well(kb,gl,kop=0, maxangle=0, rob=0.1,azimuth=0,dev=None, spacing=0.15,td=5000):
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
    blankwell = getwelldev(wella=create_random_well(kb=kb, gl=gl, stopper=td, step=spacing),step=spacing, deva=dev, kickoffpoint=kop, final_angle=maxangle, rateofbuild=rob, azimuth=azimuth)
    blankwell = remove_curves(blankwell,["DTCO","RHOB","NPHI","GR","ILD","DTS","NU"])
    return blankwell
    
    


def uniform_resample(df, key, step):
    # Save the original index name
    originalindexkey = df.index.name
    if originalindexkey:
        df = df.reset_index()
    df = df.set_index(key)
    # Create new uniform index
    uniform_spacing = step
    new_index = np.arange(df.index.min(), df.index.max() + uniform_spacing, uniform_spacing)

    # Resample using interpolation
    df = df.reindex(df.index.union(new_index)).interpolate(method="index").reindex(new_index)
    df = df.reset_index()
    if originalindexkey:
        df = df.set_index(originalindexkey)

    return df
    

def depth_shift_process(df, current_forms_dict, new_forms_dict):
    """
    Transform a dataframe by remapping its index according to provided mapping dictionaries.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numerical index
    current_forms_dict : dict
        Dictionary mapping form names to current index positions
    new_forms_dict : dict
        Dictionary mapping form names to desired new index positions
    
    Returns:
    --------
    pandas.DataFrame
        Transformed dataframe with new index positions
    """
    #df = df.fillna(0.0)
    offset = (new_forms_dict[list(new_forms_dict)[-1]]-current_forms_dict[list(current_forms_dict)[-1]])
    print(offset)
    current_forms_dict["Fin"] = float(df.index[-1])
    new_forms_dict["Fin"] = float(df.index[-1]+offset)
    print(current_forms_dict)
    print(new_forms_dict)
    
    # Convert dictionaries to sorted lists of tuples for easier processing
    current_points = sorted(current_forms_dict.items(), key=lambda x: x[1])
    new_points = sorted(new_forms_dict.items(), key=lambda x: x[1])
    
    # Extract x and y coordinates for the mapping
    current_x = [0] + [p[1] for p in current_points]  # Add 0 as starting point
    new_x = [0] + [p[1] for p in new_points]  # Add 0 as starting point

    # Create interpolation function
    transform_func = interp1d(current_x, new_x, kind='linear', bounds_error=False, 
                            fill_value=(new_x[0], new_x[-1]))
    
    # Transform the index
    new_index = transform_func(df.index)
    
    # Create new dataframe with transformed index
    df_new = df.copy()
    df_new.index = new_index
    
    # Sort by new index to ensure proper ordering
    df_new = df_new.sort_index()
    
    # Create regular spacing between min and max of new index
    num_points = len(df_new)
    regular_index = pd.Index(np.linspace(df_new.index.min(), df_new.index.max(), num_points))
    
    # Create a new dataframe with the regular index
    df_regular = pd.DataFrame(index=regular_index, columns=df_new.columns)
    
    # Copy data to the new dataframe and interpolate
    for column in df_new.columns:
        df_regular[column] = np.interp(
            regular_index,
            df_new.index,
            df_new[column].values,
            left=df_new[column].iloc[0],
            right=df_new[column].iloc[-1]
        )
    df_regular.index.names = df.index.names
    return df_regular


def get_analog_well(well,cfd,nfd,spacing=1,blankwell=None, indices="TVDM", dev= None, kop= 0, rob= 0, maxangle= 0, azimuth=0, name= "Analog", kb=0, gl=0, debug=False):
    #returns welly.well object with valid deviation data
    # Define mapping dictionaries
    df = well.df()
    df["DEPT"] = df.index.values
    if debug:
        print(df)
    if indices is not None:
        df = df.set_index(indices)
    if debug:
        print(df)
    #Resample to regular spacing, eg 1m
    uniform_spacing = spacing

    # Create new uniform index
    new_index = np.arange(df.index.min(), df.index.max() + uniform_spacing, uniform_spacing)

    # Resample using interpolation
    # Method can be 'linear', 'cubic', 'nearest', etc.
    df = df.reindex(df.index.union(new_index)).interpolate(method='index').reindex(new_index)

    #df_resampled = df.interpolate(method="index")

    if debug:
        print(df)
    

    # Transform the dataframe
    dft = depth_shift_process(df, cfd, nfd)
    dft=dft.drop(["MD"],axis=1) #MD and TVD (as well as all well specific data is invalid now 
    dft=dft.drop(["DEPT"],axis=1) #so we remove these to prevent confusion
    dftvd = uniform_resample(dft,"TVDM",spacing)
    if debug:
        print("dft fresh")
        print(dftvd)
    
    #Now we create the backbone of the proposed well with the new proposed deviation data
    if blankwell is None:
        if dev is not None:
            blankwell = getwelldev(wella=create_random_well(kb=kb, gl=gl, step=spacing),step=spacing,deva=dev)
        else:
            blankwell = getwelldev(wella=create_random_well(kb=kb, gl=gl, step=spacing),step=spacing, kickoffpoint=kop, final_angle=maxangle, rateofbuild=rob, azimuth=azimuth)
        blankwell = remove_curves(blankwell,["DTCO","RHOB","NPHI","GR","ILD","DTS","NU"])
    blankdf = blankwell.df()
    analogdevnp = blankwell.location.deviation
    analogdevdf = pd.DataFrame(analogdevnp, columns=["MD", "INC", "AZIM"])

    print(analogdevdf)
    analogdevdf.to_csv("analogdevdf.csv",index=False)
    kb = float(blankwell.header.set_index("mnemonic").loc["KB", "value"])
    gl = float(blankwell.header.set_index("mnemonic").loc["GL", "value"])
    #blankdf = uniform_resample(blankdf,"TVDM",spacing)
    if debug:
        print("Blankdf:")
        print(blankdf)
    dfmd=blankdf
    # Ensure dftvd's index is float and sorted for interpolation
    dftvd.index = dftvd.index.astype(float)
    dftvd = dftvd.sort_index()

    xp = dftvd.index.to_numpy()  # Independent variable from dftvd (TVD)
    fp = dftvd.to_numpy()  # Dependent variable(s) from dftvd (values to interpolate)

    x = dfmd["TVDM"].to_numpy()  # Values to interpolate at (TVDM from dfmd)

    # Perform interpolation using numpy.interp
    # np.interp handles 1D arrays, so we do this for each column in dftvd
    interpolated_values = np.vstack([
        np.interp(x, xp, fp[:, col_idx]) for col_idx in range(fp.shape[1])
    ]).T

    # Create a new dataframe with interpolated rows
    dft_new = pd.DataFrame(interpolated_values, index=dfmd.index, columns=dftvd.columns)

    # Optionally, include MD from dfmd for context
    dft_new["MD"] = dfmd["MD"]
    dft_new["TVDM"] = dfmd["TVDM"]
    dft_new["DEPT"] = dfmd["MD"]
    #dft_new.index.name = "DEPT"

    if debug:
        print("dft_new:")
        print(dft_new)
    
    analogwell = add_curves(blankwell,dft_new,True)
    if debug:
        print("analogwell's df")
        print(analogwell.df())

        #returning the analogwell directly is a bad idea, let's write it to a las string and return that instead
        #this way we can i) return something industry standard, ii) human readable so we can check the outcome
    
        print("Kellybushing of target well is: ",kb)
    hdf = create_header(
        name="analogwell",#f"{well_name}-{key}", 
        uwi="ANALOGWELL",#f"WELL#{key}", 
        strt=str(dft_new["MD"].iloc[0]),#str(starter), 
        stop=str(dft_new["MD"].iloc[-1]),#str(stopper), 
        step=str(dft_new["MD"].iloc[-5]-dft_new["MD"].iloc[-4]),#str(stepper),
        kb=str(kb),
        gl=str(gl),
    )
    if debug:
        print(hdf)
    #analogwell.header = hdf
    cols = list(dft_new.columns)
    cols.reverse()

    dft_new = dft_new[cols]
    if debug:
        print(dft_new)
    las_string = datasets_to_las("test.las" if debug else None, {'Curves': dft_new, 'Header': hdf}, 
                             {})
    #print(las_string)
    if debug:
        print(dft_new)
        print(analogwell)
        print(analogwell.df()["MD"])
        
        
        dft.reset_index(inplace=True)
        df.reset_index(inplace=True)
        dft.set_index("TVDM",inplace=True)
        df.set_index("TVDM",inplace=True)
        
        # Create the plot
        import matplotlib.pyplot as plt
        plt.plot(dftvd['DTHM'],dftvd.index.values,label="TVD Model", alpha=0.7)
        plt.plot(df['DTHM'],df.index.values,label="Post-drill well", alpha=0.7)
        plt.plot(dft_new['DTHM'],dft_new["TVDM"],label="Analog well")
        plt.legend()
        # Force plot to start at 0,0
        plt.xlim(240, 0)
        plt.ylim(plt.ylim()[1],0)
        # Add ticks on all edges
        ax = plt.gca()
        ax.tick_params(which='both', direction='out')  # Make ticks point inward
        ax.yaxis.set_ticks_position('both')  # Show ticks on both left and right
        ax.xaxis.set_ticks_position('both')  # Show ticks on both top and bottom
        
        # Set major and minor grid
        major_ticks = np.arange(0, max(max(dft.index.values), max(dft.index.values)) + 1000, 1000)
        minor_ticks = np.arange(0, max(max(dft.index.values), max(dft.index.values)) + 100, 100)


        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        # Add grid
        plt.grid(which='major', color='gray', linestyle='-', alpha=0.5)
        plt.grid(which='minor', color='gray', linestyle=':', alpha=0.3)

        # Save the figure
        plt.savefig("depthshiftprocessing.png")
        plt.close()
    analogwell.header = analogwell.header.drop(index=0).reset_index(drop=True)
    return analogwell,io.StringIO(las_string), analogdevdf



def fuzzymatch(df, target_key):
    """
    Find the closest matching column name from a DataFrame, regardless of similarity.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        target_key (str): The target column name to match.

    Returns:
        str: The closest matching column name from the DataFrame.
    """
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    # Find the column with the highest similarity score
    target_key_lower = target_key.lower()
    best_match = max(df.columns, key=lambda col: similarity(target_key_lower, col.lower()))

    return best_match


def convert_df_tvd(df,well):
    """
    Converts the first column of formation or other dataframes from MD to TVD, given a welly.Well object with valid deviation data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe who's first column has the measured depths we want to convert.
    well : welly.Well
        Source well object containing the original log data and wellbore information.
        Must valid deviation data.
    Returns
    -------
    pandas.DataFrame
        The dataframe with the first column converted to TVD and column name set to TopTVD
    """
    
    originalkey = list(df)[0]
    df[originalkey] = df[originalkey].apply(lambda x: find_TVD(well, x))
    print(df)
    if originalkey != 'TopTVD':
        df = df.rename(columns={originalkey: 'TopTVD'})
    return df

def extract_depth_dict(df):
    originalkey = list(df)[0]
    f_names = ("Name")#fuzzymatch(df,"Name")
    return {row[f_names]: round(float(row[originalkey]), 2) for _, row in df.iterrows()}

def get_analog(well,current_forms, target_forms, kb,gl,dev=None,kop=0,ma=0,rob=0.1,azi=0,debug=False,td=5000):
    """
    Generate an analog well by transforming log data from an existing well according to specified formation depths
    and wellbore geometry parameters. This function creates a new well object with transformed log data that 
    maintains the character of the original logs while honoring new formation tops and a new wellbore trajectory.

    The function performs several key operations:
    1. Converts MD (Measured Depth) formations to TVD (True Vertical Depth) if necessary
    2. Creates a blank wellbore with specified geometry
    3. Transforms the original well's log data to match new formation depths
    4. Generates a new well object with the transformed data

    Parameters
    ----------
    well : welly.Well
        Source well object containing the original log data and wellbore information.
        Must contain basic log curves (e.g., GR, RHOB, etc.) and valid deviation data.

    current_forms : pandas.DataFrame
        DataFrame containing the current formation tops information.
        Must include columns for depth values and formation names.
        Depths can be in either MD or TVD (specified by column name).
        Example format:
            TopTVD/MD  Name
            1690      Alpha
            2380      Beta
            ...       ...

    target_forms : pandas.DataFrame
        DataFrame containing the target formation tops information for the analog well.
        Must follow the same format as current_forms.
        These depths represent where formations should appear in the analog well.

    kb : float
        Kelly Bushing elevation in meters for the analog well.
        Used as the reference point for depth measurements.

    gl : float
        Ground Level elevation in meters for the analog well.
        Must be less than or equal to kb if onshore and equal to negetive water depth if offshore

    dev : array-like, optional
        Custom deviation survey data for the analog well.
        If provided, overrides the kop, ma, rob, and azi parameters.
        Format should be compatible with welly.Well deviation data.
        Default is None.

    kop : float, optional
        Kick-Off Point depth in meters.
        Depth at which the wellbore begins to deviate from vertical.
        Must be >= 0.
        Default is 0.

    ma : float, optional
        Maximum angle (deviation) in degrees.
        The maximum inclination angle the wellbore will reach.
        Must be between 0 and 90.
        Default is 0 (vertical well).

    rob : float, optional
        Rate Of Build in degrees per meter.
        Rate at which the wellbore builds angle from the KOP.
        Typical values range from 0.1 to 3.0 degrees/meter.
        Default is 0.1.

    azi : float, optional
        Azimuth in degrees.
        The compass direction of the wellbore deviation.
        Must be between 0 and 360.
        Default is 0 (North).

    debug : bool, optional
        If True, enables debug mode which:
        - Prints intermediate dataframes
        - Generates diagnostic plots
        - Saves additional output files
        - Provides verbose console output
        Default is False.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - welly.Well : The created analog well object with transformed log data
          and new trajectory information
        - io.StringIO : A StringIO object containing the LAS file representation
          of the analog well

    Notes
    -----
    - The function uses fuzzy matching to find the formation names column in the dataframes,
      so the column name can be "Name", "Formation_Name", etc.
    - Log data is resampled to maintain consistent spacing throughout the transformation
    - The function preserves the character and relationships between different log curves
      while shifting them to new depths
    - The transformation process interpolates depths linearly between formation tops, 
      while preserving thickness below the deepest formation top
    - Formation top names must be consistent between current_forms and target_forms
    - If a formation has pinched out, it should still be included in both dataframes
      with its depth matching either the overlying or underlying formation based on
      geological interpretation

    """
    offset1 = kb-gl
    offset0 = (float(well.header.set_index("mnemonic").loc["KB", "value"]))-float(well.header.set_index("mnemonic").loc["GL", "value"])
    
    step = well.df().index.values[1] - well.df().index.values[0]
    if debug:
        print(step)
    if list(current_forms)[0].lower()=="toptvd":
        cfd = extract_depth_dict(current_forms)
    else:
        cfd = extract_depth_dict(convert_df_tvd(current_forms,well))
        
    
    newblankwell=create_blank_well(kb=kb,gl=gl,spacing=step,maxangle=ma,kop=kop,azimuth=azi,dev=dev,td=td)
    
    if list(target_forms)[0].lower()=="toptvd":
        nfd = extract_depth_dict(target_forms)
    else:
        nfd = extract_depth_dict(convert_df_tvd(target_forms,newblankwell))

    cfd["Start"] = offset0
    nfd["Start"] = offset1
    
    return get_analog_well(well,cfd,nfd,5,blankwell=newblankwell,debug=debug)
    


if __name__=="__main__":
    import welly
    spacing = 5
    wellu = welly.Well.from_las("testlas.las")
    print(wellu)
    dev = pd.read_csv("testdev.csv")
    print(dev)

    well = getwelldev(wella=wellu,deva=dev,step=spacing)
    print(well.df())
    
    well.header = pd.concat([well.header[well.header["mnemonic"] != "KB"], pd.DataFrame({"mnemonic": ["KB"], "value": [80]})], ignore_index=True)
    well.header = pd.concat([well.header[well.header["mnemonic"] != "GL"], pd.DataFrame({"mnemonic": ["GL"], "value": [20]})], ignore_index=True)
    
    #well = getwelldev(wella=create_random_well(kb=35, gl=-200, step=spacing),step=spacing, kickoffpoint=2500, final_angle=25, rateofbuild=0.1, azimuth=270)
    form_md = """TopTVD,StratiNum,Name,GR_Cut,Struc_Top,Struc_Bot,Z_ratio,OWC,GOC,Bt,Shm_Azim,Dip_Azim,Dip_Angle
    1690,1,Alpha,70,500,2380,0.5,0,0,,,,
    2380,2,Beta,70,1600,2610,0.5,0,0,,,,
    2900,3,Gamma,70,2450,2950,0.5,2499,0,,90,,
    2950,4,Echo,60,2550,3400,0.66,3100,3050,,90,0,0
    3250,5,Delta,65,2750,3300,0.33,2950,2900,,85,2,3"""
    
    f0 = pd.read_csv("testforms.csv")
    print(f0)

    #f0 = pd.read_csv(io.StringIO(form_md), sep=",")
    cfd = extract_depth_dict(convert_df_tvd(f0,well))
    
    new_md = """topTVD,StratiNum,Formation_Name,GR_Cut,Struc_Top,Struc_Bot,Z_ratio,OWC,GOC,Bt,Shm_Azim,Dip_Azim,Dip_Angle
    1750,1,Alpha,70,500,2380,0.5,0,0,,,,
    2124,2,Beta,70,1600,2610,0.5,0,0,,,,
    2900,3,Gamma,70,2450,2950,0.5,2499,0,,90,,
    3150,4,Echo,60,2550,3400,0.66,3100,3050,,90,0,0
    4200,5,Delta,65,2750,3300,0.33,2950,2900,,85,2,3"""
    
    f1 = pd.read_csv("testforms2.csv")
    print(f1)
    
    #f1 = pd.read_csv(io.StringIO(new_md), sep=",")
    #newblankwell=create_blank_well(kb=20,gl=10,spacing=15,maxangle=90,kop=4000,azimuth=90)
    
    new_well,stringlas,devdas = get_analog(well,f0,f1,kb=20,gl=-200,ma=90,kop=1000,azi=90,debug=False)
    freshnew = getwelldev(string_las=stringlas.read(),deva=devdas,step=5)
    print(freshnew)
    print(new_well)

    output = compute_geomech(freshnew,sfs=30, tango=4000, writeFile=False,)
    outdf = output[0]
    outdf.to_csv("testout.csv")
    print(new_well.df())
    stringlas.seek(0)
    print(stringlas.read(2500))
    """
    lulu = io.StringIO(output[1])
    lulu.seek(0)
    print(lulu.read(2500))
    """

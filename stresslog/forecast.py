import stresslog as lst


well = lst.getwelldev(wella=lst.create_random_well(kb=35, gl=-200, step=50))
output = lst.plotPPzhang(well, writeFile=False)
df = output[0]
df.set_index("DEPT",inplace=True)

currentformsdict = {"A":1000.00, "B":2000.00, "C":3000.00, "D":4000, "E":5000}
newformsdict = {"A":900.00, "B":2200.00, "C":2500.00, "D":4000, "E":5000}

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def transform_dataframe_index(df, current_forms_dict, new_forms_dict):
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
    current_forms_dict["Fin"] = df.index[-1]
    new_forms_dict["Fin"] = df.index[-1]+(nfd[list(nfd)[-1]]-cfd[list(cfd)[-1]])
    print(current_forms_dict)
    print(new_forms_dict)
    print((nfd[list(nfd)[-1]]-cfd[list(cfd)[-1]]))
    
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
    
    return df_regular



# Define mapping dictionaries
cfd = {"A":1000.00, "B":2000.00, "C":3000.00, "D":4000.00, "E":5000.00}
nfd = {"A":900.00, "B":2200.00, "C":2500.00, "D":4000.00, "E":4500.00}

# Transform the dataframe
dft = transform_dataframe_index(df, cfd, nfd)

import matplotlib.pyplot as pltf
pltf.plot(dft.index.values,dft['MD'])
pltf.gca().set_aspect('equal')
pltf.savefig("test.png")
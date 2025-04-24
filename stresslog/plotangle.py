"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import matplotlib.pyplot as plt
import numpy as np

from io import StringIO

def plot_to_svg(matplot) -> str:
    """
    Saves the last plot made using ``matplotlib.pyplot`` to a SVG string.
    
    Returns:
        The corresponding SVG string.
    """
    s = StringIO()
    matplot.savefig(s, format="svg")
    matplot.close()  # https://stackoverflow.com/a/18718162/14851404
    return s.getvalue()


def plotfracs(data):
    """
    Plot line segments representing fractures based on position and angle data.
    
    Parameters
    ----------
    data : ndarray
        A 2D numpy array with shape (n, 4) where:
        - data[:, 0] contains y-coordinates
        - data[:, 1] contains x-coordinates
        - data[:, 2] contains first set of angles
        - data[:, 3] contains second set of angles
    
    Returns
    -------
    matplotlib.pyplot
        A matplotlib pyplot object containing the plotted fractures
    """
    x = data[:, 1]
    x2 = x+180
    y = data[:, 0]
    angles = data[:, 2]
    angles2 = -data[:, 3]
    # Creating the plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 360)  # Adjusting x-axis limit to fit your data
    #ax.set_ylim(2450, 2460)  # Adjusting y-axis limit to fit your data
    #ax.set_aspect(0.2)  # This makes 1 unit in x equal to 1 unit in y

    # Plotting each line segment based on the angle
    for x_i, y_i, angle in zip(x, y, angles):
        # Calculating the end point of the line segment
        # Assuming a fixed length for each segment
        length = 1  # Adjusted length to be visible given the scale of your data
        end_x = x_i + length * np.cos(np.radians(angle))
        end_y = y_i + length * np.sin(np.radians(angle))
        
        # Plotting the line segment
        ax.plot([x_i, end_x], [y_i, end_y], linewidth=0.01, color='black')  # Added markers for clarity
        
        
    for x2_i, y_i, angle2 in zip(x2, y, angles2):
        # Calculating the end point of the line segment
        # Assuming a fixed length for each segment
        length = 1  # Adjusted length to be visible given the scale of your data
        end_x = x2_i - length * np.cos(np.radians(angle2))
        end_y = y_i - length * np.sin(np.radians(angle2))
        
        # Plotting the line segment
        ax.plot([x2_i, end_x], [y_i, end_y], linewidth=0.01, color='black')  # Added markers for clarity
        
    return plt
#plt.show()

def plotfracsQ(data):
    """
    Initialize a plot for fractures without plotting the line segments.
    
    Parameters
    ----------
    data : ndarray
        A 2D numpy array with shape (n, 4) where:
        - data[:, 0] contains y-coordinates
        - data[:, 1] contains x-coordinates
        - data[:, 2] contains first set of angles
        - data[:, 3] contains second set of angles
    
    Returns
    -------
    matplotlib.pyplot
        A matplotlib pyplot object with initialized axes set to [0, 360]
    """
    x = data[:, 1]
    x2 = x+180
    y = data[:, 0]
    angles = data[:, 2]
    angles2 = -data[:, 3]
    # Creating the plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 360)  # Adjusting x-axis limit to fit your data
    #ax.set_ylim(2450, 2460)  # Adjusting y-axis limit to fit your data
    #ax.set_aspect(0.2)  # This makes 1 unit in x equal to 1 unit in y

    return plt
#plt.show()

def cot(x):
    return np.cos(x)/np.sin(x)

def plotfrac(data,path=None, dia=8.5, debug=False):
    """
    Generate detailed fracture morphology plot with depth calculations.
    
    Parameters
    ----------
    data : tuple
        A tuple containing:
        - tvd : float
            True vertical depth
        - fr : ndarray
            Array of fracture indicators
        - angles : ndarray
            Array of angles
        - minangle : float
            Minimum angle value
        - maxangle : float
            Maximum angle value
    path : str, optional
        File path to save the plot. If None, plot is not saved to file.
    dia  : float, optional
        Hole Diameter in inches to be used for the calculation, in inches, 8.5 by default
    
    Returns
    -------
    tuple
        A tuple containing two ndarrays:
        - cdepths : ndarray
            Corrected depths array
        - fdepths : ndarray
            Final depths array
    
    Notes
    -----
    Uses a bit diameter of 8.5 inches for calculations.
    Handles special angle cases and implements various depth corrections
    and transformations.
    """
    tvd,fr,angles,minangle,maxangle = data
    #dia = 8.5 #inches, bit
    circumference = np.pi*dia #in inches
    cm = 0.0254*circumference
    i = 0
    d = np.zeros(360)
    yj = np.zeros(360)
    depths = np.zeros(360)
    sign = np.zeros(360)
    midpoint1 = min(minangle+90,(minangle+270)%360)
    midpoint2 = max(minangle+90,(minangle+270)%360)
    spVa = 360
    spVb = 0
    while(i<360):
        if i>midpoint1 and i<midpoint2:
            d[i] = (i-(minangle+180))
        else:
            if i<midpoint1:
                d[i] = i-minangle
            else:
                d[i] = i-(360+minangle)
        if abs(d[i])==270:
            d[i]=90*(abs(d[i])/d[i])
        yj[i] = (np.tan(np.radians((90-angles[i])%360))*d[i]) #FlipVertN if abs
        sign[i] = yj[i]/abs(yj[i])
        depths[i] = (((yj[i]-180)/180)*(cm/2))
        if d[i-1]==0:
            yj[i-1] = (yj[i-2]+yj[i])/2
            spV =  yj[i-1]
            depths[i-1] = (depths[i-2]+depths[i])/2
            spVa = min(depths[i-1],spVa)
            spVb = max(depths[i-1],spVb)

        i+=1
    
    yj = yj-spV
    #depths = depths*sign #Dont FlipV if signed #AntiFlipV
    if path is not None:
        plt.figure(figsize=(10, 10))
        plt.plot(yj)
        #Setting axis limits
        plt.xlim(0, 360)
        plt.ylim(-180, 180)
        plt.title("Fracture Morphology")
        plt.savefig(path)
        plt.close()
    print(yj) if debug else None
    yj[(maxangle-10)%360:(maxangle+15)%360]=np.nan
    yj[(maxangle+170)%360:(maxangle+195)%360]=np.nan
    #depths[(maxangle-10)%360:(maxangle+15)%360]=np.nan
    #depths[(maxangle+170)%360:(maxangle+195)%360]=np.nan
    i=0
    cyj = yj
    cyj[midpoint1:midpoint2] = cyj[midpoint1:midpoint2][::-1]
    avdepths = depths.copy()
    while i<360: #ShiftVert1
        #if i<midpoint1 or i>midpoint2-1:
        #    depths[i] = depths[i]-spVa
        #else:
        #    depths[i] = depths[i]-spVb
        if abs(depths[i])>1.6:
            #depths[i] = np.nan
            avdepths[i] = np.nan
        if fr[i]<1:
            depths[i] = np.nan
        i+=1
    av1 = np.nanmean(np.concatenate([depths[0:(midpoint1-1)],depths[midpoint2:360]]))
    av2 = np.nanmean(depths[midpoint1:(midpoint2-1)])
    
    i=0
    while i<360: #ShiftVert2
        if i<midpoint1 or i>midpoint2-1:
            depths[i] = depths[i]-av1
        else:
            depths[i] = depths[i]-av2
        i+=1
    
    i=0
    fdepths=depths.copy()
    print(fdepths) if debug else None
    deldep = (np.nanmean(depths)) 
    depths = depths-deldep
    fdepths = fdepths-deldep
    cdepths = depths+tvd
    while i<360:
        if abs(depths[i])>1.6:
            cdepths[i] = np.nan
        if fr[i]<1:
            cdepths[i] = np.nan
        i+=1
    fdepths = fdepths+tvd
    m1 = np.nanmax(np.concatenate([fdepths[0:(midpoint1-1)],fdepths[midpoint2:360]]))
    m2 = np.nanmax(fdepths[midpoint1:(midpoint2-1)])
    diff = m2-m1
    print(diff) if debug else None
    fdepths[0:(midpoint1-1)] = fdepths[0:(midpoint1-1)] + diff/2
    fdepths[midpoint2:360] = fdepths[midpoint2:360] + diff/2
    fdepths[midpoint1:(midpoint2-1)]=fdepths[midpoint1:(midpoint2-1)] - diff/2
    
    cdepths[0:(midpoint1-1)] = cdepths[0:(midpoint1-1)] + diff/2
    cdepths[midpoint2:360] = cdepths[midpoint2:360] + diff/2
    cdepths[midpoint1:(midpoint2-1)]=cdepths[midpoint1:(midpoint2-1)] - diff/2
    #cdepths[midpoint1:midpoint2] = cdepths[midpoint1:midpoint2][::-1]# FlipHorzR
    #fdepths[midpoint1:midpoint2] = fdepths[midpoint1:midpoint2][::-1]# FlipHorzR
    print(fdepths) if debug else None
    print(d) if debug else None
    
    """
    while i<360:
        if fr[i]>0:
            plt.scatter(i, yj[i], color='black', marker='o')
        i+=1
    """

    return cdepths,fdepths
#plt.show()

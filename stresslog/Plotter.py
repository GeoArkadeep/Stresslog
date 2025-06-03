"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from PIL import Image

import plotly.tools as tls
import plotly.io as pio

import os

user_home = os.path.expanduser("~/Documents")

def plot_logs_labels(data, styles, points=None, pointstyles=None, y_min=None, y_max=None, width=15, height=10, label_height=20, dpi=100, output_dir = os.path.join(user_home, "Stresslog_plots"), to_plotly=False, title=None, details={"unit":"metres","type":"TVD","reference":"KB/DF","KB":0,"GL":0}, display=False):
    """
    Wrapper function that calls plot_logs three times to generate the main plot and label plots.
    
    Parameters:
    - Same as plot_logs function, except plot_labels is not included as it's handled internally.
    
    Returns:
    - fig, axes: The figure and axes objects of the main plot.
    """
    labelmax = int(y_max) #2600
    #print(labelmax)
    # Generate simulated data for label plots
    simulated_depth = np.array([labelmax])
    simulated_data = pd.DataFrame({key: np.random.rand(1) for key in data.columns}, index=simulated_depth)
    
    # Call plot_logs for bottom labels
    plot_logs(simulated_data, styles, None, None, labelmax+1, labelmax+2, 
              plot_labels=True, width=width, height=3, label_height=98, 
              dpi=dpi, output_dir=output_dir,to_plotly=to_plotly)
    
    # Call plot_logs for top labels
    plot_logs(simulated_data, styles, None, None, labelmax+1, labelmax, 
              plot_labels=True, width=width, height=3, label_height=98, 
              dpi=dpi, output_dir=output_dir,to_plotly=to_plotly)
    # Call plot_logs for the main plot without labels
    fig, axes = plot_logs(data, styles, points, pointstyles, y_min, y_max, 
                          plot_labels=False, width=width,height=height, label_height=label_height, 
                          dpi=dpi,to_plotly=to_plotly)
    plotfile = os.path.join(output_dir,"PlotFigure.png")
    if title is not None:
        fig.suptitle('Wellbore: ' + title, fontsize=18, y=0.91)
        #fig.text(0.50, 0.89, 'Wellbore: ' + title, fontsize=18, horizontalalignment='center')
        fig.text(0.05, 0.91, 'Depth in '+details["unit"]+", "+details["type"], fontsize=9, horizontalalignment='left')
        fig.text(0.05, 0.89, 'Depth Reference: '+details["reference"], fontsize=9, horizontalalignment='left')
        
        fig.text(0.95, 0.91, 'KB: '+str(details["KB"])+' '+details["unit"]+' above MSL', fontsize=9, horizontalalignment='right')
        fig.text(0.95, 0.89, 'GL: '+str(-(details["GL"]))+' '+details["unit"]+' below MSL', fontsize=9, horizontalalignment='right') if details["GL"]<0 else fig.text(0.95, 0.89, 'GL: '+str(details["GL"])+' '+details["unit"]+' above MSL', fontsize=9, horizontalalignment='right')
    
    plt.savefig(plotfile,dpi=dpi)
    if display:
        plt.show()
    plt.close()
    choptop(55,0,os.path.join(output_dir,'BottomLabel.png'))
    cutify2(os.path.join(output_dir,'PlotFigure.png'),os.path.join(output_dir,'BottomLabel.png'), os.path.join(output_dir,'WellPlot.png'), 0, 300, 0, 0)
    choptop(200,0,os.path.join(output_dir,'WellPlot.png'))
    if to_plotly:
        plotly_fig = tls.mpl_to_plotly(fig)
        normalization_info = {}
        for col, style in styles.items():
            normalization_info[col] = {
                'type': style['type'],
                'left': style['left'],
                'right': style['right']
            }
        if pointstyles:
            for col, style in pointstyles.items():
                normalization_info[col] = {
                    'type': style['type'],
                    'left': style['left'],
                    'right': style['right']
                }
        # Update Plotly figure with custom hover templates
        for trace in plotly_fig.data:
            if 'name' in trace and trace.name in normalization_info:
                col_name = trace.name
                norm_info = normalization_info[col_name]
                
                # Check if it's a line trace (mode should include 'lines')
                if 'lines' in trace.mode:
                    if norm_info['type'] == 'linear':
                        left, right = norm_info['left'], norm_info['right']
                        # Denormalize the data
                        original_data = [x * (right - left) + left for x in trace.x]
                        trace.customdata = original_data
                        hovertemplate = f"{col_name}: " + "%{customdata:.2f}<br>Depth: %{y}<extra></extra>"
                    elif norm_info['type'] == 'log':
                        hovertemplate = f"{col_name}: " + "%{x:.2e}<br>Depth: %{y}<extra></extra>"
                    trace.update(hovertemplate=hovertemplate)
                else:
                    # For scatter plots or unidentified traces, turn off hover
                    trace.update(hoverinfo='skip')
            else:
                # For any trace we can't identify, turn off hover
                trace.update(hoverinfo='skip')
        #plt.close()
        # Fix the x-axis range for both subplots
        plotly_fig.update_xaxes(fixedrange=True)

        # Share the y-axis between subplots
        plotly_fig.update_yaxes(matches='y')

        # Make sure the plot fits the width of the browser
        plotly_fig.update_layout(
            autosize=True,
            width=None,  # Remove the fixed width to make it responsive
            #height=600,  # Set a fixed height if needed, or leave it auto-sized
            margin=dict(l=30, r=30, t=30, b=30)  # Adjust margins as needed
        )
        plotly_fig.update_xaxes(showgrid=True, gridcolor = 'rgba(211, 211, 211, 0.8)', tickcolor='rgba(0, 0, 0, 0)', showticklabels=False)
        plotly_fig.update_yaxes(automargin=False,showgrid=True, gridcolor = 'rgba(211, 211, 211, 0.8)', tickcolor='rgba(0, 0, 0, 0)', tickformat='.0f', linecolor='grey', linewidth=1)
        plotly_fig.update_layout(
            yaxis=dict(showticklabels=True),  # Show labels for the leftmost y-axis
            yaxis2=dict(showticklabels=False),  # Hide labels for the second column
            yaxis3=dict(showticklabels=False),  # Hide labels for the third column
            yaxis4=dict(showticklabels=False),
            yaxis5=dict(showticklabels=False),
            yaxis6=dict(showticklabels=False),
        )
        plotly_fig.update_layout(margin=dict(l=30,r=0,t=0,b=0))
        plotly_fig.update_layout(
            paper_bgcolor='white',  # Background of the entire figure
            plot_bgcolor='white',   # Background of the plot area
        )
        #plotly_fig.update_yaxes(showticklabels=False) # hide all the yticks
        # Save as an interactive HTML file and open it in the browser
        pio.write_json(plotly_fig,os.path.join(output_dir,'plotly.json'))
        #pio.write_html(plotly_fig,'plotly.html')
    return fig, axes

def plot_logs(data, styles, points=None, pointstyles=None, y_min=None, y_max=None, plot_labels=True, width=15,height=10, label_height=20, dpi=100, output_dir = os.path.join(user_home, "Stresslog_plots"), to_plotly=False):
    """
    Plots well log data in tracks and sparse data points.

    Parameters:
    - data: A pandas DataFrame where each column corresponds to a curve to be plotted.
    - styles: A dictionary where keys are column names from the DataFrame and values are dictionaries of style attributes.
              Example: {'lithology1': {"color": "green", "linewidth": 1.5, "style": '.', "track": 0, "left": 0, "right": 150, "type": 'linear'}}
    - points: A pandas DataFrame where each column corresponds to sparse points to be plotted.
    - pointstyles: A dictionary where keys are column names from the points DataFrame and values are dictionaries of style attributes.
              Example: {'ucs': {'color': 'blue', 'pointsize': 10, 'symbol': 'o', 'track': 4, 'left': 0, 'right': 100, 'type': 'linear', 'unit': 'Mpa'}}
    - y_min: Minimum value of the y-axis (depth).
    - y_max: Maximum value of the y-axis (depth).
    - plot_labels: Boolean to specify if the labels should be plotted (default is True).
    - figsize: Tuple specifying the width and height of the figure (default is (15, 10)).
    - label_height: Height of the label rectangles (default is 20).
    - dpi: Dots per inch for the saved figure.
    """
    
    figsize=(width, height)
    if y_max is not None and y_min is not None and y_max > y_min:
        invert_yaxis = True
        pltsign = 1
    else:
        invert_yaxis = False
        pltsign = -1

    if len(data.columns) != len(styles):
        raise ValueError("The number of columns in the data must match the number of entries in the styles dictionary.")

    if points is not None and pointstyles is not None:
        if len(points.columns) != len(pointstyles):
            raise ValueError("The number of columns in the points must match the number of entries in the pointstyles dictionary.")

    n_tracks = max(max(style["track"] for style in styles.values()), 
                   max(pointstyle["track"] for pointstyle in pointstyles.values()) if pointstyles else 0) + 1
    fig, axes = plt.subplots(nrows=1, ncols=n_tracks, figsize=figsize, sharey=True)
    if n_tracks == 1:
        axes = [axes]

    # Function to add labels with style
    def add_label(ax, label, original_x_min, original_x_max, y_offset, color, linewidth, linestyle, units, log_scale=False):
        y_data_coords = y_offset

        if log_scale:
            # For log scale, keep the original label placement
            x_min = original_x_min
            x_max = original_x_max
        else:
            # For linear scale, normalize label placement to 0-1
            x_min = 0
            x_max = 1

        ax.set_axis_off()
        # Plot the representative line with the correct linestyle
        ax.plot([x_min, x_max], [y_data_coords+(0.05*pltsign), y_data_coords+(0.05*pltsign)], color=color, linewidth=linewidth, linestyle=linestyle, clip_on=False, zorder=3)

        # Plot the text above the representative line
        ax.text(x_min, y_data_coords, f"{original_x_min}", ha='left', va='center', fontsize=10, color=color, backgroundcolor='none', clip_on=False, zorder=4)
        if log_scale:
            center = np.sqrt(original_x_min * original_x_max)
        else:
            center = 0.5
        ax.text(center, y_data_coords, label, ha='center', va='center', fontsize=10, color=color, backgroundcolor='none', clip_on=False, zorder=4)
        ax.text(x_max, y_data_coords, f"{original_x_max}", ha='right', va='center', fontsize=10, color=color, backgroundcolor='none', clip_on=False, zorder=4)
        ax.text(center, y_data_coords + (0.09 * pltsign), f"{units}", ha='center', va='center', fontsize=7.5, color=color, backgroundcolor='none', clip_on=False, zorder=4)

    depth = data.index
    if y_min is None:
        y_min = min(depth)
    if y_max is None:
        y_max = max(depth)

    label_offsets = {track: y_min - (y_max - y_min) * (0.01) for i, track in enumerate(range(n_tracks))}
    
    # Dictionary to store normalization information
    #normalization_info = {}
    
    # Plotting data
    for col in data.columns:
        track = styles[col]["track"]
        ax = axes[track]
        x_data = data[col]
        style = styles[col]

        original_left = style["left"]
        original_right = style["right"]

        if style["type"] == 'log':
            ax.set_xscale('log')
            x_plot_data = x_data
            x_min = original_left
            x_max = original_right
            #normalization_info[col] = {'type': 'log', 'left': original_left, 'right': original_right}
        else:
            x_plot_data = (x_data - original_left) / (original_right - original_left)
            x_min = 0
            x_max = 1
            #normalization_info[col] = {'type': 'linear', 'left': original_left, 'right': original_right}

        ax.plot(x_plot_data, depth, label=col, color=style["color"], linewidth=style["linewidth"], linestyle=style["style"])
        # Add fill logic
        if style.get("fill") == "left":
            ax.fill_betweenx(depth, x_min, x_plot_data, color=style["color"], alpha=0.3)
        if style.get("fill") == "right":
            ax.fill_betweenx(depth, x_plot_data, x_max, color=style["color"], alpha=0.3)
        if style.get("fill_between"):
            ref_col = style["fill_between"]["reference"]
            ref_data = data[ref_col]
            if styles[ref_col]["type"] == 'log':
                ref_plot_data = ref_data
            else:
                ref_plot_data = (ref_data - styles[ref_col]["left"]) / (styles[ref_col]["right"] - styles[ref_col]["left"])
            
            fill_colors = style["fill_between"]["colors"]
            
            # Fill above reference line
            ax.fill_betweenx(depth, x_plot_data, ref_plot_data, where=x_plot_data >= ref_plot_data, facecolor=fill_colors[0], alpha=0.3)
            
            if "cmap" in style["fill_between"]:
                cmap_name = style["fill_between"]["cmap"]
                cmap = plt.get_cmap(cmap_name)
                
                colorlog = style["fill_between"]["colorlog"]
                cutoffs = style["fill_between"]["cutoffs"]
                
                # Normalize the colorlog data
                norm = mcolors.Normalize(vmin=min(cutoffs), vmax=max(cutoffs))
                
                # Generate colors based on the normalized values
                color_values = cmap(norm(data[colorlog]))
                
                # Fill below reference line with respective colors
                for color in np.unique(color_values, axis=0):
                    mask = (color_values == color).all(axis=-1)
                    ax.fill_betweenx(depth, x_plot_data, ref_plot_data, where=(x_plot_data < ref_plot_data) & mask, facecolor=color, alpha=1)
            elif "colorlog" in style["fill_between"]:
                colorlog = style["fill_between"]["colorlog"]
                cutoffs = style["fill_between"]["cutoffs"]
                cutoff_colors = style["fill_between"]["fillcolors"]
                
                colors = np.full(data[colorlog].shape, 'orange')  # Default color if no cutoff is matched
                for i, cutoff in enumerate(cutoffs):
                    color = cutoff_colors[i]
                    if i == 0:
                        colors = np.where(data[colorlog] < cutoff, color, colors)
                    else:
                        colors = np.where((data[colorlog] < cutoff) & (data[colorlog] >= cutoffs[i-1]), color, colors)
                
                # Fill below reference line with respective colors
                for color in np.unique(colors):
                    mask = colors == color
                    ax.fill_betweenx(depth, x_plot_data, ref_plot_data, where=(x_plot_data < ref_plot_data) & mask, facecolor=color, alpha=1)
            else:
                # Fallback fill behavior if neither "cmap" nor "colorlog" is present
                ax.fill_betweenx(depth, x_plot_data, ref_plot_data, where=x_plot_data < ref_plot_data, facecolor=fill_colors[1], alpha=0.3)                
        if style["type"] == 'log':
            ax.set_xlim(original_left, original_right)
            ax.set_xticks(np.geomspace(original_left, original_right, num=5))
        else:
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, num=5))

        ax.set_ylim(y_min, y_max)
        if invert_yaxis:
            ax.invert_yaxis()

        ax.set_xticklabels([])  # Hide tick labels
        ax.set_xlabel("")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if plot_labels:
            add_label(ax, col, original_left, original_right, label_offsets[track], style["color"], style["linewidth"], style["style"], style["unit"], log_scale=(style["type"] == 'log'))
            label_offsets[track] -= -1.5 * (y_max - y_min) * (label_height / 1000)
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='y', colors='white')

    # Plotting points
    if points is not None and pointstyles is not None:
        for col in points.columns:
            track = pointstyles[col]["track"]
            ax = axes[track]
            y_points = points[col].dropna().index
            x_points = points[col].dropna().values
            style = pointstyles[col]

            if style["type"] == 'log':
                ax.set_xscale('log')
                x_plot_points = x_points
                x_min = style["left"]
                x_max = style["right"]
                #normalization_info[col] = {'type': 'log', 'left': style["left"], 'right': style["right"]}
            else:
                x_plot_points = (x_points - style["left"]) / (style["right"] - style["left"])
                x_min = 0
                x_max = 1
                #normalization_info[col] = {'type': 'linear', 'left': style["left"], 'right': style["right"]}


            ax.scatter(x_plot_points, y_points, color=style["color"], s=style["pointsize"], marker=style["symbol"], zorder=5)
            
            if style.get("uptosurface", False):
                ax.vlines(x_plot_points, y_points, y_max, color=style["color"], linestyles='-', zorder=4)

            if style["type"] == 'log':
                ax.set_xlim(style["left"], style["right"])
                ax.set_xticks(np.geomspace(style["left"], style["right"], num=5))
            else:
                ax.set_xlim(0, 1)
                ax.set_xticks(np.linspace(0, 1, num=5))

            if plot_labels:
                add_label(ax, col, style["left"], style["right"], label_offsets[track], style["color"], 1, "-", style["unit"], log_scale=(style["type"] == 'log'))
                label_offsets[track] -= -1.5 * (y_max - y_min) * (label_height / 1000)

    # Set common y-axis label
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.025)
    
    if plot_labels:
        if pltsign > 0:
            #plt.tight_layout()
            plt.savefig(os.path.join(output_dir,"BottomLabel.png"), dpi=dpi)
            plt.close()
            return
        else:
            plt.savefig(os.path.join(output_dir,'TopLabel.png'), dpi=dpi)
            if to_plotly:
                plotly_fig = tls.mpl_to_plotly(fig)
                #plt.tight_layout()
                plotly_fig.update_layout(
                    autosize=True,
                    width=None,  # Remove the fixed width to make it responsive
                    #height=600,  # Set a fixed height if needed, or leave it auto-sized
                    margin=dict(l=30, r=0, t=0, b=20)  # Adjust margins as needed
                )
                plotly_fig.update_layout(
                    paper_bgcolor='white',  # Background of the entire figure
                    plot_bgcolor='white',   # Background of the plot area
                )
                plotly_fig.update_xaxes(showgrid=False, tickcolor='rgba(0, 0, 0, 0)',showticklabels=False)
                plotly_fig.update_yaxes(automargin=False,showgrid=False, tickcolor='rgba(0, 0, 0, 0)',tickformat='.0f',tickfont=dict(color='rgba(0, 0, 0, 0)'), linecolor='grey', linewidth=1)
                plotly_fig.update_layout(
                    yaxis=dict(showticklabels=True),  # Show labels for the leftmost y-axis
                    yaxis2=dict(showticklabels=False),  # Hide labels for the second column
                    yaxis3=dict(showticklabels=False),# Hide labels for the third column
                    yaxis4=dict(showticklabels=False),
                    yaxis5=dict(showticklabels=False),
                    yaxis6=dict(showticklabels=False),
                )
                plotly_fig.update_layout(margin=dict(l=50,r=50,t=0,b=0))
                pio.write_json(plotly_fig,os.path.join(output_dir,"TopPlotly.json"))
                #plotly_fig.write_image("TopPlotly.png")
            plt.close()
            return
    #print(normalization_info)
    return fig, axes
    


# Define functions to crop and combine images

def choptop(xx,yy,nm='plot1.png'):
    # Open the image
    img = Image.open(nm)

    # Get the original dimensions
    width, height = img.size

    # Define the number of pixels to crop from the top and bottom
    crop_top = xx  # Replace with your value for 'x'
    crop_bottom = yy  # Replace with your value for 'y'

    # Calculate the cropping box: (left, upper, right, lower)
    crop_box = (0, crop_top, width, height - crop_bottom)

    # Crop the image
    cropped_img = img.crop(crop_box)

    # Save the cropped image, overwriting the original
    cropped_img.save(nm)

    # Optionally, show the cropped image to verify the result
    #cropped_img.show()
    


def chopleft(xx, yy, nm='combined_plot.png'):
    
    # Open the image
    img = Image.open(nm)

    # Get the original dimensions
    width, height = img.size

    # Define the number of pixels to crop from the left and right
    crop_left = xx  # Number of pixels to crop from the left
    crop_right = yy  # Number of pixels to crop from the right

    # Calculate the cropping box: (left, upper, right, lower)
    crop_box = (crop_left, 0, width - crop_right, height)

    # Crop the image
    cropped_img = img.crop(crop_box)

    # Save the cropped image, overwriting the original
    cropped_img.save(nm)

    # Optionally, show the cropped image to verify the result
    # cropped_img.show()


    

#chopleft(50,50)
def joinery(n1,n2,n3,x1,x2):
    choptop(x1,x2,n1) #119,109
    # Open the images
    from PIL import Image
    img2 = Image.open(n1)
    img1 = Image.open(n2)

    # Get the width and height of the images
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Create a new image with the combined height of both images
    combined_img = Image.new('RGB', (max(width1, width2), height1 + height2))

    # Paste the images into the new image
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (0, height1))

    # Save the combined image
    combined_img.save(n3)

def joinery2(n1,n2,n3,x1,x2):
    choptop(x1,x2,n2) #119,109
    # Open the images
    from PIL import Image
    img2 = Image.open(n1)
    img1 = Image.open(n2)

    # Get the width and height of the images
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Create a new image with the combined height of both images
    combined_img = Image.new('RGB', (max(width1, width2), height1 + height2))

    # Paste the images into the new image
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (0, height1))

    # Save the combined image
    combined_img.save(n3)


def cutify(n1,n2,n3,x1,x2,x3,x4):
    joinery(n1,n2,n3,x1,x2)
    # Example usage
    chopleft(x3, x4, n3)#120,120
    # Display the combined image
    img1 = Image.open(n3)
    img1.save(n3)
    #img1.show()
    
def cutify2(n1,n2,n3,x1,x2,x3,x4):
    joinery2(n2,n1,n3,x1,x2)
    # Example usage
    chopleft(x3, x4, n3)#120,120
    # Display the combined image
    img1 = Image.open(n3)
    img1.save(n3)
    #img1.show()

def chopify(n1,x1,x2,x3,x4):
    choptop(x1,x2,n1) #119,109
    # Example usage
    chopleft(x3, x4, n1)#120,120
    # Display the combined image
    img1 = Image.open(n1)
    img1.save(n1)
    #img1.show()

#cutify2('plot.png','BottomLabel.png','combined.png',119,109,120,120)
"""
# Example usage of the modified function
data = pd.DataFrame({
    'log1': np.random.random(100) * 150,
    'log2': np.random.random(100) * 200,
}, index=np.linspace(0, 1000, 100))

styles = {
    'log1': {"color": "green", "linewidth": 1.5, "style": '-', "track": 0, "left": 0, "right": 150, "type": 'linear', "unit": "m/s"},
    'log2': {"color": "blue", "linewidth": 1.5, "style": '-', "track": 1, "left": 0, "right": 200, "type": 'linear', "unit": "m/s"},
}

points = pd.DataFrame({
    'point1': np.random.random(10) * 100,
    'point2': np.random.random(10) * 50,
}, index=np.linspace(0, 1000, 10))

pointstyles = {
    'point1': {'color': 'red', 'pointsize': 50, 'symbol': 'o', 'track': 0, 'left': 0, 'right': 100, 'type': 'linear', 'unit': "Mpa", 'uptosurface': True},
    'point2': {'color': 'purple', 'pointsize': 50, 'symbol': 'o', 'track': 1, 'left': 0, 'right': 50, 'type': 'linear', 'unit': "Mpa"},
}

x = plot_logs(data, styles, points, pointstyles, y_min=0, y_max=1000)
"""

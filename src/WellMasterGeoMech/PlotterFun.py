import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
#from PIL import Image

def plot_logs(data, styles, y_min=None, y_max=None, plot_labels=True, figsize=(15, 10), label_height=20, dpi=100):
    """
    Plots well log data in tracks.

    Parameters:
    - data: A pandas DataFrame where each column corresponds to a curve to be plotted.
    - styles: A dictionary where keys are column names from the DataFrame and values are dictionaries of style attributes.
              Example: {'lithology1': {"color": "green", "linewidth": 1.5,"style":'.',"track":0,"left":0,"right":150,"type":'linear'}}
    - invert_yaxis: Boolean to specify if the y-axis should be inverted (default is True).
    - y_min: Minimum value of the y-axis (depth).
    - y_max: Maximum value of the y-axis (depth).
    - plot_labels: Boolean to specify if the labels should be plotted (default is True).
    - figsize: Tuple specifying the width and height of the figure (default is (15, 10)).
    - label_height: Height of the label rectangles (default is 20).
    """
    if y_max>y_min:
        invert_yaxis=True
        pltsign = 1
    else:
         invert_yaxis=False
         pltsign = -1
         
    if len(data.columns) != len(styles):
        raise ValueError("The number of columns in the data must match the number of entries in the styles dictionary.")
    n_tracks = max(style["track"] for style in styles.values()) + 1
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

        # Draw a white rectangle as background for the label
        rect = Rectangle((x_min, y_data_coords - label_height / 2), x_max - x_min, label_height, 
                         linewidth=0, edgecolor='none', facecolor='white', clip_on=False, zorder=2)
        ax.add_patch(rect)

        # Plot the representative line with the correct linestyle
        ax.plot([x_min, x_max], [y_data_coords, y_data_coords], color=color, linewidth=linewidth, linestyle=linestyle, clip_on=False, zorder=3)

        # Plot the text above the representative line
        ax.text(x_min, y_data_coords, f"{original_x_min}", ha='left', va='center', fontsize=10, color=color, backgroundcolor='white', clip_on=False, zorder=4)
        if log_scale:
            center = np.sqrt(original_x_min * original_x_max)
        else:
            center = 0.5
        ax.text(center, y_data_coords, label, ha='center', va='center', fontsize=10, color=color, backgroundcolor='white', clip_on=False, zorder=4)
        ax.text(x_max, y_data_coords, f"{original_x_max}", ha='right', va='center', fontsize=10, color=color, backgroundcolor='white', clip_on=False, zorder=4)
        ax.text(center, y_data_coords+(.14*pltsign), f"{units}", ha='center', va='center', fontsize=7.5, color=color, backgroundcolor='white', clip_on=False, zorder=4)

        
    depth = data.index
    if y_min is None:
        y_min = min(depth)
    if y_max is None:
        y_max = max(depth)

    label_offsets = {track: y_min - (y_max - y_min) * (0.01) for i, track in enumerate(range(n_tracks))}

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
        else:
            x_plot_data = (x_data - original_left) / (original_right - original_left)
            x_min = 0
            x_max = 1

        ax.plot(x_plot_data, depth, label=col, color=style["color"], linewidth=style["linewidth"], linestyle=style["style"])
        # Add fill logic
        if style.get("fill") == "left":
            ax.fill_betweenx(depth, x_min, x_plot_data, color=style["color"], alpha=0.3)
        if style.get("fill") == "right":
            ax.fill_betweenx(depth, x_plot_data, x_max, color=style["color"], alpha=0.3)
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
            add_label(ax, col, original_left, original_right, label_offsets[track], style["color"], style["linewidth"], style["style"],style["unit"], log_scale=(style["type"] == 'log'))
            label_offsets[track] -= -1.5 * (y_max - y_min) * (label_height / 1000)
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='y', colors='white')
            # Adjusted for label height

    # Set common y-axis label
    fig.text(0.04, 0.5, 'Depth', va='center', rotation='vertical', fontsize=12)
    fig.subplots_adjust(wspace=0.025)  # Adjust the width space between subplots
    fig.subplots_adjust(hspace=0)  # Adjust the height space between subplots
    # plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    if plot_labels:
        if pltsign>0:
            plt.savefig('BottomLabel.png', dpi=dpi)
        else:
            plt.savefig('TopLabel.png', dpi=dpi)
    else:
        plt.savefig('plot.png', dpi=dpi)
        simulated_depth = np.array([2600])
        simulated_data = pd.DataFrame({key: np.random.rand(len(data)) for key in data.columns}, index=simulated_depth)
        plot_logs(simulated_data, styles, y_min=2601, y_max=2602, plot_labels=True, figsize=(15, 1.5), label_height=200)
        plot_logs(simulated_data, styles, y_min=2601, y_max=2600, plot_labels=True, figsize=(15, 1.5), label_height=200)
        return fig, axes

"""
# Example usage
depth = np.linspace(2500, 2800, 300)
data = pd.DataFrame({
    'lithology1': np.random.rand(300) * 100,
#    'lithology2': np.random.rand(300) * 100,
#    'lithology3': np.random.rand(300) * 100,
    'resistivity1': np.random.rand(300) * .1 + 1,
    'resistivity2': np.random.rand(300) * .1 + 1,
    'resistivity3': np.random.rand(300) * .1 + 1,
    'resistivity4': np.random.rand(300) * .1 + 1,
    'porosity': np.random.rand(300) * 0.25 + 0.45,
    'density': np.random.rand(300) * 0.5 + 1.9,
    'saturation': np.random.rand(300) * 0.5+0.9,
    'quartz': np.random.rand(300),
    'shale': np.random.rand(300),
    'lime': np.random.rand(300),
    'coal': np.random.rand(300)
}, index=depth)

# Ensure cumulative volume sums to 1.0 for each sample
tv = data[['quartz', 'shale', 'lime', 'coal']].sum(axis=1)
data['quartz'] /= tv
data['shale']=(data['shale'] / tv) +data['quartz']
data['lime']=(data['lime'] / tv) +data['shale']
data['coal']=(data['coal'] / tv) +data['lime']

# Update styles to include new components
styles = {
    "lithology1": {"color": "green", "linewidth": 1.5, "style": ':', "track": 0, "left": 0, "right": 150, "type": 'linear',"unit":'gAPI',"fill":"left"},
    #"lithology2": {"color": "blue", "linewidth": 1.5, "style": '--', "track": 0, "left": 0, "right": 150, "type": 'linear',"unit":'gAPI',"fill":"none"},
    #"lithology3": {"color": "orange", "linewidth": 1.5, "style": '-', "track": 0, "left": 0, "right": 150, "type": 'linear',"unit":'gAPI',"fill":"none"},
    "resistivity1": {"color": "red", "linewidth": 1.5, "style": '-', "track": 1, "left": 0.2, "right": 200, "type": 'log',"unit":'ohm/m',"fill":"none"},
    "resistivity2": {"color": "magenta", "linewidth": 1.5, "style": '-.', "track": 1, "left": 0.2, "right": 200, "type": 'log',"unit":'ohm/m',"fill":"none"},
    "resistivity3": {"color": "cyan", "linewidth": 1.5, "style": '--', "track": 1, "left": 0.2, "right": 200, "type": 'log',"unit":'ohm/m',"fill":"none"},
    "resistivity4": {"color": "purple", "linewidth": 1.5, "style": ':', "track": 1, "left": 0.2, "right": 200, "type": 'log',"unit":'ohm/m',"fill":"none"},
    "porosity": {"color": "blue", "linewidth": 1.5, "style": ':', "track": 2, "left": 0.54, "right": -0.06, "type": 'linear',"unit":'p.u.',"fill":"none"},
    "density": {"color": "brown", "linewidth": 1.5, "style": '-', "track": 2, "left": 1.8, "right": 2.8, "type": 'linear',"unit":'g/cc',"fill":"none"},
    "saturation": {"color": "black", "linewidth": 0.5, "style": ':', "track": 4, "left": 0, "right": 1, "type": 'linear',"unit":'',"fill":"right"},
    "quartz": {"color": "yellow", "linewidth": 0.5, "style": '-', "track": 3, "left": 0, "right": 1, "type": 'linear',"unit":'',"fill":"none"},
    "shale": {"color": "green", "linewidth": 0.5, "style": '-', "track": 3, "left": 0, "right": 1, "type": 'linear',"unit":'',"fill":"none"},
    "lime": {"color": "aqua", "linewidth": 0.5, "style": '-', "track": 3, "left": 0, "right": 1, "type": 'linear',"unit":'',"fill":"none"},
    "coal": {"color": "black", "linewidth": 0.5, "style": '-', "track": 3, "left": 0, "right": 1, "type": 'linear',"unit":'',"fill":"none"}
}

print(data)
# Plot the main image without labels
plot_logs(data, styles, y_min=2670, y_max=2800, plot_labels=False, figsize=(15, 10))

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
def joinery():
    choptop(119,109,'plot.png')
    # Open the images
    from PIL import Image
    img2 = Image.open('plot.png')
    img1 = Image.open('TopLabel.png')

    # Get the width and height of the images
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Create a new image with the combined height of both images
    combined_img = Image.new('RGB', (max(width1, width2), height1 + height2))

    # Paste the images into the new image
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (0, height1))

    # Save the combined image
    combined_img.save('combined_plot.png')



joinery()
# Example usage
chopleft(120, 120)
# Display the combined image
img1 = Image.open('combined_plot.png')
img1.show()
"""
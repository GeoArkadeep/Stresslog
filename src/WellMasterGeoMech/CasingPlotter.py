def plotCasing(depths,ax,length=50):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    matplotlib.use("svg")
    # Shoe depths data and initial setup
    shoe_depths_md = depths
    sorted_depths = np.sort(shoe_depths_md)
    triangle_base = 0.5
    triangle_height = length
    gap = 0.3

    # Calculate positions
    positions = np.arange(len(sorted_depths)) * gap
    symmetric_depths = np.concatenate([sorted_depths, sorted_depths[::-1]])
    symmetric_positions = np.concatenate([positions, positions[-1] + gap + positions])

    # Modified function to add half-triangles at the bottom of a line


    def add_half_triangle(ax, base_center, base_width, height, bottom_y, position_idx, total_positions, color='k'):
        # Determine whether to draw the left or right half based on position
        half_base_width = base_width / 2
        is_left_half = position_idx < total_positions / 2

        if is_left_half:
            # Draw left half of the triangle
            coordinates = [
                (base_center, bottom_y),
                (base_center, bottom_y - height),
                (base_center - half_base_width, bottom_y)
            ]
        else:
            # Draw right half of the triangle
            coordinates = [
                (base_center, bottom_y),
                (base_center + half_base_width, bottom_y),
                (base_center, bottom_y - height)
            ]

        triangle = plt.Polygon(coordinates, closed=True, color=color)
        ax.add_patch(triangle)

    # Plotting adjustments
#    fig, ax = plt.subplots()
    num_lines = len(symmetric_depths)
    total_width_needed = num_lines * gap
#    fig.set_size_inches(total_width_needed * 0.5, 6)

    # Plot lines and add half-triangles at the bottom
    for idx, (pos, depth) in enumerate(zip(symmetric_positions, symmetric_depths)):
        ax.plot([pos, pos], [0, depth], 'k-', lw=2)
        add_half_triangle(ax, pos, triangle_base, triangle_height, depth, idx, len(symmetric_positions))
    # Function to draw a small horizontal line at the bottom of the vertical line
    def add_cover_line(ax, base_center, base_width, bottom_y, position_idx, total_positions, color='k'):
        half_base_width = base_width / 2
        is_left_half = position_idx < total_positions / 2

        if is_left_half:
            # Extend to the left of the base center
            start_x = base_center - half_base_width
            end_x = base_center
        else:
            # Extend to the right of the base center
            start_x = base_center
            end_x = base_center + half_base_width
        
        # Draw the horizontal line
        ax.plot([start_x, end_x], [bottom_y, bottom_y], color=color, lw=2)

    # Modify the plotting loop to include add_cover_line call
    for idx, (pos, depth) in enumerate(zip(symmetric_positions, symmetric_depths)):
        ax.plot([pos, pos], [0, depth], 'k-', lw=2)
        add_cover_line(ax, pos, triangle_base, depth, idx, len(symmetric_positions))  # Cover the line end
        add_half_triangle(ax, pos, triangle_base, triangle_height, depth, idx, len(symmetric_positions))
    
    return ax
    """
    # Axis adjustments
    plt.ylim(-triangle_height, max(symmetric_depths) * 1.05)
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.savefig('casingtrial',dpi=300)
    plt.show()
    """
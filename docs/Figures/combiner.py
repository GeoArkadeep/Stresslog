from PIL import Image
def overlay_images(base_path, overlay_path, output_path, 
                  x_position=0, y_position=0,
                  x_scale=1.0, y_scale=1.0,
                  transparency=0.75,
                  white_threshold=45):  # New parameter for white detection
    """
    Overlay one image on top of another with transformation controls.
    White pixels in the overlay image become completely transparent.
    
    Parameters:
    base_path (str): Path to the base image
    overlay_path (str): Path to the overlay image
    output_path (str): Path where the result will be saved
    x_position (int): X-axis position for overlay (default: 0)
    y_position (int): Y-axis position for overlay (default: 0)
    x_scale (float): X-axis scaling factor (default: 1.0)
    y_scale (float): Y-axis scaling factor (default: 1.0)
    transparency (float): Transparency level 0-1 (default: 0.75)
    white_threshold (int): RGB difference from white to consider transparent (default: 2)
    """
    
    # Open the images
    base_img = Image.open(base_path).convert('RGBA')
    overlay_img = Image.open(overlay_path).convert('RGBA')
    
    # Scale the overlay image
    new_width = int(overlay_img.width * x_scale)
    new_height = int(overlay_img.height * y_scale)
    overlay_img = overlay_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a new RGBA image with the same size as the base image
    result = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
    
    # Apply transparency to the overlay
    overlay_data = overlay_img.getdata()
    transparent_overlay = []
    
    for item in overlay_data:
        # Check if the pixel is close to white
        is_white = all(abs(255 - x) <= white_threshold for x in item[:3])
        
        if is_white:
            # Make white pixels completely transparent
            transparent_overlay.append((255, 255, 255, 0))
        else:
            # Apply normal transparency to non-white pixels
            new_alpha = int(item[3] * transparency)
            transparent_overlay.append((item[0], item[1], item[2], new_alpha))
    
    overlay_img.putdata(transparent_overlay)
    
    # Paste the base image
    result.paste(base_img, (0, 0))
    
    # Paste the overlay at the specified position
    result.paste(overlay_img, (x_position, y_position), overlay_img)
    
    # Save the result
    result.save(output_path, 'PNG')
    
    return result

# Example usage in Python console:

# Use the function
result = overlay_images(
    'FMI-2630.png',        # Base image path
    'PlotBHI3.png',       # Overlay image path
    'overlay2.png',         # Output path
    x_position=-168,       # Move overlay 100 pixels right
    y_position=-52,        # Move overlay 50 pixels down
    x_scale=0.157,         # Stretch horizontally by 50%
    y_scale=0.091,         # Compress vertically by 20%
    transparency=0.800     # 75% transparency
)
import os
from PIL import Image

def reduce_png_resolution(folder: str):
    EXCEPTIONS = {'WellPlot.png', 'overlay.png', 'FMI-2630.png'}
    output_folder = os.path.join(folder, "resized")
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return
    
    for filename in os.listdir(folder):
        if filename.lower().endswith('.png') and filename not in EXCEPTIONS:
            file_path = os.path.join(folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            user_input = input(f"Resize {filename}? (y/N): ").strip().lower()
            if user_input != 'y':
                print(f"Skipping {filename}.")
                continue
            
            scale_factor = input("Enter the scale factor (default: 10): ")
            scale_factor = int(scale_factor) if scale_factor.isdigit() else 10
            
            try:
                with Image.open(file_path) as img:
                    new_width = max(img.width // scale_factor, 1)
                    new_height = max(img.height // scale_factor, 1)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    img.save(output_path)
                print(f"Reduced resolution of {filename} from ({img.width}, {img.height}) to ({new_width}, {new_height}).")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    folder = input("Enter the path to the folder: ")
    reduce_png_resolution(folder)
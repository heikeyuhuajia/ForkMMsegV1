import os
from PIL import Image
from tqdm import tqdm

# Define the paths of folder A and folder B
folder_a_path = '/home/wyuan/code/ForkMMsegV1/workDirs/DiffExp/BeijingBuilding256/segneXt_mscan_tiny/test'
folder_b_path = '/home/wyuan/code/ForkMMsegV1/workDirs/DiffExp/BeijingBuilding256/segneXt_mscan_tiny/test_rgb'

# Create folder B if it doesn't exist
if not os.path.exists(folder_b_path):
    os.makedirs(folder_b_path)

# Get the list of image files in folder A
image_files = [f for f in os.listdir(folder_a_path) if f.endswith('.png') or f.endswith('.jpg')]

# Process each image
for image_file in tqdm(image_files, desc='Processing images'):
    # Load the image
    image_path = os.path.join(folder_a_path, image_file)
    image = Image.open(image_path)

    # Process the pixels
    pixels = image.load()
    width, height = image.size
    for x in range(width):
        for y in range(height):
            if pixels[x, y] == 1:
                pixels[x, y] = 255

    # Save the processed image to folder B
    processed_image_path = os.path.join(folder_b_path, image_file)
    image.save(processed_image_path)

# Display the progress bar
tqdm.close()

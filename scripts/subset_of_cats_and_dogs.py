import os
import shutil

# Define the source and destination folders
path_to_folder = '/home/mpkuse/Downloads/kagglecatsanddogs_5340/PetImages/kaggle/'
source_folders = [path_to_folder+"/cat", path_to_folder+"/dog"]
destination_folder = path_to_folder+"/subset_full/test"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Number of images to copy
offset=10500
num_images = 2000

# Copy the images
for folder in source_folders:
    category = os.path.basename(folder)  # Get 'Cat' or 'Dog'
    dest_category_folder = os.path.join(destination_folder, category)
    os.makedirs(dest_category_folder, exist_ok=True)
    
    # List all images in the folder
    images = sorted([img for img in os.listdir(folder) if img.endswith('.jpg')])[offset:offset+num_images]
    
    for image in images:
        src_path = os.path.join(folder, image)
        dest_path = os.path.join(dest_category_folder, image)
        shutil.copy(src_path, dest_path)

print(f"Successfully copied {num_images} images from each folder to {destination_folder}")

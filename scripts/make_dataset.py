import os
import shutil

# Define the source directory where images are located
base = "/home/mpkuse/Downloads/kagglecatsanddogs_5340/PetImages/kaggle/"
source_dir =  base + 'train' # Replace with your folder path
cat_dir = os.path.join(base, "cat")  # Directory for cat images
dog_dir = os.path.join(base, "dog")  # Directory for dog images

# Create the destination directories if they do not exist
os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

# Loop through the files in the source directory
for filename in os.listdir(source_dir):
    # Skip directories, we only want files
    if os.path.isdir(os.path.join(source_dir, filename)):
        continue
    
    # Check if the file starts with 'cat.' or 'dog.'
    if filename.startswith("cat."):
        # Create the new filename without the 'cat.' prefix
        new_filename = filename[4:]
        # Move the file to the 'cat' folder with the new name
        shutil.move(os.path.join(source_dir, filename), os.path.join(cat_dir, new_filename))
    elif filename.startswith("dog."):
        # Create the new filename without the 'dog.' prefix
        new_filename = filename[4:]
        # Move the file to the 'dog' folder with the new name
        shutil.move(os.path.join(source_dir, filename), os.path.join(dog_dir, new_filename))

print("Files have been successfully moved and renamed!")
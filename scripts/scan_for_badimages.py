import os
from PIL import Image
from tensorflow.keras.preprocessing import image


base_dir = "/home/mpkuse/Downloads/kagglecatsanddogs_5340/PetImages/subset1/validation"
base_dir="/home/mpkuse/Downloads/kagglecatsanddogs_5340/PetImages/kaggle/dog"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify that the file is a valid image
        except (IOError, SyntaxError):
            print(f"Removing invalid file: {file_path}")
            os.remove(file_path)

# Walk through the directory and check each image
problematic_images = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            with Image.open(file_path) as img:
                img_mode = img.mode  # Get image mode (e.g., RGB, L)
                if img_mode not in ['RGB', 'L', 'RGBA']:  # Only support RGB, Grayscale, or RGBA
                    channels = len(img.getbands())  # Get number of channels
                    # if channels != 1 and channels != 3 and channels != 4:
                    if channels == 2:
                        problematic_images.append(file_path)
        except Exception as e:
            print(f"Error opening {file_path}: {e}")

# Print all problematic images
if problematic_images:
    print("Problematic images with unsupported number of channels:")
    for image in problematic_images:
        print(image)
else:
    print("No problematic images found.")
exit()

image_paths = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        file_path = os.path.join(root, file)
        image_paths.append(file_path)

# Try loading each image and printing path
for img_path in image_paths:
    try:
        img = image.load_img(img_path)  # This will load the image without error if it's valid
        img = image.img_to_array(img)
        # print(f"Valid image: {img_path}, shape: {img.shape}")
    except Exception as e:
        print(f"Error with image {img_path}: {e}")
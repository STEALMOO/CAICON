import os
import shutil

# Define source folders
train_images_folder = '/root/caicon/resized_train/images'
val_images_folder = '/root/caicon/resized_val/images'

# Define destination folder
merged_images_folder = '/root/caicon/merged_images/images'

# Create the destination folder if it doesn't exist
os.makedirs(merged_images_folder, exist_ok=True)

# Copy images from train folder to the merged folder
for file_name in os.listdir(train_images_folder):
    src_path = os.path.join(train_images_folder, file_name)
    dst_path = os.path.join(merged_images_folder, file_name)
    shutil.copy(src_path, dst_path)

# Copy images from val folder to the merged folder
for file_name in os.listdir(val_images_folder):
    src_path = os.path.join(val_images_folder, file_name)
    dst_path = os.path.join(merged_images_folder, file_name)
    shutil.copy(src_path, dst_path)

print("Images have been merged successfully!")
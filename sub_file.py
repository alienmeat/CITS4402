import os
import re
import shutil


# This script will create sub file
# Base path where the volumes will be created
base_path = 'archive/BraTS2020_training_data/content/data/'
# Define the directory where the .h5 files are located
source_directory = 'archive/BraTS2020_training_data/content/data'

# Ensure the base path exists
os.makedirs(base_path, exist_ok=True)

# Create the subdirectories for volumes 1 to 369
for i in range(1, 370):
    volume_path = os.path.join(base_path, f'volume_{i}')
    os.makedirs(volume_path, exist_ok=True)

# List the directories to confirm creation
created_directories = os.listdir(base_path)
created_directories.sort()

created_directories

# Define the directory where the .h5 files are located

# List all .h5 files in the directory
all_h5_files = [f for f in os.listdir(source_directory) if f.endswith('.h5')]

# This regex pattern will match the volume number in the file name
pattern = re.compile(r'volume_(\d+)')

# Move each .h5 file to the corresponding volume folder
for file in all_h5_files:
    match = pattern.search(file)
    if match:
        volume_number = match.group(1)
        # Construct the destination directory based on the volume number
        destination_directory = os.path.join(base_path, f'volume_{volume_number}')
        # Move the file
        shutil.move(os.path.join(source_directory, file), destination_directory)

# Check the first few files to ensure they are moved correctly
# We will print out the contents of the first 10 directories
moved_files_check = {}

for i in range(1, 369):
    volume_path = os.path.join(base_path, f'volume_{i}')
    if os.path.exists(volume_path):
        shutil.copy(os.path.join(source_directory, file), destination_directory)



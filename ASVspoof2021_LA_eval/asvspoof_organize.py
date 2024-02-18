import shutil
import os

# Set the paths
metadata_path = 'keys/LA/CM/trial_metadata.txt'
source_folder = 'flac/'
destination_folder = 'flac/training_spoofed/'

# Make sure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

with open(metadata_path, 'r') as file:
    for line in file:
        if 'spoof' in line:
            # Extract the filename
            filename = line.split()[1] + '.flac'  # Assuming the second field is the filename without the extension
            source_file_path = os.path.join(source_folder, filename)
            destination_file_path = os.path.join(destination_folder, filename)

            # Move the file
            shutil.move(source_file_path, destination_file_path)
            print(f"Moved: {filename}")

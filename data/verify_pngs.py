import os
from PIL import Image
import sys

destination_directory = sys.argv[1]

def check_corrupted_files(directory):
    corrupted = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify() 
            except (IOError, SyntaxError) as e:
                print(f"Corrupted file: {filename}")
                corrupted.append(filename)
    return corrupted

corrupted_files = check_corrupted_files(destination_directory)
if corrupted_files:
    print("Corrupted or incomplete files found:", corrupted_files)
else:
    print("No corrupted files found.")

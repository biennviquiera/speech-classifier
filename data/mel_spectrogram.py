import os
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 3:
    print("Usage: script.py <source_directory> <destination_directory")
    sys.exit(1)  

source_directory = sys.argv[1]
destination_directory = sys.argv[2]

# Function to process and save a single audio file
def process_and_save(file_path, destination_path):
    y, sr = librosa.load(file_path, sr=None)
    # Generate the Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    # Convert to dB
    S_DB = librosa.power_to_db(S, ref=np.max)
    # Normalize S_DB to 0-1 range
    S_DB_normalized = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min())
    # Save the normalized Mel-spectrogram to a NumPy file
    np.save(destination_path, S_DB_normalized)

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Process each FLAC file in the source directory
for filename in os.listdir(source_directory):
    print(f"processing file {filename}")
    file_path = os.path.join(source_directory, filename)
    # Construct the output file path (change extension to .npy)
    output_filename = os.path.splitext(filename)[0] + '.npy'
    destination_path = os.path.join(destination_directory, output_filename)
    # Process and save the Mel-spectrogram
    process_and_save(file_path, destination_path)

print("All Mel-spectrograms have been processed and saved.")
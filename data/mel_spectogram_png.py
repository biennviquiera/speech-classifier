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

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

for filename in os.listdir(source_directory):
    output_file_path = os.path.join(destination_directory, os.path.splitext(filename)[0] + '.png')

    if os.path.exists(output_file_path):
        print("already exists")
        continue
    if filename.endswith('.flac'):
        file_path = os.path.join(source_directory, filename)
        y, sr = librosa.load(file_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        # Convert to dB
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        # Save the Mel-spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()
        
        # Construct the output file path
        plt.savefig(output_file_path)
        plt.close()

print("All Mel-spectrograms have been saved to:", destination_directory)

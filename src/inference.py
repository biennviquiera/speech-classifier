import torch
import numpy as np
import sys

from basic_model import CNNWithGAP

if len(sys.argv) < 2:
    print("Usage: inference.py <target npy file>")
    sys.exit(1)  

target_spectrogram = sys.argv[1]

model = CNNWithGAP(num_channels=16, output_size=2)
model.load_state_dict(torch.load('cnn_with_gap_model.pth'))
model.eval()

mel_spec = np.load(target_spectrogram)
new_audio_mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

with torch.no_grad():
    output = model(new_audio_mel_spec_tensor)
    predicted_class = torch.argmax(output, dim=1)
    print(predicted_class)

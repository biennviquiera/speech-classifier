# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from dataset import MelSpectrogramDataset
from basic_model import CNNWithGAP

def collate_fn(batch):
    """Custom collate function for handling variable-size Mel-spectrograms."""
    mel_specs, labels = zip(*batch)
    
    # Find the longest Mel-spectrogram in the batch assuming shape is [1, H, W]
    max_length = max(spec.shape[2] for spec in mel_specs)
    
    # Pad each Mel-spectrogram to match the longest one
    mel_specs_padded = [F.pad(spec, (0, max_length - spec.shape[2])) for spec in mel_specs]
    
    # Stack the padded Mel-spectrograms and labels
    mel_specs_padded = torch.stack(mel_specs_padded)
    labels = torch.tensor(labels)
    
    return mel_specs_padded, labels

paths_train = np.loadtxt('data/paths_train.txt', dtype=str)
paths_val = np.loadtxt('data/paths_val.txt', dtype=str)
paths_test = np.loadtxt('data/paths_test.txt', dtype=str)

labels_train = np.loadtxt('data/labels_train.txt', dtype=int)
labels_val = np.loadtxt('data/labels_val.txt', dtype=int)
labels_test = np.loadtxt('data/labels_test.txt', dtype=int)

train_dataset = MelSpectrogramDataset(paths_train, labels_train)
val_dataset = MelSpectrogramDataset(paths_val, labels_val)
test_dataset = MelSpectrogramDataset(paths_test, labels_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

model = CNNWithGAP(num_channels=16)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
total_steps = len(train_loader)
print('start bruh')
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for i, (mel_specs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        outputs = model(mel_specs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()  # Update model parameters
        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / total_steps
    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}\n')

torch.save(model.state_dict(), 'cnn_with_gap_model.pth')
print('Model saved successfully!')

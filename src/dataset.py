import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import data_utils

class MelSpectrogramDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load Mel-spectrogram (as numpy array), convert to tensor
        mel_spec = np.load(self.paths[idx])
        mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32)
        
        # Add a channel dimension: (C, H, W) = (1, H, W) for CNNs
        if mel_spec_tensor.ndim == 2:  # If the tensor is 2D (H, W), add a channel dimension (C)
            mel_spec_tensor = mel_spec_tensor.unsqueeze(0)  # Shape becomes (1, H, W)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_spec_tensor, label
    
# paths_train = np.loadtxt('data/paths_train.txt', dtype=str)
# labels_train = np.loadtxt('data/labels_train.txt', dtype=int)

# paths_val = np.loadtxt('data/paths_val.txt', dtype=str)
# labels_val = np.loadtxt('data/labels_val.txt', dtype=int)

# paths_test = np.loadtxt('data/paths_test.txt', dtype=str)
# labels_test = np.loadtxt('data/labels_test.txt', dtype=int)

# train_dataset = MelSpectrogramDataset(paths_train, labels_train)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# val_dataset = MelSpectrogramDataset(paths_val, labels_val)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# test_dataset = MelSpectrogramDataset(paths_test, labels_test)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# for mel_specs, labels in train_loader:
#     pass

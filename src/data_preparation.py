# Call this function from the project root directory!
import glob
import os
import numpy as np

path_to_genuine = 'data/genuine_mel_spectrograms'
path_to_spoofed = 'data/spoofed_mel_spectrograms'

paths_real = sorted(glob.glob(os.path.join(path_to_genuine, "*.npy")))
paths_fake = sorted(glob.glob(os.path.join(path_to_spoofed, "*.npy")))

labels_real = [1] * len(paths_real)
labels_fake = [0] * len(paths_fake)

paths = paths_real + paths_fake
labels = labels_real + labels_fake

# Save paths and labels for persistence
with open('data/labels.txt', 'w') as f:
    for path, label in zip(paths, labels):
        f.write(f"{path} {label}\n")

paths = np.array(paths)
labels = np.array(labels)

indices = np.arange(len(paths))
np.random.shuffle(indices)

# Shuffle according to indices arrangement
paths = paths[indices]
labels = labels[indices]

num_train_idx = int(len(paths) * 0.8)
num_val_idx = int(len(paths) * 0.9)

paths_train, paths_val, paths_test = paths[:num_train_idx], paths[num_train_idx:num_val_idx], paths[num_val_idx:]
labels_train, labels_val, labels_test = labels[:num_train_idx], labels[num_train_idx:num_val_idx], labels[num_val_idx:]

# Save for persistence
np.savetxt('data/paths_train.txt', paths_train, fmt='%s')
np.savetxt('data/labels_train.txt', labels_train, fmt='%d')
np.savetxt('data/paths_val.txt', paths_val, fmt='%s')
np.savetxt('data/labels_val.txt', labels_val, fmt='%d')
np.savetxt('data/paths_test.txt', paths_test, fmt='%s')
np.savetxt('data/labels_test.txt', labels_test, fmt='%d')

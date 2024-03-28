'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''

import numpy as np

def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')
            # If there was nothing to read
            if path == '':
                break
            path_list.append(path)

    return path_list


def pad_mel_spectrogram(mel_spec, max_length, pad_value=0):
    """Pads or truncates a Mel-spectrogram to a fixed length."""
    current_length = mel_spec.shape[1]
    if current_length < max_length:
        padding = np.full((mel_spec.shape[0], max_length - current_length), pad_value)
        mel_spec = np.concatenate((mel_spec, padding), axis=1)
    elif current_length > max_length:
        mel_spec = mel_spec[:, :max_length]
    return mel_spec
#!/usr/bin/env python
import numpy as np

WINDOW_SIZE = 11

def hamming_smoothing(signal, window_size):
    padded = np.r_[signal[window_size-1:0:-1], signal, signal[-2:-window_size-1:-1]] # pad the signal at two ends
    window = np.hamming(window_size)
    smoothed = np.convolve(window/window.sum(), padded, mode='valid')
    return smoothed[window_size/2-1:-window_size/2]

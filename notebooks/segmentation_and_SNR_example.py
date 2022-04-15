import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import sys
sys.path.append(os.path.abspath('../src'))
from segmentation import segment_cough, compute_SNR
file = "../sample_recordings/cough.wav"
x,fs = librosa.load(file, sr=None)
plt.plot(x)
plt.title("Input cough signal")
cough_segments, cough_mask = segment_cough(x,fs)
plt.plot(x)
plt.plot(cough_mask)
plt.title("Segmentation Output")
fig, axs = plt.subplots(len(cough_segments),1, figsize=(7,10))
for i in range(0,len(cough_segments)):
    axs[i].plot(cough_segments[i])
    axs[i].set_title("Cough segment " + str(i))
cough_segments, cough_mask = segment_cough(x,fs, cough_padding=0)
plt.plot(x)
plt.plot(cough_mask)
plt.title("Segmentation Output")
fig, axs = plt.subplots(len(cough_segments),1, figsize=(7,10))
for i in range(0,len(cough_segments)):
    axs[i].plot(cough_segments[i])
    axs[i].set_title("Cough segment " + str(i))
snr = compute_SNR(x,fs)
print("The SNR of the cough signal is {0}".format(snr))

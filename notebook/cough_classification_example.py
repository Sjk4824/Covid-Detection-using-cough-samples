import os
import sys
sys.path.append(os.path.abspath('../src'))
from feature_class import features
from DSP import classify_cough
from scipy.io import wavfile
import pickle
data_folder = '../sample_recordings'
loaded_model = pickle.load(open(os.path.join('../models', 'cough_classifier'), 'rb'))
loaded_scaler = pickle.load(open(os.path.join('../models','cough_classification_scaler'), 'rb'))
filename = 'cough.wav'
fs, x = wavfile.read(data_folder+'/'+filename)
probability = classify_cough(x, fs, loaded_model, loaded_scaler)
print("The file {0} has a {1}\% probability of being a cough".format(filename,round(probability*100,2)))
filename = 'not_cough.wav'
fs, x = wavfile.read(data_folder+'/'+filename)
probability = classify_cough(x, fs, loaded_model, loaded_scaler)
print("The file {0} has a {1}\% probability of being a cough".format(filename,round(probability*100,2)))

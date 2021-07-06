import os
import csv
import pickle
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def display_signal(signal):
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def physiological_preprocessing(physiological_data, sampling_rate=128):
    '''
    Preprocesss ppg and gsr and epoches data based on triggers
    Then concat ppg and gsr
    '''

    preprocessed_gsr = gsr_noise_cancelation(physiological_data,
                                             sampling_rate)
    data = normalization(np.array(preprocessed_gsr))
    # display_signal(normalization(np.array(preprocessed_gsr)))

    return data


def gsr_noise_cancelation(data, sampling_rate, low_pass=0.1, high_pass=15):
    '''
    '''
    nyqs = sampling_rate * 0.5
    # Removing high frequency noises
    b, a = signal.butter(5, [low_pass / nyqs, high_pass / nyqs], 'bands')
    output = signal.filtfilt(b, a, np.array(data, dtype=np.float))

    # Removing rapid transient artifacts
    final_output = signal.medfilt(output, kernel_size=5)
    return final_output


def normalization(data):
    # Normalization
    min = np.amin(data)
    max = np.amax(data)
    output = (data - min) / (max - min)
    return output

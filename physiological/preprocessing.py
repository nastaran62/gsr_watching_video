import os
import csv
import pickle
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def display_signal(signal):
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def get_frequency(signal):
    sample_rate = 128
    N = (2 - 0) * sample_rate
    frequency = np.linspace(0.0, 50, int(N/2))
    freq_data = fft(signal)
    y = 2/N * np.abs(freq_data[0:np.int(N/2)])
    plt.plot(frequency, y)
    plt.title('Frequency domain Signal')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Amplitude')
    plt.show()


def physiological_preprocessing(physiological_data, sampling_rate=128):
    '''
    Preprocesss ppg and gsr and epoches data based on triggers
    Then concat ppg and gsr
    '''
    # display_signal(physiological_data)
    # get_frequency(physiological_data)
    preprocessed_gsr = gsr_noise_cancelation(physiological_data,
                                             sampling_rate)
    # display_signal(preprocessed_gsr)
    #data = normalization(np.array(preprocessed_gsr))
    # display_signal(normalization(np.array(preprocessed_gsr)))
    normalized = baseline_normalization(preprocessed_gsr[sampling_rate*3:],
                                        preprocessed_gsr[0:3*sampling_rate],
                                        128)
    # display_signal(normalized)
    return normalized


def gsr_noise_cancelation(data, sampling_rate, low_pass=0.1, high_pass=15):
    '''
    '''
    nyqs = sampling_rate * 0.5
    # Removing high frequency noises
    b, a = signal.butter(5, [low_pass / nyqs, high_pass / nyqs], 'bands')
    output = signal.filtfilt(b, a, np.array(data, dtype=np.float))

    # Removing rapid transient artifacts
    output = signal.medfilt(output, kernel_size=5)
    return output


def baseline_normalization(data, baseline, sampling_rate=128):
    mean = np.mean(baseline)
    # return data - mean
    length = int(baseline.shape[0] / sampling_rate)
    all = []
    for i in range(length):
        all.append(baseline[i*sampling_rate:(i+1)*sampling_rate])
    baseline = np.mean(np.array(all), axis=0)

    window_count = round(data.shape[0] / sampling_rate)
    for i in range(window_count):
        data[i*sampling_rate:(i+1)*sampling_rate] -= baseline
    return data

def normalization(data):
    # Normalization
    min = np.amin(data)
    max = np.amax(data)
    output = (data - min) / (max - min)
    return output 
import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy
from pywt import wavedec
from neurokit.bio.bio_eda import eda_process

def prop_neg_derivatives(arr):
    x = (arr<0).sum()/np.product(arr.shape)
    return x

def get_gsr_features(gsr_data):
    #phasic, tonic = extract_gsr_components(data[i, 0, :], 128)
    #phasic_features = [np.mean(phasic), np.std(phasic)]
    #tonic_features = [np.mean(tonic), np.std(tonic)]
    gsr_features = [np.mean(gsr_data), np.std(gsr_data)]
    diff = np.diff(gsr_data, n=1)
    diff2 = np.diff(gsr_data, n=2)
    d1 = prop_neg_derivatives(diff)
    d2 = prop_neg_derivatives(diff2)
    diff_features = [np.mean(diff), np.std(diff)]
    diff_features2 = [np.mean(diff2), np.std(diff2)]
    feature = \
        gsr_features + diff_features + diff_features2 +d1 +d2
    
    # _get_frequency_features(gsr_data)
    # [gsr_entropy]
    return np.array(feature)



def _get_frequency_features(data):

    (cA, cD) = pywt.dwt([1, 26, 3, 4, 5, 6], 'db1')

    bands = [cA, cD]
    all_features = []
    for band in range(len(bands)):
        power = np.sum(bands[band]**2)
        entropy = np.sum((bands[band]**2)*np.log(bands[band]**2))
        all_features.extend([power, entropy])
    return all_features


def _get_multimodal_statistics(signal_data):
    mean = np.mean(signal_data, axis=1)
    std = np.std(signal_data, axis=1)
    return [mean, std]


def extract_gsr_components(gsr_data, sampling_rate):
    processed_eda = eda_process(gsr_data, sampling_rate=sampling_rate)
    eda = processed_eda['df']
    phasic = eda["EDA_Phasic"]
    tonic = eda["EDA_Tonic"]
    return np.array(phasic), np.array(tonic)
    
    filtered = eda["EDA_Filtered"]
    print(filtered)
    fig1, ax1 = plt.subplots(4, sharex=False, sharey=False)
    ax1[0].plot(filtered)
    ax1[0].set(ylabel="filtered")

    ax1[1].plot(phasic)
    ax1[1].set(ylabel="phasic")
    maxima, = scipy.signal.argrelextrema(np.array(phasic), np.greater)
    minima, = scipy.signal.argrelextrema(np.array(phasic), np.less)
    ax1[1].scatter(maxima, np.array(phasic)[maxima], linewidth=0.03, s=50, c='r')
    ax1[1].scatter(minima, np.array(phasic)[minima], linewidth=0.03, s=50, c='g')

    ax1[2].plot(tonic)
    ax1[2].set(ylabel="tonic")
    maxima, = scipy.signal.argrelextrema(np.array(tonic), np.greater)
    minima, = scipy.signal.argrelextrema(np.array(tonic), np.less)
    ax1[2].scatter(maxima, np.array(tonic)[maxima], linewidth=0.03, s=50, c='r')
    ax1[2].scatter(minima, np.array(tonic)[minima], linewidth=0.03, s=50, c='r')

    plt.show()
    #return np.array(phasic), np.array(tonic)

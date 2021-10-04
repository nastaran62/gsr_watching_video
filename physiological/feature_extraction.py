from os import minor
import numpy as np
import matplotlib.pyplot as plt
import scipy
from neurokit.bio.bio_eda import eda_process
from scipy.fftpack import fft


def prop_neg_derivatives(arr):
    x = (arr < 0).sum()/np.product(arr.shape)
    return x


def get_local_maxima(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Reterns local maximums
    '''
    return [data[i] for i in scipy.signal.argrelextrema(data, np.greater)[0]]


def get_local_minima(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Reterns local minimums
    '''
    return [data[i] for i in scipy.signal.argrelextrema(data, np.less)[0]]


def get_frequency_peak(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Reterns frequency of occurrence of local extremes
    '''
    local_maxima = get_local_maxima(data)
    local_minima = get_local_minima(data)

    freq_extremes = len(local_maxima) + len(local_minima)

    return [freq_extremes]


def get_max_amp_peak(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns the highest value of the determined maximums if it exists. Otherwise it returns zero
    '''
    local_maxima = list(get_local_maxima(data)) + [0]
    return [max(local_maxima)]


def get_var_amp_peak(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns variance of amplitude values calculated for local extremes
    '''
    amplitude_of_local_maxima = np.absolute(get_local_maxima(data))
    amplitude_of_local_minima = np.absolute(get_local_minima(data))
    if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
        return [0]
    variance = np.var(list(amplitude_of_local_maxima) + list(amplitude_of_local_minima))

    return [variance]


def std_amp_peak(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns the standard deviation calculated for local extremes
    '''

    local_extremes = list(get_local_maxima(data)) + list(get_local_minima(data))
    if len(local_extremes) == 0:
        return [0]
    return [np.std(local_extremes)]


def skewness_amp_peak(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Retruns the skewness calculated for amplitude of local extremes
    '''
    amplitude_of_local_maxima = np.absolute(get_local_maxima(data))
    amplitude_of_local_minima = np.absolute(get_local_minima(data))
    if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
        return [0]
    skewness = scipy.stats.skew(list(amplitude_of_local_maxima) + list(amplitude_of_local_minima))

    return [skewness]


def kurtosis_amp_peak(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Retruns the kurtosis calculated for amplitude of local extremes
    '''
    amplitude_of_local_maxima = np.absolute(get_local_maxima(data))
    amplitude_of_local_minima = np.absolute(get_local_minima(data))
    if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
        return [0]
    kurtosis = scipy.stats.kurtosis(list(amplitude_of_local_maxima) +
                                    list(amplitude_of_local_minima))

    return [kurtosis]


def max_abs_amp_peak(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Retruns the kurtosis calculated for amplitude of local extremes
    '''
    amplitude_of_local_maxima = np.absolute(get_local_maxima(data))
    amplitude_of_local_minima = np.absolute(get_local_minima(data))
    if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
        return [0]
    max_val = max(list(amplitude_of_local_maxima) + list(amplitude_of_local_minima))

    return [max_val]


def variance(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns the variance of the data
    '''
    var = np.var(data)

    return [var]


def standard_deviation(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns the standard deviation of the data
    '''
    std = np.std(data)

    return [std]


def skewness(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns the skewness of the data
    '''
    skewness = scipy.stats.skew(data)

    return [skewness]


def kurtosis(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns kurtosis calculated from the data
    '''
    kurtosis = scipy.stats.kurtosis(data)

    return [kurtosis]


def sum_of_positive_derivative(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Retruns the sum of positive values of the first derivative of the data
    '''
    first_derivative = np.diff(data, n=1)
    pos_sum = sum(d for d in first_derivative if d > 0)

    return [pos_sum]


def sum_of_negative_derivative(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns the sum of the negative values of the first derivative of the data
    '''
    first_derivative = np.diff(data, n=1)
    neg_sum = sum(d for d in first_derivative if d < 0)

    return [neg_sum]


def mean(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns the mean of the data
    '''
    mean = np.mean(data)

    return [mean]


def median(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns the median of the data
    '''
    median = np.median(data)

    return [median]


def range(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Retruns the range of the data
    '''
    range = max(data) - min(data)

    return [range]


def maximum(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns the maximum of the data
    '''
    return [max(data)]


def minimum(data):
    '''
    :type data: List[int] - Accepts an array of numbers
    :rtype: List[int] - Returns the minimum of the data
    '''
    return [min(data)]

def group_one(gsr_data):
    return get_frequency_peak(gsr_data) + range(gsr_data) + sum_of_negative_derivative(gsr_data) + standard_deviation(gsr_data)

def group_two(gsr_data):
    return get_max_amp_peak(gsr_data) + mean(gsr_data) + median(gsr_data) + maximum(gsr_data) + minimum(gsr_data)

def group_three(gsr_data):
    return get_var_amp_peak(gsr_data) + std_amp_peak(gsr_data) + max_abs_amp_peak(gsr_data) + variance(gsr_data) + \
        standard_deviation(gsr_data) + sum_of_positive_derivative(gsr_data) + mean(gsr_data) + median(gsr_data) + range(gsr_data) + maximum(gsr_data) + minimum(gsr_data)

def group_four(gsr_data):
    return skewness_amp_peak(gsr_data)

def group_five(gsr_data):
    return kurtosis_amp_peak(gsr_data) + kurtosis(gsr_data)

def group_six(gsr_data):
    return max_abs_amp_peak(gsr_data) + get_var_amp_peak(gsr_data) + std_amp_peak(gsr_data) + max_abs_amp_peak(gsr_data) + variance(gsr_data) + standard_deviation(gsr_data) + sum_of_positive_derivative(gsr_data) + range(gsr_data)

def group_seven(gsr_data):
    return max_abs_amp_peak(gsr_data) + mean(gsr_data) + median(gsr_data) + maximum(gsr_data) + minimum(gsr_data)

def group_eight(gsr_data):
    return skewness_amp_peak(gsr_data)

def ratio_of_minimum_maximum(gsr_data):
    ratio = min(gsr_data)/min(gsr_data)
    return [ratio]

def wearable_emotion_recognition_system_based_feature_extraction(gsr_data):
    # Time domain
    time_domain_features = mean(gsr_data) + median(gsr_data) + standard_deviation(gsr_data) + maximum(gsr_data) + minimum(gsr_data) + ratio_of_minimum_maximum(gsr_data)
    time_domain_first_derivative = np.diff(time_domain_features, n=1)
    time_domain_first_derivative_features = mean(time_domain_first_derivative) + median(time_domain_first_derivative) + standard_deviation(time_domain_first_derivative) + maximum(time_domain_first_derivative) + minimum(time_domain_first_derivative) + ratio_of_minimum_maximum(time_domain_first_derivative)
    time_domain_second_derivative = np.diff(time_domain_features, n=2)
    time_domain_second_derivative_features = mean(time_domain_second_derivative) + median(time_domain_second_derivative) + standard_deviation(time_domain_second_derivative) + maximum(time_domain_second_derivative) + minimum(time_domain_second_derivative) + ratio_of_minimum_maximum(time_domain_second_derivative)

    final_time_domain_features = time_domain_features + time_domain_first_derivative.tolist() + time_domain_first_derivative_features + time_domain_second_derivative.tolist() + time_domain_second_derivative_features

    # Frequency domian
    freq_data = fft(gsr_data).astype(np.float32).tolist()
    final_frequency_domain_features = mean(freq_data) + median(freq_data) + standard_deviation(freq_data) + maximum(freq_data) + minimum(freq_data) + range(freq_data)

    return final_time_domain_features + final_frequency_domain_features

def get_gsr_features(gsr_data):
    gsr_features = []

    # gsr_features += group_one(gsr_data)

    # gsr_features += group_two(gsr_data)

    # gsr_features += group_three(gsr_data)

    # gsr_features += group_four(gsr_data)

    # gsr_features += group_five(gsr_data)
    
    # gsr_features += group_six(gsr_data)
    
    # gsr_features += group_seven(gsr_data)

    gsr_features += group_eight(gsr_data)

    # gsr_features += wearable_emotion_recognition_system_based_feature_extraction(gsr_data)

    return np.array(gsr_features)


def _get_multimodal_statistics(signal_data):
    mean = np.mean(signal_data, axis=1)
    std = np.std(signal_data, axis=1)
    return [mean, std]


def extract_gsr_components(gsr_data, sampling_rate):
    # freq_data = fft(gsr_data)
    processed_eda = eda_process(gsr_data, sampling_rate=sampling_rate)
    eda = processed_eda['df']
    # phasic = np.array(eda["EDA_Phasic"])
    tonic = np.array(eda["EDA_Tonic"])
    # raw_features = get_gsr_features(gsr_data)
    # phasic_features = get_gsr_features(phasic)
    tonic_features = get_gsr_features(tonic)
    # freq_data_features = get_gsr_features(freq_data)
    features = []
    # features += raw_features.tolist() 
    # features += phasic_features.tolist()
    features += tonic_features.tolist() 
    # features += freq_data_features.astype(np.float32).tolist()
    return np.array(features)
    '''
    filtered = eda["EDA_Filtered"]

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
    '''

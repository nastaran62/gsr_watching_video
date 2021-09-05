from os import minor
import numpy as np
import matplotlib.pyplot as plt
import scipy
from neurokit.bio.bio_eda import eda_process


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


def get_gsr_features(gsr_data):
    #phasic, tonic = extract_gsr_components(data[i, 0, :], 128)
    #phasic_features = [np.mean(phasic), np.std(phasic)]
    #tonic_features = [np.mean(tonic), np.std(tonic)]
    # gsr_features = [np.mean(gsr_data), np.std(gsr_data)]
    # diff = np.diff(gsr_data, n=1)
    # diff2 = np.diff(gsr_data, n=2)
    # diff_features = [np.mean(diff), np.std(diff)]
    # diff_features2 = [np.mean(diff2), np.std(diff2)]
    # d1 = prop_neg_derivatives(diff)
    # d2 = prop_neg_derivatives(diff2)

    # feature = \
    #     gsr_features  # + diff_features + diff_features2 + d1 + d2
    # # _get_frequency_features(gsr_data)
    # # [gsr_entropy]
    gsr_features = mean(gsr_data) + \
        standard_deviation(gsr_data) + get_frequency_peak(gsr_data) + \
        get_max_amp_peak(gsr_data) + sum_of_negative_derivative(gsr_data)
    return np.array(gsr_features)


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

import numpy as np
from load_data import load_all_physiological, load_all_labels, load_labels, load_deap_data
from physiological.preprocessing import physiological_preprocessing
from lstm_classification import lstm_classification
from cnn_lstm_classification import cnn_lstm_classification
from simple_classification import feature_classification, kfold_testing


PART_SECONDS = 5
LABEL_TYPE = "arousal"
GSR_SAMPLING_RATE = 128
IGNORE_TIME = 8

if LABEL_TYPE == "emotion":
    CLASSES = [0, 1, 2, 3, 4]
else:
    CLASSES = [0, 1]


def prepare_experimental_data():
    path = "data/prepared_data"
    participant_list = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                        32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43]
    label_path = "data/labels/self_report.csv"
    labels = load_all_labels(label_path)
    all_labels = \
        load_labels(labels, participant_list, type=LABEL_TYPE)
    labels = all_labels
    # CLASSES COUNT
    for i in range(len(CLASSES)):
        print("class count", CLASSES[i], (np.array(all_labels) == CLASSES[i]).sum())

    physiological_data = load_all_physiological(path, participant_list)

    participants, trials = np.array(all_labels).shape
    all_processed_physiological = []
    for p in range(participants):
        all_trials_physiological = []
        for t in range(trials):
            # preprocessing
            # Ignores 8 seconds from the start of each trial
            data = physiological_data[p, t, IGNORE_TIME*GSR_SAMPLING_RATE:, 1]
            preprocessed_physiological = \
                physiological_preprocessing(data,
                                            sampling_rate=GSR_SAMPLING_RATE)

            all_trials_physiological.append(preprocessed_physiological)

        all_processed_physiological.append(all_trials_physiological)
    physiological_data = np.array(all_processed_physiological)

    return physiological_data, labels


def prepare_deap_data():
    # Loading deap dataset
    gsr_data, labels = \
        load_deap_data(label_type=LABEL_TYPE)

    # CLASSES COUNT
    for i in range(len(CLASSES)):
        print("class count", CLASSES[i], (labels == CLASSES[i]).sum())

    all_processed_physiological = []
    for p in range(gsr_data.shape[0]):
        all_trials_physiological = []
        for t in range(gsr_data.shape[1]):
            # preprocessing
            # Ignores IGNORE_TIME seconds from the start of each trial
            data = gsr_data[p, t, 0, IGNORE_TIME*GSR_SAMPLING_RATE:]
            preprocessed_physiological = \
                physiological_preprocessing(data,
                                            sampling_rate=GSR_SAMPLING_RATE)

            all_trials_physiological.append(preprocessed_physiological)

        all_processed_physiological.append(all_trials_physiological)
    gsr_data = np.array(all_processed_physiological)

    return gsr_data, labels


# Loading deap dataset
physiological_data, labels = prepare_deap_data()

# Loading experimental dataset
#physiological_data, labels = prepare_experimental_data()

# lstm_classification(physiological_data, labels, PART_SECONDS,
#                    CLASSES, sampling_rate=GSR_SAMPLING_RATE)
#cnn_lstm_classification(physiological_data, labels, CLASSES)
# feature_classification(physiological_data, labels, PART_SECONDS,
#                       CLASSES, sampling_rate=GSR_SAMPLING_RATE)
kfold_testing(physiological_data, labels, PART_SECONDS,
              CLASSES, sampling_rate=GSR_SAMPLING_RATE)

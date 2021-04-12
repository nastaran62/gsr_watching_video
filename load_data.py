import pathlib
import pandas as pd
import os
import csv
import pickle
import numpy as np
import collections
import random


def balancing_data(data, labels, classes):
    '''
    '''
    joint = list(zip(data, labels))
    random.shuffle(joint)
    data, labels = zip(*joint)

    class_0_data = []
    class_0_labels = []
    class_1_data = []
    class_1_labels = []
    each_label_count = len(labels) / len(classes)
    i = 0
    while i < len(labels):

        if labels[i] == classes[0]:
            if len(class_0_data) < each_label_count:
                class_0_data.append(data[i])
                class_0_labels.append(classes[0])
        elif labels[i] == classes[1]:
            if len(class_1_data) < each_label_count:
                class_1_data.append(data[i])
                class_1_labels.append(classes[1])
        i += 1
    # Oversample each list which is shorter
    if len(class_0_data) < each_label_count:
        i = 0
        while len(class_0_data) < each_label_count:
            class_0_data.append(class_0_data[i])
            class_0_labels.append(classes[0])
            i += 1
    if len(class_1_data) < each_label_count:
        i = 0
        while len(class_1_data) < each_label_count:
            class_1_data.append(class_1_data[i])
            class_1_labels.append(classes[1])
            i += 1

    final_data = class_0_data + class_1_data
    final_labels = class_0_labels + class_1_labels

    joint = list(zip(final_data, final_labels))
    random.shuffle(joint)
    final_data, final_labels = zip(*joint)

    return final_data, final_labels


def load_all_labels(path):
    all_participants = {}
    data_frame = pd.read_csv(path)
    for index, row in data_frame.iterrows():
        participant_number = row["participant_id"][1:]
        if participant_number not in all_participants.keys():
            label_dict = {"stimulus_id": [],
                          "emotion": [],
                          "intensity": [],
                          "valence": [],
                          "arousal": [],
                          "dominance": []}
            all_participants[participant_number] = label_dict

        all_participants[participant_number]["stimulus_id"].append(row["stimulus_id"])
        if row["emotion"] == 1:
            all_participants[participant_number]["emotion"].append(0)
        elif row["emotion"] == 3:
            all_participants[participant_number]["emotion"].append(1)
        elif row["emotion"] == 4:
            all_participants[participant_number]["emotion"].append(2)
        elif row["emotion"] == 5:
            all_participants[participant_number]["emotion"].append(3)
        elif row["emotion"] == 6:
            all_participants[participant_number]["emotion"].append(4)

        all_participants[participant_number]["intensity"].append(row["intensity"])
        if row["valence"] < 4:
            all_participants[participant_number]["valence"].append(0)
        elif row["valence"] >= 4:
            all_participants[participant_number]["valence"].append(1)
        if row["arousal"] > 4:
            all_participants[participant_number]["arousal"].append(1)
        else:
            all_participants[participant_number]["arousal"].append(0)
        if row["dominance"] >= 4:
            all_participants[participant_number]["dominance"].append(1)
        else:
            all_participants[participant_number]["dominance"].append(0)
    return all_participants


def load_all_physiological(path, all_participants):
    all_physiologcal = []
    for participant_number in all_participants:
        all_trials_data = []
        trials_path = os.path.join(path,
                                   "p{0}".format(str(participant_number).zfill(2)),
                                   "physiological_csv")
        all_trials = os.listdir(trials_path)
        all_trials.sort()
        for trial in all_trials:
            trial_path = os.path.join(trials_path, trial)
            trial_data = np.loadtxt(trial_path, delimiter=',')
            all_trials_data.append(trial_data)
        all_physiologcal.append(np.array(all_trials_data))
    print(np.array(all_physiologcal).shape)
    return np.array(all_physiologcal)


def load_labels(labels, participant_list, type="emotion"):
    label_array = []
    for participant_number in participant_list:
        label_array.append(labels[str(participant_number)][type])
    return np.array(label_array)


def load_deap_data(label_type="arousal"):
    path = "deap_data/prepared_data"
    all_participants = os.listdir(path)
    all_participants.sort()
    all_participants_gsr = []
    all_labels = []
    for item in all_participants:
        gsr_file_path = os.path.join(os.path.join(path, item), "gsr.csv")
        labels_path = os.path.join(os.path.join(path, item), "labels.csv")
        gsr_data = pickle.load(open(gsr_file_path, "rb"))
        (arousal, valence, dominance) = pickle.load(open(labels_path, "rb"))

        if label_type == "arousal":
            labels = arousal
        elif label_type == "valence":
            labels = valence
        elif label_type == "dominance":
            labels = dominance

        # Balancing data per participant
        #gsr_data, labels = balancing_data(gsr_data, labels, [0, 1])

        all_labels.append(labels)
        all_participants_gsr.append(gsr_data)

    return np.array(all_participants_gsr), np.array(all_labels)


'''
if __name__ == "__main__":
    load_deap_data()

    main_path = "data/prepared_data"
    label_path = "data/labels/self_report.csv"
    participant_list = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                        32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43]
    physiological = load_all_physiological(main_path, participant_list)

    # eeg = load_all_eeg(main_path, participant_list)
    labels = load_all_labels(label_path)
    all_labels = \
        load_labels(labels, participant_list, type="emotion")
    print("all_labels", all_labels.shape)
'''

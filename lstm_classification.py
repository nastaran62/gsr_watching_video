import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pickle

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, \
    precision_recall_fscore_support
from sklearn.utils import class_weight

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

from physiological.feature_extraction import get_gsr_features
from physiological.preprocessing import physiological_preprocessing
from load_data import load_all_physiological, load_all_labels, load_labels, load_deap_data
from utils import validate_predictions


def prepare_experimental_data(classes, label_type, sampling_rate, ignore_time):
    label_path = "data/labels/self_report.csv"
    participant_list = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                        32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43]
    labels = load_all_labels(label_path)
    all_labels = \
        load_labels(labels, participant_list, type=label_type)
    labels = all_labels
    # CLASSES COUNT
    for i in range(len(classes)):
        print("class count", classes[i], (np.array(all_labels) == classes[i]).sum())

    main_path = "data/prepared_data"
    physiological_data = load_all_physiological(main_path, participant_list)

    participants, trials = np.array(all_labels).shape
    all_processed_physiological = []
    for p in range(participants):
        all_trials_physiological = []
        for t in range(trials):
            # preprocessing
            # Ignores 8 seconds from the start of each trial
            data = physiological_data[p, t, ignore_time*sampling_rate:, 0]
            preprocessed_physiological = \
                physiological_preprocessing(data,
                                            sampling_rate=sampling_rate)

            all_trials_physiological.append(preprocessed_physiological)

        all_processed_physiological.append(all_trials_physiological)
    physiological_data = np.array(all_processed_physiological)

    return physiological_data, labels


def prepare_deap_data(classes, label_type, sampling_rate, ignore_time):
    # Loading deap dataset
    gsr_data, labels = \
        load_deap_data(label_type=label_type)

    # CLASSES COUNT
    for i in range(len(classes)):
        print("class count", classes[i], (labels == classes[i]).sum())

    all_processed_physiological = []
    for p in range(gsr_data.shape[0]):
        all_trials_physiological = []
        for t in range(gsr_data.shape[1]):
            # preprocessing
            # Ignores IGNORE_TIME seconds from the start of each trial
            data = gsr_data[p, t, 0, ignore_time*sampling_rate:]
            preprocessed_physiological = \
                physiological_preprocessing(data,
                                            sampling_rate=sampling_rate)

            all_trials_physiological.append(preprocessed_physiological)

        all_processed_physiological.append(all_trials_physiological)
    physiological_data = np.array(all_processed_physiological)

    return physiological_data, labels


def lstm_classification(classes, label_type, sampling_rate, part_seconds, ignore_time):
    '''
    Classify data using lstm method
    '''
    # Loading deap dataset
    physiological_data, labels = \
        prepare_deap_data(classes, label_type, sampling_rate, ignore_time)

    # Loading experimental dataset
    # gsr_data, labels = \
    #    prepare_experimental_data(classes, label_type, sampling_rate, ignore_time)

    participants, trials = np.array(labels).shape
    all_physiological_features = []
    i = 0

    participants, trials, points = physiological_data.shape
    part_length = part_seconds * sampling_rate
    part_count = int(points / part_length)
    all_participants_labels = []
    for p in range(participants):
        all_trials_physiological = []
        all_trial_labels = []
        for t in range(trials):
            physiological_parts = []
            for i in range(part_count):
                gsr_data = \
                    physiological_data[p,
                                       t,
                                       i * part_length:(i+1)*part_length]
                gsr_labels = labels[p, t]

                physiological_parts.append(get_gsr_features(gsr_data))
                i += 1
            all_trial_labels.append(gsr_labels)
            all_trials_physiological.append(physiological_parts)
        all_participants_labels.append(all_trial_labels)
        all_physiological_features.append(all_trials_physiological)
    physiological_data = np.array(all_physiological_features)
    all_participants_labels = np.array(all_participants_labels)

    physiological_train, physiological_test, \
        y_train, y_test = \
        normal_train_test_split(physiological_data, all_participants_labels)

    #physiological_train, physiological_test, \
    #   y_train, y_test = \
    #    leave_one_trial_out_split(physiological_data,
    #                              np.array(all_participants_labels), trial_out=0)

    preds_physiological = \
        physiological_lstm_classifier(physiological_train, physiological_test,
                                      y_train, y_test, classes)
    print(validate_predictions(preds_physiological, y_test, classes))


def feature_scaling_for_lstm(train, test, method="standard"):
    scaler = StandardScaler()
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    train_samples, train_times, train_features = train.shape
    test_samples, test_times, test_features = test.shape
    train = train.reshape(train_samples*train_times, train_features)
    test = test.reshape(test_samples*test_times, test_features)
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    train = train.reshape(train_samples, train_times, train_features)
    test = test.reshape(test_samples, test_times, test_features)
    return train, test


def physiological_lstm_classifier(train_x, test_x, train_y, test_y, classes):
    '''
    Classifying physiological features
    '''
    print("train_x", train_x.shape)
    train_x, test_x = \
        feature_scaling_for_lstm(train_x, test_x, method="standard")
    class_weights = \
        class_weight.compute_class_weight('balanced',
                                          np.unique(train_y),
                                          train_y)
    class_weights = dict(enumerate(class_weights))
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    print("start_classification")
    checkpoint = \
        ModelCheckpoint("models/physiological_model.h5",
                        monitor='val_accuracy',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=False,
                        mode='auto',
                        period=1)
    early_stopping = \
        EarlyStopping(monitor='val_loss',
                      patience=100,
                      verbose=1,
                      mode='auto')
    model = simple_lstm((train_x.shape[1], train_x.shape[2]),
                        80,  # lstm layers
                        len(classes),  # number of classes
                        dropout=0.5)
    model.summary()

    model.fit(np.array(train_x),
              np.array(train_y),
              batch_size=32,
              epochs=1000,
              class_weight=class_weights,
              validation_data=(np.array(test_x),
                               np.array(test_y)),
              callbacks=[checkpoint, early_stopping])
    physiological_model = load_model("models/physiological_model.h5")
    preds_physiological = physiological_model.predict_proba(np.array(test_x))
    return preds_physiological


def simple_lstm(input_shape, lstm_layers, num_classes, dropout=0.7):
    '''
    Model definition
    '''
    model = Sequential()
    model.add(LSTM(lstm_layers, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(30))
    # model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy",  # categorical_crossentropy
                  optimizer=optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    return model


def normal_train_test_split(physiological_data, labels):
    '''
    mixing all trials from all participnats and Spliting them to train and test
    '''
    physiological_features = \
        physiological_data.reshape(-1, *physiological_data.shape[-2:])
    labels = \
        np.array(labels).reshape(-1)
    physiological_features, labels = \
        shuffle(np.array(physiological_features),
                np.array(labels))

    physiological_train, physiological_test, y_train, y_test = \
        train_test_split(np.array(physiological_features),
                         np.array(labels),
                         test_size=0.3,
                         random_state=200,
                         stratify=labels)

    return physiological_train, physiological_test, \
        y_train, y_test


def leave_one_subject_out_split(physiological_data, labels, subject_out=0):
    '''
    Splitting data based on leave one subject out strategy for lstm classification
    One participant's trials are in test set and others in train set
    '''
    participants_count, trials, parts, features = physiological_data.shape

    physiological_train = \
        np.concatenate((physiological_data[0:subject_out, :, :, :],
                        physiological_data[subject_out+1:participants_count, :, :, :]),
                       axis=0)
    y_train = \
        np.concatenate((labels[0:subject_out, :],
                        labels[subject_out+1:participants_count, :]),
                       axis=0)

    physiological_test = physiological_data[subject_out, :, :, :]
    y_test = labels[subject_out, :]

    physiological_train = \
        physiological_train.reshape(-1, *physiological_train.shape[-2:])

    physiological_test = \
        physiological_test.reshape(-1, *physiological_test.shape[-2:])
    y_test = \
        np.array(y_test).reshape(-1)
    y_train = \
        np.array(y_train).reshape(-1)

    physiological_train, y_train = \
        shuffle(physiological_train,
                y_train)

    return physiological_train, physiological_test, \
        y_train, y_test


def leave_one_trial_out_split(physiological_data, labels, trial_out=0):
    '''
    Splitting data based on leave one trial out strategy for lstm classification
    One trial of each participant is in test set and others in train set
    '''
    participants_count, trials, parts, features = physiological_data.shape

    physiological_train = \
        np.concatenate((physiological_data[:, 0:trial_out, :, :],
                        physiological_data[:, trial_out+1:trials, :, :]),
                       axis=1)
    y_train = \
        np.concatenate((labels[:, 0:trial_out],
                        labels[:, trial_out+1:trials]),
                       axis=1)

    physiological_test = physiological_data[:, trial_out, :, :]
    y_test = labels[:, trial_out]

    physiological_train = \
        physiological_train.reshape(-1, *physiological_train.shape[-2:])

    physiological_test = \
        physiological_test.reshape(-1, *physiological_test.shape[-2:])
    y_test = \
        np.array(y_test).reshape(-1)
    y_train = \
        np.array(y_train).reshape(-1)

    physiological_train, y_train = \
        shuffle(physiological_train,
                y_train)

    return physiological_train, physiological_test, \
        y_train, y_test


def visualize_physiological_data(physiological_features, emotion_labels):
    dict = {"gsr_mean": physiological_features[:, 0],
            "gsr_std": physiological_features[:, 1],
            "ppg_mean": physiological_features[:, 2],
            "ppg_std": physiological_features[:, 3],
            "emotions": emotion_labels, }

    data = pd.DataFrame(dict)
    sns.set_style("whitegrid")
    # sns.FacetGrid(data, hue="emotions", size=4) \
    #   .map(plt.scatter, "gsr_mean", "gsr_std") \
    #   .add_legend()
    sns.pairplot(data, hue="emotions", size=4)

    plt.show()

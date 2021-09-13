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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

from physiological.feature_extraction import get_gsr_features
from utils import validate_predictions


def lstm_classification(physiological_data, labels, part_seconds, classes, sampling_rate=128):
    '''
    Classify data using lstm method
    '''
    participants, trials = np.array(labels).shape
    participants, trials, points = physiological_data.shape
    part_length = part_seconds * sampling_rate
    part_count = int(points / part_length)
    all_physiological_features = []
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

    mean = np.mean(train)
    std = np.std(train)
    train = (train - mean) / std
    test = (test - mean) / std
    '''
    scaler = StandardScaler()
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    train_samples, train_times, train_features = train.shape
    test_samples, test_times, test_features = test.shape
    train = train.reshape(train_samples, train_times*train_features)
    test = test.reshape(test_samples, test_times*test_features)
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    train = train.reshape(train_samples, train_times, train_features)
    test = test.reshape(test_samples, test_times, test_features)
    '''
    return train, test


def physiological_lstm_classifier(train_x, test_x, train_y, test_y, classes):
    '''
    Classifying physiological features
    '''
    print("train_x", train_x.shape)
    if not os.path.exists("models"):
        os.mkdir("models")
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

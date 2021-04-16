
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import numpy as np
import pickle

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.models import Sequential
from keras.models import Model
from keras import optimizers
from keras.layers import Dense, Dropout, LSTM

from sklearn.utils import class_weight
from utils import validate_predictions


def cnn_lstm_classification(physiological_data, labels, classes):
    print(physiological_data.shape)
    print(labels.shape)
    train_x, test_x, \
        train_y, test_y = \
        normal_train_test_split(physiological_data, labels)
    preds_physiological = cnn_lstm(train_x, test_x, train_y, test_y)
    print(validate_predictions(preds_physiological, test_y, classes))


def cnn_lstm(train_x, test_x, train_y, test_y):
    if not os.path.exists("models"):
        os.path.mkdir("models")
    class_weights = \
        class_weight.compute_class_weight('balanced',
                                          np.unique(train_y),
                                          train_y)
    class_weights = dict(enumerate(class_weights))
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    # define model
    verbose, epochs, batch_size = 1, 25, 32
    print(train_x.shape)
    print(train_y.shape)
    n_timesteps, n_outputs = train_x.shape[1], train_y.shape[1]

    n_steps = 10
    n_features = 1
    # reshape data into time steps of sub-sequences
    n_length = int(n_timesteps/n_steps)
    train_x = train_x.reshape((train_x.shape[0], n_steps, n_length, n_features))
    test_x = test_x.reshape((test_x.shape[0], n_steps, n_length, n_features))
    print(train_x.shape)
    print(test_x.shape)

    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
                              input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(80))
    model.add(Dropout(0.5))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    checkpoint = \
        ModelCheckpoint("models/physiological_cnn_model.h5",
                        monitor='val_accuracy',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=False,
                        mode='auto',
                        period=1)
    early_stopping = \
        EarlyStopping(monitor='val_loss',
                      patience=50,
                      verbose=1,
                      mode='auto')
    model.summary()
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
              class_weight=class_weights,
              validation_data=(np.array(test_x),
                               np.array(test_y)),
              callbacks=[checkpoint, early_stopping])
    # evaluate model
    _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    physiological_model = load_model("models/physiological_cnn_model.h5")
    preds_physiological = physiological_model.predict_proba(np.array(test_x))
    print(accuracy)
    return preds_physiological


def normal_train_test_split(physiological_data, labels):
    physiological_features = \
        physiological_data.reshape(-1, physiological_data.shape[-1])
    labels = \
        np.array(labels).reshape(-1)
    physiological_features, labels = \
        shuffle(np.array(physiological_features),
                np.array(labels))

    # Trial-based splitting
    physiological_train, physiological_test, y_train, y_test = \
        train_test_split(np.array(physiological_features),
                         np.array(labels),
                         test_size=0.3,
                         random_state=200,
                         stratify=labels)

    return physiological_train, physiological_test, \
        y_train, y_test

    def leave_one_subject_out_split(physiological_data, labels, subject_out=0):
        print(physiological_data.shape)
        participants_count, trials, features = physiological_data.shape

        physiological_train = \
            np.concatenate((physiological_data[0:subject_out, :, :],
                            physiological_data[subject_out+1:participants_count, :, :]),
                           axis=0)
        print(labels.shape)
        y_train = \
            np.concatenate((labels[0:subject_out, :],
                            labels[subject_out+1:participants_count, :]),
                           axis=0)

        physiological_test = physiological_data[subject_out, :, :]
        y_test = labels[subject_out, :]

        physiological_train = \
            physiological_train.reshape(-1, physiological_train.shape[-1])

        physiological_test = \
            physiological_test.reshape(-1, physiological_test.shape[-1])
        y_test = \
            np.array(y_test).reshape(-1)
        y_train = \
            np.array(y_train).reshape(-1)

        physiological_train, y_train = \
            shuffle(physiological_train,
                    y_train)

        scaler = MinMaxScaler()
        # Fit on training set only.
        # Apply transform to both the training set and the test set.
        scaler.fit(physiological_train)
        physiological_train = scaler.transform(physiological_train)
        physiological_test = scaler.transform(physiological_test)

        return physiological_train, physiological_test, \
            y_train, y_test

    def leave_one_trial_out_split(physiological_data, labels, trial_out=0):
        participants_count, trials, features = physiological_data.shape

        physiological_train = \
            np.concatenate((physiological_data[:, 0:trial_out, :],
                            physiological_data[:, trial_out+1:trials, :]),
                           axis=1)
        y_train = \
            np.concatenate((labels[:, 0:trial_out],
                            labels[:, trial_out+1:trials]),
                           axis=1)

        physiological_test = physiological_data[:, trial_out, :]
        y_test = labels[:, trial_out]

        physiological_train = \
            physiological_train.reshape(-1, physiological_train.shape[-1])

        physiological_test = \
            physiological_test.reshape(-1, physiological_test.shape[-1])
        y_test = \
            np.array(y_test).reshape(-1)
        y_train = \
            np.array(y_train).reshape(-1)

        physiological_train, y_train = \
            shuffle(physiological_train,
                    y_train)

        scaler = MinMaxScaler()
        # Fit on training set only.
        # Apply transform to both the training set and the test set.
        scaler.fit(physiological_train)
        physiological_train = scaler.transform(physiological_train)
        physiological_test = scaler.transform(physiological_test)

        return physiological_train, physiological_test, \
            y_train, y_test

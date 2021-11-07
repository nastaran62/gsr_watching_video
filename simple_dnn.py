import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.utils import class_weight
from utils import validate_predictions
import tensorflow as tf


def simple_dnn_classification(physiological_data, labels, classes):
    print(physiological_data.shape)
    print(labels.shape)
    train_x, test_x, \
        train_y, test_y = \
        normal_train_test_split(physiological_data, labels)
    preds_physiological = simple_dnn(train_x, test_x, train_y, test_y)
    print(validate_predictions(preds_physiological, test_y, classes))


def simple_dnn(train_x, test_x, train_y, test_y):

    print("Train x shape", train_x.shape)
    print("Train y shape", train_y.shape)
    print("Test x shape", test_x.shape)
    print("Test y shape", test_y.shape)
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


    print("Train x shape", train_x.shape)
    print("Train y shape", train_y.shape)
    print("Test x shape", test_x.shape)
    print("Test y shape", test_y.shape)
    verbose, epochs, batch_size = 1, 100, 1


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64,input_shape=(train_x.shape[1],), activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # fit network
    checkpoint = \
        ModelCheckpoint("models/physiological_simple_dnn_model.h5",
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

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
              class_weight=class_weights,
              validation_data=(np.array(test_x),
                               np.array(test_y)),
              callbacks=[checkpoint, early_stopping])
    model.summary()
    # evaluate model
    _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    physiological_model = load_model("models/physiological_simple_dnn_model.h5")
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
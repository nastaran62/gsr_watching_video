from lstm_classification import lstm_classification
from cnn_lstm_classification import cnn_lstm_classification
from simple_classification import feature_classification


PART_SECONDS = 1
LABEL_TYPE = "arousal"
GSR_SAMPLING_RATE = 128
IGNORE_TIME = 8

if LABEL_TYPE == "emotion":
    CLASSES = [0, 1, 2, 3, 4]
else:
    CLASSES = [0, 1]

#lstm_classification(CLASSES, LABEL_TYPE, GSR_SAMPLING_RATE, PART_SECONDS, IGNORE_TIME)
#cnn_lstm_classification(CLASSES, LABEL_TYPE, GSR_SAMPLING_RATE, PART_SECONDS, IGNORE_TIME)
feature_classification(CLASSES, LABEL_TYPE, GSR_SAMPLING_RATE, PART_SECONDS, IGNORE_TIME)

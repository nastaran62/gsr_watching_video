import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, \
    mean_squared_error, classification_report, accuracy_score


def validate_predictions(preds_y, test_Y, classes):
    predicted_labels = []
    predicted_indexes = np.argmax(preds_y, axis=1)
    predicted_labels = []
    for item in predicted_indexes:
        predicted_labels.append(classes[item])
    predicted_labels = np.array(predicted_labels)
    test_y = np.array(test_Y)  # np.argmax(test_Y, axis=1)

    accuracy = \
        accuracy_score(test_y,
                       predicted_labels)
    print(confusion_matrix(test_y, predicted_labels))
    precision, recall, f_score, support = \
        precision_recall_fscore_support(test_y,
                                        predicted_labels,
                                        average='weighted')
    print("precision={}, recall={}, f_score={}, support={}".format(precision,
                                                                   recall,
                                                                   f_score,
                                                                   support))

    return accuracy, precision, recall, f_score

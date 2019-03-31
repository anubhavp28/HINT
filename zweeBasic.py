from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import tensorflow
import PIL
import keras.backend as K

#def auc(y_true, y_pred):
#    auc = tensorflow.metrics.auc(y_true, y_pred)[1]
#    K.get_session().run(tensorflow.local_variables_initializer())
#    return auc

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def matthews_correlation(y_true, y_pred):
    """Matthews correlation metric.
# Aliases

    It is only computed as a batch-wise average, not globally.
    Computes the Matthews correlation coefficient measure for quality
    of binary classification problems.
    """
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

class TestLangClass:
    def __init__(self):
        self.HYPERPARAMETER = {}
    
TestLang = TestLangClass()

def hyperparameter(value, name):
    global TestLang
    if type(value) != tuple:
        TestLang.HYPERPARAMETER[name] = value
    else:
        TestLang.HYPERPARAMETER[name] = str(value)
    return value

def predict_func(func):
    def wrapper():
        import os

        global TestLang
        TestLang.NO_OF_CORRECT_PREDICTIONS = 0
        TestLang.NO_OF_WRONG_PREDICTIONS = 0
        TestLang.TOTAL_PREDICTION_MADE = 0

        for (dirpath, dirnames, filenames) in os.walk('test_dataset'):
            for filename in filenames:
                label = filename.split('.')[0].lower()        
                predicted_label = func(os.path.join('test_dataset', filename))
                print(predicted_label, predicted_label == label)
                TestLang.TOTAL_PREDICTION_MADE += 1
                TestLang.NO_OF_CORRECT_PREDICTIONS += int(predicted_label == label)
                TestLang.NO_OF_WRONG_PREDICTIONS += int(predicted_label != label)

        print("TOTAL_PREDICTION_MADE", TestLang.TOTAL_PREDICTION_MADE)
        print("NO_OF_CORRECT_PREDICTIONS", TestLang.NO_OF_CORRECT_PREDICTIONS)
        print("NO_OF_WRONG_PREDICTIONS", TestLang.NO_OF_WRONG_PREDICTIONS)
    return wrapper

from pymongo import MongoClient
client = MongoClient('3.94.109.234')
db = client.ci
print("Prev build")
import pymongo
builds = list(db.builds.find().sort([('build_time', pymongo.DESCENDING)]))

TestLang.prev = []
def use_prev(i):
    for j in range(i):    
        prev_build = builds[j]
        print(prev_build)
        t = TestLangClass()
        for k, v in prev_build.items():
            if k.isupper():
                setattr(t, k, v)
        TestLang.prev.append(t)
    
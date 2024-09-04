import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

# Define the Hamming score function
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            union_len = len(set_true.union(set_pred))
            if union_len == 0:
                tmp_a = 0  # Handle the case where both sets are empty. Set to 0 instead of NaN to avoid division.
            else:
                tmp_a = len(set_true.intersection(set_pred)) / float(union_len)
        acc_list.append(tmp_a)
    return np.mean(acc_list)

# Hamming score metric wrapper for Keras
def hamming_score_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(tf.round(y_pred), tf.bool)
    return tf.py_function(hamming_score, (y_true, y_pred), tf.double)

# Weighted F1 Score metric
def weighted_f1_score_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    return tf.py_function(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0), (y_true, y_pred), tf.double)

# Sampled F1 Score metric
def sampled_f1_score_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    return tf.py_function(lambda y_true, y_pred: f1_score(y_true, y_pred, average='samples', zero_division=0), (y_true, y_pred), tf.double)

# Precision metric
def precision_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    return tf.py_function(lambda y_true, y_pred: precision_score(y_true, y_pred, average='samples', zero_division=0), (y_true, y_pred), tf.double)

# Recall metric
def recall_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    return tf.py_function(lambda y_true, y_pred: recall_score(y_true, y_pred, average='samples', zero_division=0), (y_true, y_pred), tf.double)

# Exact Match Ratio metric (also known as Subset Accuracy)
def exact_match_ratio_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    return tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(y_true, y_pred), axis=1), tf.float32))

# Accuracy metric
def accuracy_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    return tf.py_function(accuracy_score, (y_true, y_pred), tf.double)


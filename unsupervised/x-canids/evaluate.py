# Date: 05-31-2023
# Author: Mario Freund
# Purpose: Evaluate x-canids

import argparse
import sys
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def detect_intrusions(dataset, reconstruction, O, O_i, window, signals):
    # convert attack_dataset to numpy array
    length = 0
    i = 0
    for element in dataset:
        length += 1
    dataset_np = np.empty((length, window, signals))
    for element in dataset.as_numpy_iterator():
        dataset_np[i] = element
        i += 1
    
    # determine signalwise loss
    squared_error = np.square(dataset_np - reconstruction)

    for idx in range(squared_error.shape[0]):
        x = squared_error[idx]
        x = np.sum(x, axis=0) / window
        squared_error[idx] = x

    # calculate error vectors and intrusion score -> make prediction
    predictions = np.empty((length))
    for idx in range(squared_error.shape[0]):
        x = squared_error[idx]
        x = x / O_i
        if (np.max(x) >= O):
            predictions[idx] = 1
        else:
            predictions[idx] = 0

    return predictions.astype(int)

def evaluate_attack(model, batch_size, attack_path, O, O_i, read_tfrecord_feature, read_tfrecord_label, window, signals):
    print("reading attack data from disk.....")
    attack_files = []
    for file in os.listdir(attack_path):
        if file.endswith(".tfrecords"):
            attack_files.append(attack_path + file)
    
    raw_attack_dataset = tf.data.TFRecordDataset(attack_files, num_parallel_reads=len(attack_files))
    attack_dataset = raw_attack_dataset.map(read_tfrecord_feature)
    labels = raw_attack_dataset.map(read_tfrecord_label)

    # convert to numpy
    length = 0
    i = 0
    for label in labels:
        length += 1
    labels_np = np.empty((length))
    for label in labels.as_numpy_iterator():
        labels_np[i] = label
        i += 1
    labels_np = labels_np.astype(int)

    attack_dataset = attack_dataset.batch(batch_size, drop_remainder=False)

    print("predicting intrusions.....")
    reconstruction = model.predict(attack_dataset)
    attack_dataset = attack_dataset.unbatch()
    predictions = detect_intrusions(attack_dataset, reconstruction, O, O_i, window, signals)
    #print (predictions)

    print("calculating confusion matrix.....")
    tn, fp, fn, tp = confusion_matrix(labels_np, predictions).ravel()

    print("evaluation of attack data.....")
    print("tn: {}".format(tn))
    print("fp: {}".format(fp))
    print("fn: {}".format(fn))
    print("tp: {}".format(tp))

    fnr = fn / (tp + fn)
    fpr = fp / (fp + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

    print("fnr: {}".format(fnr))
    print("fpr: {}".format(fpr))
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("f1: {}".format(f1))

def evaluate_benign(model, batch_size, benign_path, O, O_i, read_tfrecord, window, signals):
    print("reading benign data from disk.....")
    benign_files = []
    for file in os.listdir(benign_path):
        if file.endswith(".tfrecords"):
            benign_files.append(benign_path + file)
    
    raw_benign_dataset = tf.data.TFRecordDataset(benign_files, num_parallel_reads=len(benign_files))
    pre_benign_dataset = raw_benign_dataset.map(read_tfrecord)

    # labels as to numpy
    length = 0
    for element in pre_benign_dataset:
        length += 1
    labels_np = np.zeros((length))

    #if (length > 100000):
    #    benign_dataset = pre_benign_dataset.take(75000)
    #else:
    benign_dataset = pre_benign_dataset
    
    benign_dataset = benign_dataset.batch(batch_size, drop_remainder=False)

    print("predicting intrusions.....")
    reconstruction = model.predict(benign_dataset)
    benign_dataset = benign_dataset.unbatch()
    predictions = detect_intrusions(benign_dataset, reconstruction, O, O_i, window, signals)

    print("calculating confusion matrix.....")
    tn, fp, fn, tp = confusion_matrix(labels_np, predictions).ravel()

    print("evaluation of benign data.....")
    print("tn: {}".format(tn))
    print("fp: {}".format(fp))

    fpr = fp / (fp + tn)

    print("fpr: {}".format(fpr))


def main(attack_path, benign_path, model_path, threshold_path, window, signals, batch_size, percentile):
    # obtain model
    model = tf.keras.models.load_model(model_path)

    input_dim = signals * window
    
    feature_description = {
            'X': tf.io.FixedLenFeature([input_dim], tf.float32)
    }

    feature_description_label = {
            'X': tf.io.FixedLenFeature([input_dim], tf.float32),
            'Y': tf.io.FixedLenFeature([window], tf.int64)
    }

    def read_tfrecord_feature(example):

        data = tf.io.parse_single_example(example, feature_description)
        x = data['X']
        feature = tf.reshape(x, shape=[window, signals])
        feature = tf.debugging.assert_all_finite(feature, 'Input must be finite')
        return feature

    def read_tfrecord_label(example):

        data = tf.io.parse_single_example(example, feature_description_label)
        y = data['Y'] # label
        return tf.reduce_max(y)

    max_rs = np.load(threshold_path+'max_rs.npy')
    O_i = np.load(threshold_path+'O_i.npy')
    O = np.percentile(max_rs, percentile*100)
    print("O : {}".format(O))

    if attack_path:
        evaluate_attack(model, batch_size, attack_path, O, O_i, read_tfrecord_feature, read_tfrecord_label, window, signals)
    if benign_path:
        evaluate_benign(model, batch_size, benign_path, O, O_i, read_tfrecord_feature, window, signals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_path', type=str)
    parser.add_argument('--benign_path', type=str)
    parser.add_argument('--model_path', type=str, default="Data/results/")
    parser.add_argument('--threshold_path', type=str, default="Data/thresholds/")
    parser.add_argument('--window', type=int, default=150)
    parser.add_argument('--signals', type=int, default=202)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--percentile', type=float, default=0.96)
    args = parser.parse_args()

    main(args.attack_path, args.benign_path, args.model_path, args.threshold_path, args.window, args.signals, args.batch_size, args.percentile)
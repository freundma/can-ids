# Date: 10-02-2023
# Author: Mario Freund
# Purpose: Evaluate x-canids
#   --attack_path: A path to tfrecord files with labeled samples as returned by preprocessing_labeled.py as string
#   --benign_path: A path to tfrecord files with unlabeled (benign) samples as returned by preprocessing_unlabeled.py as string
#   --model_path: A path to the model as returned by train.py as string
#   --threshold_path: A path to the max_rs and O_is as returned by threshold.py as string
#   --loss_path: A path were an example error rate vector should be exported to as string
#   --window: The window size to be used as int
#   --signals: The number of signals as int
#   --batch_size: The batch size to be used as int

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

def detect_intrusions(dataset, reconstruction, O, O_i, window, bytes):
    # convert attack_dataset to numpy array
    length = 0
    i = 0
    for element in dataset:
        length += 1
    dataset_np = np.empty((length, window, bytes))
    for element in dataset.as_numpy_iterator():
        dataset_np[i] = element
        i += 1
    
    # determine signalwise loss
    squared_error = np.square(dataset_np - reconstruction)

    loss_vectors = np.empty((length, bytes))
    for idx in range(squared_error.shape[0]):
        x = squared_error[idx]
        x = np.sum(x, axis=0) / window
        loss_vectors[idx] = x

    # calculate error vectors and intrusion score -> make prediction
    predictions = np.empty((length))
    for idx in range(loss_vectors.shape[0]):
        x = loss_vectors[idx]
        x = x / O_i
        if (np.max(x) >= O):
            predictions[idx] = 1
        else:
            predictions[idx] = 0

    return predictions.astype(int), loss_vectors

def evaluate_attack(model, batch_size, attack_path, percentiles, O, O_i, read_tfrecord_feature, read_tfrecord_label, window, bytes, loss_path):
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

    i = 0
    for o in O:

        predictions, error_rates = detect_intrusions(attack_dataset, reconstruction, o, O_i, window, bytes)
        if (i == 0): # save error rates once in the beginning
            np.save(loss_path+'error_rates.npy', error_rates)

        
        print("percentile: {}".format(percentiles[i]))
        print("calculating confusion matrix.....")
        tn, fp, fn, tp = confusion_matrix(labels_np, predictions).ravel()

        print("evaluation of attack data.....")
        print("tn: {}".format(tn))
        print("fp: {}".format(fp))
        print("fn: {}".format(fn))
        print("tp: {}".format(tp))

        fnr = fn / (tp + fn)
        fpr = fp / (fp + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))

        print("fnr: {}".format(fnr))
        print("fpr: {}".format(fpr))
        print("precision: {}".format(precision))
        print("recall: {}".format(recall))
        print("f1: {}".format(f1))
        print("---------------------------------------")
        i += 1

def evaluate_benign(model, batch_size, benign_path, percentiles, O, O_i, read_tfrecord, window, bytes, loss_path):
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

    # split into multiple parts to avoid OOM

    # truncate to multiple of batch size
    length_of_b = (length // batch_size) * batch_size
    benign_dataset = pre_benign_dataset.take(length_of_b)
    benign_dataset_copy = pre_benign_dataset.take(length_of_b) # trick to keep the dataset is whole
    labels_np = np.zeros((length_of_b)) #labels to numpy

    print("predicting intrusions.....")
    # determine part size
    part_size = (800*batch_size)
    iterations = length_of_b // part_size
    rest = length_of_b % part_size

    reconstruction = np.empty((length_of_b, window, bytes))

    for i in range(iterations):
        benign_dataset_part = benign_dataset.take(part_size)
        benign_dataset_part = benign_dataset_part.batch(batch_size)
        reconstruction_part_np = model.predict(benign_dataset_part)
        reconstruction[i*part_size:(i+1)*part_size] = reconstruction_part_np
        benign_dataset = benign_dataset.skip(part_size)

    # rest
    benign_dataset_rest = benign_dataset.take(rest)
    benign_dataset_rest = benign_dataset_rest.batch(batch_size)
    reconstruction_rest_np = model.predict(benign_dataset_rest)
    reconstruction[(iterations)*part_size:] = reconstruction_rest_np

    i = 0
    for o in O:
    
        predictions, error_rates = detect_intrusions(benign_dataset_copy, reconstruction, o, O_i, window, bytes)
        if (i == 0): # save error rates once in the beginning
            np.save(loss_path+'error_rates.npy', error_rates)

        print("percentile: {}".format(percentiles[i]))
        print("calculating confusion matrix.....")
        tn, fp, fn, tp = confusion_matrix(labels_np, predictions).ravel()

        print("evaluation of benign data.....")
        print("tn: {}".format(tn))
        print("fp: {}".format(fp))

        fpr = fp / (fp + tn)

        print("fpr: {}".format(fpr))
        print("---------------------------------------")
        i += 1


def main(attack_path, benign_path, model_path, threshold_path, loss_path, window, bytes, batch_size):
    # obtain model
    model = tf.keras.models.load_model(model_path)
    print(model.summary())

    input_dim = bytes * window
    
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
        feature = tf.reshape(x, shape=[window, bytes])
        feature = tf.debugging.assert_all_finite(feature, 'Input must be finite')
        return feature

    def read_tfrecord_label(example):

        data = tf.io.parse_single_example(example, feature_description_label)
        y = data['Y'] # label
        return tf.reduce_max(y)

    max_rs = np.load(threshold_path+'max_rs.npy')
    O_i = np.load(threshold_path+'O_i.npy')
    percentiles = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 1]
    O = []
    for p in percentiles:
        O.append(np.percentile(max_rs, p*100))
    print("percentiles: {}".format(percentiles))
    print("O: {}".format(O))

    if attack_path:
        evaluate_attack(model, batch_size, attack_path, percentiles, O, O_i, read_tfrecord_feature, read_tfrecord_label, window, bytes, loss_path)
    if benign_path:
        evaluate_benign(model, batch_size, benign_path, percentiles, O, O_i, read_tfrecord_feature, window, bytes, loss_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_path', type=str)
    parser.add_argument('--benign_path', type=str)
    parser.add_argument('--model_path', type=str, default="Data/results/")
    parser.add_argument('--threshold_path', type=str, default="Data/thresholds/")
    parser.add_argument('--loss_path', type=str, default="Data/losses/")
    parser.add_argument('--window', type=int, default=150)
    parser.add_argument('--bytes', type=int, default=244)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args.attack_path, args.benign_path, args.model_path, args.threshold_path, args.loss_path, args.window, args.bytes, args.batch_size)
# Date: 10-02-2023
# Author: Mario Freund
# Purpose: Split all tfrecord data into training, validation, test
# Commandline arguments:
#   --inpath: A path to one or more tfrecord files as produced by preprocessing_unlabeled.py as string
#   --outpath: A path where to store the datasplit as string; the directory must contain the subfolders /train/, /val/, /test/
#   --window: The used window size as int
#   --bytes: The number of bytes as int
#   --train_ratio: The train ratio as float < 1
#   --val_ratio: The validation ratio as float < 1
#   --test_ratio: The test ratio as float < 1
# Note: The train, validation, and test ratios should add up to 1.

import tensorflow as tf
import argparse
from tqdm import tqdm
import os
import numpy as np

def convert_to_numpy(dataset, bytes, length):
    dataset_np = np.empty((length, bytes))
    i = 0
    for element in dataset.as_numpy_iterator():
        dataset_np[i] = element
        i += 1
    return dataset_np

def apply_sliding_window(dataset_np, window, bytes):
    dataset_np_sw = np.lib.stride_tricks.sliding_window_view(dataset_np, (window, bytes))
    dataset_np_sw = dataset_np_sw.reshape(-1, window, bytes)
    dataset_np_sw = dataset_np_sw.reshape(-1, window*bytes)
    return dataset_np_sw

def write_to_tfrecord(X, path, samples_per_file, file):
    i = 1
    writer = tf.io.TFRecordWriter(path.format(file + "_0"))
    for idx in tqdm(range(X.shape[0])):
        x = X[idx]
        example = tf.train.Example(features=tf.train.Features(feature={
            'X': tf.train.Feature(float_list=tf.train.FloatList(value=x))
        }))
        writer.write(example.SerializeToString())
        if ((i % samples_per_file) == 0):
                writer = tf.io.TFRecordWriter(path.format(file + "_" + str(int(i/samples_per_file))))
        i += 1

def main(inpath, outpath, window, bytes, train_ratio, val_ratio, test_ratio):
    # get all tfrecord files
    files = []
    for file in os.listdir(inpath):
        if file.endswith(".tfrecords"):
            files.append(inpath+file)

    # feature description
    feature_description = {
        'S': tf.io.FixedLenFeature([bytes], tf.float32)
    }

    def read_tfrecord(example):

        data = tf.io.parse_single_example(example, feature_description)
        s = data['S']
        s = tf.debugging.assert_all_finite(s, 'Input must be finite')
        return s

    # prepare tf outpaths
    train_path = outpath + 'train/train_{}.tfrecords'
    val_path = outpath + 'val/val_{}.tfrecords'
    test_path = outpath + 'test/test_{}.tfrecords'

    total_train_samples = 0
    total_val_samples = 0
    total_test_samples = 0

    for file in files:
        # extract
        dataset_raw = tf.data.TFRecordDataset(file)
        dataset = dataset_raw.map(read_tfrecord)

        # train test val split

        # get length of data
        length_of_data = 0
        for element in dataset:
            length_of_data += 1

        train_size = int(length_of_data*train_ratio)

        val_size = int(length_of_data*val_ratio)

        test_size = int(length_of_data*test_ratio)

        train = dataset.take(train_size)

        val = dataset.skip(train_size)
        test = val.skip(val_size)
        test = test.take(test_size) # small trick to avoid having too many samples, this drops a few samples. But we can live with that

        val = val.take(val_size)

        # convert train to numpy
        train_np = convert_to_numpy(train, bytes, train_size)
        val_np = convert_to_numpy(val, bytes, val_size)
        test_np = convert_to_numpy(test, bytes, test_size)

        # apply sliding window
        X_train = apply_sliding_window(train_np, window, bytes)
        total_train_samples += X_train.shape[0]

        X_val = apply_sliding_window(val_np, window, bytes)
        total_val_samples += X_val.shape[0]

        X_test = apply_sliding_window(test_np, window, bytes)
        total_test_samples += X_test.shape[0]

        # write to tf_records
        file_name = os.path.basename(file)
        file_prefix = os.path.splitext(file_name)[0]
        samples_per_file = 900

        print("Writing datasplit for file {}.....".format(file_name))
        write_to_tfrecord(X_train, train_path, samples_per_file, file_prefix)
        write_to_tfrecord(X_val, val_path, samples_per_file, file_prefix)
        write_to_tfrecord(X_test, test_path, samples_per_file, file_prefix)
    
    print("Total samples: {}".format(total_test_samples+total_train_samples+total_val_samples))
    print("Total train samples: {}".format(total_train_samples))
    print("Total validation samples: {}".format(total_val_samples))
    print("Total test samples: {}".format(total_test_samples))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='Data/TFRecords/')
    parser.add_argument('--outpath', type=str, default='Data/datasplit/')
    parser.add_argument('--window', type=int, default=150)
    parser.add_argument('--bytes', type=int, default=244)
    parser.add_argument('--train_ratio', type=float, default=4/6)
    parser.add_argument('--val_ratio', type=float, default=1/6)
    parser.add_argument('--test_ratio', type=float, default=1/6)
    args = parser.parse_args()

    main(args.inpath, args.outpath, args.window, args.bytes, args.train_ratio, args.val_ratio, args.test_ratio)
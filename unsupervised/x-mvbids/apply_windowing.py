# Date: 07-05-2023
# Author: Mario Freund
# Purpose: Convert MVB signal vectors into training samples with sliding window.

import tensorflow as tf
import argparse
from tqdm import tqdm
import os
import numpy as np

def convert_to_numpy(dataset, signals, length):
    dataset_np = np.empty((length, signals))
    i = 0
    for element in dataset.as_numpy_iterator():
        dataset_np[i] = element
        i += 1
    return dataset_np

def apply_sliding_window(dataset_np, window, signals):
    dataset_np_sw = np.lib.stride_tricks.sliding_window_view(dataset_np, (window, signals))
    dataset_np_sw = dataset_np_sw.reshape(-1, window, signals)
    dataset_np_sw = dataset_np_sw.reshape(-1, window*signals)
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

def main(infile, outpath, window, signals):
    # feature description
    feature_description = {
        'S': tf.io.FixedLenFeature([signals], tf.float32)
    }

    def read_tfrecord(example):

        data = tf.io.parse_single_example(example, feature_description)
        s = data['S']
        s = tf.debugging.assert_all_finite(s, 'Input must be finite')
        return s
    
    outpath = outpath + '{}.tfrecords'

    # extract
    dataset_raw = tf.data.TFRecordDataset(infile)
    dataset = dataset_raw.map(read_tfrecord)

    # convert to numpy to be able to apply sliding window
    length_of_data = 0
    for element in dataset:
        length_of_data += 1

    data_np = convert_to_numpy(dataset, signals, length_of_data)
    X = apply_sliding_window(data_np, window, signals)

    file_name = os.path.basename(infile)
    file_prefix = os.path.splitext(file_name)[0]

    # write to tf_records
    samples_per_file = 900

    print("Writing datasplit for file {}.....".format(file_name))
    write_to_tfrecord(X, outpath, samples_per_file, file_prefix)

    print("Total samples: {}".format(X.shape[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str)
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--window', type=int, default=64)
    parser.add_argument('--signals', type=int, default=271)
    args = parser.parse_args()

    main(args.infile, args.outpath, args.window, args.signals)


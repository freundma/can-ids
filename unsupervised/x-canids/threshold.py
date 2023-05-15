# Date: 05-11-2023
# Author: Mario Freund
# Purpose: Determine the intrusion threshold for x-canids

import argparse
import tensorflow as tf

# Multi GPU setup
mirrored_strategy = tf.distribute.MirroredStrategy()

def main(model_path, train_data, val_data, window, signals, percentile):
    # obtain model
    model = tf.keras.models.load_model(model_path)

    input_dim = signals * window
    feature_description = {
        'X': tf.io.FixedLenFeature([input_dim], tf.float32)
    }

    def read_tfrecord(example):

        data = tf.io.parse_single_example(example, feature_description)
        x = data['X']
        feature = tf.reshape(x, shape=[window, signals])
        feature = tf.debugging.assert_all_finite(feature, 'Input must by finite')
        return feature

    # obtain training data
    raw_train_dataset = tf.data.TFRecordDataset(train_data)
    train_dataset = raw_train_dataset.map(read_tfrecord)

    #obtain validation data
    raw_val_dataset = tf.data.TFRecordDataset(val_data)
    val_dataset = raw_val_dataset.map(read_tfrecord)

    # calculate loss vectors l with l = {l_1, l_2, ...., l_x}
    # after this we have an l for every S containing the loss of each signal

    # calculate Oi = mean(l_i) + 3*sigmar_i -> a threshold for every signal

    # take validation set and calculate error vectors r with r = {r_i | r_i = l_i/O_i for i = 1...x} -> results in {r_1,....,r_x}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default= "./data/results/")
    parser.add_argument('--train_data', type=str, defeault="./data/TFRecord.tfrecords")
    parser.add_argument('--val_data', type=str, default="./data/TFRecord_val.tfrecords")
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--signals', type=int, default=664)
    parser.add_argument('--percentile', type=float, default=0.96)
    args=parser.parse_args()

    main(args.model_path, args.train_data, args.val_data, args.window, args.signals, args.percentile)
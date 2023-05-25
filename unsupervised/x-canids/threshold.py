# Date: 05-11-2023
# Author: Mario Freund
# Purpose: Determine the intrusion threshold for x-canids

import argparse
import tensorflow as tf
import numpy as np

# Multi GPU setup
# mirrored_strategy = tf.distribute.MirroredStrategy()

def main(model_path, train_data, val_data, window, signals, batch_size, percentile):
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
        feature = tf.debugging.assert_all_finite(feature, 'Input must be finite')
        return feature

    # obtain training data
    raw_train_dataset = tf.data.TFRecordDataset(train_data)
    train_dataset = raw_train_dataset.map(read_tfrecord)
    train_dataset.batch(batch_size)

    #obtain validation data
    raw_val_dataset = tf.data.TFRecordDataset(val_data)
    val_dataset = raw_val_dataset.map(read_tfrecord)

    # calculate loss vectors l with l = {l_1, l_2, ...., l_x}
    # after this, we have an l for every S containing the loss of each signal

    # predict
    s_ = model.predict(train_dataset)
    s_.unbatch()

    # convert train dataset to numpy
    # find out length
    length_of_s = 0
    for element in train_dataset:
        length_of_s += 1
    
    # convert
    s = np.empty((length_of_s, window, signals))
    i = 0
    for element in train_dataset.as_numpy_iterator():
        s[i] = element
        i+=1

    # loss
    s_squared_error = np.square(s - s_)

    # sum up and divide by window size
    for idx in range(s_squared_error.shape[0]):
        x = s_squared_error[idx] # S
        x = np.sum(x, axis=0) / window
        s_squared_error[idx] = x

    # calculate Oi = mean(l_i) + 3*sigmar_i -> a threshold for every signal
    l_i_mean = np.mean(s_squared_error, axis=0)
    l_i_var = np.var(s_squared_error, axis=0)

    O_i = l_i_mean + 3*l_i_var

    # find out length
    length_of_val = 0
    for element in val_dataset:
        length_of_val += 1

    val_np = np.empty((length_of_val, window, signals))

    i = 0
    # take validation set and calculate error vectors r with r = {r_i | r_i = l_i/O_i for i = 1...x} -> results in {r_1,....,r_x}
    for element in val_dataset.as_numpy_iterator():
        element = np.sum(element, axis=0) / window
        element = element / O_i
        val_np[i] = element
        i += 1
    
    # calculate max(r) for every r
    max_rs = []
    for idx in range(val_np.shape[0]):
        r = val_np[idx]
        max_rs.append(max(r))

    O = np.percentile(max_rs, percentile)

    print("O is: {}".format(O))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default= "./data/results/")
    parser.add_argument('--train_data', type=str, default="./data/TFRecord.tfrecords")
    parser.add_argument('--val_data', type=str, default="./data/TFRecord_val.tfrecords")
    parser.add_argument('--window', type=int, default=150)
    parser.add_argument('--signals', type=int, default=197)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--percentile', type=float, default=0.96)
    args=parser.parse_args()

    main(args.model_path, args.train_data, args.val_data, args.window, args.signals, args.batch_size, args.percentile)
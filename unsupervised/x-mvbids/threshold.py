# Date: 07-03-2023
# Author: Mario Freund
# Purpose: Determine the intrusion threshold for x-mvbids

import argparse
import tensorflow as tf
import numpy as np
import os

# Multi GPU setup
mirrored_strategy = tf.distribute.MirroredStrategy()

def main(model_path, data_path, outpath, window, signals, batch_size):
    with mirrored_strategy.scope():
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
        train_path = data_path + 'train/'
        val_path = data_path + 'val/'

        train_files = []
        for file in os.listdir(train_path):
            if file.endswith(".tfrecords"):
                train_files.append(train_path + file)

        val_files = []
        for file in os.listdir(val_path):
            if file.endswith(".tfrecords"):
                val_files.append(val_path + file)

        raw_train_dataset = tf.data.TFRecordDataset(train_files, num_parallel_reads=len(train_files))
        pre_train_dataset = raw_train_dataset.map(read_tfrecord)

        # convert train dataset to numpy

        # prepare numpy
        # find out length
        length_of_train = 0
        for element in pre_train_dataset:
            length_of_train += 1

        # truncate to multiple of batch size
        length_of_s = (length_of_train // batch_size) * batch_size
        train_dataset = pre_train_dataset.take(length_of_s)

        # convert
        s = np.empty((length_of_s, window, signals))
        i = 0
        for element in train_dataset.as_numpy_iterator():
            s[i] = element
            i+=1

        s_ = np.empty((length_of_s, window, signals))

        # split data for inference to avoid OOM
        part_size = (800*batch_size)
        iterations = length_of_s // part_size
        rest = length_of_s % part_size

        for i in range(iterations):
            dataset_part = train_dataset.take(part_size)
            dataset_part = dataset_part.batch(batch_size)
            dataset_part_np = model.predict(dataset_part)
            s_[i*part_size:(i+1)*part_size] = dataset_part_np
            train_dataset = train_dataset.skip(part_size)

        dataset_rest = train_dataset.take(rest)
        dataset_rest = dataset_rest.batch(batch_size)
        dataset_rest_np = model.predict(dataset_rest)
        s_[(iterations)*part_size:] = dataset_rest_np

        #obtain validation data
        raw_val_dataset = tf.data.TFRecordDataset(val_files, num_parallel_reads=len(val_files))
        pre_val_dataset = raw_val_dataset.map(read_tfrecord)

        # convert val dataset to numpy

        # find out length
        length_of_validation = 0
        for element in pre_val_dataset:
            length_of_validation += 1

        # truncate to multiple of batch size
        length_of_v = (length_of_validation // batch_size) * batch_size
        val_dataset = pre_val_dataset.take(length_of_v)

        # convert
        v = np.empty((length_of_v, window, signals))
        i = 0
        for element in val_dataset.as_numpy_iterator():
            v[i] = element
            i+=1

        v_ = np.empty((length_of_v, window, signals))

        # split data for inference to avoid OOM
        part_size = (400*batch_size)
        iterations = length_of_v // part_size
        rest = length_of_v % part_size

        for i in range(iterations):
            dataset_part = val_dataset.take(part_size)
            dataset_part = dataset_part.batch(batch_size)
            dataset_part_np = model.predict(dataset_part)
            v_[i*part_size:(i+1)*part_size] = dataset_part_np
            val_dataset = val_dataset.skip(part_size)

        dataset_rest = val_dataset.take(rest)
        dataset_rest = dataset_rest.batch(batch_size)
        dataset_rest_np = model.predict(dataset_rest)
        v_[(iterations)*part_size:] = dataset_rest_np

        # calculate loss vectors l with l = {l_1, l_2, ...., l_x}
        # after this, we have an l for every S containing the loss of each signal

        # loss
        s_squared_error = np.square(s - s_)
        v_squared_error = np.square(v - v_)

        # sum up and divide by window size
        for idx in range(s_squared_error.shape[0]):
            x = s_squared_error[idx] # S
            x = np.sum(x, axis=0) / window
            s_squared_error[idx] = x

        # calculate Oi = mean(l_i) + 3*sigmar_i -> a threshold for every signal
        l_i_mean = np.mean(s_squared_error, axis=0)
        l_i_std = np.std(s_squared_error, axis=0)

        O_i = l_i_mean + 3*l_i_std

        # calculate signal losses of validation: sum up and divide by window size
        for idx in range(v_squared_error.shape[0]):
            x = v_squared_error[idx]
            x = np.sum(x, axis=0) / window
            v_squared_error[idx] = x

        # calculate error vectors r with r = {r_i | r_i = l_i/O_i for i = 1....x}
        for idx in range(v_squared_error.shape[0]):
            x = v_squared_error[idx]
            x = x / O_i
            v_squared_error[idx] = x
        # calculate max(r) for every r
        max_rs = np.empty((v_squared_error.shape[0]))
        for idx in range(v_squared_error.shape[0]):
            r = v_squared_error[idx]
            max_rs[idx] = np.max(r)

        O = np.percentile(max_rs, 0.98*100) # example output

        np.save(outpath+'max_rs.npy', max_rs)
        np.save(outpath+'O_i.npy', O_i)

        print("Saved max errors and O_is.....")
        print("O with percentile of 0.98 would be: {}".format(O))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default= "Data/results/")
    parser.add_argument('--data_path', type=str, default="Data/datasplit/")
    parser.add_argument('--outpath', type=str, default="Data/thresholds/")
    parser.add_argument('--window', type=int, default=64)
    parser.add_argument('--signals', type=int, default=271)
    parser.add_argument('--batch_size', type=int, default=64)
    args=parser.parse_args()

    main(args.model_path, args.data_path, args.outpath, args.window, args.signals, args.batch_size)
# Date: 10-11-2023
# Author: Mario Freund
# Purpose: Script to predict batches of data and measure inference

import argparse
import tensorflow as tf
import os

def main(data_path, model_path, window, signals, batch_size):
    # obtain model
    ("Obtaining model.....")
    model = tf.keras.models.load_model(model_path)
    print(model.summary())

    input_dim = signals * window

    feature_description = {
            'X': tf.io.FixedLenFeature([input_dim], tf.float32)
    }

    def read_tfrecord_feature(example):
        data = tf.io.parse_single_example(example, feature_description)
        x = data['X']
        feature = tf.reshape(x, shape=[window, signals])
        feature = tf.debugging.assert_all_finite(feature, 'Input must be finite')
        return feature
    
    print("Reading data from disk.....")
    files = []
    for file in os.listdir(data_path):
        if file.endswith(".tfrecords"):
            files.append(data_path + file)

    raw_dataset = tf.data.TFRecordDataset(files, num_parallel_reads=len(files))
    dataset = raw_dataset.map(read_tfrecord_feature)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    reconstruction = model.predict(dataset)
    
    print ("Inference done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--window', type=int, default=150)
    parser.add_argument('--signals', type=int, default=202)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args.data_path, args.model_path, args.window, args.signals, args.batch_size)
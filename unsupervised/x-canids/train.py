# Date: 05-05-2023
# Author: Mario Freund
# Purpose: Train x-canids classifier with benign preprocessed data

import numpy as np
import argparse
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

def x_canids_model(window, num_signals):
    model = Sequential()
    model.add((LSTM(2*num_signals, activation='relu',
                                 input_shape=(window, num_signals), return_sequences=True)))
    model.add((LSTM(num_signals + int(0.05*num_signals), activation='relu',
                                 return_sequences=False)))
    model.add(RepeatVector(window))
    model.add((LSTM(2*num_signals, activation='relu',
                                 return_sequences=True)))
    model.add((LSTM(2*num_signals, activation='relu',
                                 return_sequences=True)))
    model.add(TimeDistributed(Dense(num_signals)))
    return model

def main(infile, outfile, window, num_signals, epochs, batch_size):
    # Read TFRecord
    filenames = [infile]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    
    input_dim = num_signals * window
    feature_description = {
        'X': tf.io.FixedLenFeature([input_dim], tf.float32)
    }

    def read_tfrecord(example):

        data = tf.io.parse_single_example(example, feature_description)
        x = data['X']
        feature = tf.reshape(x, shape=[window, num_signals])
        label = feature
        return (feature, label) # label = feature because of reconstruction
    dataset = raw_dataset.map(read_tfrecord)
    dataset = dataset.batch(batch_size)

    model = x_canids_model(window, num_signals)
    model.compile(optimizer='adam', loss='mse')
    model.build((None, window, num_signals))
    #print(model.summary())

    callback=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    
    model.fit(
        x=dataset,
        epochs=epochs,
        callbacks=[callback],
    )
    
    model.save(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default="./data/NewTFRecord.tfrecords")
    parser.add_argument('--outfile', type=str, default="./Results/")
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--signals', type=int, default=664)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args.infile, args.outfile, args.window, args.signals, args.epochs, args.batch_size)
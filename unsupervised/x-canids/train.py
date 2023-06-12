# Date: 05-05-2023
# Author: Mario Freund
# Purpose: Train x-canids classifier with benign preprocessed data

import argparse
import sys
import os
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# Multi GPU setup
mirrored_strategy = tf.distribute.MirroredStrategy()

def x_canids_model(window, num_signals, latent_space_size):
    with mirrored_strategy.scope():
        model = Sequential()
        model.add(Bidirectional(LSTM(num_signals, activation='tanh',
                                 input_shape=(window, num_signals), return_sequences=True)))
        model.add(Bidirectional(LSTM(latent_space_size, activation='tanh',
                                 return_sequences=False)))
        model.add(RepeatVector(window))
        model.add(Bidirectional(LSTM(num_signals, activation='tanh',
                                 return_sequences=True)))
        model.add(Bidirectional(LSTM(num_signals, activation='tanh',
                                 return_sequences=True)))
        model.add(TimeDistributed(Dense(num_signals)))
    return model

def x_canids_model_stock(window, num_signals):
    model = Sequential()
    model.add((Bidirectional(LSTM(107, activation='tanh',
                                  input_shape=(window, num_signals), return_sequences=True))))
    model.add((Bidirectional(LSTM(125, activation='tanh',
                                  return_sequences=False))))
    model.add(RepeatVector(window))
    model.add(Bidirectional(LSTM(107, activation='tanh',
                                 return_sequences=True)))
    model.add(Bidirectional(LSTM(107, activation='tanh',
                                 return_sequences=True)))
    model.add(TimeDistributed(Dense(num_signals)))
    return model

def x_canids_model_alternative(window, num_signals):
    model = Sequential()
    model.add((Bidirectional(LSTM(128, activation='tanh',
                                  input_shape=(window, num_signals), return_sequences=True))))
    model.add((Bidirectional(LSTM(64, activation='tanh',
                                  return_sequences=False))))
    model.add(RepeatVector(window))
    model.add(Bidirectional(LSTM(64, activation='tanh',
                                 return_sequences=True)))
    model.add(Bidirectional(LSTM(128, activation='tanh',
                                 return_sequences=True)))
    model.add(TimeDistributed(Dense(num_signals)))
    return model
    

def main(inpath, outpath, window, num_signals, epochs, batch_size, latent_space_size, checkpoint_path, tensorboard_path, lr):
    # declare training, validation tfrecord files from data split
    train_path = inpath + 'train/'
    val_path = inpath + 'val/'

    train_files = []
    for file in os.listdir(train_path):
        if file.endswith(".tfrecords"):
            train_files.append(train_path + file)

    val_files = []
    for file in os.listdir(val_path):
        if file.endswith(".tfrecords"):
            val_files.append(val_path + file)

    # Read TFRecords
    with mirrored_strategy.scope():
        raw_train_dataset = tf.data.TFRecordDataset(train_files, num_parallel_reads=len(train_files))
        raw_val_dataset = tf.data.TFRecordDataset(val_files, num_parallel_reads=len(val_files))
    
        input_dim = num_signals * window
        feature_description = {
            'X': tf.io.FixedLenFeature([input_dim], tf.float32)
        }

        def read_tfrecord(example):

            data = tf.io.parse_single_example(example, feature_description)
            x = data['X']
            feature = tf.reshape(x, shape=[window, num_signals])
            feature = tf.debugging.assert_all_finite(feature, 'Input must be finite')
            tf.debugging.assert_non_negative(feature, 'Input must be positive')
            label = tf.identity(feature)
            return (feature, label) # label = feature because of reconstruction, unsupervised learning
        
        # prepare datasets
        train_dataset = raw_train_dataset.map(read_tfrecord)
        train_dataset = train_dataset.shuffle(25000)
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        val_dataset = raw_val_dataset.map(read_tfrecord)
        val_dataset = val_dataset.shuffle(25000)
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

        model = x_canids_model(window, num_signals, latent_space_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=optimizer, loss=loss)
        model.build((None, window, num_signals))
        print(model.summary())

        # callbacks
        checkpoint_filepath = checkpoint_path + "/weights.{epoch:02d}-{val_loss:2f}.hdf5"
        callback_early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        callback_tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_path,
            write_graph=False
        )
        callback_nan = tf.keras.callbacks.TerminateOnNaN()
    
        model.fit(
            x=train_dataset,
            epochs=epochs,
            callbacks=[callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_nan],
            validation_data=val_dataset
        )
    
        model.save(outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default="Data/datasplit/")
    parser.add_argument('--outpath', type=str, default="Data/results/")
    parser.add_argument('--window', type=int, default=150)
    parser.add_argument('--signals', type=int, default=197)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_space_size', type=int, default=230)
    parser.add_argument('--checkpoint_path', type=str, default="Data/results/checkpoints")
    parser.add_argument('--tensorboard_path', type=str, default="Data/results/tensorboards")
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    args = parser.parse_args()

    main(args.inpath, args.outpath, args.window, args.signals, args.epochs, args.batch_size, args.latent_space_size, args.checkpoint_path, args.tensorboard_path, args.learning_rate)
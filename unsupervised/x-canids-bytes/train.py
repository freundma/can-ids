# Date: 10-02-2023
# Author: Mario Freund
# Purpose: Train x-canids classifier with benign preprocessed byte-based data
# Commandline arguments:
#   --inpath: A path to a datasplit as produced by train_val_test_split.py as string; the subfolders /train/ and /val/ must be included
#   --outpath: A path were to save the trained model when finished as string
#   --window: The used window size as int
#   --bytes: The number of bytes as int
#   --epochs: The number of epcohs to train as int
#   --batch_size: The batch size to use as int
#   --latent_space_size: The latent space size as int divided by two (for a latent space size of 500 type 250)
#   --checkpoint_path: A path where to save checkpoints of the model as string
#   --tensorboard_path: A path where to store a tensorboard with training statistics as string
#   --learning_rate: The learning rate to be used as float
#   --from model: A flag whether to continue with a trained model. It will be loaded from the outpath


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

def x_canids_model(window, num_bytes, latent_space_size):
    with mirrored_strategy.scope():
        model = Sequential()
        model.add(Bidirectional(LSTM(num_bytes, activation='tanh',
                                 input_shape=(window, num_bytes), return_sequences=True)))
        model.add(Bidirectional(LSTM(latent_space_size, activation='tanh',
                                 return_sequences=False)))
        model.add(RepeatVector(window))
        model.add(Bidirectional(LSTM(num_bytes, activation='tanh',
                                 return_sequences=True)))
        model.add(Bidirectional(LSTM(num_bytes, activation='tanh',
                                 return_sequences=True)))
        model.add(TimeDistributed(Dense(num_bytes)))
    return model

def x_canids_model_stock(window, num_bytes):
    model = Sequential()
    model.add((Bidirectional(LSTM(107, activation='tanh',
                                  input_shape=(window, num_bytes), return_sequences=True))))
    model.add((Bidirectional(LSTM(125, activation='tanh',
                                  return_sequences=False))))
    model.add(RepeatVector(window))
    model.add(Bidirectional(LSTM(107, activation='tanh',
                                 return_sequences=True)))
    model.add(Bidirectional(LSTM(107, activation='tanh',
                                 return_sequences=True)))
    model.add(TimeDistributed(Dense(num_bytes)))
    return model

def x_canids_model_alternative(window, num_bytes):
    model = Sequential()
    model.add((Bidirectional(LSTM(128, activation='tanh',
                                  input_shape=(window, num_bytes), return_sequences=True))))
    model.add((Bidirectional(LSTM(64, activation='tanh',
                                  return_sequences=False))))
    model.add(RepeatVector(window))
    model.add(Bidirectional(LSTM(64, activation='tanh',
                                 return_sequences=True)))
    model.add(Bidirectional(LSTM(128, activation='tanh',
                                 return_sequences=True)))
    model.add(TimeDistributed(Dense(num_bytes)))
    return model
    

def main(inpath, outpath, window, num_bytes, epochs, batch_size, latent_space_size, checkpoint_path, tensorboard_path, lr, from_model):
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
    
        input_dim = num_bytes * window
        feature_description = {
            'X': tf.io.FixedLenFeature([input_dim], tf.float32)
        }

        def read_tfrecord(example):

            data = tf.io.parse_single_example(example, feature_description)
            x = data['X']
            feature = tf.reshape(x, shape=[window, num_bytes])
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

        if (not from_model):
            model = x_canids_model(window, num_bytes, latent_space_size)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            loss = tf.keras.losses.MeanSquaredError()
            model.compile(optimizer=optimizer, loss=loss)
            model.build((None, window, num_bytes))
        else:
            model = tf.keras.models.load_model(outpath)
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

        if (from_model):
            initial = 2000-epochs
        else:
            initial = 0
    
        model.fit(
            x=train_dataset,
            epochs=epochs,
            callbacks=[callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_nan],
            validation_data=val_dataset,
            initial_epoch=initial #if we start from an already trained model e.g. start from 501
        )
    
        model.save(outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default="Data/datasplit/")
    parser.add_argument('--outpath', type=str, default="Data/results/")
    parser.add_argument('--window', type=int, default=150)
    parser.add_argument('--bytes', type=int, default=244)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_space_size', type=int, default=285)
    parser.add_argument('--checkpoint_path', type=str, default="Data/results/checkpoints")
    parser.add_argument('--tensorboard_path', type=str, default="Data/results/tensorboards")
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--from_model', action='store_true')
    args = parser.parse_args()

    main(args.inpath, args.outpath, args.window, args.bytes, args.epochs, args.batch_size, args.latent_space_size, args.checkpoint_path, args.tensorboard_path, args.learning_rate, args.from_model)
# Date: 05-05-2023
# Author: Mario Freund
# Purpose: Train x-canids classifier with benign preprocessed data

import argparse
import tensorflow as tf
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
        model.add(Bidirectional(LSTM(num_signals, activation='relu',
                                 input_shape=(window, num_signals), return_sequences=True)))
        model.add(Bidirectional(LSTM(latent_space_size, activation='relu',
                                 return_sequences=False)))
        model.add(RepeatVector(window))
        model.add(Bidirectional(LSTM(num_signals, activation='relu',
                                 return_sequences=True)))
        model.add(Bidirectional(LSTM(num_signals, activation='relu',
                                 return_sequences=True)))
        model.add(TimeDistributed(Dense(num_signals)))
    return model

def x_canids_model_stock(window, num_signals):
    model = Sequential()
    model.add((Bidirectional(LSTM(107, activation='relu',
                                  input_shape=(window, num_signals), return_sequences=True))))
    model.add((Bidirectional(LSTM(125, activation='relu',
                                  return_sequences=False))))
    model.add(RepeatVector(window))
    model.add(Bidirectional(LSTM(107, activation='relu',
                                 return_sequences=True)))
    model.add(Bidirectional(LSTM(107, activation='relu',
                                 return_sequences=True)))
    model.add(TimeDistributed(Dense(num_signals)))
    return model
    

def main(infile, outfile, window, num_signals, epochs, batch_size, latent_space_size, checkpoint_path, tensorboard_path):
    # Read TFRecord
    with mirrored_strategy.scope():
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
            feature = tf.debugging.assert_all_finite(feature, 'Input must by finite')
            tf.debugging.assert_non_negative(feature, 'Input must be positive')
            label = tf.identity(feature)
            return (feature, label) # label = feature because of reconstruction, unsupervised learning
        dataset = raw_dataset.map(read_tfrecord)
        dataset = dataset.batch(batch_size)

        model = x_canids_model(window, num_signals, latent_space_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=0.5, clipvalue=1.0)
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=optimizer, loss=loss)
        model.build((None, window, num_signals))
        print(model.summary())

        # callbacks
        checkpoint_filepath = checkpoint_path + "/weights.{epoch:02d}-{loss:2f}.hdf5"
        callback_early_stopping=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
        callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True
        )
        callback_tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_path,
            write_graph=False
        )
        callback_nan = tf.keras.callbacks.TerminateOnNaN()
    
        model.fit(
            x=dataset,
            epochs=epochs,
            callbacks=[callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_nan],
        )
    
        model.save(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default="./data/NewTFRecord.tfrecords")
    parser.add_argument('--outfile', type=str, default="./data/results/")
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--signals', type=int, default=664)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_space_size', type=int, default=125)
    parser.add_argument('--checkpoint_path', type=str, default="./data/results/checkpoints")
    parser.add_argument('--tensorboard_path', type=str, default="./data/results/tensorboards")
    args = parser.parse_args()

    main(args.infile, args.outfile, args.window, args.signals, args.epochs, args.batch_size, args.latent_space_size, args.checkpoint_path, args.tensorboard_path)
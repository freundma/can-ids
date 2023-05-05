# Date: 05-05-2023
# Author: Mario Freund
# Purpose: Train x-canids classifier with benign preprocessed data

import numpy as np
import argparse
import tensorflow as tf

def read_tfrecord(example):
    return

def main(infile, outfile, window, num_signals, epochs, batch_size):
    # Read TFRecord
    filenames = [infile]
    raw_dataset = tf.data.TFRecordDataset(filenames)

    return None

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default="./data/TFRecord.tfrecords")
    parser.add_argument('--outfile', type=str, default="./Results/")
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--signals', type=int, default=664)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args.infile, args.outfile, args.window, args.signals, args.epochs, args.batch_size)
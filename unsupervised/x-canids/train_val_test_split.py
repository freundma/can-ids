# Date: 05-24-2023
# Author: Mario Freund
# Purpose: Split all tfrecord data into training, validation, test

import tensorflow as tf
import argparse
from tqdm import tqdm
import os

def main(inpath, outpath, train_ratio, val_ratio, test_ratio):
    # get all tfrecord files
    files = []
    for file in os.listdir(inpath):
        if file.endswith(".tfrecords"):
            files.append(inpath+file)
    
    # prepare tf writers
    train_path = outpath + 'train.tfrecords'
    val_path = outpath + 'val.tfrecords'
    test_path = outpath + 'test.tfrecords'

    train_writer = tf.io.TFRecordWriter(train_path)
    val_writer = tf.io.TFRecordWriter(val_path)
    test_writer = tf.io.TFRecordWriter(test_path)

    # for each dataset read samples and split into train, test, validation
    dataset = tf.data.TFRecordDataset(files)

    # get total amount of samples, hack because this metadata gets not stored
    num_samples = 0
    for element in dataset:
        num_samples += 1
    
    train_size = int(num_samples*train_ratio)
    val_size = int(num_samples*val_ratio)
    test_size = int(num_samples*test_ratio)

    print("total samples: {}".format(num_samples))
    print("train samples: {}".format(train_size))
    print("validation samples: {}".format(val_size))
    print("test size: {}".format(test_size))

    # split data
    train = dataset.take(train_size)

    val = dataset.skip(train_size)
    test = val.skip(val_size)

    val = val.take(val_size)

    # write data
    print("writing train data.....")
    for element in tqdm(train):
        train_writer.write(tf.io.serialize_tensor(element).numpy())
    print("writing validation data.....")
    for element in tqdm(val):
        val_writer.write(tf.io.serialize_tensor(element).numpy())
    print("writing test data.....")
    for element in tqdm(test):
        test_writer.write(tf.io.serialize_tensor(element).numpy())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='Data/TFRecords/')
    parser.add_argument('--outpath', type=str, default='Data/datasplit/')
    parser.add_argument('--train_ratio', type=float, default=4/6)
    parser.add_argument('--val_ratio', type=float, default=1/6)
    parser.add_argument('--test_ratio', type=float, default=1/6)
    args = parser.parse_args()

    main(args.inpath, args.outpath, args.train_ratio, args.val_ratio, args.test_ratio)

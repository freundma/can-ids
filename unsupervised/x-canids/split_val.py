import tensorflow as tf
import argparse
from tqdm import tqdm
import os

def main(inpath, outpath):
    val_path = outpath + 'val/val_{}.tfrecords'
    samples_per_file = 900

    files = []
    for file in os.listdir(inpath):
        if file.endswith(".tfrecords"):
            files.append(inpath+file)

    dataset = tf.data.TFRecordDataset(files)
    print("writing validation data.....")
    i = 1
    val_writer = tf.io.TFRecordWriter(val_path.format(0))
    for element in tqdm(dataset):
        val_writer.write(element.numpy())
        if ((i % samples_per_file) == 0):
            val_writer = tf.io.TFRecordWriter(val_path.format(int(i/samples_per_file)))
        i += 1

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='Data/TFRecords/')
    parser.add_argument('--outpath', type=str, default='Data/datasplit/')
    args = parser.parse_args()
    main (args.inpath, args.outpath)
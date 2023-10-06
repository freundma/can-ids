# Date: 10-02-2023
# Author: Mario Freund
# Purpose: Preprocess ambient byte extracted data of road dataset as done in x-canids with labels

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
import sys
import json
import os
np.set_printoptions(threshold=sys.maxsize)

def convert_to_int(x):
    if (type(x) == str):
        return int(x, 16)
    return x

def compute_offsets(offsets):
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i-1]
    return offsets

def scale_s(s, unique_id_list, min_dict, max_dict):
    scaled_s = s.copy() # copy
    offset = 0
    for i in range(len(unique_id_list)):
        # get minimums and maximums of byte of each id
        mins_i = min_dict[str(unique_id_list[i])]
        maxs_i = max_dict[str(unique_id_list[i])]
        for j in range(len(mins_i)): # scale
            if(maxs_i[j] == mins_i[j]): # constant value
                scaled_s[offset+j] = 1.0
            else:
                scaled_s[offset+j] = (scaled_s[offset+j] - mins_i[j]) / (maxs_i[j] - mins_i[j]) # s^_i  = (s_i - min_i) / (max_i - min_i)
        offset += len(mins_i)
    return scaled_s

def get_s(df, t, delta_t, offsets, unique_id_list, min_dict, max_dict, cache, const_dict, exclude_constant_bytes):
    clean = True
    t_minus_one = t - delta_t
    df_t = df.loc[(df['timestamp'] <= t) & (df['timestamp'] > t_minus_one)]
    # get latest bytes of each id
    s = np.empty(offsets[-1]) # total amount of bytes
    i = 0
    offset = 0
    labels = []
    for id in unique_id_list:
        df_id = df_t.loc[df_t['id'] == id]
        if (df_id.empty): # take cached value
            if ('bytes' in cache[str(id)]):
                s[offset:offsets[i]] = cache[str(id)]['bytes']
                label = cache[str(id)]['label']
                labels.append(label)
            else:
                clean = False
            offset = offsets[i]
            i += 1
            continue
        index = df_id['timestamp'].idxmax() # latest value
        df_id = df_id.loc[[index]]
        df_id = df_id.drop(['timestamp','id'], axis=1)
        label = df_id['label'].values[0]
        labels.append(label)
        df_id = df_id.drop(['label'], axis=1)
        df_id = df_id.dropna(axis=1)
        if (exclude_constant_bytes):
            for byte in const_dict[str(id)]:
                df_id = df_id.drop(['data{}'.format(byte-1)], axis=1) # drop constant byte
        bytes = df_id.to_numpy().flatten()
        cache[str(id)]['bytes'] = bytes # cache bytes
        cache[str(id)]['label'] = label # cache label
        s[offset:offsets[i]] = bytes
        offset = offsets[i]
        i += 1

    if (clean):
        return scale_s(s, unique_id_list, min_dict, max_dict), cache, max(labels)
    else:
        return None, cache, None

def get_constant_bytes(df, id):
    df_id = df.loc[df['id']==id]
    df_id = df_id.drop(['label','timestamp','id'], axis=1)
    df_id = df_id.dropna(axis=1) 
    # just relevant byte columns left
    constant_bytes_list = []
    constant_bytes = 0
    byte = 1
    for col in df_id:
        max = df_id[col].max()
        min = df_id[col].min()
        if (max == min):
            constant_bytes += 1
            constant_bytes_list.append(byte)
            byte += 1
            continue
        byte += 1
    return constant_bytes, constant_bytes_list

def info_constant_bytes(df, id):
    df_id = df.loc[df['id']==id]
    df_id = df_id.drop(['label','timestamp','id'], axis=1)
    df_id = df_id.dropna(axis=1) 
    # just relevant byte columns left
    constant_bytes = 0
    for col in df_id:
        max = df_id[col].max()
        min = df_id[col].min()
        if (max == min):
            constant_bytes += 1
    return constant_bytes

def main(inpath, outpath, delta_t, w, exclude_constant_bytes, constant_byte_file, min_max_file):
    files = []
    for file in os.listdir(inpath):
        if file.endswith(".csv"):
            files.append(inpath+file)
    for inputfile in files:
        df = pd.read_csv(inputfile, dtype={
                'label': bool,
                'timestamp': float,
                'id': str,
                'dlc': int,
                'data0': str,
                'data1': str,
                'data2': str,
                'data3': str,
                'data4': str,
                'data5': str,
                'data6': str,
                'data7': str,
            })

        df = df.drop(['dlc'], axis=1)
        # convert hex payloads to integer
        for i in range(8):
            df['data{}'.format(i)] = df['data{}'.format(i)].apply(convert_to_int)
        df = df[df.id != '671'] # exlude id with unregular bytes
        unique_id_list = df['id'].unique()
        unique_id_list.sort()
        unique_id_list = list(unique_id_list)

        min_dict = {}
        max_dict = {}
        const_dict = {}
        offsets = []
        const_list = []
        constant_bytes_total = 0

        # extract global minimums and maximums of training, validation, test, and attack data
        f = open(min_max_file)
        mins_maxs_pack = json.load(f)
        min_dict = mins_maxs_pack["mins"]
        max_dict = mins_maxs_pack["maxs"]

        if (constant_byte_file):
            f = open(constant_byte_file)
            const_byte_pack = json.load(f)
            const_dict = const_byte_pack["constant_bytes"]
            offsets = const_byte_pack["offsets"]
        cache = {}
        # initialize cache
        for id in unique_id_list:
            cache[str(id)] = {}
        max_t = df['timestamp'].max()
        min_t = df['timestamp'].min()
        
        # if we have an external file saying which bytes to exclude we do that
        # if we exclude constant bytes, but don't have an external file, the constant bytes are determined here
        # if we don't exlude constant bytes we just count them and leave them in the set
        for id in unique_id_list:
            if (exclude_constant_bytes):
                if (constant_byte_file):
                    constant_bytes = len(const_dict[str(id)])
                else:
                    constant_bytes, const_list = get_constant_bytes(df, id)
                    const_dict[str(id)] = const_list
            else:
                constant_bytes = info_constant_bytes(df, id)
            constant_bytes_total += constant_bytes
            if (not constant_byte_file):
                offsets.append(len(min_dict[str(id)]))

        # ambient dyno drive winter: 465 constant bytes
        # ambient dyno drive basic long: 363
        # ambient dyno drive basic short: 377
        # ambient dyno drive exercise all bits: 125
        # ambient highway street driving long: 293

        if (not constant_byte_file):
            offsets = compute_offsets(offsets)

        # prepare writing to disk
        file_name = os.path.basename(inputfile)
        file = os.path.splitext(file_name)

        outfile = file[0] + ".tfrecords"
        writer = tf.io.TFRecordWriter(outpath+outfile)

        # loop over timestamp steps
        s_len = offsets[-1]
        steps = 0
        total_s_len = int((max_t - min_t + delta_t)/delta_t)
        total_s = np.empty([total_s_len, s_len])
        total_label = np.empty([total_s_len,1])
        print("extracting {}.........".format(inputfile))
        for t in (tqdm(np.arange(min_t + delta_t, max_t + delta_t, delta_t))):

            s, cache, label = get_s(df, t, delta_t, offsets, unique_id_list, min_dict, max_dict, cache, const_dict, exclude_constant_bytes)
        
            if (not(s is None)):
                total_s[steps] = s # collect all scaled vectors s
                total_label[steps] = label
                steps += 1
        
        total_s = total_s[:steps] # truncate to true number of s
        X = np.lib.stride_tricks.sliding_window_view(total_s, (w, s_len)) # sliding window every delt_t seconds e.g. 0.01
        X = X.reshape(-1, w, s_len) 
        X = X.reshape(-1, w*s_len)
        Y = np.lib.stride_tricks.sliding_window_view(total_label, (w, 1))
        Y = Y.reshape(-1, w, 1)
        Y = Y.reshape(-1, w)
        Y = Y.astype(int)
        assert not np.any(np.isnan(X))
        print("writing TFRecord.........")
        for idx in tqdm(range(X.shape[0])):
            x = X[idx]
            y = Y[idx]
            example = tf.train.Example(features=tf.train.Features(feature={
                'X': tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                'Y': tf.train.Feature(int64_list=tf.train.Int64List(value=y))
            }))
            writer.write(example.SerializeToString())

        print("window: {}, number of bytes: {}".format(w, s_len))
        if (exclude_constant_bytes):
            pack = {}
            pack["constant_bytes"] = const_dict
            pack["offsets"] = offsets
            print("number of constant bytes excluded: {}".format(constant_bytes_total))
            with open ('Data/constant_bytes.json', 'w') as f:
                json.dump(pack, f)
        else:
            print("number of constant bytes: {}".format(constant_bytes_total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default="./ambient_street_driving_long.csv")
    parser.add_argument('--outpath', type=str, default="./")
    parser.add_argument('--timesteps', type=float, default=0.01)
    parser.add_argument('--windowsize', type=int, default=150)
    parser.add_argument('--exclude_constant_bytes', action='store_true')
    parser.add_argument('--constant_byte_file', type=str)
    parser.add_argument('--min_max_file', type=str, default="Data/ranges/min_max_merge.json")
    args = parser.parse_args()

    main(args.inpath, args.outpath, args.timesteps, args.windowsize, args.exclude_constant_bytes, args.constant_byte_file, args.min_max_file)
# Date: 05-02-2023
# Author: Mario Freund
# Purpose: Preprocess ambient signal extracted data of road dataset as done in x-canids

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
import sys
np.set_printoptions(threshold=sys.maxsize)

def compute_offsets(offsets):
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i-1]
    return offsets

def scale_s(s, unique_id_list, min_dict, max_dict):
    scaled_s = s # copy
    offset = 0
    for i in range(len(unique_id_list)):
        # get minimums and maximums of signal of each ID
        mins_i = min_dict[str(unique_id_list[i])]
        maxs_i = max_dict[str(unique_id_list[i])]
        for j in range(len(mins_i)): # scale
            if(maxs_i[j] == mins_i[j]):
                scaled_s[offset+j] = 1.0
            else:
                scaled_s[offset+j] = (scaled_s[offset+j] - mins_i[j]) / (maxs_i[j] - mins_i[j]) # s^_i  = (s_i - min_i) / (max_i - min_i)
        offset += len(mins_i)
    return scaled_s

def get_s(df, t, delta_t, offsets, unique_id_list, min_dict, max_dict, cache, all_seen):
    clean = True
    t_minus_one = t - delta_t
    df_t = df.loc[(df['Time'] <= t) & (df['Time'] > t_minus_one)]
    # get latest signals of each ID
    s = np.empty(offsets[-1]) # total amount of signals
    i = 0
    offset = 0
    for id in unique_id_list:
        df_id = df_t.loc[df_t['ID'] == id]
        if (df_id.empty): # take cached value
            if (str(id) in cache):
                s[offset:offsets[i]] = cache[str(id)]
            else:
                clean = False
            offset = offsets[i]
            i += 1
            continue
        index = df_id['Time'].idxmax() # latest value
        df_id = df_id.loc[[index]]
        df_id = df_id.drop(['Time','ID'], axis=1)
        df_id = df_id.dropna(axis=1)
        signals = df_id.to_numpy().flatten()
        cache[str(id)] = signals # cache signals
        s[offset:offsets[i]] = signals
        offset = offsets[i]
        i += 1

    if (clean):
        return scale_s(s, unique_id_list, min_dict, max_dict), cache
    else:
        return None, cache

def range_of_signals(df, id):
    df_id = df.loc[df['ID']==id]
    df_id = df_id.drop(['Time','ID'], axis=1)
    df_id = df_id.dropna(axis=1) 
    # just relevant signal columns left
    mins = []
    maxs = []
    for col in df_id:
        maxs.append(df_id[col].max())
        mins.append(df_id[col].min())
    return mins, maxs

def main(inputfile, outfile, delta_t, w):
    df = pd.read_csv(inputfile, dtype={
        'Label': bool,
        'Time': float,
        'ID': int,
        'Signal_1_of_ID': float,
        'Signal_2_of_ID': float,
        'Signal_3_of_ID': float,
        'Signal_4_of_ID': float,
        'Signal_5_of_ID': float,
        'Signal_6_of_ID': float,
        'Signal_7_of_ID': float,
        'Signal_8_of_ID': float,
        'Signal_9_of_ID': float,
        'Signal_10_of_ID': float,
        'Signal_11_of_ID': float,
        'Signal_12_of_ID': float,
        'Signal_13_of_ID': float,
        'Signal_14_of_ID': float,
        'Signal_15_of_ID': float,
        'Signal_16_of_ID': float,
        'Signal_17_of_ID': float,
        'Signal_18_of_ID': float,
        'Signal_19_of_ID': float,
        'Signal_20_of_ID': float,
        'Signal_21_of_ID': float,
        'Signal_22_of_ID': float,
    })

    df = df.drop(['Label'], axis=1)
    unique_id_list = df['ID'].unique()
    unique_id_list.sort()
    unique_id_list = list(unique_id_list)

    min_dict = {}
    max_dict = {}
    cache = {}
    offsets = []
    max_t = df['Time'].max()
    
    # get min max of each
    for id in unique_id_list:
        mins, maxs = range_of_signals(df, id)
        min_dict[str(id)] = mins
        max_dict[str(id)] = maxs
        offsets.append(len(mins))

    offsets = compute_offsets(offsets)

    # prepare writing to disk
    results_path = outfile + ".tfrecords"
    writer = tf.io.TFRecordWriter(results_path)

    # loop over time steps
    steps = 0
    s_w = np.empty([w, offsets[-1]])
    all_seen = False
    for t in (tqdm(np.arange(delta_t, max_t + delta_t, delta_t))):
        if (((steps % w) == 0) and (steps > 0)): # wait until we have w vectors
            s_w = np.reshape(s_w, w*offsets[-1]) # e.g. 200*664
            # write to TF_Record
            example = tf.train.Example(features=tf.train.Features(feature={
                'X': tf.train.Feature(float_list=tf.train.FloatList(value=s_w))
            }))
            writer.write(example.SerializeToString())
            s_w = np.empty([w, offsets[-1]])

        s, cache = get_s(df, t, delta_t, offsets, unique_id_list, min_dict, max_dict, cache, all_seen)
    
        if (not(s is None)):
            all_seen = True
            s_w[steps % w] = s
            steps += 1

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default="./ambient_street_driving_long.csv")
    parser.add_argument('--outfile', type=str, default="./")
    parser.add_argument('--timesteps', type=float, default=0.01)
    parser.add_argument('--windowsize', type=int, default=200)
    args = parser.parse_args()

    main(args.infile, args.outfile, args.timesteps, args.windowsize)
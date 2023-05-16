# Date: 05-16-2023
# Author: Mario Freund
# Purpose: Preprocess ambient signal extracted data of road dataset as done in x-canids with labels

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
import sys
import json
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
            if(maxs_i[j] == mins_i[j]): # constant value
                scaled_s[offset+j] = 1.0
            else:
                scaled_s[offset+j] = (scaled_s[offset+j] - mins_i[j]) / (maxs_i[j] - mins_i[j]) # s^_i  = (s_i - min_i) / (max_i - min_i)
                assert (scaled_s[offset+j] <= 1)
                assert (scaled_s[offset+j] >= 0)
        offset += len(mins_i)
    return scaled_s

def get_s(df, t, delta_t, offsets, unique_id_list, min_dict, max_dict, cache, const_dict, exclude_constant_signals):
    clean = True
    t_minus_one = t - delta_t
    df_t = df.loc[(df['Time'] <= t) & (df['Time'] > t_minus_one)]
    # get latest signals of each ID
    s = np.empty(offsets[-1]) # total amount of signals
    i = 0
    offset = 0
    label = 0
    for id in unique_id_list:
        df_id = df_t.loc[df_t['ID'] == id]
        if (df_id.empty): # take cached value
            if (str(id) in cache):
                s[offset:offsets[i]] = cache[str(id)]["signals"]
                label = cache[str(id)]["label"]
            else:
                clean = False
            offset = offsets[i]
            i += 1
            continue
        index = df_id['Time'].idxmax() # latest value
        df_id = df_id.loc[[index]]
        df_id = df_id.drop(['Time','ID'], axis=1)
        label = df_id['Label']
        df_id = df_id.drop(['Label'], axis=1)
        df_id = df_id.dropna(axis=1)
        if (exclude_constant_signals):
            for signal in const_dict[str(id)]:
                df_id = df_id.drop(['Signal_{}_of_ID'.format(signal)], axis=1) # drop constant signal
        signals = df_id.to_numpy().flatten()
        cache[str(id)]["signals"] = signals # cache signals
        cache[str(id)]["label"] = label # cache label
        s[offset:offsets[i]] = signals
        offset = offsets[i]
        i += 1

    if (clean):
        return scale_s(s, unique_id_list, min_dict, max_dict), cache, label
    else:
        return None, cache

def range_of_signals_constant(df, id):
    df_id = df.loc[df['ID']==id]
    df_id = df_id.drop(['Time','ID'], axis=1)
    df_id = df_id.dropna(axis=1) 
    # just relevant signal columns left
    mins = []
    maxs = []
    constant_signals_list = []
    constant_signals = 0
    signal = 1
    for col in df_id:
        max = df_id[col].max()
        min = df_id[col].min()
        if (max == min):
            constant_signals += 1
            constant_signals_list.append(signal)
            signal += 1
            continue
        maxs.append(max)
        mins.append(min)
        signal += 1
    return mins, maxs, constant_signals, constant_signals_list

def range_of_signals_constant_from_json(df, id, constant_signals_list):
    df_id = df.loc[df['ID']==id]
    df_id = df_id.drop(['Time','ID'], axis=1)
    df_id = df_id.dropna(axis=1) 
    # just relevant signal columns left
    mins = []
    maxs = []
    constant_signals = 0
    signal = 1
    for col in df_id:
        max = df_id[col].max()
        min = df_id[col].min()
        if (signal in constant_signals_list):
            constant_signals += 1
            signal += 1
            continue
        maxs.append(max)
        mins.append(min)
        signal += 1
    return mins, maxs, constant_signals, constant_signals_list

def range_of_signals(df, id):
    df_id = df.loc[df['ID']==id]
    df_id = df_id.drop(['Time','ID'], axis=1)
    df_id = df_id.dropna(axis=1) 
    # just relevant signal columns left
    mins = []
    maxs = []
    constant_signals = 0
    for col in df_id:
        max = df_id[col].max()
        min = df_id[col].min()
        if (max == min):
            constant_signals += 1
        maxs.append(max)
        mins.append(min)
    return mins, maxs, constant_signals

def main(inputfile, outfile, delta_t, w, exclude_constant_signals, constant_signal_file, truncate_to):
    df = pd.read_csv(inputfile, dtype={
        'Label': int,
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

    unique_id_list = df['ID'].unique()
    unique_id_list.sort()
    unique_id_list = list(unique_id_list)

    min_dict = {}
    max_dict = {}
    const_dict = {}
    offsets = []
    const_list = []
    constant_signals_total = 0
    if (constant_signal_file):
        f = open(constant_signal_file)
        const_signal_pack = json.load(f)
        const_dict = const_signal_pack["constant_signals"]
        offsets = const_signal_pack["offsets"]
    cache = {}
    max_t = df['Time'].max()
    
    # get min max of each
    # if we have an external file saying which signals to exclude we do that
    # if we exclude constant signals, but don't have an external file, the constant signals are determined here
    # if we don't exlude constant signals we just count them and leave them in the set
    for id in unique_id_list:
        if (exclude_constant_signals):
            if (constant_signal_file):
                mins, maxs, constant_signals, const_list = range_of_signals_constant_from_json(df, id, const_dict[str(id)])
            else:
                mins, maxs, constant_signals, const_list = range_of_signals_constant(df, id)
        else:
            mins, maxs, constant_signals = range_of_signals(df, id)
        constant_signals_total += constant_signals
        min_dict[str(id)] = mins
        max_dict[str(id)] = maxs
        const_dict[str(id)] = const_list
        if (not constant_signal_file):
            offsets.append(len(mins))

    # ambient dyno drive winter: 465 constant signals
    # ambient dyno drive basic long: 363
    # ambient dyno drive basic short: 377
    # ambient dyno drive exercise all bits: 125
    # ambient highway street driving long: 293

    if (not constant_signal_file):
        offsets = compute_offsets(offsets)

    # prepare writing to disk
    results_path = outfile + ".tfrecords"
    writer = tf.io.TFRecordWriter(results_path)

    # loop over time steps
    s_len = offsets[-1]
    if (truncate_to):
        s_len = truncate_to
    steps = 0
    total_s_len = int((max_t + delta_t)/delta_t)
    total_s = np.empty([total_s_len, s_len])
    total_label = np.empty([total_s_len,1])
    print("extracting.........")
    for t in (tqdm(np.arange(delta_t, max_t + delta_t, delta_t))):

        s, cache, label = get_s(df, t, delta_t, offsets, unique_id_list, min_dict, max_dict, cache, const_dict, exclude_constant_signals)
    
        if (not(s is None)):
            if (truncate_to):
                s = s[:truncate_to]
            total_s[steps] = s # collect all scaled vectors s
            total_label[steps] = label
            steps += 1
    
    total_s = total_s[:steps] # truncate to true number of s
    X = np.lib.stride_tricks.sliding_window_view(total_s, (w, s_len)) # sliding window every delt_t seconds e.g. 0.01
    X = X.reshape(-1, w, s_len) 
    X = X.reshape(-1, w*s_len)
    Y = np.lib.stride_tricks.sliding_window_view(total_label, (w, 1))
    Y = Y. reshape(-1, w, 1)
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

    print("window: {}, number of signals: {}".format(w, s_len))
    if (exclude_constant_signals):
        pack = {}
        pack["constant_signals"] = const_dict
        pack["offsets"] = offsets
        print("number of constant signals excluded: {}".format(constant_signals_total))
        with open ('Data/constant_signals.json', 'w') as f:
            json.dump(pack, f)
    else:
        print("number of constant signals: {}".format(constant_signals_total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default="./ambient_street_driving_long.csv")
    parser.add_argument('--outfile', type=str, default="./")
    parser.add_argument('--timesteps', type=float, default=0.01)
    parser.add_argument('--windowsize', type=int, default=200)
    parser.add_argument('--exclude_constant_signals', action='store_true')
    parser.add_argument('--constant_signal_file', type=str)
    parser.add_argument('--truncate_to', type=int)
    args = parser.parse_args()

    main(args.infile, args.outfile, args.timesteps, args.windowsize, args.exclude_constant_signals, args.constant_signal_file, args.truncate_to)
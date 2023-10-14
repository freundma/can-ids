# Date: 06-28-2023
# Author: Mario Freund
# Purpose: Preprocess ambient mvb data after the concept of x-canids
# Note: Complete constant payloads are expected be excluded before To execute this file, mostly proprietary MVB datasets are necessary. The field address shall contain the port address or device address.
#       Each signal of address field must have the hex form <0xbeef>. The slave frame lines must have been joint with the master frame lines beforehand to join the address of the
#       master frame with the corresponding slave frame.
# Commandline arguments:
#   --infile: A path to a csv file in master/ slave form with split payload as string
#       format of csv: Time,Control,Type,Frame Type,Payload,Address,Signal_1_of_Address,...,Signal_25_of_Address
#   --outfile: A path where to save the output tfrecord file as string
#   --timesteps: The timesteps in which the feature extraction should be executed (parameter t of X-CANDIS) as float
#   --constant_signal_file: A path to a constant signal file as produced by extract_constant_signals.py as string
#   --min_max_file: A path to the byte field ranges that are supposed to be used for the min max scaling as string

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
import sys
import json

def convert_to_int(x):
    if (type(x) == str):
        return int(x, 16)
    return x

def scale_s(s, unique_address_list, min_dict, max_dict):
    scaled_s = s # copy
    offset = 0
    for i in range(len(unique_address_list)):
        # get minimums and maximums of signal of each ID
        mins_i = min_dict[str(unique_address_list[i])]
        maxs_i = max_dict[str(unique_address_list[i])]
        for j in range(len(mins_i)): # scale
            if(maxs_i[j] == mins_i[j]): # constant value
                scaled_s[offset+j] = 1.0
            else:
                scaled_s[offset+j] = (scaled_s[offset+j] - mins_i[j]) / (maxs_i[j] - mins_i[j]) # s^_i  = (s_i - min_i) / (max_i - min_i)
        offset += len(mins_i)
    return scaled_s

def get_s(df, t, delta_t, offsets, unique_address_list, min_dict, max_dict, cache, const_dict):
    clean = True
    t_minus_one = t - delta_t
    df_t = df.loc[(df['Time'] <= t) & (df['Time'] > t_minus_one)]
    # get latest signals of each ID
    s = np.empty(offsets[-1]) # total amount of signals
    i = 0
    offset = 0
    for address in unique_address_list:
        df_address = df_t.loc[df_t['Address'] == address]
        if (df_address.empty): # take cached value
            if (str(address) in cache):
                s[offset:offsets[i]] = cache[str(address)]
            else:
                clean = False
            offset = offsets[i]
            i += 1
            continue
        index = df_address['Time'].idxmax() # latest value
        df_address = df_address.loc[[index]]
        df_address = df_address.drop(['Time','Address'], axis=1)
        df_address = df_address.dropna(axis=1)
        for signal in const_dict[str(address)]:
            df_address = df_address.drop(['Signal_{}_of_Address'.format(signal)], axis=1) # drop constant signal
        signals = df_address.to_numpy().flatten()
        cache[str(address)] = signals # cache signals
        s[offset:offsets[i]] = signals
        offset = offsets[i]
        i += 1

    if (clean):
        return scale_s(s, unique_address_list, min_dict, max_dict), cache
    else:
        return None, cache

def main(infile, outfile, delta_t, constant_signal_file, min_max_file):
    df = pd.read_csv(infile, dtype={
        'Time': float,
        'Control': str,
        'Type': str,
        'Frame Type': str,
        'Payload': str,
        'Address': str,
        'Signal_1_of_Address': str,
        'Signal_2_of_Address': str,
        'Signal_3_of_Address': str,
        'Signal_4_of_Address': str,
        'Signal_5_of_Address': str,
        'Signal_6_of_Address': str,
        'Signal_7_of_Address': str,
        'Signal_8_of_Address': str,
        'Signal_9_of_Address': str,
        'Signal_10_of_Address': str,
        'Signal_11_of_Address': str,
        'Signal_12_of_Address': str,
        'Signal_13_of_Address': str,
        'Signal_14_of_Address': str,
        'Signal_15_of_Address': str,
        'Signal_16_of_Address': str,
        'Signal_17_of_Address': str,
        'Signal_18_of_Address': str,
        'Signal_19_of_Address': str,
        'Signal_20_of_Address': str,
        'Signal_21_of_Address': str,
        'Signal_22_of_Address': str,
        'Signal_23_of_Address': str,
        'Signal_24_of_Address': str,
        'Signal_25_of_Address': str
    })

    # The feature extraction only takes slave frames into account
    df = df.loc[df['Frame Type'] == 'Slave frame']
    df = df.drop(['Control','Type','Frame Type','Payload'], axis=1)

    # convert hex payloads to integer
    for i in range(1, 25):
        df['Signal_{}_of_Address'.format(i)] = df['Signal_{}_of_Address'.format(i)].apply(convert_to_int)

    unique_address_list = df['Address'].unique()
    unique_address_list.sort()
    unique_address_list = list(unique_address_list)

    constant_signals_total = 0

    # get mins and maxs for scaling process
    f = open(min_max_file)
    mins_maxs_pack = json.load(f)
    min_dict = mins_maxs_pack["mins"]
    max_dict = mins_maxs_pack["maxs"]

    # get constant signals
    f = open(constant_signal_file)
    const_signal_pack = json.load(f)
    const_dict = const_signal_pack["constant_signals"]
    offsets = const_signal_pack["offsets"]
    
    cache = {}
    max_t = df['Time'].max()
    min_t = df['Time'].min()

    for address in unique_address_list:
        constant_signals = len(const_dict[address])
        constant_signals_total += constant_signals

    # prepare writing to disk
    results_path = outfile + ".tfrecords"
    writer = tf.io.TFRecordWriter(results_path)

     # loop over time steps
    s_len = offsets[-1]
    steps = 0
    total_s_len = int((max_t + delta_t)/delta_t)
    total_s = np.empty([total_s_len, s_len])
    print("extracting.........")
    for t in (tqdm(np.arange(min_t + delta_t, max_t + delta_t, delta_t))):

        s, cache = get_s(df, t, delta_t, offsets, unique_address_list, min_dict, max_dict, cache, const_dict)
    
        if (not(s is None)):
            total_s[steps] = s # collect all scaled vectors s
            steps += 1
    
    total_s = total_s[:steps] # truncate to true number of s
    assert not np.any(np.isnan(total_s))
    assert not np.any(np.isinf(total_s))
    print("writing TFRecord.........")
    for idx in tqdm(range(total_s.shape[0])):
        s = total_s[idx]
        example = tf.train.Example(features=tf.train.Features(feature={
            'S': tf.train.Feature(float_list=tf.train.FloatList(value=s))
        }))
        writer.write(example.SerializeToString())

    print("number of signals: {}".format(s_len))
    print("number of constant signals excluded: {}".format(constant_signals_total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--timesteps', type=float, default=0.016)
    parser.add_argument('--constant_signal_file', type=str, default='Data/constant_signals.json')
    parser.add_argument('--min_max_file', type=str, default='Data/ranges/min_max_merge.json')
    args = parser.parse_args()

    main(args.infile, args.outfile, args.timesteps, args.constant_signal_file, args.min_max_file)

# Date: 05-02-2023
# Author: Mario Freund
# Purpose: Preprocess ambient signal extracted data of road as done in x-canids

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

def compute_offsets(offsets):
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i-1]
    return offsets

def scale_s(s, unique_id_list, min_dict, max_dict):
    scaled_s = s
    i = 0
    for s_i in np.nditer(s):
        min_i = min_dict[str(unique_id_list[i])]
        max_i = max_dict[str(unique_id_list[i])]
        scaled_s[i] = (s_i - min_i) / (max_i - min_i)
        i += 1
    return scaled_s

def get_s(df, t, offsets, unique_id_list, min_dict, max_dict):
    df_t = df.loc[df['Time'] <= t]
    unique_id_list_t = df_t['ID'].unique().tolist()
    if(len(unique_id_list) != len(unique_id_list_t)):
        return None # have to see each ID at least once
    # get latest signals of each ID
    s = np.empty(sum(offsets)) # total amount of signals
    i = 0
    offset = 0
    for id in unique_id_list:
        df_id = df_t.loc[df_t['ID'] == id]
        df_id = df_id[df.Time == df.Time.max()] # latest value
        df_id = df_id.drop(['Time','ID'])
        df_id = df_id.dropna(axis=1)
        signals = df.to_numpy.flatten()
        s[offset:offsets[i]] = signals
        offset += offsets[i]
        i += 1
    
    return scale_s(s, unique_id_list, min_dict, max_dict)

def range_of_signals(df, id):
    df_id = df.loc[df['ID']==id]
    df_id = df_id.drop('ID')
    df_id = df_id.dropna(axis=1) 
    # just relevant columns left
    mins, maxs = []
    for col in df_id:
        maxs.append(df_id[col].max())
        mins.append(df_id[col].min())
    return mins, maxs

def main(inputfile, outdir, t, w):
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

    df = df.drop('Label')
    unique_id_list = df['ID'].unique().tolist().sort()

    # get min max of each 
    min_dict = {}
    max_dict = {}
    offsets = []
    max_t = df['Time'].max
    
    for id in unique_id_list:
        mins, maxs = range_of_signals(df, id)
        min_dict[str(id)] = mins
        max_dict[str(id)] = maxs
        offsets.append(len(mins))

    offsets = compute_offsets(offsets)

    # loop over time steps
    steps = 0
    for t in range(0, max_t, t):
        steps += 1
        s = get_s(df, t, offsets, unique_id_list, min_dict, max_dict)
        if ((steps % w) == 0):
            # TODO









    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default="./ambient_street_driving_long.csv")
    parser.add_argument('--outdir', type=str, default="./")
    parser.add_argument('--timesteps', type=float, default=0.01)
    parser.add_argument('--windowsize', type=int, default=40)
    args = parser.parse_args()

    main(args.infile, args.outdir, args.timesteps, args.windowsize)
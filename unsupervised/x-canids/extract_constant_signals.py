# Date: 06-15-2023
# Author: Mario Freund
# Purpose: Extract constant signals from dataset

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
import sys
import json

def compute_offsets(offsets):
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i-1]
    return offsets

def get_constant_signals(df, id):
    df_id = df.loc[df['ID']==id]
    df_id = df_id.drop(['Time','ID'], axis=1)
    df_id = df_id.dropna(axis=1) 
    # just relevant signal columns left
    constant_signals_list = []
    signals_per_id = 0
    constant_signals = 0
    signal = 1
    for col in df_id:
        max = df_id[col].max()
        min = df_id[col].min()
        if (max == min and (id != 208) and (id !=1255) and (id != 1760)): # evaluation will be conducted on those IDs
            constant_signals += 1
            constant_signals_list.append(signal)
            signal += 1
            continue
        signals_per_id += 1
        signal += 1
    return constant_signals, constant_signals_list, signals_per_id

def main (infile, outfile, syncan):
    if (syncan):
        df = pd.read_csv(infile, dtype={
        'Label': bool,
        'Time': float,
        'ID': str,
        'Signal_1_of_ID': float,
        'Signal_2_of_ID': float,
        'Signal_3_of_ID': float,
        'Signal_4_of_ID': float,
        })
    else:
        df = pd.read_csv(infile, dtype={
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
    if (not syncan):
        df = df[df.ID != 1649] # exlude ID with unregular signals
    unique_id_list = df['ID'].unique()
    unique_id_list.sort()
    unique_id_list = list(unique_id_list)

    const_list = []
    const_dict = {}
    offsets = []
    constant_signals_total = 0

    for id in unique_id_list:
        constant_signals, const_list, signals_per_id = get_constant_signals(df, id)
        const_dict[str(id)] = const_list
        offsets.append(signals_per_id)
        constant_signals_total += constant_signals

    offsets = compute_offsets(offsets)

    print("number of constant signals: {}".format(constant_signals_total))
    pack = {}
    pack["constant_signals"] = const_dict
    pack["offsets"] = offsets
    with open (outfile, 'w') as f:
        json.dump(pack, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default="ambient_street_driving_long.csv")
    parser.add_argument('--outfile', type=str, default="Data/constant_signals.json")
    parser.add_argument('--syncan', action='store_true')
    args = parser.parse_args()

main(args.infile, args.outfile, args.syncan)


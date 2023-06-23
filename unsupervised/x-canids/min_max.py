# Date: 06-06-2023
# Author: Mario Freund
# Purpose: Determine mins and maxs of signals in dataset, external constant signal file needs to be provided

import pandas as pd
import argparse
import json
import os

def range_of_signals(df, id, constant_signals_list):
    df_id = df.loc[df['ID']==id]
    df_id = df_id.drop(['Time','ID'], axis=1)
    df_id = df_id.dropna(axis=1) 
    # just relevant signal columns left

    mins = []
    maxs = []
    signal = 1

    for col in df_id:
        if (signal in constant_signals_list):
            signal += 1
            continue
        max = df_id[col].max()
        min = df_id[col].min()
        maxs.append(max)
        mins.append(min)
        signal += 1
    
    return mins, maxs


def main(inpath, outpath, constant_signal_file, syncan):
    files = []
    for file in os.listdir(inpath):
        if file.endswith(".csv"):
            files.append(inpath + file)
    for infile in files:
        if (syncan):
            df = pd.read_csv(infile, dtype={
                'Label': bool,
                'Time': float,
                'ID': str,
                'Signal_1_of_ID': float,
                'Signal_2_of_ID': float,
                'Signal_3_of_ID': float,
                'Signal_4_of_ID': float
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

        min_dict = {}
        max_dict = {}
        const_dict = {}
        
        # get mins, maxs

        f = open(constant_signal_file)
        const_signal_pack = json.load(f)
        const_dict = const_signal_pack["constant_signals"]
        for id in unique_id_list:
            mins, maxs = range_of_signals(df, id, const_dict[str(id)])
            min_dict[str(id)] = mins
            max_dict[str(id)] = maxs

        # save to file
        pack = {}
        pack["mins"] = min_dict
        pack["maxs"] = max_dict

        # outfile
        file_name = os.path.basename(infile)
        file = os.path.splitext(file_name)

        outfile = "min_max_" + file[0] + ".json"

        with open (outpath+outfile, 'w') as f:
            json.dump(pack, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default="Data/csv_with_street/")
    parser.add_argument('--outpath', type=str, default="Data/ranges/")
    parser.add_argument('--constant_signal_file', type=str)
    parser.add_argument('--syncan', action='store_true')
    args = parser.parse_args()

    main(args.inpath, args.outpath, args.constant_signal_file, args.syncan)
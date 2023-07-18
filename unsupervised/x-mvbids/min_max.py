# Date: 06-28-2023
# Author: Mario Freund
# Purpose: Determine mins and maxs of signals in dataset, external constant signal file needs to be provided

import pandas as pd
import argparse
import json
import os

def convert_to_int(x):
    if (type(x) == str):
        return int(x, 16)
    return x

def range_of_signals(df, address, constant_signals_list):
    df_address = df.loc[df['Address']==address]
    df_address = df_address.drop(['Time','Address'], axis=1)
    df_address = df_address.dropna(axis=1) 
    # just relevant signal columns left

    mins = []
    maxs = []
    signal = 1

    for col in df_address:
        if (signal in constant_signals_list):
            signal += 1
            continue
        max = float(df_address[col].max())
        min = float(df_address[col].min())
        maxs.append(max)
        mins.append(min)
        signal += 1
    
    return mins, maxs

def main(inpath, outpath, constant_signal_file):
    files = []
    for file in os.listdir(inpath):
        if file.endswith(".csv"):
            files.append(inpath + file)
    for infile in files:
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
        df = df.loc[df['Frame Type'] == 'Slave frame']
        df = df.drop(['Control','Type','Frame Type','Payload'], axis=1)

        # convert hex payloads to integer
        for i in range(1, 25):
            df['Signal_{}_of_Address'.format(i)] = df['Signal_{}_of_Address'.format(i)].apply(convert_to_int)
    
        unique_address_list = df['Address'].unique()
        unique_address_list.sort()
        unique_address_list = list(unique_address_list)

        min_dict = {}
        max_dict = {}

        # get minimums, maximums
        f = open(constant_signal_file)
        constant_signal_pack = json.load(f)
        const_dict = constant_signal_pack["constant_signals"]
        for address in unique_address_list:
            mins, maxs = range_of_signals(df, address, const_dict[address])
            min_dict[address] = mins
            max_dict[address] = maxs

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
    parser.add_argument('--inpath', type=str, default='Data/csv/')
    parser.add_argument('--outpath', type=str, default='Data/ranges/')
    parser.add_argument('--constant_signal_file', type=str)
    args = parser.parse_args()

    main(args.inpath, args.outpath, args.constant_signal_file)
# Date: 06-26-2023
# Author: Mario Freund
# Purpose: Find out singel constant signals in the already split up payload of mvb data

import sys
import pandas as pd
import json
import argparse

infile = sys.argv[1]

def compute_offsets(offsets):
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i-1]
    return offsets

def get_constant_signals(df, address):
    df_address = df.loc[(df['Address'] == address)]
    df_address = df_address.drop(['Time', 'Frame Type', 'Payload', 'Address'], axis=1)
    df_address = df_address.dropna(axis=1)

    constant_signals_list = []
    signals_per_address = 0
    constant_signals = 0
    signal = 1
    for col in df_address:
        df_address[col].apply(lambda x: int(x, 16))
        max = df_address[col].max()
        min = df_address[col].min()
        if (max == min):
            constant_signals += 1
            constant_signals_list.append(signal)
            signal += 1
            continue
        signals_per_address += 1
        signal += 1
    return constant_signals, constant_signals_list, signals_per_address

def main(infile, outfile):
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
    df = df.drop(['Control','Type'], axis=1)

    unique_address_list = df['Address'].unique()
    unique_address_list.sort()
    unique_address_list = list(unique_address_list)

    const_list = []
    const_dict = {}
    offsets = []
    constant_signals_total = 0

    for address in unique_address_list:
        constant_signals, const_list, signals_per_address = get_constant_signals(df, address)
        const_dict[address] = const_list
        offsets.append(signals_per_address)
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
    parser.add_argument('--infile', type=str)
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

main(args.infile, args.outfile)
# Date: 09-27-2023
# Author: Mario Freund
# Purpose: Extract constant byte fields from dataset
# Commandline arguments:
#   --infile: A path to a csv file like one of the byte-based road dataset as string
#       format of csv: label,timestamp,id,dlc,data0,...,data7
#   --outfile: A path were to output the resulting json file to as string

import pandas as pd
import argparse
import json

def compute_offsets(offsets):
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i-1]
    return offsets

def get_constant_bytes(df, id):
    num_bytes = 0
    df_id = df.loc[df['id']==id]
    df_id = df_id.drop(['timestamp','id'], axis=1)
    df_id = df_id.dropna(axis=1) 
    # just relevant byte columns left
    constant_bytes_list = []
    bytes_per_id = 0
    constant_bytes = 0
    byte = 1
    for col in df_id:
        num_bytes += 1
        df_id[col].apply(lambda x: int(x,16))
        max = df_id[col].max()
        min = df_id[col].min()
        if ((max == min) and (id != '0D0') and (id !='4E7') and (id != '6E0')): # evaluation will be conducted on those ids
            constant_bytes += 1
            constant_bytes_list.append(byte)
            byte += 1
            continue
        bytes_per_id += 1
        byte += 1
    return constant_bytes, constant_bytes_list, bytes_per_id, num_bytes

def main (infile, outfile):
    df = pd.read_csv(infile, dtype={
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

    df = df.drop(['label'], axis=1)
    df = df.drop(['dlc'], axis=1)
    df = df[df.id != '671'] # exlude id with unregular bytes
    unique_id_list = df['id'].unique()
    unique_id_list.sort()
    unique_id_list = list(unique_id_list)

    const_list = []
    const_dict = {}
    offsets = []
    constant_bytes_total = 0
    bytes_total = 0

    for id in unique_id_list:
        constant_bytes, const_list, bytes_per_id, num_bytes = get_constant_bytes(df, id)
        const_dict[str(id)] = const_list
        offsets.append(bytes_per_id)
        constant_bytes_total += constant_bytes
        bytes_total += num_bytes

    offsets = compute_offsets(offsets)

    print("number of byte streams: {}".format(len(unique_id_list)))
    print("number of bytes: {}".format(bytes_total))
    print("number of constant bytes: {}".format(constant_bytes_total))
    pack = {}
    pack["constant_bytes"] = const_dict
    pack["offsets"] = offsets
    with open (outfile, 'w') as f:
        json.dump(pack, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default="ambient_street_driving_long.csv")
    parser.add_argument('--outfile', type=str, default="Data/constant_bytes.json")
    args = parser.parse_args()

main(args.infile, args.outfile)


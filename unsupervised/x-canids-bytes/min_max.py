# Date: 09-28-2023
# Author: Mario Freund
# Purpose: Determine mins and maxs of byte values in dataset, external constant bytes file needs to be provided
# Command line arguments:
#   --infile: A path to a csv file like one of the byte-based road dataset as string
#       format of csv: label,timestamp,id,dlc,data0,...,data7
#   --outpath: A path where to output the byte ranges per csv as string
#   --constant_signal_file: A path to a constant byte file as created by extract_constant_bytes.py as string


import pandas as pd
import argparse
import json
import os

def convert_to_int(x):
    if (type(x) == str):
        return int(x, 16)
    return x

def range_of_bytes(df, id, constant_bytes_list):
    df_id = df.loc[df['id']==id]
    df_id = df_id.drop(['timestamp','id'], axis=1)
    df_id = df_id.dropna(axis=1) 
    # just relevant byte columns left

    mins = []
    maxs = []
    byte = 1

    for col in df_id:
        if (byte in constant_bytes_list):
            byte += 1
            continue
        max = float(df_id[col].max())
        min = float(df_id[col].min())
        maxs.append(max)
        mins.append(min)
        byte += 1
    
    return mins, maxs


def main(inpath, outpath, constant_byte_file):
    files = []
    for file in os.listdir(inpath):
        if file.endswith(".csv"):
            files.append(inpath + file)
    for infile in files:
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

        # convert hex payloads to integer
        for i in range(8):
            df['data{}'.format(i)] = df['data{}'.format(i)].apply(convert_to_int)

        unique_id_list = df['id'].unique()
        unique_id_list.sort()
        unique_id_list = list(unique_id_list)

        min_dict = {}
        max_dict = {}
        const_dict = {}
        
        # get mins, maxs

        f = open(constant_byte_file)
        const_byte_pack = json.load(f)
        const_dict = const_byte_pack["constant_bytes"]
        for id in unique_id_list:
            mins, maxs = range_of_bytes(df, id, const_dict[str(id)])
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
    parser.add_argument('--constant_byte_file', type=str)
    args = parser.parse_args()

    main(args.inpath, args.outpath, args.constant_byte_file)
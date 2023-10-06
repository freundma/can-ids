# Date: 09-28-2023
# Author: Mario Freund
# Purpose: Merge mins and maxs from multiple datasets

import pandas as pd
import argparse
import json
import os

def main(inpath, outpath):

    files = []
    for file in os.listdir(inpath):
        if file.endswith(".json"):
            files.append(inpath + file)

    # initialize
    f = open(files[0])
    mins_maxs_pack = json.load(f)
    mins = mins_maxs_pack["mins"]
    maxs = mins_maxs_pack["maxs"]

    for file in files[1:]:
        f = open(file)
        mins_maxs_pack = json.load(f)
        file_mins = mins_maxs_pack["mins"]
        file_maxs = mins_maxs_pack["maxs"]

        # replace min by smaller min
        for id in file_mins:
            for i in range(len(file_mins[id])):
                if file_mins[id][i] < mins[id][i]:
                    mins[id][i] = file_mins[id][i]

        # replace max by bigger max
        for id in file_maxs:
            for i in range(len(file_maxs[id])):
                if file_maxs[id][i] > maxs[id][i]:
                    maxs[id][i] = file_maxs[id][i]

    # print to file
    pack = {}
    pack["mins"] = mins
    pack["maxs"] = maxs

    with open (outpath + "min_max_merge.json", 'w') as f:
        json.dump(pack, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default="Data/ranges/")
    parser.add_argument('--outpath', type=str, default="Data/ranges/")
    args = parser.parse_args()

    main(args.inpath, args.outpath)
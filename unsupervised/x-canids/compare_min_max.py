# Date: 06-21-2023
# Author: Mario Freund
# Purpose: Compare min max ranges of files

import json
import argparse

def main(infile1, infile2):
    f1 = open(infile1)
    f2 = open(infile2)

    pack1 = json.load(f1)
    pack2 = json.load(f2)

    mins1, mins2 = pack1["mins"], pack2["mins"]
    maxs1, maxs2 = pack1["maxs"], pack2["mins"]

    for id in (mins1):
        for i in range(len(mins1[id])):
            if (mins2[id][i] < mins1[id][i]):
                print("min divergence for ID {} with index {} - reference: {}, other file: {}".format(id, i, mins1[id][i], mins2[id][i]))
    
    for id in (maxs1):
        for i in range(len(maxs1[id])):
            if (maxs2[id][i] > maxs1[id][i]):
                print("max divergence for ID {} with index {} - reference: {}, other file: {}".format(id, i, maxs1[id][i], maxs2[id][i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=str)
    parser.add_argument('--infile', type=str)
    args = parser.parse_args()

    main(args.reference, args.infile)
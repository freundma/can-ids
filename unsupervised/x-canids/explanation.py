# Date: 09-27-2023
# Author: Mario Freund
# Purpose: Visualize explainability of x-canids
# Commandline arguments:
#   --error_path: A path were to take the example error rates from as returned by evaluate.py as string
#   --threshold_path: A path were to take the max_rs and O_is from as returned by threshold.py as string
#   --q: The percentile to use as sensitivity parameter for intrusion detection 0<=q<=1 as float
#   --min_max_file: The file were to take the signal-ranges from (.json) as returned by min_max.py as string

import numpy as np
import argparse
import matplotlib.pyplot as plt
import json

def main(error_path, threshold_path, q, min_max_file):
    # load numpy arrays and derive threshold
    loss_vectors = np.load(error_path+'error_rates.npy')
    max_rs = np.load(threshold_path+'max_rs.npy')
    O_i = np.load(threshold_path+'O_i.npy')
    O = np.percentile(max_rs, q*100)

    print("Theta : {}".format(O))

    # load min max file for signal mapping
    f = open(min_max_file)
    mins_maxs_pack = json.load(f)
    min_dict = mins_maxs_pack["mins"]
    signals = []

    for id in min_dict:
        for i in range (len(min_dict[id])):
            signals.append(id+'_'+str(i))
    
    # pick first frame exceeding threshold
    for idx in range (loss_vectors.shape[0]):
        x = loss_vectors[idx]
        x = x / O_i
        if (np.max(x) >= O):
            signal = signals[np.argmax(x)]
            print(signal + " caused intrusion alert!")
            # draw error rates
            plt.bar(range(len(x)), x,align='center', label= 'error rate per signal')
            plt.axhline(y=O, color = 'r', linestyle = '-', label='intrusion threshold for q=1')
            #plt.xticks(range(len(signals)), signals, size='small')
            plt.xlabel("Signals")
            plt.ylabel("Error rate")
            plt.legend()
            plt.savefig('Data/road/explanation.png',dpi=400)
            exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_path', type=str, default="Data/losses/")
    parser.add_argument('--threshold_path', type=str, default = "Data/thresholds/")
    parser.add_argument('--q', type=float, default=0.99)
    parser.add_argument('--min_max_file', type=str, default="Data/ranges/min_max_merge.json") # is used to identify signals
    args = parser.parse_args()

    main(args.error_path, args.threshold_path, args.q, args.min_max_file)

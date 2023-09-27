# Date: 09-27-2023
# Author: Mario Freund
# Purpose: Visualize explainability of x-canids

import numpy as np
import argparse
import matplotlib.pyplot as plt
import json

def main(error_path, threshold_path, q, min_max_file):
    # load numpy arrays and derive threshold
    error_rates = np.load(error_path+'error_rates.npy')
    max_rs = np.load(threshold_path+'max_rs.npy')
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
    for idx in range (error_rates.shape[0]):
        if (np.max(error_rates[idx]) >= O):
            signal = signals[np.argmax(error_rates[idx])]
            print(signal + " caused intrusion alert!")
            # draw error rates
            plt.bar(range(len(error_rates[idx]), error_rates[idx]),align='center')
            plt.axhline(y=O, color = 'r', linestyle = '-')
            plt.xticks(range(len(signals)), signals, size='small')
            plt.xlabel("Signals")
            plt.ylabel("Error rate")
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_path', type=str, default="Data/losses/")
    parser.add_argument('--threshold_path', type=str, default = "Data/thresholds/")
    parser.add_argument('--q', type=float, default=0.99)
    parser.add_argument('--min_max_file', type=str, default="Data/ranges/min_max_merge.json") # is used to identify signals
    args = parser.parse_args()

    main(args.error_path, args.threshold_path, args.q, args.min_max_file)

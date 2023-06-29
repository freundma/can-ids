# Date: 06-14-2023
# Author: Mario Freund
# Purpose: Analyze timing of master frames (message or process data) in MVB data

import sys
import pandas as pd

def get_f_code(payload): # payload is hexstring (e.g. 0x40b3)
    return payload[2]

def get_address(payload):
    if (int(payload[2],16) < 5 or payload[2] == 'c'): # process data or message data
        return payload[3:]
    else:
        return ""

file = sys.argv[1]
print("analyzing {}..............................".format(file), file=sys.stderr)

df = pd.read_csv(file)
df = df.filter(['Time', 'Frame Type', 'Payload'])
df = df.loc[df['Frame Type']=='Master frame']
df['FCode'] = df['Payload'].apply(get_f_code)
df['Address'] = df['Payload'].apply(get_address)
df = df.loc[df['Address'] != ""]

unique_address_list = df['Address'].unique()
unique_address_list.sort()
unique_address_list = list(unique_address_list)
min_mean_delta_t = 10
max_mean_delta_t = 0

for address in unique_address_list:
    df_address = df.loc[df['Address'] == address]
    df_address = df_address.drop(['Frame Type','Payload','FCode','Address'], axis=1)
    df_address = df_address.diff()
    df_address = df_address.dropna(axis=0)
    mean = df_address['Time'].mean()
    if (mean > max_mean_delta_t):
        max_mean_delta_t = mean
    if (mean < min_mean_delta_t):
        min_mean_delta_t = mean
    std = df_address['Time'].std()
    print("Address: {}   Average delta t: {}     Standard deviation delta t: {}".format(address, mean, std))
print ("Max mean delta t: {}    Min mean delta t: {}".format(max_mean_delta_t, min_mean_delta_t))


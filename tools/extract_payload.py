# Date: 06-23-2023
# Author: Mario Freund
# Purpose: Split up payloads from slave frames into fields

import pandas as pd
import json
import sys
import numpy as np
from tqdm import tqdm

format_string = 'Signal_{}_of_Address'

def get_reduce_bytes(payload, num_bytes):
    hex_digits = 2*num_bytes # two hex digits are one byte
    if (payload == -1):
        ret = "0x" + payload # sometimes we don't know the exact length of the last field
    else:
        ret = "0x" + payload[:hex_digits]
    payload = payload[hex_digits:]
    return ret, payload


def split_payload(df, address, payload_desc):
    slave_indices = df.index[(df['Address'] == address) & (df['Frame Type'] == 'Slave frame')]
    for i in tqdm(slave_indices):
        payload = df.loc[i, 'Payload']
        payload = payload[2:] # cut off "0x"
        signal = 1
        for p in payload_desc:
            df.loc[i, format_string.format(signal)], payload = get_reduce_bytes(payload, p)
            signal += 1
        signal = 1
    return df


def get_address(payload):
    try:
        if (int(payload[2],16) < 5 or payload[2] == 'c'): # process data or message data
            return payload[3:]
        else:
            return ""
    except:
        return ""
    
infile = sys.argv[1]
outfile = sys.argv[2]
payload_details_file = sys.argv[3]
constant_payload_file = sys.argv[4]

df = pd.read_csv(infile)

df = df[df['Address'].notnull()]
df = df.reset_index(drop=True)

with open(payload_details_file, 'r') as f:
    payload_details = json.load(f)

with open(constant_payload_file, 'r') as f:
    constant_payload = json.load(f)

# drop master/ slave frames with constant payload
for address in constant_payload:
    df = df.drop(df[df['Address'] == address].index)

for address in payload_details:
    print ("splitting up payload for address {}.....".format(address))
    df = split_payload(df, address, payload_details[address])

df.to_csv(outfile, index=False)



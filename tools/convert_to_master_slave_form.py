# Date: 06-27-2023
# Author: Mario Freund
# Purpose: Drop Event frames and join slave frames with master frames
# Usage: python convert_to_master_slave_form.py FH1_parsed_timestamps.csv FH1_master_slave.csv

import sys
import pandas as pd
from tqdm import tqdm
import json

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
df = pd.read_csv(infile)

# get addresses
df = df.loc[df['Type'] != 'Event']
df['Address'] = df['Payload'].apply(get_address)
df = df.reset_index(drop=True)

# join slave frame with address of master frame
indices = df.index[df['Frame Type'] == 'Slave frame']
for i in indices:
    df.loc[i, 'Address'] = df.loc[i-1, 'Address']

df.to_csv(outfile, index=False)

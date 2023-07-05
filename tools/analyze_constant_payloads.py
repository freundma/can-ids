# Date: 06-21-2023
# Author: Mario Freund
# Purpose: Check how many constant signals exist in MVB data

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
    

file = sys.argv[1]
df = pd.read_csv(file)

df_slave = df.loc[(df['Frame Type']=='Slave frame') & (df['Address'].notnull())]

unique_address_list = df_slave['Address'].unique()
unique_address_list.sort()
unique_address_list = list(unique_address_list)

constant_payload = {}

for i in tqdm(range(len(unique_address_list))):
    indices = df.index[(df['Address'] == unique_address_list[i]) & (df['Frame Type'] == 'Slave frame' )].tolist()
    first = df.loc[indices[0], 'Payload']
    constant = True
    for j in indices:
        if (df.loc[j, 'Payload'] != first):
            constant = False
            break
    if (constant):
        constant_payload[str(unique_address_list[i])] = "constant"

with open ('constant_payload.json', 'w') as f:
    json.dump(constant_payload, f, indent=4)
    
           



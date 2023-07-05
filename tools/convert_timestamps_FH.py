# Date: 06-12-2023
# Author: Mario Freund
# Purpose: Convert ticks to real timestamps and start from 0

import sys
import pandas as pd

def shift(timestamp):
    if (timestamp < 0):
        timestamp += MAX_TIMESTAMP
    return timestamp

infile = sys.argv[1]
outfile = sys.argv[2]

TIMESTAMP_RESOLUTION_US = 48 # ticks per us
TIMESTAMP_RESOLUTION_S = 48*(10**6) # ticks per s
MAX_TIMESTAMP = int("0xffffffff", 16)/TIMESTAMP_RESOLUTION_S

df = pd.read_csv(infile, dtype={
    'Time(ticks)': str,
    'Control': str,
    'Type': str,
    'Frame': str,
    'Payload': str
})

df['Time(ticks)'] = df['Time(ticks)'].apply(lambda x: int(x, 16))
first = df.loc[0, 'Time(ticks)']
df['Time(ticks)'] = df['Time(ticks)'].apply(lambda x: x - first)


df = df.rename(columns={'Time(ticks)': 'Time'})
df['Time'] = df['Time'].apply(lambda x: x/TIMESTAMP_RESOLUTION_S)

# shift values
df['Time'] = df['Time'].apply(shift)
df = df.sort_values('Time', ascending=True)

df.to_csv(outfile, index=False)


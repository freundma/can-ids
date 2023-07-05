# Date: 06-12-2023
# Author: Mario Freund
# Purpose: Convert ticks to real timestamps and start from 0

import sys
import pandas as pd
from tqdm import tqdm

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

# shift values, code reference: https://stackoverflow.com/questions/23151246/iterrows-pandas-get-next-rows-value/23151722#23151722
# switch frames in wrong oder
row_iterator = df.iterrows()
swap_first = []
swap_second = []
i_last, last = next(row_iterator)
for i, row in tqdm(row_iterator):
    if (row['Time'] < df.at[i_last,'Time'] and (abs(df.at[i_last,'Time'] - row['Time']) < (MAX_TIMESTAMP - 1e-03))): # very stupid heuristic to check the for the timestamp overflow
        # no timestamp overflow -> swap entries
        swap_first.append(i_last)
        swap_second.append(i)
    i_last = i

for i in range (len(swap_first)):
    df.iloc[swap_first[i]], df.iloc[swap_second[i]] = df.iloc[swap_second[i]].copy(), df.iloc[swap_first[i]].copy()

# shift
row_iterator = df.iterrows()
print("first period....")
i_last, last = next(row_iterator)
for i, row in tqdm(row_iterator):
    if (row['Time'] < df.at[i_last,'Time']):
        df.at[i, 'Time'] = row['Time'] + MAX_TIMESTAMP
    i_last = i

row_iterator = df.iterrows()
print("second period....")
i_last, last = next(row_iterator)
for i, row in tqdm(row_iterator):
    if (row['Time'] < df.at[i_last,'Time']):
        df.at[i, 'Time'] = row['Time'] + MAX_TIMESTAMP
    i_last = i

row_iterator = df.iterrows()
print("third period....")
i_last, last = next(row_iterator)
for i, row in tqdm(row_iterator):
    if (row['Time'] < df.at[i_last,'Time']):
        df.at[i, 'Time'] = row['Time'] + MAX_TIMESTAMP
    i_last = i

row_iterator = df.iterrows()
print("fourth period....")
i_last, last = next(row_iterator)
for i, row in tqdm(row_iterator):
    if (row['Time'] < df.at[i_last,'Time']):
        df.at[i, 'Time'] = row['Time'] + MAX_TIMESTAMP
    i_last = i

df.to_csv(outfile, index=False)


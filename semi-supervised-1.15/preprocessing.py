"""
Used to convert .csv into tfrecord format
"""
import pandas as pd
import numpy as np
import glob
import dask.dataframe as dd
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import argparse

attributes = ['Timestamp', 'canID', 'DLC', 
                           'Data0', 'Data1', 'Data2', 
                           'Data3', 'Data4', 'Data5', 
                           'Data6', 'Data7', 'Flag']

def fill_flag(sample):
    if not isinstance(sample['Flag'], str):
        col = 'Data' + str(sample['DLC'])
        sample['Flag'] = sample[col]
    return sample

def convert_canid_bits(cid):
    try:
        s = bin(int(str(cid), 16))[2:].zfill(29)
        bits = list(map(int, list(s)))
        return bits
    except:
        return None

def convert_dez(hex):
    try:
        if (str(hex) != str(np.NaN)):
            return(int(hex, 16))
        return(0) # replace NaN by 0
    except:
        return None

def fill_row_hcrl(sample):
    number = int(sample['DLC'])
    if number != 8: # fill up with np.NaN
        sample['Flag'] = sample['Data' + str(number)]
        sample['Data' + str(number)] = np.NaN

def payload_sum(sample):
    sum = 0
    for i in range(8):
        sum += sample['data'+i]
    sample['sum'] = sum

mad = lambda x: x.mad()
    
def preprocess(file_name, alt_features):
    df = dd.read_csv(file_name, header=None, names=attributes)
    print('Reading from {}: DONE'.format(file_name))
    print('Dask processing: -------------')
    df = df.apply(fill_flag, axis=1)
    pd_df = df.compute()
    pd_df = pd_df[['Timestamp', 'canID', 'Flag']].sort_values('Timestamp',  ascending=True)
    pd_df['canBits'] = pd_df.canID.apply(convert_canid_bits)
    pd_df['Flag'] = pd_df['Flag'].apply(lambda x: True if x == 'T' else False)
    print('Dask processing: DONE')
    print('Aggregate data -----------------')
    as_strided = np.lib.stride_tricks.as_strided  
    win = 29
    s = 29
    #Stride is counted by bytes
    feature = as_strided(pd_df.canBits, ((len(pd_df) - win) // s + 1, win), (8*s, 8)) 
    label = as_strided(pd_df.Flag, ((len(pd_df) - win) // s + 1, win), (1*s, 1))
    df = pd.DataFrame({
        'features': pd.Series(feature.tolist()),
        'label': pd.Series(label.tolist())
    }, index= range(len(feature)))

    df['label'] = df['label'].apply(lambda x: 1 if any(x) else 0)
    print('Preprocessing: DONE')
    print('#Normal: ', df[df['label'] == 0].shape[0])
    print('#Attack: ', df[df['label'] == 1].shape[0])
    return df[['features', 'label']].reset_index().drop(['index'], axis=1)

def preprocess_altformat(file_name, total_normal, total_attack, alt_features):
    df = dd.read_csv(file_name, dtype={
        'label': bool,
        'timestamp': float, 
        'id': str,
        'dlc': int,
        'data0': str,
        'data1': str,
        'data2': str,
        'data3': str,
        'data4': str,
        'data5': str,
        'data6': str,
        'data7': str})
    print('Reading from {}: DONE'.format(file_name))
    print('Dask processing: -------------')
    pd_df = df.compute()
    if (alt_features): # use alternative feature extraction
        print('Using alternative feature extraction------------------')
        pd_df = pd_df.sort_values('timestamp', ascending=True)
        pd_df['id'] = pd_df['id'].apply(convert_dez) # convert to decimal values
        for i in range (8):
            pd_df['data' + str(i)] = pd_df['data' + str(i)].apply(convert_dez)
        pd_df = pd_df.apply(payload_sum, axis=1)
        pd_df['time_d'] = pd_df['timestamp'].diff()
        #pd_df['time_mean'] = pd_df['timestamp'].rolling(16, center=True).mean()
        pd_df['time_var'] = pd_df['timestamp'].rolling(16, min_periods=1, center=True).var()
        pd_df['time_d_mean'] = pd_df['time_d'].rolling(16, min_periods=1, center=True).mean()
        pd_df['time_d_mad'] = pd_df['time_d'].rolling(16, min_periods=1, center=True).apply(mad)
        as_strided = np.lib.stride_tricks.as_strided  
        win = 16
        s = 16
        # TODO
        feature = as_strided(pd_df.id, ((len(pd_df)- win) // s + 1, win), (pd_df.id.itemsize*s, pd_df.id.itemsize))
        label = as_strided(pd_df.label, ((len(pd_df) - win) // s + 1, win), (1*s, 1)) 
    else:
        pd_df = pd_df[['label','timestamp','id']].sort_values('timestamp',  ascending=True)
        pd_df['id'] = pd_df.id.apply(convert_canid_bits)
        print('Dask processing: DONE')
        print('Aggregate data -----------------')
        as_strided = np.lib.stride_tricks.as_strided  
        win = 29
        s = 29
        #Stride is counted by bytes
        feature = as_strided(pd_df.id, ((len(pd_df) - win) // s + 1, win), (8*s, 8)) 
        label = as_strided(pd_df.label, ((len(pd_df) - win) // s + 1, win), (1*s, 1))
        df = pd.DataFrame({
            'features': pd.Series(feature.tolist()),
            'label': pd.Series(label.tolist())
        }, index= range(len(feature)))
        df['label'] = df['label'].apply(lambda x: 1 if any(x) else 0)
        print('Preprocessing: DONE')
        print('#Normal: ', df[df['label'] == 0].shape[0])
        total_normal = total_normal + df[df['label'] == 0].shape[0]
        print('#Attack: ', df[df['label'] == 1].shape[0])
        total_attack = total_attack + df[df['label'] == 1].shape[0]
    return df[['features', 'label']].reset_index().drop(['index'], axis=1), total_normal, total_attack

def serialize_example(x, y):
    """converts x, y to tf.train.Example and serialize"""
    #Need to pay attention to whether it needs to be converted to numpy() form
    input_features = tf.train.Int64List(value = np.array(x).flatten())
    label = tf.train.Int64List(value = np.array([y]))
    features = tf.train.Features(
        feature = {
            "input_features": tf.train.Feature(int64_list = input_features),
            "label" : tf.train.Feature(int64_list = label)
        }
    )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

def write_tfrecord(data, filename):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        tfrecord_writer.write(serialize_example(row['features'], row['label']))
    tfrecord_writer.close()    

def main(indir, outdir, attacks, alt_features):
    total_normal = 0
    total_attack = 0
    data_info = {}
    for attack in attacks:
        print('Attack: {} ==============='.format(attack))
        if (len(attacks) > 4):
            finput = '{}/{}.csv'.format(indir, attack)
            df, total_normal, total_attack = preprocess_altformat(finput, total_normal, total_attack, alt_features)
        else:
            finput = '{}/{}_dataset.csv'.format(indir, attack, alt_features)
            df = preprocess(finput)
        print("Writing...................")
        foutput_attack = '{}/{}'.format(outdir, attack)
        foutput_normal = '{}/Normal_{}'.format(outdir, attack)
        df_attack = df[df['label'] == 1]
        df_normal = df[df['label'] == 0]
        write_tfrecord(df_attack, foutput_attack)
        write_tfrecord(df_normal, foutput_normal)
        
        data_info[foutput_attack] = df_attack.shape[0]
        data_info[foutput_normal] = df_normal.shape[0]
        
    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./Data/Car-Hacking")
    parser.add_argument('--outdir', type=str, default="./Data/TFRecord/")
    parser.add_argument('--attack_type', type=str, default='hcrl')
    parser.add_argument('--alt_features', action='store_true')
    args = parser.parse_args()
    
    if args.attack_type == 'hcrl':
        attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM']
    elif (args.attack_type == 'tu'):
        attack_types = ['diagnostic', 'dosattack', 'fuzzing_canid', 'fuzzing_payload', 'replay']
    elif (args.attack_type == 'road_without_masquerade'):
        attack_types = ['ambient_dyno_drive_basic_long', # some ambient data to fill up the normal data gap
                        'correlated_signal_attack_1',
                        'correlated_signal_attack_2',
                        'correlated_signal_attack_3',
                        'fuzzing_attack_1',
                        'fuzzing_attack_2',
                        'fuzzing_attack_3',
                        'max_speedometer_attack_1',
                        'max_speedometer_attack_2',
                        'max_speedometer_attack_3',
                        'reverse_light_off_attack_1',
                        'reverse_light_off_attack_2',
                        'reverse_light_off_attack_3',
                        'reverse_light_on_attack_1',
                        'reverse_light_on_attack_2',
                        'reverse_light_on_attack_3'
                        ]
    elif (args.attack_type == 'road_with_masquerade'):
        attack_types = ['ambient_dyno_drive_basic_long', # some ambient data to fill up the normal data gap
                        'correlated_signal_attack_1_masquerade',
                        'correlated_signal_attack_1',
                        'correlated_signal_attack_2_masquerade',
                        'correlated_signal_attack_2',
                        'correlated_signal_attack_3_masquerade',
                        'correlated_signal_attack_3',
                        'fuzzing_attack_1',
                        'fuzzing_attack_2',
                        'fuzzing_attack_3',
                        'max_engine_coolant_temp_attack_masquerade',
                        'max_speedometer_attack_1_masquerade',
                        'max_speedometer_attack_1',
                        'max_speedometer_attack_2_masquerade',
                        'max_speedometer_attack_2',
                        'max_speedometer_attack_3_masquerade',
                        'max_speedometer_attack_3',
                        'reverse_light_off_attack_1_masquerade',
                        'reverse_light_off_attack_1',
                        'reverse_light_off_attack_2_masquerade',
                        'reverse_light_off_attack_2',
                        'reverse_light_off_attack_3_masquerade',
                        'reverse_light_off_attack_3',
                        'reverse_light_on_attack_1_masquerade',
                        'reverse_light_on_attack_1',
                        'reverse_light_on_attack_2_masquerade',
                        'reverse_light_on_attack_2',
                        'reverse_light_on_attack_3_masquerade',
                        'reverse_light_on_attack_3'
                        ]
    elif (args.attack_type == 'road_just_masquerade'):
        attack_types = ['ambient_dyno_drive_basic_long', # some ambient data to fill up the normal data gap
                        'correlated_signal_attack_1_masquerade',
                        'correlated_signal_attack_2_masquerade',
                        'correlated_signal_attack_3_masquerade',
                        'max_engine_coolant_temp_attack_masquerade',
                        'max_speedometer_attack_1_masquerade',
                        'max_speedometer_attack_2_masquerade',
                        'max_speedometer_attack_3_masquerade',
                        'reverse_light_off_attack_1_masquerade',
                        'reverse_light_off_attack_2_masquerade',
                        'reverse_light_off_attack_3_masquerade',
                        'reverse_light_on_attack_1_masquerade',
                        'reverse_light_on_attack_2_masquerade',
                        'reverse_light_on_attack_3_masquerade',
                        ]
    else:
        attack_types = [args.attack_type]

    main(args.indir, args.outdir, attack_types, args.alt_features)
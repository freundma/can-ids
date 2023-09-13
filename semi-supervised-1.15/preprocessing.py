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
from PIL import Image
import argparse

attributes = ['Timestamp', 'canID', 'DLC', 
                           'Data0', 'Data1', 'Data2', 
                           'Data3', 'Data4', 'Data5', 
                           'Data6', 'Data7', 'Flag']
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        if (df_norm[column].min() == df_norm[column].max()):
            df_norm[column] = 1.0
        else:
            df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())

    return df_norm
    
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
    
def convert_bytearray(hex):
    try:
        byte_list = []
        byte_pieces =  bytearray.fromhex(hex)
        for b in byte_pieces:
            byte_list.append(int(b))
        return byte_list
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
        sum += sample['data'+str(i)]
    sample['sum'] = sum
    return sample

mad = lambda x: x.mad()
    
def preprocess(file_name, print_attack_frame, attack):
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
    if (print_attack_frame):
        df_attack = df.loc[df['label'] == 1]
        df_normal = df.loc[df['label'] == 0]
        index_attack = df_attack['label'].idxmin() # pick one value
        index_normal = df_normal['label'].idxmin() # pick one value
        bits_attack = np.array(df_attack.iloc[index_attack,0]).flatten()
        bits_normal = np.array(df_normal.iloc[index_normal,0]).flatten()
        #df_bits = df_attack.loc[[index]]
        #df_bits = df_bits.drop(['label'],axis=1)
        #bits = df_bits.to_numpy().flatten()
        carr = np.array([(255,255,255), (0,0,0)], dtype='uint8')
        data_attack = carr[bits_attack].reshape(-1,29,3)
        data_normal = carr[bits_normal].reshape(-1,29,3)
        img_attack = Image.fromarray(data_attack, 'RGB')
        img_normal = Image.fromarray(data_normal, 'RGB')
        img_attack.save('Data/ID_image_' + attack + '.png','PNG')
        img_normal.save('Data/ID_image_' + attack + '_normal.png','PNG')
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
    if (alt_features == 'simple'): # use alternative simple feature extraction
        print('Using alternative simple feature extraction------------------')
        #pd_df = pd_df.sort_values('timestamp', ascending=True)
        pd_df = pd_df.drop('timestamp', axis=1)
        pd_df['id'] = pd_df['id'].apply(convert_dez) # convert to decimal values
        pd_df['sum'] = 0
        for i in range (8):
            pd_df['data' + str(i)] = pd_df['data' + str(i)].apply(convert_dez)
            pd_df['sum'] = pd_df['sum'] + pd_df['data' + str(i)]
        pd_df[['id','dlc','data0','data1','data2','data3','data4','data5','data6','data7','sum']] = min_max_scaling(pd_df[['id','dlc','data0','data1','data2','data3','data4','data5','data6','data7','sum']])
        pd_df['features'] = pd_df[['id','dlc','data0','data1','data2','data3','data4','data5','data6','data7','sum']].values.tolist()
        as_strided = np.lib.stride_tricks.as_strided
        win = 32 # window
        s = 32 # step
        feature = as_strided(pd_df.features, ((len(pd_df) - win) // s + 1, win), (8*s, 8)) # float is 8 bytes
        label = as_strided(pd_df.label, ((len(pd_df) - win) // s + 1, win), (1*s, 1)) # bool is 1 byte
        df = pd.DataFrame({
            'features': pd.Series(feature.tolist()),
            'label': pd.Series(label.tolist())
        }, index= range(len(feature)))
        df['label'] = df['label'].apply(lambda x: 1 if any(x) else 0)
        df_concat = df.loc[df['label'] == 1]
        for i in range (5):
            df = pd.concat([df, df_concat])
        print('Preprocessing: DONE')
        print('#Normal: ', df[df['label'] == 0].shape[0])
        total_normal = total_normal + df[df['label'] == 0].shape[0]
        print('#Attack: ', df[df['label'] == 1].shape[0])
        total_attack = total_attack + df[df['label'] == 1].shape[0]
        return df[['features', 'label']].reset_index().drop(['index'], axis=1), total_normal, total_attack
    elif (alt_features == 'complex'): # use alternative feature extraction
        print('Using alternative complex feature extraction------------------')
        pd_df = pd_df.sort_values('timestamp', ascending=True)
        pd_df['id'] = pd_df['id'].apply(convert_dez) # convert to decimal values
        pd_df['sum'] = 0
        pd_df['timestamp'] = pd_df['timestamp'] * 10000000
        for i in range (8):
            pd_df['data' + str(i)] = pd_df['data' + str(i)].apply(convert_dez)
            pd_df['sum'] = pd_df['sum'] + pd_df['data' + str(i)]
        pd_df.to_csv('complex_feature_extraction_sum.csv', index=False)
        pd_df['time_d'] = pd_df['timestamp'].diff()
        #pd_df['time_mean'] = pd_df['timestamp'].rolling(16, center=True).mean()
        pd_df['time_var'] = pd_df['timestamp'].rolling(16, min_periods=1, center=True).var()
        pd_df['time_d_mean'] = pd_df['time_d'].rolling(16, min_periods=1, center=True).mean()
        pd_df['time_d_mad'] = pd_df['time_d'].rolling(16, min_periods=1, center=True).apply(mad)
        pd_df.to_csv('complex_feature_extraction_sum.csv', index=False)
        exit(0)
        as_strided = np.lib.stride_tricks.as_strided  
        win = 16
        s = 16
        # TODO
        feature = as_strided(pd_df.id, ((len(pd_df)- win) // s + 1, win), (pd_df.id.itemsize*s, pd_df.id.itemsize))
        label = as_strided(pd_df.label, ((len(pd_df) - win) // s + 1, win), (1*s, 1))
        return None
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

def serialize_example(x, y, alt_features):
    """converts x, y to tf.train.Example and serialize"""
    #Need to pay attention to whether it needs to be converted to numpy() form
    if (alt_features == 'original'):
        input_features = tf.train.Int64List(value = np.array(x).flatten())
        label = tf.train.Int64List(value = np.array([y]))
        features = tf.train.Features(
            feature = {
                "input_features": tf.train.Feature(int64_list = input_features),
                "label" : tf.train.Feature(int64_list = label)
            }
        )
    else:
        input_features = tf.train.FloatList(value = np.array(x).flatten())
        label = tf.train.Int64List(value = np.array([y]))
        features = tf.train.Features(
            feature = {
                "input_features": tf.train.Feature(float_list = input_features),
                "label" : tf.train.Feature(int64_list = label)
            }
        )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

def write_tfrecord(data, filename, alt_features):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        tfrecord_writer.write(serialize_example(row['features'], row['label'], alt_features))
    tfrecord_writer.close()    

def main(indir, outdir, attacks, alt_features, dataset, print_attack_frame):
    total_normal = 0
    total_attack = 0
    data_info = {}
    for attack in attacks:
        print('Attack: {} ==============='.format(attack))
        if (dataset != 'hcrl' ):
            finput = '{}/{}.csv'.format(indir, attack)
            df, total_normal, total_attack = preprocess_altformat(finput, total_normal, total_attack, alt_features)
        else:
            finput = '{}/{}_dataset.csv'.format(indir, attack, alt_features)
            df = preprocess(finput, print_attack_frame, attack)
        print("Writing...................")
        foutput_attack = '{}/{}'.format(outdir, attack)
        foutput_normal = '{}/Normal_{}'.format(outdir, attack)
        df_attack = df[df['label'] == 1]
        df_normal = df[df['label'] == 0]
        write_tfrecord(df_attack, foutput_attack, alt_features)
        write_tfrecord(df_normal, foutput_normal, alt_features)
        
        data_info[foutput_attack] = df_attack.shape[0]
        data_info[foutput_normal] = df_normal.shape[0]
        
    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./Data/Car-Hacking")
    parser.add_argument('--outdir', type=str, default="./Data/TFRecord/")
    parser.add_argument('--attack_type', type=str, default='hcrl')
    parser.add_argument('--alt_features_simple', action='store_true')
    parser.add_argument('--alt_features_complex', action='store_true')
    parser.add_argument('--print_attack_frame', action='store_true')
    args = parser.parse_args()
    
    if args.attack_type == 'hcrl':
        attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM']
    elif (args.attack_type == 'tu'):
        attack_types = ['diagnostic', 'dosattack', 'fuzzing_canid', 'fuzzing_payload', 'replay']
    elif (args.attack_type == 'road_without_masquerade'):
        attack_types = ['road_without_masquerade_32',
                        #'ambient_street_driving_long', # some ambient data to fill up the normal data gap
                        #'correlated_signal_attack_1',
                        #'correlated_signal_attack_2',
                        #'correlated_signal_attack_3',
                        #'fuzzing_attack_1',
                        #'fuzzing_attack_2',
                        #'fuzzing_attack_3',
                        #'max_speedometer_attack_1',
                        #'max_speedometer_attack_2',
                        #'max_speedometer_attack_3',
                        #'reverse_light_off_attack_1',
                        #'reverse_light_off_attack_2',
                        #'reverse_light_off_attack_3',
                        #'reverse_light_on_attack_1',
                        #'reverse_light_on_attack_2',
                        #'reverse_light_on_attack_3'
                        ]
    elif (args.attack_type == 'road_with_masquerade'):
        attack_types = ['ambient_street_driving_long', # some ambient data to fill up the normal data gap
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
        attack_types = [#'ambient_dyno_drive_basic_long', # some ambient data to fill up the normal data gap
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

    if (args.alt_features_simple):
        main(args.indir, args.outdir, attack_types, 'simple', args.attack_type, args.print_attack_frame)
    elif (args.alt_features_complex):
        main(args.indir, args.outdir, attack_types, 'complex', args.attack_type, args.print_attack_frame)
    else:
       main(args.indir, args.outdir, attack_types, 'original', args.attack_type, args.print_attack_frame)
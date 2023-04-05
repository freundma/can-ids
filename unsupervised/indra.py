# Date: 03-29-2023
# Author: Mario Freund
# Purpose: Adjust INDRA IDS to normal python code and different feature extraction and working code
# Reference: https://github.com/EPIC-CSU/vehicle-cybersecurity/blob/main/INDRA-tensorflow-colab.ipynb

import sys
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import matplotlib
from pathlib import Path

def get_INDRA_model(num_features, subsequence_length, linear_out_dim, drp_out, context_dim):
    inputs = tf.keras.Input(shape=(subsequence_length, num_features)) # shape = (subsequence_length, data_dimension/num_features)
    # encoder
    x = tf.keras.layers.Dense(linear_out_dim, activation='tanh')(inputs)
    if drp_out: x = tf.keras.layers.Dropout(0.2)(x)
    enc_out, enc_hidden = tf.keras.layers.GRU(context_dim, return_sequences=True, return_state=True, activation="tanh")(x)
    if drp_out:
        enc_out = tf.keras.layers.Dropout(0.2)(enc_out)
        enc_hidden = tf.keras.layers.Dropout(0.2)(enc_hidden)
    # decoder
    dec_out , dec_hidden = tf.keras.layers.GRU(lin_out_dim, return_sequences=True, return_state=True, activation="tanh")(enc_out)
    if drp_out:
        dec_out = tf.keras.layers.Dropout(0.2)(dec_out)
        dec_hidden = tf.keras.layers.Dropout(0.2)(dec_hidden)
    outputs = tf.keras.layers.Dense(num_features, activation='tanh')(dec_out)
    model = tf.keras.Model(inputs, outputs)
    return model

def csv2df(dir_path):
    data_frames = []
    csv_path = dir_path + "/train_1.csv"
    df_temp = pd.read_csv(csv_path)
    data_frames.append(df_temp)
    for i in range(2, 5):
        csv_path = dir_path + "/train_" + str(i) + ".csv"
        df_temp = pd.read_csv(csv_path, header=None, names=["Label",  "Time", "ID", "Signal1",  "Signal2",  "Signal3",  "Signal4"])
        data_frames.append(df_temp)
    df = pd.concat(data_frames)
    return df

def prepare_dataset(dir_path):
    df = csv2df(dir_path)
    df = df.dropna(axis=1,how='all')
    #assert df.isnull().values.any() == False
    df = df.iloc[:,2:len(df.columns)] # put label and timestamp away
    df['ID'] = df['ID'].apply(lambda x: int(x[2])) # cast ID to numeric value
    return df

def min_max_scaling(df, column):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling on column
    df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())

    return df_norm

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# settings
settings = {
    "LinDec" : {
        "num_layers" : 1,
        "num_epochs" : 200,
        "batch_size" : 128,
        "sub_seq_len": 20,
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        "save_epochs": 3,
        "patience": 8,
        "learning_rate" : 0.0001,
        "context_dim" : 512,
        "L_func" : "MSE",
        "optimizer" : "ADAM",
        "max_plt_count" : 8,
        "IS_config" : "avg"

    },
    "ED_2L" : {
        "num_layers" : 1,
        "num_epochs" : 200,
        "batch_size" : 128,
        "sub_seq_len": 20,
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        "save_epochs": 3,
        "patience": 8,
        "learning_rate" : 0.0001,
        "lin_out_dim" : 128,
        "context_dim" : 64,
        "L_func" : "MSE",
        "optimizer" : "ADAM",
        "max_plt_count" : 8,
        "IS_config" : "avg"

    },
    "1L_ED" : {
        "num_layers" : 1,
        "num_epochs" : 200,
        "batch_size" : 128,
        "sub_seq_len": 20,
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        "save_epochs": 3,
        "patience": 8,
        "learning_rate" : 0.0001,
        "lin_out_dim" : 128,
        "context_dim" : 64,
        "L_func" : "MSE",
        "optimizer" : "ADAM",
        "max_plt_count" : 8,
        "IS_config" : "avg"

    },
    "attacks" : {
        "no_attack" : 0,
        "plateau" : 1,
        "continuous" : 2,
        "playback" : 3,
        "suppress" : 4,
        "flooding" : 5
    },
}

# arguments
args = {}
args["settings"] = settings
args["dataset"] = Path(Path.cwd(), 'SynCAN')
args["output_model"] = Path(Path.cwd(),'output_model')
args["plot_dir"] = Path(Path.cwd(),'plots')
args["stats_dir"] = Path(Path.cwd(),'stats')
args["msg_id"] = "2"
args["config"] = "DR_ED_2L"
args["resume"] = False

if args["config"] not in ['ED_2L' , 'ED_2L_AL', 'DR_ED_2L', 'DR_ED_2L_AL']:
    print("Invalid settings configuration")
    assert False
else:
    if "_AL" in args["config"]:
        lr_factor = 10
    else:
        lr_factor = 1

    settings_config = "ED_2L"

drp_out = 1 if "DR_" in args["config"] else 0

# hyper parameters
settings = args["settings"]
num_layers_ = settings[settings_config]['num_layers']
num_epochs = settings[settings_config]['num_epochs']
batch_size = settings[settings_config]['batch_size']
sub_seq_len = settings[settings_config]['sub_seq_len']
train_split = settings[settings_config]['train_split']
val_split = settings[settings_config]['val_split']
test_split = settings[settings_config]['test_split']
learning_rate = lr_factor * settings[settings_config]['learning_rate']
context_dim = settings[settings_config]['context_dim']
lin_out_dim = settings[settings_config]['lin_out_dim']
L_func = settings[settings_config]['L_func']
patience_ = settings[settings_config]['patience']
optimizer = settings[settings_config]['optimizer']
test_str = str(args["config"])

print('Settings:')
print('--------')
print('Msg ID: id{0}'.format(args["msg_id"]))
print('num_epochs {0}'.format(num_epochs))
print('batch_size {0}'.format(batch_size))
print('sub_seq_len {0}'.format(sub_seq_len))
print('learning_rate {0}'.format(learning_rate))
print('patience_ {0}'.format(patience_))
print('dropout {0}'.format(drp_out))
print('configuration {0}'.format(args["config"]))

# pre-processing dataset
directory = Path(args["dataset"])
data_set = prepare_dataset(str(directory))
data_set = data_set.fillna(0) # fill nan with 0, not the best as some ECUs really send signal 0
data_set = min_max_scaling(data_set, 'ID') # scale ID between 0 and 1
num_features = len(data_set.columns)
print("num feauters: " + str(num_features))

# reduce data set to a multiple of window size
num_samples_org = len(data_set.index)
num_ss = num_samples_org//sub_seq_len
data_set = data_set.iloc[:(num_ss*sub_seq_len)]

num_samples_adj = len(data_set.index)   # num adjusted samples
train_size = int(num_samples_adj*train_split)  # data samples used for training
val_size = int(num_samples_adj*val_split)  # % of data used for validation
test_size = int(num_samples_adj*test_split) # data samples used for test

# train data
train_data = data_set[:train_size]
truncate_index = (train_data.shape[0]//(sub_seq_len*num_features))
truncate_index *= (sub_seq_len*num_features)
train_data = train_data[:truncate_index]
train_data = train_data.to_numpy()
# sliding window (never happened on the original tensorflow jupyter notebook)
train_data = np.lib.stride_tricks.sliding_window_view(train_data, (sub_seq_len, num_features)).reshape(-1, sub_seq_len, num_features)

# validation data
val_data = data_set[train_size:train_size+val_size]
truncate_index = (val_data.shape[0]//(sub_seq_len*num_features))
truncate_index *= (sub_seq_len*num_features)
val_data = val_data[:truncate_index]
val_data = val_data.to_numpy()
# sliding window (never happened on the original tensorflow jupyter notebook)
val_data = np.lib.stride_tricks.sliding_window_view(val_data, (sub_seq_len, num_features)).reshape(-1, sub_seq_len, num_features)

# test data
test_data = data_set[train_size+val_size:]
truncate_index = (test_data.shape[0]//(sub_seq_len*num_features))
truncate_index *= (sub_seq_len*num_features)
test_data = test_data.to_numpy()
# sliding window (never happened on the original tensorflow jupyter notebook)
test_data = np.lib.stride_tricks.sliding_window_view(test_data, (sub_seq_len, num_features)).reshape(-1, sub_seq_len, num_features)

if L_func == "BCE":
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metrics = [tf.keras.metrics.BinaryCrossentropy()]
    monitor = "val_binary_crossentropy_error"
elif L_func == "MSE":
    loss_fn = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]
    monitor = "val_mean_squared_error"
elif L_func == "RMSE":
    loss_fn = tf.keras.losses.RootMeanSquaredError()
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
    monitor = "val_root_mean_squared_error"
else:
    print("Invalid loss function")
    assert False

if optimizer == "ADAM":
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
elif optimizer == "SGD":
    optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate)
elif optimizer == "RMSprop":
    optimiser = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, clipvalue=0.5)
else:
    print("Invalid optimiser")
    assert False

# Building and compiling the model
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=3,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="saved_model",
        monitor="val_loss",
        save_best_only=True,
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir="./tensorboard_logs",
    )
]
model = get_INDRA_model(num_features=num_features, subsequence_length=sub_seq_len,
                        linear_out_dim=lin_out_dim, drp_out=True, context_dim=context_dim)
model.compile(optimizer=optimizer, 
            loss=loss_fn,
            metrics=metrics)
#tf.keras.utils.plot_model(model, show_shapes=True)

history = model.fit(train_data,train_data,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(val_data, val_data),
    callbacks=callbacks_list,
    shuffle="batch")





#!/bin/sh
python preprocessing.py --indir ./Data/hcrl --outdir ./Data/TFRecord --attack_type hcrl
python train_test_split.py --indir ./Data/TFRecord --outdir ./Data --attack_type all --train_ratio 0.7 --train_label_ratio 0.1 --val_ratio 0.15 --test_ratio 0.15
python train.py --model CAAE --data_dir ./Data --batch_size 64 --epochs 100 --is_train


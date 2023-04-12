python preprocessing.py --indir Data/hcrl --outdir Data/hcrl/TFRecord --attack_type 'hcrl'
python train_test_split.py --indir Data/hcrl/TFRecord --outdir Data/hcrl --attack_type 'hcrl' --train_ratio 0.7 --train_label_ratio 0.1 --val_ratio 0.15 --test_ratio 0.15
python train.py --model CAAE --data_dir Data/hcrl/Train_0.7_Labeled_0.1 --batch_size 64 --epochs 100 --is_train --res_path Data/hcrl/Results


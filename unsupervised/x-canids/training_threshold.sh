python train.py --inpath Data/road/datasplit/complete/ --outpath Data/road/results/model-09-28/ --batch_size 1024 --latent_space_size 177 --checkpoint_path Data/road/results/checkpoints/09-28-2023/ --tensorboard_path Data/road/results/tensorboards/
python threshold.py --model_path Data/road/results/model-09-28/ --data_path Data/road/datasplit/complete/ --outpath Data/road/thresholds/model-09-28-max_rs/
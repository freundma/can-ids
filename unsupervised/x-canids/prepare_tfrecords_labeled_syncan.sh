python preprocessing_labeled.py --infile Data/syncan/csv_attack/test_continuous.csv --outfile Data/syncan/TFRecords_attack/TFRecords_test_continuous --timesteps 15 --windowsize 3 --min_max_file Data/syncan/ranges/min_max_merge.json --syncan
python preprocessing_labeled.py --infile Data/syncan/csv_attack/test_flooding.csv --outfile Data/syncan/TFRecords_attack/TFRecords_test_flooding --timesteps 15 --windowsize 3 --min_max_file Data/syncan/ranges/min_max_merge.json --syncan
python preprocessing_labeled.py --infile Data/syncan/csv_attack/test_plateau.csv --outfile Data/syncan/TFRecords_attack/TFRecords_test_plateau --timesteps 15 --windowsize 3 --min_max_file Data/syncan/ranges/min_max_merge.json --syncan
python preprocessing_labeled.py --infile Data/syncan/csv_attack/test_playback.csv --outfile Data/syncan/TFRecords_attack/TFRecords_test_playback --timesteps 15 --windowsize 3 --min_max_file Data/syncan/ranges/min_max_merge.json --syncan
python preprocessing_labeled.py --infile Data/syncan/csv_attack/test_suppress.csv --outfile Data/syncan/TFRecords_attack/TFRecords_test_suppress --timesteps 15 --windowsize 3 --min_max_file Data/syncan/ranges/min_max_merge.json --syncan

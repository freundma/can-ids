# Prepare TFRecords
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_dyno_drive_winter.csv --outfile Data/TFRecords/TFRecord_dyno_drive_winter_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_dyno_drive_basic_short.csv --outfile Data/TFRecords/TFRecord_dyno_drive_basic_short_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_dyno_drive_basic_long.csv --outfile Data/TFRecords/TFRecord_dyno_drive_basic_long_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_dyno_drive_benign_anomaly.csv --outfile Data/TFRecords/TFRecord_dyno_drive_benign_anomaly_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_dyno_drive_extended_long.csv --outfile Data/TFRecords/TFRecord_dyno_drive_extended_long_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_dyno_drive_extended_short.csv --outfile Data/TFRecords/TFRecord_dyno_drive_extended_short_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_dyno_drive_radio_infotainment.csv --outfile Data/TFRecords/TFRecord_dyno_drive_radio_infotainment_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_dyno_exercise_all_bits.csv --outfile Data/TFRecords/TFRecord_dyno_exercise_all_bits_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_dyno_idle_radio_infotainment.csv --outfile Data/TFRecords/TFRecord_dyno_idle_radio_infotainment_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_dyno_reverse.csv --outfile Data/TFRecords/TFRecord_dyno_reverse_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_highway_street_driving_diagnostics.csv --outfile Data/TFRecords/TFRecord_highway_street_driving_diagnostics_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json
python preprocessing_unlabeled.py --infile Data/csv_with_street/ambient_highway_street_driving_long.csv --outfile Data/TFRecords/TFRecord_highway_street_driving_long_signals202 --exclude_constant_signals --constant_signal_file Data/constant_signals.json

# Apply Datasplit
python train_val_test_split.py

# can-ids

This repository contains implementations of X-CANIDS by Jeong et al.[3] It is recommended to read the paper before using the repository. It is also recommended to read my [master's thesis](https://osm.hpi.de/theses/master) in the context of which this code was written.

## Structure
It is split into three sub parts: x-canids, x-canids-bytes, and x-mvbids.

The first one contains the implementation of X-CANIDS for signal-translated CAN datasets. It is specialized to the datasets [SynCAN](https://github.com/etas/SynCAN) by Hanselmann et al. [2] and [ROAD](https://0xsam.com/road/) by Bridges et al. [1]. However, it can also be adapted easily to other datasets. The second one contains an adaption to the byte-based datasets of [ROAD](https://0xsam.com/road/). The third one contains an adaption of the method to MVB data. The used MVB data is proprietary and cannot be provided.

All three types of IDS follow the same general pattern of execution:
1. extract_constant_signals.py -> indentify constant signals
2. min-max.py -> determine ranges of each monitored signal
3. merge-min-max.py -> generate common ranges for multiple files by merging the min-max-files
4. preprocessing_labeled.py -> applies the feature extraction of X-CANIDS and includes labels for testing
5. preprocessing_unlabeled.py -> applies the feature extraction of X-CANIDS and does not include labels
6. train_val_test_split.py -> applies a train val test split on the extracted features
7. train.py -> trains the model
8. threshold.py -> determines the thresholds for intrusion detection based on validation and training data
9. evaluate.py -> uses the model and thresholds to run evaluation on the attack and bening test data

Extra files:
* apply_windowing.py -> can be used to apply windowing on a standalone tfrecord file with s-vectors
* explanation.py -> can be used to make the loss of the first positive classified attack sample visible and can be adjusted for further use

## Setup
Each of the subfolders contains a Dockerfile that can be used for building a docker image with tensorflow and the other according requirements. It is recommended to run a container based on this image and mount a folder for external provision of datasets and to export the training results. A port mapping can be used to publish tensorboards. [NVIDIA container](https://developer.nvidia.com/nvidia-container-runtime) runtime is necessary:

```bash
cd can-ids/unsupervised/x-canids
docker build . -t can-ids-unsupervised
docker run -it --publish 6006:6006 --name can-ids-unsupervised-training --gpus all --mount type=bind,src="$(pwd)"/datasets,dst=/ids/Data can-ids-unsupervised bash
```
## Usage
The usage of each script can be derived directly in the python script.

## Additional notes
A main difference between the literature evaluation and my evaluation is that the [ROAD](https://0xsam.com/road/) dataset does not provide the ranges of the signals in the dbc-file. Thus, the ranges need to be determined before on the training datasets.

## References

[1]: Verma, Miki E., et al. "Addressing the lack of comparability & testing in CAN intrusion detection research: A comprehensive guide to CAN IDS data & introduction of the ROAD dataset." arXiv preprint arXiv:2012.14600 (2020).

[2]: Hanselmann, Markus, et al. "CANet: An unsupervised intrusion detection system for high dimensional CAN bus data." Ieee Access 8 (2020): 58194-58205.

[3]: Jeong, Seonghoon, et al. "X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network." arXiv preprint arXiv:2303.12278 (2023).
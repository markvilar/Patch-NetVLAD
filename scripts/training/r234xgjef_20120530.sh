#!/usr/bin/bash

python train_benthic.py \
    --data_path "/home/martin/data/configs/training/r234xgjef_20120530.ini" \
    --resume_path "/home/martin/data/models/benthic_encoder.pth.tar" \
    --save_path "/home/martin/pCloudDrive/data/selected/sessions" \
    --config_path "/home/martin/dev/patch-netvlad/configs/train_benthic.ini" \
    --identifier "r234xgjef_20120530"

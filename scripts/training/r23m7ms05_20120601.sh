#!/usr/bin/bash

python train_benthic.py \
    --data_path "/home/martin/data/configs/training/r23m7ms05_20120601.ini" \
    --resume_path "/home/martin/data/models/benthic_encoder.pth.tar" \
    --save_path "/home/martin/pCloudDrive/data/selected/sessions" \
    --config_path "/home/martin/dev/patch-netvlad/configs/train_benthic.ini" \
    --identifier "r23m7ms05_20120601"

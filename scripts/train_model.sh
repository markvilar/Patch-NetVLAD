#!/usr/bin/bash

SAVE_PATH=$1
MODEL_PATH=$2
CONFIG_PATH=$3
DATA_PATH=$4
IDENTIFIER=$5

echo $SAVE_PATH
echo $MODEL_PATH
echo $CONFIG_PATH
echo $DATA_PATH
echo $IDENTIFIER
echo $EPOCH

python train_benthic.py \
    --data_path "$DATA_PATH" \
    --resume_path "$MODEL_PATH" \
    --save_path "$SAVE_PATH" \
    --config_path "$CONFIG_PATH" \
    --identifier "$IDENTIFIER" \
    --epochs 30

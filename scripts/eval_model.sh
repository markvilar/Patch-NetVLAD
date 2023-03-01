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

python eval_benthic.py \
    --save_path "$SAVE_PATH" \
    --resume_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --data_path "$DATA_PATH" \
    --identifier "$IDENTIFIER"

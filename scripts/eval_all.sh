# !/usr/bin/bash
SAVE_PATH="/home/martin/pCloudDrive/data/selected/eval_isolated"
MODEL_DIR="/home/martin/pCloudDrive/data/selected/training_isolated"
CONFIG_PATH="/home/martin/dev/patch-netvlad/configs/perf_benthic.ini"
DATA_DIR="/home/martin/data/configs"

echo "\n---------------------------------"
echo "---------- Henderson 1 ----------"
echo "---------------------------------\n"

bash ./scripts/eval_model.sh \
    "$SAVE_PATH" \
    "$MODEL_DIR/henderson_01/checkpoints/model_best.pth.tar" \
    "$CONFIG_PATH" \
    "$DATA_DIR/evaluation/r7jjss8n6_20131022.ini" \
    "henderson_01_eval"

echo "\n---------------------------------"
echo "---------- Henderson 2 ----------"
echo "---------------------------------\n"

bash ./scripts/eval_model.sh \
    "$SAVE_PATH" \
    "$MODEL_DIR/henderson_02/checkpoints/model_best.pth.tar" \
    "$CONFIG_PATH" \
    "$DATA_DIR/evaluation/r7jjskxq6_20131022.ini" \
    "henderson_02_eval"

echo "\n---------------------------------"
echo "---------- Henderson 3 ----------"
echo "---------------------------------\n"

bash ./scripts/eval_model.sh \
    "$SAVE_PATH" \
    "$MODEL_DIR/henderson_03/checkpoints/model_best.pth.tar" \
    "$CONFIG_PATH" \
    "$DATA_DIR/evaluation/r7jjssbhh_20131022.ini" \
    "henderson_03_eval"

echo "\n--------------------------------"
echo "---------- Scott Reef ----------"
echo "--------------------------------\n"

bash ./scripts/eval_model.sh \
    "$SAVE_PATH" \
    "$MODEL_DIR/scott_reef/checkpoints/model_best.pth.tar" \
    "$CONFIG_PATH" \
    "$DATA_DIR/evaluation/qtqxshxst_20150328_02.ini" \
    "scott_reef_eval"

echo "\n------------------------------"
echo "---------- Lanterns ----------"
echo "------------------------------\n"

bash ./scripts/eval_model.sh \
    "$SAVE_PATH" \
    "$MODEL_DIR/lanterns/checkpoints/model_best.pth.tar" \
    "$CONFIG_PATH" \
    "$DATA_DIR/evaluation/r234xgjef_20140616.ini" \
    "lanterns_eval"

echo "\n----------------------------------"
echo "---------- Mistakencape ----------"
echo "----------------------------------\n"

bash ./scripts/eval_model.sh \
    "$SAVE_PATH" \
    "$MODEL_DIR/mistakencape/checkpoints/model_best.pth.tar" \
    "$CONFIG_PATH" \
    "$DATA_DIR/evaluation/r23m7ms05_20140616.ini" \
    "mistakencape_eval"

echo "\n----------------------------------"
echo "---------- St. Helens 1 ----------"
echo "----------------------------------\n"

bash ./scripts/eval_model.sh \
    "$SAVE_PATH" \
    "$MODEL_DIR/st_helens_01/checkpoints/model_best.pth.tar" \
    "$CONFIG_PATH" \
    "$DATA_DIR/evaluation/r29kz9ff0_20130611.ini" \
    "st_helens_01_eval"

echo "\n----------------------------------"
echo "---------- St. Helens 2 ----------"
echo "----------------------------------\n"

bash ./scripts/eval_model.sh \
    "$SAVE_PATH" \
    "$MODEL_DIR/st_helens_02/checkpoints/model_best.pth.tar" \
    "$CONFIG_PATH" \
    "$DATA_DIR/evaluation/r29kz9dg9_20130611.ini" \
    "st_helens_02_eval"

echo "\n--------------------------------------"
echo "---------- Elephant Rocks 1 ----------"
echo "--------------------------------------\n"

bash ./scripts/eval_model.sh \
    "$SAVE_PATH" \
    "$MODEL_DIR/elephant_rocks_01/checkpoints/model_best.pth.tar" \
    "$CONFIG_PATH" \
    "$DATA_DIR/evaluation/r29mrd124_20110612.ini" \
    "elephant_rocks_01_eval"

echo "\n--------------------------------------"
echo "---------- Elephant Rocks 2 ----------"
echo "--------------------------------------\n"

bash ./scripts/eval_model.sh \
    "$SAVE_PATH" \
    "$MODEL_DIR/elephant_rocks_02/checkpoints/model_best.pth.tar" \
    "$CONFIG_PATH" \
    "$DATA_DIR/evaluation/r29mrd5h4_20110612.ini" \
    "elephant_rocks_02_eval"

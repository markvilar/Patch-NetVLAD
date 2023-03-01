# !/usr/bin/bash
SAVE_PATH="/home/martin/pCloudDrive/data/selected/training_new"
MODEL_PATH="/home/martin/data/models/mapillary_WPCA4096.pth.tar"
CONFIG_PATH="/home/martin/dev/patch-netvlad/configs/train_benthic.ini"
DATA_PATH="/home/martin/data/configs/training"

echo "\n------------------------------------------"
echo "---------- Starting Henderson 1 ----------"
echo "------------------------------------------\n"

bash ./scripts/train_model.sh \
    "$SAVE_PATH" \
    "$MODEL_PATH" \
    "$CONFIG_PATH" \
    "$DATA_PATH/r7jjss8n6_20121013.ini" \
    "r7jjss8n6_20121013"

echo "\n------------------------------------------"
echo "---------- Starting Henderson 2 ----------"
echo "------------------------------------------\n"

bash ./scripts/train_model.sh \
    "$SAVE_PATH" \
    "$MODEL_PATH" \
    "$CONFIG_PATH" \
    "$DATA_PATH/r7jjskxq6_20121013.ini" \
    "r7jjskxq6_20121013"

echo "\n------------------------------------------"
echo "---------- Starting Henderson 3 ----------"
echo "------------------------------------------\n"

bash ./scripts/train_model.sh \
    "$SAVE_PATH" \
    "$MODEL_PATH" \
    "$CONFIG_PATH" \
    "$DATA_PATH/r7jjssbhh_20121013.ini" \
    "r7jjssbhh_20121013"

echo "\n-----------------------------------------"
echo "---------- Starting Scott Reef ----------"
echo "-----------------------------------------\n"

bash ./scripts/train_model.sh \
    "$SAVE_PATH" \
    "$MODEL_PATH" \
    "$CONFIG_PATH" \
    "$DATA_PATH/qtqxshxst_20150328_01.ini" \
    "qtqxshxst_20150328_01"

echo "\n---------------------------------------"
echo "---------- Starting Lanterns ----------"
echo "---------------------------------------\n"

bash ./scripts/train_model.sh \
    "$SAVE_PATH" \
    "$MODEL_PATH" \
    "$CONFIG_PATH" \
    "$DATA_PATH/r234xgjef_20120530.ini" \
    "r234xgjef_20120530"

echo "\n-------------------------------------------"
echo "---------- Starting Mistakencape ----------"
echo "-------------------------------------------\n"

bash ./scripts/train_model.sh \
    "$SAVE_PATH" \
    "$MODEL_PATH" \
    "$CONFIG_PATH" \
    "$DATA_PATH/r23m7ms05_20120601.ini" \
    "r23m7ms05_20120601"

echo "\n-------------------------------------------"
echo "---------- Starting St. Helens 1 ----------"
echo "-------------------------------------------\n"

bash ./scripts/train_model.sh \
    "$SAVE_PATH" \
    "$MODEL_PATH" \
    "$CONFIG_PATH" \
    "$DATA_PATH/r29kz9ff0_20090613_02.ini" \
    "r29kz9ff0_20090613_02"

echo "\n-------------------------------------------"
echo "---------- Starting St. Helens 2 ----------"
echo "-------------------------------------------\n"

bash ./scripts/train_model.sh \
    "$SAVE_PATH" \
    "$MODEL_PATH" \
    "$CONFIG_PATH" \
    "$DATA_PATH/r29kz9dg9_20090613_02.ini" \
    "r29kz9dg9_20090613_02"

echo "\n-----------------------------------------------"
echo "---------- Starting Elephant Rocks 1 ----------"
echo "-----------------------------------------------\n"

bash ./scripts/train_model.sh \
    "$SAVE_PATH" \
    "$MODEL_PATH" \
    "$CONFIG_PATH" \
    "$DATA_PATH/r29mrd124_20090613_02.ini" \
    "r29mrd124_20090613_02"

echo "\n-----------------------------------------------"
echo "---------- Starting Elephant Rocks 2 ----------"
echo "-----------------------------------------------\n"

bash ./scripts/train_model.sh \
    "$SAVE_PATH" \
    "$MODEL_PATH" \
    "$CONFIG_PATH" \
    "$DATA_PATH/r29mrd5h4_20090613.ini" \
    "r29mrd5h4_20090613"

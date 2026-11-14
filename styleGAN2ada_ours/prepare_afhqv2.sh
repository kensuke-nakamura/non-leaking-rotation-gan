#!/bin/bash


dir_input=/nas/dataset/afhqv2
dir_output=/nas/dataset/zipped

python3 dataset_tool.py --source=$dir_input             --dest=$dir_output/afhqv2-512x512.zip
wait
python3 dataset_tool.py --source=$dir_input/train/cat  --dest=$dir_output/afhqv2cat-512x512.zip
wait
python3 dataset_tool.py --source=$dir_input/train/dog --dest=$dir_output/afhqv2dog-512x512.zip
wait
python3 dataset_tool.py --source=$dir_input/train/wild --dest=$dir_output/afhqv2wild-512x512.zip
wait

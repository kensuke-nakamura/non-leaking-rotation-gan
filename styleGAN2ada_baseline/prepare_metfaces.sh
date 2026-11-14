

dir_input=/nas/dataset
dir_output=/nas/dataset/zipped

python3 dataset_tool.py --source=$dir_input/metfaces/images --dest=$dir_output/metfaces-1024x1024.zip
wait

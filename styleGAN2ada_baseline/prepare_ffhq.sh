

dir_input=/nas/dataset/ffhq
dir_output=/nas/dataset/zipped

# Original 1024x1024 resolution.
python3 dataset_tool.py --source=$dir_input/images1024x1024 --dest=$dir_output/ffhq-1024x1024.zip

# Scaled down 256x256 resolution.
python3 dataset_tool.py --source=$dir_input/images1024x1024 --dest=$dir_putput/ffhq-256x256.zip --resolution=256x256

wait

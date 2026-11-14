
#
# StyleGAN2-ada + ours
#

# The official configuration:
# https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md

# directory of data
path='/hdd1/dataset/stylegan3'

# Styelgan2-Ada 
for dataset in ffhq-512x512.zip afhqv2-512x512.zip  metfaces-512x512.zip
do
for trial in 1 2 3 4 5
do
python3 train.py --data=$path/$dataset --outdir=../training-runs --cfg=stylegan2 --gpus=4 --batch=32 --gamma=1.6384 --map-depth=2 --glr=0.0025 --dlr=0.0025 --kimg 10000 --trial $trial  
wait
done
done




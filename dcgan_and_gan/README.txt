
# install anaconda environment

conda deactivate
conda env remove -n dcgan
bash requirements.sh
conda activate dcgan
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.device_count())"
pip install prdc
pip install lmdb

# run demo

script_demo.sh

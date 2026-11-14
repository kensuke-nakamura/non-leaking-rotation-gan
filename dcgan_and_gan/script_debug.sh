



inifile='./option/celeba_dcgan2_gr.ini'


# rotation and hue change with p=1 (demonstration of leaking)
cuda=0; ini=`python3 edit_ini_file.py $inifile --Transform__probability 0.80 --Transform__option 'rot-a4,hue-a4' `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
wait


python3 evaluate_saved_generator2.py
wait

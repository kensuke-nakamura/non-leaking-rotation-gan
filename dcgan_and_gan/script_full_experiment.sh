# Full experiment script for DCGAN-gr.
#
# Note: FID values shown during training are temporary. Final evaluation metrics
# are computed by evaluate_saved_generator2.py and saved in ./result/evaluation/.


for inifile in './option/celeba_dcgan2_gr.ini' './option/oxflower_dcgan2_gr.ini' './option/food-101_dcgan2_gr.ini' './option/stanford-cars_dcgan2_gr.ini' './option/lsun-church_dcgan2_gr.ini' './option/lsun-conference_dcgan2_gr.ini'
do
	for i in 0 1 2 3 4
	do
		# baseline
		cuda=0; ini=`python3 edit_ini_file.py $inifile                                                                 --Optimize__trial_id_number $((4*i+1)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=1; ini=`python3 edit_ini_file.py $inifile                                                                 --Optimize__trial_id_number $((4*i+2)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=2; ini=`python3 edit_ini_file.py $inifile                                                                 --Optimize__trial_id_number $((4*i+3)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=3; ini=`python3 edit_ini_file.py $inifile                                                                 --Optimize__trial_id_number $((4*i+4)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		wait
		# nvidia-ada (rot4 and hue4 with Ada-Heuristic)
		cuda=0; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'Ada'           --Optimize__trial_id_number $((4*i+1)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=1; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'Ada'           --Optimize__trial_id_number $((4*i+2)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=2; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'Ada'           --Optimize__trial_id_number $((4*i+3)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=3; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'Ada'           --Optimize__trial_id_number $((4*i+4)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		wait
		# ours 
		cuda=0; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'rot-a3,hue-a3' --Optimize__trial_id_number $((4*i+1)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=1; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'rot-a3,hue-a3' --Optimize__trial_id_number $((4*i+2)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=2; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'rot-a3,hue-a3' --Optimize__trial_id_number $((4*i+3)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=3; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'rot-a3,hue-a3' --Optimize__trial_id_number $((4*i+4)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		wait
		# our rot only# Full experiment script for DCGAN-gr.
# FID values shown during training are temporary. Final evaluation metrics
# are computed by evaluate_saved_generator2.py and saved in ./result/evaluation/.
		cuda=0; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'rot-a3'        --Optimize__trial_id_number $((4*i+1)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=1; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'rot-a3'        --Optimize__trial_id_number $((4*i+2)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=2; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'rot-a3'        --Optimize__trial_id_number $((4*i+3)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=3; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'rot-a3'        --Optimize__trial_id_number $((4*i+4)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		wait
		# our hue only
		cuda=0; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'hue-a3'        --Optimize__trial_id_number $((4*i+1)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=1; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'hue-a3'        --Optimize__trial_id_number $((4*i+2)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=2; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'hue-a3'        --Optimize__trial_id_number $((4*i+3)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		cuda=3; ini=`python3 edit_ini_file.py $inifile  --Transform__probability 1 --Transform__option 'hue-a3'        --Optimize__trial_id_number $((4*i+4)) `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
		wait
	done

wait
done

python3 evaluate_saved_generator2.py
wait

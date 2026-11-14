
#
# baesline w/o transformation
#
CUDA_VISIBLE_DEVICES=0 python3 main.py --n_element 2   --P 0.00             --label 'p=0, n=2' &
CUDA_VISIBLE_DEVICES=1 python3 main.py --n_element 3   --P 0.00             --label 'p=0, n=3' &
CUDA_VISIBLE_DEVICES=2 python3 main.py --n_element 4   --P 0.00             --label 'p=0, n=4' &
CUDA_VISIBLE_DEVICES=3 python3 main.py --n_element 5   --P 0.00             --label 'p=0, n=5' &
wait

#
# {n}-cyclic data,   P is termined by {n} so as to simulate 'p'=1/3, 2/3, 3/3
#
n=2
CUDA_VISIBLE_DEVICES=0 python3 main.py --n_element $n  --P 0.16666666666667 --label 'p=1/3, n=2' &
CUDA_VISIBLE_DEVICES=1 python3 main.py --n_element $n  --P 0.33333333333333 --label 'p=2/3, n=2' & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --n_element $n  --P 0.50             --label 'p=1, n=2'   & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --n_element $n  --P 1                --label '(ours) P=1, n=2'   &
wait
n=3
CUDA_VISIBLE_DEVICES=0 python3 main.py --n_element $n  --P 0.22222222222222 --label 'p=1/3, n=3' &
CUDA_VISIBLE_DEVICES=1 python3 main.py --n_element $n  --P 0.44444444444444 --label 'p=2/3, n=3' & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --n_element $n  --P 0.66666666666666 --label 'p=1, n=3'   & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --n_element $n  --P 1                --label '(ours) P=1, n=3'   &
wait
n=4
CUDA_VISIBLE_DEVICES=0 python3 main.py --n_element $n  --P 0.25             --label 'p=1/3, n=4' &
CUDA_VISIBLE_DEVICES=1 python3 main.py --n_element $n  --P 0.50             --label 'p=2/3, n=4' & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --n_element $n  --P 0.75             --label 'p=1, n=4'   & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --n_element $n  --P 1                --label '(ours) P=1, n=4'   &
wait
n=5
CUDA_VISIBLE_DEVICES=0 python3 main.py --n_element $n  --P 0.26666666666667 --label 'p=1/3, n=5' &
CUDA_VISIBLE_DEVICES=1 python3 main.py --n_element $n  --P 0.53333333333333 --label 'p=2/3, n=5' & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --n_element $n  --P 0.8              --label 'p=1, n=5'   & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --n_element $n  --P 1                --label '(ours) P=1, n=5'   &
wait

wait 
sh concatenate_images.sh
wait





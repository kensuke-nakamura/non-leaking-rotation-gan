



# 20 trials
for i in 0 1 2 3 4
do
	# Expected training time:
	#  celeba          4h
	#  food-101        6h
	#  lsun-church     2h
	#  oxflower        3h
	#  stanford-cars   3.5h
	#  lsun-conference 2h
	for dataset in 'celeba' 'food-101' 'lsun-church' 'oxflower' 'stanford-cars' 'lsun-conference'
	do
		if     [ "$dataset" = "celeba" ];          then  epoch=50
		elif   [ "$dataset" = "food-101" ];        then  epoch=100
		elif   [ "$dataset" = "lsun-church" ];     then  epoch=100
		elif   [ "$dataset" = "lsun-conference" ]; then  epoch=30
		elif   [ "$dataset" = "mnist" ];           then  epoch=100
		elif   [ "$dataset" = "oxflower" ];        then  epoch=500
		elif   [ "$dataset" = "stanford-cars" ];   then  epoch=300
		else epoch=10
		fi

		# DRAGAN
		# baseline, Ada, and ours
		CUDA_VISIBLE_DEVICES=0 python3 training_dragan.py             --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_dragan.py             --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_dragan.py             --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_dragan.py             --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		CUDA_VISIBLE_DEVICES=0 python3 training_dragan.py   --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_dragan.py   --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_dragan.py   --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_dragan.py   --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		CUDA_VISIBLE_DEVICES=0 python3 training_dragan.py   --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_dragan.py   --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_dragan.py   --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_dragan.py   --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		# WGAN-GP
		CUDA_VISIBLE_DEVICES=0 python3 training_wgan_gp.py            --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_wgan_gp.py            --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_wgan_gp.py            --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_wgan_gp.py            --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		CUDA_VISIBLE_DEVICES=0 python3 training_wgan_gp.py  --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_wgan_gp.py  --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_wgan_gp.py  --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_wgan_gp.py  --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		CUDA_VISIBLE_DEVICES=0 python3 training_wgan_gp.py  --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_wgan_gp.py  --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_wgan_gp.py  --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_wgan_gp.py  --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		# WGAN-DIV
		CUDA_VISIBLE_DEVICES=0 python3 training_wgan_div.py           --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_wgan_div.py           --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_wgan_div.py           --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_wgan_div.py           --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		CUDA_VISIBLE_DEVICES=0 python3 training_wgan_div.py --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_wgan_div.py --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_wgan_div.py --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_wgan_div.py --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		CUDA_VISIBLE_DEVICES=0 python3 training_wgan_div.py --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_wgan_div.py --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_wgan_div.py --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_wgan_div.py --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		# DCGAN
		CUDA_VISIBLE_DEVICES=0 python3 training_dcgan.py              --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_dcgan.py              --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_dcgan.py              --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_dcgan.py              --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		CUDA_VISIBLE_DEVICES=0 python3 training_dcgan.py    --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_dcgan.py    --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_dcgan.py    --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_dcgan.py    --ada     --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		CUDA_VISIBLE_DEVICES=0 python3 training_dcgan.py    --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+1)) &
		CUDA_VISIBLE_DEVICES=1 python3 training_dcgan.py    --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+2)) &
		CUDA_VISIBLE_DEVICES=2 python3 training_dcgan.py    --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+3)) &
		CUDA_VISIBLE_DEVICES=3 python3 training_dcgan.py    --nonleak --dataset $dataset --epoch $epoch  --trial $((4*i+4)) &
		wait
		#done
	done
done

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

inifile='./option/celeba_dcgan2_gr.ini'
probabilities=(0.00 0.50 0.80 1.00)
mkdir -p ./result/log
python3 -c 'import torch, torchvision; from torchvision.ops import nms; assert torch.cuda.device_count() >= 4, "Four CUDA GPUs are required"'

pids=()

for gpu in "${!probabilities[@]}"; do
    (
        probability="${probabilities[$gpu]}"

        for trial in {1..10}; do
            run_name="fixedp_p${probability}_trial${trial}"

            ini=$(python3 edit_ini_file.py "$inifile" \
                --Transform__probability "$probability" \
                --Transform__option 'rot-a4,hue-a4' \
                --Optimize__trial_id_number "$trial" \
                --new_option_name "${run_name}.ini")

            echo "Starting ${run_name} on GPU ${gpu}"
            CUDA_VISIBLE_DEVICES="$gpu" python3 -u main.py "$ini" \
                2>&1 | tee "./result/log/${run_name}.stdout.log"
        done
    ) &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if (( status != 0 )); then
    exit "$status"
fi

# Evaluate FID and IS for the best checkpoint of each completed trial.
CUDA_VISIBLE_DEVICES=0 python3 evaluate_saved_generator2.py --pattern 'fixedp_p*' --checkpoint best --overwrite \
    2>&1 | tee './result/log/fixedp_evaluation.stdout.log'

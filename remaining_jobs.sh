#!/bin/bash
cd BASES

# Define the screens, corresponding GPU devices, and victim values
declare -a screens=("a" "b" "c")
declare -a gpus=("0" "0" "1")
declare -a victims=("Salman2020Do_R18" "Standard_R50" "Standard_R50")
declare -a seeds=("1" "1" "10")

# Iterate over screens, GPUs, and victim values
for i in "${!screens[@]}"; do
    screen_name="${screens[$i]}"
    gpu_device="${gpus[$i]}"
    victim_value="${victims[$i]}"
    seed_value="${seeds[$i]}"

    echo "Attaching to screen $screen_name, setting CUDA_VISIBLE_DEVICES=$gpu_device, using victim=$victim_value, and running command."

    screen -dmS "$screen_name" bash -c "
        export CUDA_VISIBLE_DEVICES=$gpu_device;
        python query_w_bb.py --victim $victim_value --start_epoch 0 --eps 4 --n_wb 10  --helpers_path /home/mdjilani/robustblack/utils_robustblack --exp_root /mnt/data/data/mdjilani/bases_exp --adv_root /mnt/data/data/mdjilani/bases_adv --iterw 20 --seed $seed_value --data_path /mnt/data/data/mdjilani/dataset/val --gpu cuda --comet_proj RQ1 -untargeted
        exec bash"
done
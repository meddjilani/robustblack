my_list=("Standard_R50" "Liu2023Comprehensive_ConvNeXt-L" "Salman2020Do_R18" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L")

test_path="/raid/data/mdjilani/dataset/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"


for seed in 1; do
  for target in "${my_list[@]}"; do

    cd LGV
    python LGV-MI-FGSM.py --eps 0.0156862745 --comet_proj RQ2 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model Peng2023Robust --batch_size 32 --lgv_models "/raid/data/mdjilani/${seed}lgv_models_robustPeng2023Robust" -robust

    cd ..
  done
done
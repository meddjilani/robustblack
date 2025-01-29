my_list=("Standard_R50" "Liu2023Comprehensive_ConvNeXt-L" "Salman2020Do_R18" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L")

test_path="/home/mdjilani/datasets/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"


for seed in 1 10 42; do
  for target in "${my_list[@]}"; do

    cd LGV
    python LGV-MI-FGSM.py --eps 0.0156862745 --comet_proj RQ2 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model wide_resnet101_2 --batch_size 32 --lgv_models "/home/mdjilani/datasets/lgv_models/_FullTrain_${seed}_lgv_models_wide_resnet101_2_128"

    cd ..
  done
done
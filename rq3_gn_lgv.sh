my_list=( "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

test_path="/raid/data/mdjilani/dataset/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"

for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    cd LGV
    python LGV-MI-FGSM.py --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model Engstrom2019Robustness --batch_size 64 --lgv_models "/raid/data/mdjilani/lgv_models_robust" --comet_proj RQ3 -robust

    cd ../GHOST
    python GN-MI-FGSM.py --seed $seed --target $target --data_path $test_path --gpu cuda --model Engstrom2019Robustness --batch_size 64 --comet_proj RQ3 -robust

    cd ..
  done
done

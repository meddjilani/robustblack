my_list=( "Liu2023Comprehensive_Swin-L" )

test_path="/raid/data/mdjilani/dataset/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"

for seed in 42; do
  for target in "${my_list[@]}"; do

    cd TREMBA
    python attack.py --comet_proj RQ3 --seed $seed --model_name $target --device cuda --config config/attack_untarget.json --data_path $test_path --helpers_path $helpers_path --generator_name Imagenet_Wong2020Fast_Engstrom2019Robustness_Debenedetti2022Light_XCiT-M12_untarget --save_path /raid/data/mdjilani/tremba_save_path_robust

    cd ..
  done
done
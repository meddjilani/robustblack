my_list=( "Wong2020Fast" "Engstrom2019Robustness" )

test_path="/raid/data/mdjilani/dataset/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"

for seed in 42; do
  for target in "${my_list[@]}"; do

    cd TREMBA
    python attack.py --comet_proj RQ1 --seed $seed --model_name $target --device cuda --config config/attack_untarget.json --data_path $test_path --helpers_path $helpers_path --generator_name Imagenet_VGG16_Resnet18_Squeezenet_Googlenet_untarget --save_path /raid/data/mdjilani/tremba_save_path_vanilla

    cd ..
  done
done

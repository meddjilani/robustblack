my_list=( "Standard_R50" "Salman2020Do_R18" "Salman2020Do_R50" "Salman2020Do_50_2" "Singh2023Revisiting_ViT-S-ConvStem" "Liu2023Comprehensive_ConvNeXt-B" "Liu2023Comprehensive_Swin-B" "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

test_path="/raid/data/mdjilani/dataset/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"

for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    cd TREMBA
    python attack.py --comet_proj RQ1 --seed $seed --model_name $target --device cuda --config config/attack_untarget.json --data_path $test_path --helpers_path $helpers_path --generator_name Imagenet_VGG16_Resnet18_Squeezenet_Googlenet_untarget --save_path /raid/data/mdjilani/tremba_save_path_vanilla

    cd ..
  done
done

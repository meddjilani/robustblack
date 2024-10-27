my_list=("Liu2023Comprehensive_Swin-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_ConvNeXt-L", "Liu2023Comprehensive_Swin-B" "Liu2023Comprehensive_ConvNeXt-B" "Singh2023Revisiting_ViT-S-ConvStem" "Salman2020Do_50_2" "Salman2020Do_R50", "Salman2020Do_R18", "Standard_R50")

test_path="/raid/data/mdjilani/dataset/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"


for seed in 1 10 42; do
  for target in "${my_list[@]}"; do

    cd LGV
    python LGV-MI-FGSM.py --eps 0.0156862745 --comet_proj RQ2 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model wide_resnet50_2 --batch_size 64 --lgv_models "/raid/data/mdjilani/${seed}lgv_modelswide_resnet50_2"

    cd ..

#    cd GHOST
#    python GN-MI-FGSM.py --eps 0.0156862745 --comet_proj RQ2 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model wide_resnet50_2 --batch_size 64
#
#    cd ..
  done
done

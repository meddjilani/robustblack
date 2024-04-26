my_list=( "Standard_R50" "Salman2020Do_R18" "Salman2020Do_R50" "Salman2020Do_50_2" "Singh2023Revisiting_ViT-S-ConvStem" "Liu2023Comprehensive_ConvNeXt-B" "Liu2023Comprehensive_Swin-B" "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

train_path="/raid/data/mdjilani/dataset/Imagenet/Sample_49000"  # Removed leading /
test_path="/raid/data/mdjilani/dataset/Imagenet/Sample_1000"  # Removed leading /


for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    cd GHOST
    python GN-MI-FGSM.py --comet_proj RQ1 --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet101_2 --batch_size 64

    cd LGV
    python LGV-MI-FGSM.py --comet_proj RQ1 --seed $seed --target $target --data_path $test_path --train_path $train_path --gpu cuda --model wide_resnet101_2 --batch_size 64 --lgv_models "/raid/data/mdjilani/lgv_models"

    cd ..
  done
done

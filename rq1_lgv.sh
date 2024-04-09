export CUDA_VISIBLE_DEVICES="0"  # Assuming this is intended

my_list=( "Standard_R50" "Ding2020MMA" "Rony2019Decoupling" "Salman2020Do_50_2" "Chen2024Data_WRN_50_2" "Singh2023Revisiting_ViT-B-ConvStem" "Liu2023Comprehensive_ConvNeXt-B" "Liu2023Comprehensive_Swin-B" "Liu2023Comprehensive_ConvNeXt-L" "Liu2023Comprehensive_Swin-L" )

train_path="/mnt/data/data/mdjilani/dataset/Imagenet/Sample_49000"  # Removed leading /
test_path="/mnt/data/data/mdjilani/dataset/Imagenet/Sample_1000"  # Removed leading /
exp_root="/mnt/data/data/mdjilani/bases_exp"  # Removed leading /
adv_root="/mnt/data/data/mdjilani/bases_adv"  # Removed leading /

for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    cd LGV
    python LGV-MI-FGSM.py --seed $seed --target $target --data_path $test_path --train_path $train_path --gpu cuda --model wide_resnet101_2 --batch_size 64 --lgv_models "/mnt/data/data/mdjilani/lgv_models"
    
    cd ..

  done
done

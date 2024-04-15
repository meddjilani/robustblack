export CUDA_VISIBLE_DEVICES="1,2"  # Assuming this is intended

my_list=( "Standard_R50" "Salman2020Do_R18" "Salman2020Do_50_2" "Debenedetti2022Light_XCiT-S12" "Debenedetti2022Light_XCiT-L12" "Singh2023Revisiting_ViT-S-ConvStem" "Liu2023Comprehensive_ConvNeXt-B" "Liu2023Comprehensive_Swin-B" "Liu2023Comprehensive_ConvNeXt-L" "Liu2023Comprehensive_Swin-L" )

train_path="/mnt/data/data/mdjilani/dataset/Imagenet/Sample_49000"  # Removed leading /
test_path="/mnt/data/data/mdjilani/dataset/Imagenet/Sample_1000"  # Removed leading /
exp_root="/mnt/data/data/mdjilani/bases_exp"  # Removed leading /
adv_root="/mnt/data/data/mdjilani/bases_adv"  # Removed leading /

for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    cd GHOST
    python GN-MI-FGSM.py --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet101_2 --batch_size 64

    cd ../BASES
    python query_w_bb.py --seed $seed --victim $target --gpu cuda --iterw 20 --data_path $test_path --exp_root $exp_root --adv_root $adv_root -untargeted

    cd ../TREMBA
    python attack.py --seed $seed --model_name $target --device cuda --config config/attack_untarget.json

    cd ..
  done
done

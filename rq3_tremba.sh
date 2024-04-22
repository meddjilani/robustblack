my_list=( "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

train_path="/raid/data/mdjilani/dataset/Imagenet/Sample_49000"  # Removed leading /
test_path="/raid/data/mdjilani/dataset/Imagenet/Sample_1000"  # Removed leading /

for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    cd ../TREMBA
    python attack.py --comet_proj RQ3 --seed $seed --model_name $target --device cuda --config config/attack_untarget.json

    cd ..
  done
done

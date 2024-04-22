my_list=( "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

train_path="/raid/data/mdjilani/dataset/Imagenet/Sample_49000"  # Removed leading /
test_path="/raid/data/mdjilani/dataset/Imagenet/Sample_1000"  # Removed leading /
exp_root="/raid/data/mdjilani/bases_exp"  # Removed leading /
adv_root="/raid/data/mdjilani/bases_adv"  # Removed leading /

for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    cd BASES
    python query_w_bb.py --exp_root $exp_root --adv_root $adv_root --n_wb 3 --iterw 20 --seed $seed --victim $target --data_path $test_path --gpu cuda --models Wong2020Fast Engstrom2019Robustness Debenedetti2022Light_XCiT-M12 --comet_proj RQ3 -robust -untargeted

    cd ..
  done
done

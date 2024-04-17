my_list=( "Standard_R50" "Ding2020MMA" "Rony2019Decoupling" "Salman2020Do_50_2" "Chen2024Data_WRN_50_2" "Singh2023Revisiting_ViT-B-ConvStem" "Liu2023Comprehensive_ConvNeXt-B" "Liu2023Comprehensive_Swin-B" "Liu2023Comprehensive_ConvNeXt-L" "Liu2023Comprehensive_Swin-L" )

train_path="/raid/data/mdjilani/dataset/Imagenet/Sample_49000"  # Removed leading /
test_path="/raid/data/mdjilani/dataset/Imagenet/Sample_1000"  # Removed leading /

for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    cd BASES
    python query_w_bb.py --iterw 20 --seed $seed --victim $target --data_path $test_path --gpu cuda --models Wong2020Fast Engstrom2019Robustness Debenedetti2022Light_XCiT-M12 --comet_proj RQ3 -robust -untargeted

    cd ..
  done
done

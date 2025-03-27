my_list=( "Salman2020Do_R18" "Salman2020Do_R50" "Salman2020Do_50_2" "Singh2023Revisiting_ViT-S-ConvStem" "Liu2023Comprehensive_ConvNeXt-B" "Liu2023Comprehensive_Swin-B" "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

test_path="/raid/data/mdjilani/dataset/val"
test_path="/home/mdjilani/datasets/val"

for seed in 42; do
  for target in "${my_list[@]}"; do

    python ADMIX.py --steps 50 --eps 0.062745 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 16

  done
done

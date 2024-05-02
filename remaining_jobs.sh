my_list=( "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

test_path="/raid/data/mdjilani/dataset/val"

for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    python ADMIX.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 32

  done
done

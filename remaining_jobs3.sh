my_list=( "Standard_R50" "Salman2020Do_R18" "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

test_path="/raid/data/mdjilani/dataset/val"

for seed in 42; do
  for target in "${my_list[@]}"; do

    python ADMIX.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model Peng2023Robust --batch_size 2 -robust

  done
done
my_list=("Liu2023Comprehensive_ConvNeXt-L")

test_path="/raid/data/mdjilani/dataset/val"

for seed in 1 10 42; do
  for target in "${my_list[@]}"; do

    python ADMIX.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet50_2 --batch_size 16

  done
done

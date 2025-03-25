my_list=( "Standard_R50"  "Salman2020Do_R50" )

test_path="/raid/data/mdjilani/dataset/val"
test_path="/home/mdjilani/datasets/val"

for seed in 42; do
  for target in "${my_list[@]}"; do

    python ADMIX.py --eps 0.062745 --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet50_2 --batch_size 8

  done
done

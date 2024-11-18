my_list=( "Standard_R50" "Salman2020Do_R18" "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

test_path="/raid/data/mdjilani/dataset/val"

for seed in 10; do
  for target in "${my_list[@]}"; do

    python MI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model Peng2023Robust --batch_size 32 -robust

    python SGM-MI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model Peng2023Robust --batch_size 32 -robust

    python DI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model Peng2023Robust --batch_size 32 -robust

    python TI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model Peng2023Robust --batch_size 32 -robust

    python VMI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model Peng2023Robust --batch_size 32 -robust

    python VNI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model Peng2023Robust --batch_size 32 -robust

    cd pytorch-gd-uap
    python train.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model Peng2023Robust --batch_size 32 -robust

    cd ..
  done
done
my_list=( "Standard_R50"  "Salman2020Do_R50" )

test_path="/home/mdjilani/datasets/val"

for seed in 42; do
  for target in "${my_list[@]}"; do

    python MI-FGSM.py --eps 0.062745 --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet50_2 --batch_size 64

    python SGM-MI-FGSM.py --eps 0.062745 --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet50_2 --batch_size 64

    python DI-FGSM.py --eps 0.062745 --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet50_2 --batch_size 64

    python TI-FGSM.py --eps 0.062745 --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet50_2 --batch_size 64

    python VMI-FGSM.py --eps 0.062745 --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet50_2 --batch_size 64

    python VNI-FGSM.py --eps 0.062745 --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet50_2 --batch_size 64

    cd pytorch-gd-uap
    python train.py --eps 0.062745 --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet50_2 --batch_size 64

    cd ..
  done
done

my_list=( "Salman2020Do_R18" "Salman2020Do_R50" "Salman2020Do_50_2" "Singh2023Revisiting_ViT-S-ConvStem" "Liu2023Comprehensive_ConvNeXt-B" "Liu2023Comprehensive_Swin-B" "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

test_path="/raid/data/mdjilani/dataset/val"

for seed in 42; do
  for target in "${my_list[@]}"; do

    python MI-FGSM.py --eps 0.03125 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    python SGM-MI-FGSM.py --eps 0.03125 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    python DI-FGSM.py --eps 0.03125 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    python TI-FGSM.py --eps 0.03125 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    python VMI-FGSM.py --eps 0.03125 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    python VNI-FGSM.py --eps 0.03125 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    cd pytorch-gd-uap
    python train.py --eps 0.03125 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    cd ..
  done
done

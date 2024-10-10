my_list=( "Liu2023Comprehensive_Swin-B" "Liu2023Comprehensive_ConvNeXt-B" "Singh2023Revisiting_ViT-S-ConvStem" "Salman2020Do_50_2" "Salman2020Do_R50")

test_path="/raid/data/mdjilani/dataset/val"

for seed in 42; do
  for target in "${my_list[@]}"; do

    python MI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    python SGM-MI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    python DI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    python TI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    python VMI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    python VNI-FGSM.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    cd pytorch-gd-uap
    python train.py --eps 0.0156862745 --seed $seed --target $target --data_path $test_path --gpu cuda --model resnet50 --batch_size 64

    cd ..
  done
done

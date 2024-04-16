my_list=( "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

test_path="/mnt/data/data/mdjilani/dataset/Imagenet/Sample_1000"  # Removed leading /

for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    python MI-FGSM.py --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet101_2 --batch_size 64

    python DI-FGSM.py --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet101_2 --batch_size 64

    python TI-FGSM.py --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet101_2 --batch_size 64

    python VMI-FGSM.py --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet101_2 --batch_size 64

    python VNI-FGSM.py --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet101_2 --batch_size 64

    python ADMIX.py --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet101_2 --batch_size 64

    cd pytorch-gd-uap
    python gduap.py --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet101_2 --batch_size 64

    cd ..
  done
done

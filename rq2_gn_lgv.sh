my_list=( "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

train_path="/mnt/data/data/mdjilani/dataset/Imagenet/Sample_49000"  # Removed leading /
test_path="/mnt/data/data/mdjilani/dataset/Imagenet/Sample_1000"  # Removed leading /

for seed in 42 1 10; do
  for target in "${my_list[@]}"; do

    cd LGV
    python LGV-MI-FGSM.py --seed $seed --target $target --data_path $test_path --train_path $train_path --gpu cuda --model wide_resnet101_2 --batch_size 64 --lgv_models "/mnt/data/data/mdjilani/lgv_models_robust" --comet_proj RQ2

    cd GHOST
    python GN-MI-FGSM.py --seed $seed --target $target --data_path $test_path --gpu cuda --model wide_resnet101_2 --batch_size 64 --comet_proj RQ2

    cd ..
  done
done

my_list=( "Salman2020Do_R18" "Salman2020Do_R50" "Salman2020Do_50_2" "Singh2023Revisiting_ViT-S-ConvStem" "Liu2023Comprehensive_ConvNeXt-B" "Liu2023Comprehensive_Swin-B" "Liu2023Comprehensive_ConvNeXt-L" "Bai2024MixedNUTS" "Liu2023Comprehensive_Swin-L" )

test_path="/home/mdjilani/datasets/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"


for seed in 42; do
  for target in "${my_list[@]}"; do

    cd LGV
    python LGV-MI-FGSM.py --steps 20 --eps 0.062745 --comet_proj RQ1 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model resnet50 --batch_size 64 --lgv_models "/home/mdjilani/datasets/lgv_models/_FullTrain_42_lgv_models_resnet50_256"

    cd ..

    cd GHOST
    python GN-MI-FGSM.py --steps 20 --eps 0.062745 --comet_proj RQ1 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model resnet50 --batch_size 64

    cd ..
  done
done
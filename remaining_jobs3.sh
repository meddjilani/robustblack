my_list=( "Standard_R50"  "Salman2020Do_R50" )

test_path="/home/mdjilani/datasets/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"


for seed in 42; do
  for target in "${my_list[@]}"; do

    cd LGV
    python LGV-MI-FGSM.py --steps 50 --eps 0.062745 --comet_proj RQ1 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model wide_resnet50_2 --batch_size 64 --lgv_models "/home/mdjilani/datasets/lgv_models/_FullTrain_42_lgv_models_wide_resnet50_2_128"

    cd ..

    cd GHOST
    python GN-MI-FGSM.py --steps 50 --eps 0.062745 --comet_proj RQ1 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model wide_resnet50_2 --batch_size 64

    cd ..
  done
done
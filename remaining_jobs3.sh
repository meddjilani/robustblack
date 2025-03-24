my_list=( "Standard_R50"  "Salman2020Do_R50" )

test_path="/raid/data/mdjilani/dataset/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"


for seed in 42; do
  for target in "${my_list[@]}"; do

    cd LGV
    python LGV-MI-FGSM.py --eps 0.03125 --comet_proj RQ1 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model wide_resnet50_2 --batch_size 64 --lgv_models "/raid/data/mdjilani/_FullTrain_42_lgv_models_resnet50_256"

    cd ..

    cd GHOST
    python GN-MI-FGSM.py --eps 0.03125 --comet_proj RQ1 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model wide_resnet50_2 --batch_size 64

    cd ..
  done
done
my_list=( "Wong2020Fast" "Engstrom2019Robustness" )

test_path="/raid/data/mdjilani/dataset/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"


for seed in 42; do
  for target in "${my_list[@]}"; do

    cd LGV
    python LGV-MI-FGSM.py --eps 0.0156862745 --comet_proj RQ2 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model resnet50 --batch_size 64 --lgv_models "/raid/data/mdjilani/seed0"

    cd ..

    cd GHOST
    python GN-MI-FGSM.py --eps 0.0156862745 --comet_proj RQ2 --seed $seed --target $target --data_path $test_path --helpers_path $helpers_path --gpu cuda --model resnet50 --batch_size 64

    cd ..
  done
done

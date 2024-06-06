my_list=( "Liu2023Comprehensive_ConvNeXt-L" )

test_path="/raid/data/mdjilani/dataset/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"

exp_root="/raid/data/mdjilani/bases_exp_rob"
adv_root="/raid/data/mdjilani/bases_adv_rob"

for seed in 42; do
  for target in "${my_list[@]}"; do

    cd BASES
    python query_w_bb.py --eps 4 --models Engstrom2019Robustness Wong2020Fast  --helpers_path $helpers_path --exp_root $exp_root --adv_root $adv_root --iterw 20 --seed $seed --victim $target --data_path $test_path --gpu cuda --comet_proj RQ3 -robust -untargeted

    cd ..
  done
done
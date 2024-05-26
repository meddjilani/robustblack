my_list=( "Bai2024MixedNUTS" )

test_path="/raid/data/mdjilani/dataset/val"
helpers_path="/home/mdjilani/robustblack/utils_robustblack"

exp_root="/raid/data/mdjilani/bases_exp"
adv_root="/raid/data/mdjilani/bases_adv"

for seed in 42; do
  for target in "${my_list[@]}"; do

    cd BASES
    python query_w_bb.py --eps 4 --n_wb 10  --helpers_path $helpers_path --exp_root $exp_root --adv_root $adv_root --iterw 20 --seed $seed --victim $target --data_path $test_path --gpu cuda --comet_proj RQ1 -untargeted

    cd ..
  done
done
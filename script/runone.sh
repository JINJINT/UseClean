#!/bin/bash

#===== main experiments
# baseline
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod none --seed $3 --info seed$3info$4 --train_num $6 --test_num $6 --dev_num $6
# baseline-clean
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod CLin --score nerloss --cutoff clean --seed $3 --info seed$3info$4 --train_num $6 --test_num $6 --dev_num $6
# clpaper (use true noise rate for in dynamic cutoff)
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod CLin --score nerloss --cutoff heuri --seed $3 --info seed$3info$4 --train_num $6 --test_num $6 --dev_num $6
# clpaper withclean (use true noise rate for in dynamic cutoff)
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod CLin --score nerloss --cutoff heuri --injectclean true --seed $3 --info seed$3info$4withclean --cleanprop $5 --train_num $6 --test_num $6 --dev_num $6
# useclean 
for score in useclean usecleanhead usecleantail
do
for conf in nerloss diff
do
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod CLin --score $score --usecleanscore $conf --cutoff fake --seed $3 --info seed$3info$4 --warm true --weight true --cleanprop $5 --train_num $6 --test_num $6 --dev_num $6
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod CLin --score $score --usecleanscore $conf --cutoff fitmix --seed $3 --info seed$3info$4 --warm true --weight true --cleanprop $5 --train_num $6 --test_num $6 --dev_num $6
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod CLin --score $score --usecleanscore $conf --cutoff goracle --seed $3 --info seed$3info$4 --warm true --weight true --cleanprop $5 --train_num $6 --test_num $6 --dev_num $6
done
done
# coreg
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod coreg --seed $3 --info seed$3info$4 --train_num $6 --test_num $6 --dev_num $6 
# metaweight
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod metaweight --seed $3 --info seed$3info$4 --train_num $6 --test_num $6 --dev_num $6
# contrast
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod none --contrastive true --seed $3 --info seed$3info$4contrast --train_num $6 --test_num $6 --dev_num $6 

#=====other experiments
# CLout correct
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod CLout --score nerloss --cutoff heuri --modify correct --info seed$3info$4 --train_num $6 --test_num $6 --dev_num $6 
# CLout weight
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod CLout --score nerloss --cutoff heuri --modify weight --info seed$3info$4 --train_num $6 --test_num $6 --dev_num $6 
# CLout rank
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod CLout --score nerloss --cutoff heuri --modify rank --info seed$3info$4 --train_num $6 --test_num $6 --dev_num $6 
# UseClean balanced
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset $2 --clmethod CLin --score useclean --tau 0.05 --cutoff goracle --info seed$3info$4 --train_num $6 --test_num $6 --dev_num $6 



# CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikigold_dist --clmethod none --contrastive true --seed 1 --info bert1contrast --train_num 100 --test_num 20 --dev_num 20
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikigold_dist --clmethod none --seed 1 --info bert1diag --train_num 100 --test_num 20 --dev_num 20 --diag true
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikigold_dist --clmethod CLin --score nerloss --cutoff heuri --injectclean true --seed 1 --info bert1diagwithclean --train_num 100 --test_num 20 --dev_num 20 --diag true
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikigold_dist --clmethod CLin --score useclean --cutoff fitmix --cleanprop 0.1 --cleanepochs 50 --num_epochs 20 --seed 1 --info bert1finalwarmweight --diag true  --warm true --weight true
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikigold_dist --clmethod CLin --score useclean --cleanprop 0.1 --cleanepochs 10 --num_epochs 0 --seed 1 --info bert1all --diag true
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikigold_dist --clmethod CLin --tau 0.5 --score useclean --cutoff goracle --cleanprop 0.1 --cleanepochs 20 --num_epochs 5 --seed 1 --info bert1diagbal --diag true
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikigold_dist --clmethod coreg --num_epochs 5 --seed 1 --info bert1diag --diag true
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikigold_dist --clmethod metaweight --num_epochs 5 --seed 1 --info bert1diag --diag true
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikigold_dist --clmethod CLout --score nerloss --cutoff heuri --modify correct --num_epochs 5 --seed 1 --info bert1 --train_num 100 --test_num 20 --dev_num 20


CUDA_VISIBLE_DEVICES=0 python main.py --dataset massive_en_us__noise_bias_level1 --clmethod CLin --score useclean --usecleanscore nerloss --cutoff fitmix --cleanprop 0.03 --cleanepochs 50 --num_epochs 20 --seed 1 --info bert1warmweight --diag true  --warm true --weight true
CUDA_VISIBLE_DEVICES=0 python main.py --dataset massive_en_us__noise_bias_level1 --clmethod CLin --score useclean --usecleanscore nerloss --cutoff goracle --cleanprop 0.03 --cleanepochs 50 --num_epochs 20 --seed 1 --info bert1warmweight --diag true  --warm true --weight true
CUDA_VISIBLE_DEVICES=0 python main.py --dataset massive_en_us__noise_bias_level1 --clmethod CLin --score useclean --usecleanscore nerloss --cutoff goracle --cleanprop 0.03 --cleanepochs 50 --num_epochs 20 --seed 1 --info bert1 --diag true 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset massive_en_us__noise_bias_level1 --clmethod CLin --score useclean --usecleanscore nerloss --encoder bilstm --learning_rate 0.01 --cutoff fitmix --cleanprop 0.03 --cleanepochs 50 --num_epochs 30 --seed 1 --info bilstm1warmweight --diag true  --warm true --weight true
CUDA_VISIBLE_DEVICES=0 python main.py --dataset massive_en_us__noise_bias_level1 --clmethod CLin --score useclean --usecleanscore nerloss --encoder bilstm --learning_rate 0.01 --cutoff goracle --cleanprop 0.03 --cleanepochs 50 --num_epochs 30 --seed 1 --info bilstm1warmweight --diag true  --warm true --weight true
CUDA_VISIBLE_DEVICES=0 python main.py --dataset massive_en_us__noise_bias_level1 --clmethod CLin --score useclean --usecleanscore nerloss --encoder bilstm --learning_rate 0.01 --cutoff goracle --cleanprop 0.03 --cleanepochs 50 --num_epochs 30 --seed 1 --info bilstm1 --diag true 
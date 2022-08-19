#!/bin/bash

#===== main experiments
for data in massive_en_us__noise_bias_level0.7 massive_en_us__noise_bias_level1 massive_en_us__noise_miss_level0.7 massive_en_us__noise_miss_level1 massive_en_us__noise_extend_level0.7 massive_en_us__noise_extend_level1 massive_dist
do
# baseline
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod none --seed $2 --info bert$2
# baseline-clean
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff clean --seed $2 --info bert$2
# clpaper (use true noise rate for in dynamic cutoff)
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff heuri --seed $2 --info bert$2
# clpaper withclean (use true noise rate for in dynamic cutoff)
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff heuri --injectclean true --seed $2 --info bert$2withclean --cleanprop 0.03
# useclean 
for score in useclean usecleanhead usecleantail
do
for conf in nerloss diff
do
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score $score --usecleanscore $conf --cutoff fitmix --seed $2 --info bert$2warmweight --warm true --weight true --cleanprop 0.03
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score $score --usecleanscore $conf --cutoff goracle --seed $2 --info bert$2warmweight --warm true --weight true --cleanprop 0.03 
done
done
done

for data in conll_dist conll_transfer
do
# baseline
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod none --seed $2 --info bert$2
# baseline-clean
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff clean --seed $2 --info bert$2
# clpaper (use true noise rate for in dynamic cutoff)
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff heuri --seed $2 --info bert$2
# clpaper withclean (use true noise rate for in dynamic cutoff)
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff heuri --injectclean true --seed $2 --info bert$2withclean --cleanprop 0.01
# useclean 
for score in useclean usecleanhead usecleantail
do
for conf in nerloss diff
do
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score $score --usecleanscore $conf --cutoff fitmix --seed $2 --info bert$2warmweight --warm true --weight true --cleanprop 0.01
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score $score --usecleanscore $conf --cutoff goracle --seed $2 --info bert$2warmweight --warm true --weight true --cleanprop 0.01 
done
done
done

for data in massive_transeasy
do
# baseline
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod none --seed $2 --info bert$2
# baseline-clean
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff clean --seed $2 --info bert$2
# clpaper (use true noise rate for in dynamic cutoff)
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff heuri --seed $2 --info bert$2
# clpaper withclean (use true noise rate for in dynamic cutoff)
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff heuri --injectclean true --seed $2 --info bert$2withclean --cleanprop 0.05
# useclean 
for score in useclean usecleanhead usecleantail
do
for conf in nerloss diff
do
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score $score --usecleanscore $conf --cutoff fitmix --seed $2 --info bert$2warmweight --warm true --weight true --cleanprop 0.05
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score $score --usecleanscore $conf --cutoff goracle --seed $2 --info bert$2warmweight --warm true --weight true --cleanprop 0.05 
done
done
done


for data in wikigold_dist
do
# baseline
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod none --seed $2 --info bert$2
# baseline-clean
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff clean --seed $2 --info bert$2
# clpaper (use true noise rate for in dynamic cutoff)
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff heuri --seed $2 --info bert$2
# clpaper withclean (use true noise rate for in dynamic cutoff)
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score nerloss --cutoff heuri --injectclean true --seed $2 --info bert$2withclean --cleanprop 0.1
# useclean 
for score in useclean
do
for conf in nerloss diff
do
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score $score --usecleanscore $conf --cutoff fitmix --seed $2 --info bert$2warmweight --warm true --weight true --cleanprop 0.1
CUDA_VISIBLE_DEVICES=$1 python main.py --dataset data --clmethod CLin --score $score --usecleanscore $conf --cutoff goracle --seed $2 --info bert$2warmweight --warm true --weight true --cleanprop 0.1 
done
done
done


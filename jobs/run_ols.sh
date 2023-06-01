#!/bin/bash
cd ../scripts

NUM_RUNS=32
GPU_IDS=( 0 1 2 3 )
NUM_GPUS=${#GPU_IDS[@]}


## Parameters

# DATAS=( "synthetic" "mnist" "cifar10" "eurosat" "fashion" "arxiv")
DATAS=( "synthetic" "mnist" "cifar10" "eurosat" "fashion" )
# DATAS=( "synthetic" "mnist" "cifar10" )
# DATAS=( "mnist" "cifar10" )
# DATAS=( "mnist" )
# DATAS=( "cifar10" )
# DATAS=( "shl" )
# DATAS=( "arxiv" )
# DATAS=( "cifar10" )
# DATAS=( "eurosat" )

SHIFTS=( "monotone" "square" "sinusoidal" "bernouli" )
# SHIFTS=( "bernouli" "sinusoidal")
# SHIFTS=( "bernouli" )
# SHIFTS=( "monotone" )

USE_SOURCE_TEST=1
# SOURCE_TEST_RATIOS=( 0.01 0.04 0.08 0.16 0.32 0.64 )
SOURCE_TEST_RATIOS=( 0.02 1 )
# SOURCE_TEST_RATIOS=( 1 )
# SOURCE_TEST_RATIOS=( 0.02 )
# SOURCE_TEST_RATIOS=( 0.1 )

# CAL_TYPES=( "ts" "vs" )
# CAL_TYPES=( "vs" )
CAL_TYPES=( "ts" )

USE_CURRENT_MARG_EST=0

# NUM_ONLINE_SAMPLES=( 1 10 50 100 200 )
NUM_ONLINE_SAMPLES=( 10 )

# SHIFT_EST_TYPES=( "bbse" "rlls" "mlls" )
# SHIFT_EST_TYPES=( "rlls" "mlls" )
# SHIFT_EST_TYPES=( "bbse" "rlls" )
# SHIFT_EST_TYPES=( "mlls" )
SHIFT_EST_TYPES=( "bbse" )

SEEDS=( 8610 1308 2011 )
# SEEDS=( 2011 )

declare -A MODELS=(
    ["synthetic"]="logreg"
    ["mnist"]="fcn_early"
    # ["mnist"]="randforest"
    ["cifar10"]="resnet18"
    # ["cifar10"]="randforest"
    ["eurosat"]="resnet18"
    ["fashion"]="mlp"
    ["shl"]="mlp"
    ["arxiv"]="bert"
)

counter=0
for shift_est_type in "${SHIFT_EST_TYPES[@]}"; do
for cal_type in "${CAL_TYPES[@]}"; do
for num_online_samples in "${NUM_ONLINE_SAMPLES[@]}"; do
for st_ratio in "${SOURCE_TEST_RATIOS[@]}"; do
for data_name in "${DATAS[@]}"; do
for shift in "${SHIFTS[@]}"; do
for seed in "${SEEDS[@]}"; do

	# Get GPU id.
	gpu_idx=$((counter % $NUM_GPUS))
	gpu_id=${GPU_IDS[$gpu_idx]}

    cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python ../scripts/run_ols.py \
    '-d' ${data_name} '-m' ${MODELS[$data_name]} '-s' ${shift} \
    '--use-source-test' ${USE_SOURCE_TEST} \
	'--source-test-ratio' ${st_ratio} '--seed' ${seed} \
    '-t' '1000' '--do-all' '1' '--save' '1' \
    '--num-online-samples' '${num_online_samples}' \
    '--calibration-type' '${cal_type}' \
    --use-current-marginal-estimate '${USE_CURRENT_MARG_EST}' \
    --shift-estimator-type '${shift_est_type}' \
    "

    echo $cmd
	eval ${cmd} &

    sleep 10

	counter=$((counter+1))
	if ! ((counter % NUM_RUNS)); then
	  wait
	fi
done
done
done
done
done
done
done

#!/bin/bash

DATASETS=("BDGP")
LAMBDAS=("0.01" "0.05" "0.1" "0.3" "0.5" "0.7" "1")

mkdir -p experiments # 创建experiments文件夹

for dataset in "${DATASETS[@]}"; do
    mkdir -p "experiments/${dataset}" # 在experiments文件夹中创建数据集文件夹
    for lambda1 in "${LAMBDAS[@]}"; do
        for lambda2 in "${LAMBDAS[@]}"; do
            log_file="experiments/${dataset}/lambda1_${lambda1}-lambda2_${lambda2}.log" # log文件路径
            echo "Running with lambda1=$lambda1, lambda2=$lambda2 on $dataset dataset" >> "${log_file}"
            # 运行模型，将输出追加到log文件
            python train_SSGC0.py --lambda1 "$lambda1" --lambda2 "$lambda2" --dataset "$dataset" >> "${log_file}"
            echo "Finished running" >> "${log_file}"
        done
    done
done
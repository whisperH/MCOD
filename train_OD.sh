#!/bin/bash

# python train_OD.py -c ../../configs/Hela/AutoEncoder.yaml data_dir datasets/Fake train_status false infer_status false eval_status true infer_datafile fake-shuffle_ratio-0.4-repeat_time-2.gz

# 定义待执行的任务函数
run_task () {
    # 需要执行的任务命令
    modelname=$1
    filename=$2
    # 执行任务
    CUDA_VISIBLE_DEVICES=3 python3 train_OD.py -c ../../configs/Hela/AE/$modelname data_dir datasets/Fake train_status false infer_status false eval_status true infer_datafile $filename
#    CUDA_VISIBLE_DEVICES=3 python3 train_OD.py -c ../../configs/Hela/AE/$modelname data_dir datasets/Fake train_status false infer_status false eval_status true infer_datafile fake-shuffle_ratio-0.01-repeat_time-$filename.gz
}

# 定义变量来存储要启动的进程数
max_processes=5

# 初始化计数器
count=0

# model list
#model_list=("GMM.yaml" "DIF.yaml" "LSCP.yaml" "LOF.yaml" "LMDD.yaml" "KNN.yaml" "IForest.yaml" "HBOS.yaml" "FeatBagging.yaml" "ECOD.yaml" "DeepSVDD.yaml" "COF.yaml" "CBLOF.yaml" "AutoEncoder.yaml" "ABOD.yaml" "OCSVM.yaml" "PCA.yaml" "SOGAAL.yaml" "VAE.yaml" )
model_list=("AE_MEM10_S20_CLS.yaml")

# get the files in the fake folder
folder="./datasets/Fake"
files=$(ls $folder)
for imodel in ${model_list[@]}
do
  for file in ${files}
    do
      # 同时运行的进程数超过了最大值，则等待一个进程完成
      ((++count)) && ((count==$max_processes)) && wait -n && ((count--))
      # 执行函数，并将任务命令传递给它
      run_task "$imodel" "$file" &
    done
done

#files=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
#for imodel in ${model_list[@]}
#do
#  for file in ${files[@]}
#    do
#      # 同时运行的进程数超过了最大值，则等待一个进程完成
#      ((++count)) && ((count==$max_processes)) && wait -n && ((count--))
#      # 执行函数，并将任务命令传递给它
#      run_task "$imodel" "$file" &
#    done
#done

wait
#!/bin/bash
set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
username=`whoami`

config_path=$1
save_path=$2
face_path=$3

mode=PTI
gpu_numbers=4
function main
{
  python tools/train_pose_facial_multi_gpus.py \
                         --config_path ${config_path} \
                         --save_path ${save_path} \
                         --path ${face_path} \
                         --gpu_numbers ${gpu_numbers}
}
_timestamp=`date +%Y%m%d%H`
main

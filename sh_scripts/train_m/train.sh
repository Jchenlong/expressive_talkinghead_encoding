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

  python tools/w_pivot_training_init.py \
                         --config_path ${config_path} \
                         --save_path ${save_path} \
                         --path ${face_path}

  python -m torch.distributed.launch --nproc_per_node=${gpu_numbers} --master_port 15575 tools/w_pivot_training_multi_gpus.py ${gpu_numbers} ${config_path} ${save_path} ${face_path} ${mode}


  python tools/train_pose_facial_multi_gpus.py \
                         --config_path ${config_path} \
                         --save_path ${save_path} \
                         --path ${face_path} \
                         --gpu_numbers ${gpu_numbers}


  python -m torch.distributed.launch --nproc_per_node=${gpu_numbers} --master_port 15575 tools/pivot_training_multi_gpus.py ${gpu_numbers} ${config_path} ${save_path} ${face_path} ${mode}

}
_timestamp=`date +%Y%m%d%H`
main

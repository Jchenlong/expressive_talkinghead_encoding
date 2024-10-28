set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

gt_path=$1
exp_dir=$2
facial_path=${exp_dir}/facial_ft
decoder_path=${exp_dir}/pti_ft_512/facial_snapshots_ft_512
if [ -e ${facial_path} ]; then
        echo "facial_path"
else
  facial_path=${exp_dir}/facial
  echo "facial"
  decoder_path=${exp_dir}/pti_ft_512/snapshots
fi

function main
{
    python tools/f_space_training.py \
                                 --gt_path ${gt_path} \
                                 --facial_path ${facial_path} \
                                 --config_path /data1/chenlong/github/v65/expressive_talkinghead_encoding/sh_scripts/train_set_v7_encoder/config.yaml \
                                 --save_path ${exp_dir}/f_space/encoder \
                                 --gpus 4 \
                                 --decoder_path ${decoder_path} 
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
export MASTER_PORT=25557
main

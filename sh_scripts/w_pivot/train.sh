set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

gt_path=$1
exp_dir=$2
config_path=$3

w_pti_save_path=${exp_dir}/pti

function main
{   
    python tools/pivot_training.py \
                           --gt_path ${gt_path} \
                           --latent_path ${exp_dir}/w \
                           --config_path ${config_path}/config.yaml \
                           --save_path ${w_pti_save_path}
}

_timestamp=`date +%Y%m%d%H`
main

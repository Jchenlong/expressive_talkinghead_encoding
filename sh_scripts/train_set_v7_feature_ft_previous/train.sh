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
save_feature_ft_path=${exp_dir}/f_space/feature_ft
f_resume_path=${exp_dir}/f_space/feature

function main
{
    python tools/f_space_detailed_training_previous.py \
                                 --gt_path ${gt_path} \
                                 --facial_path ${facial_path} \
                                 --config_path ./sh_scripts/train_set_v7_feature_ft_previous/config.yaml \
                                 --save_path ${save_feature_ft_path} \
                                 --gpus 4 \
                                 --decoder_path ${decoder_path} \
                                 --resume_path ${f_resume_path}
}

_timestamp=`date +%Y%m%d%H`
export MASTER_PORT=15586
main

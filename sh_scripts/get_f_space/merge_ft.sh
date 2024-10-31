set -e

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

feature_path=${exp_dir}/f_space/feature_ft/snapshots
f_decoder_path=${exp_dir}/f_space/decoder_ft
to_path=${exp_dir}/f_space/f_space_ft.pt
echo 'merge_feature'
function main
{
    CUDA_VISIBLE_DEVICES=0 python merge_more2one_ft.py ${feature_path} ${to_path}
}

_timestamp=`date +%Y%m%d%H`
main
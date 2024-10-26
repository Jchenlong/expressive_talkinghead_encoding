set -e

gt_path=/data1/chenlong/online_model_set/video/eR6CMmTu_2/video_split/smooth
exp_dir=/data1/chenlong/0517/video/0822/eR6CMmTu_2/eR6CMmTu_2_exp_set_v4_depoly/exp
facial_path=${exp_dir}/facial_ft
decoder_path=${exp_dir}/pti_ft_512/facial_snapshots_ft_512

if [ -e ${facial_path} ]; then
        echo "facial_path"
else
  facial_path=${exp_dir}/facial
  echo "facial"
  decoder_path=${exp_dir}/pti_ft_512/snapshots
fi

feature_path=${exp_dir}/f_space_bdinv_012/feature
f_decoder_path=${exp_dir}/f_space_bdinv_012/decoder
to_path=${exp_dir}/f_space_bdinv_012/f_space.pt

function main
{
    CUDA_VISIBLE_DEVICES=3 python merge_more2one.py ${feature_path} ${to_path}
}

_timestamp=`date +%Y%m%d%H`
main
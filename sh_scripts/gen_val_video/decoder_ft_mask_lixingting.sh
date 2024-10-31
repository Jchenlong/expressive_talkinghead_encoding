set -e

gt_path=/data1/chenlong/online_model_set/video/5UWuxkBQ_20240103/video_split/smooth
exp_dir=/data1/chenlong/0517/video/0822/5UWuxkBQ_20240103/5UWuxkBQ_20240103_exp_set_v4_depoly/exp
expname=5UWuxkBQ_20240103
facial_path=${exp_dir}/facial_ft
decoder_path=${exp_dir}/pti_ft_512/facial_snapshots_ft_512

if [ -e ${facial_path} ]; then
        echo "facial_path"
else
  facial_path=${exp_dir}/facial
  echo "facial"
  decoder_path=${exp_dir}/pti_ft_512/snapshots
fi

f_encoder_path=${exp_dir}/f_space/encoder/snapshots
f_decoder_path=${exp_dir}/f_space/decoder_ft/snapshots
#f_decoder_path=${exp_dir}/f_space/decoder_ft/snapshots
f_space_path=${exp_dir}/f_space/f_space_ft.pt
save_path=/data1/chenlong/yuyuhang/interface_out/1029
echo ${expname}
function main
{
    python tools/test_feature_decoder_soft_mask.py \
                                 --expname ${expname} \
                                 --save_path ${save_path} \
                                 --latest_decoder_path ${f_decoder_path} \
                                 --f_space_path ${f_space_path}
}

_timestamp=`date +%Y%m%d%H`
main
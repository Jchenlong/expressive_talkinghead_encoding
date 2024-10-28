set -e

gt_path=$1
exp_dir=$2
expname=$3
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
f_decoder_path=${exp_dir}/f_space/decoder/snapshots
f_space_path=${exp_dir}/f_space/f_space.pt
save_path=/data1/chenlong/yuyuhang/interface_out/1025/1025
echo ${expname}
function main
{
    python tools/test_feature_decoder_mask.py \
                                 --expname ${expname} \
                                 --save_path ${save_path} \
                                 --latest_decoder_path ${f_decoder_path} \
                                 --f_space_path ${f_space_path}
}

_timestamp=`date +%Y%m%d%H`
main
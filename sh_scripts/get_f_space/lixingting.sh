set -e

gt_path=/data1/chenlong/online_model_set/video/5UWuxkBQ_20240103/video_split/smooth
exp_dir=/data1/chenlong/0517/video/0822/5UWuxkBQ_20240103/5UWuxkBQ_20240103_exp_set_v4_depoly/exp
facial_path=${exp_dir}/facial_ft
decoder_path=${exp_dir}/pti_ft_512/facial_snapshots_ft_512

if [ -e ${facial_path} ]; then
        echo "facial_path"
else
  facial_path=${exp_dir}/facial
  echo "facial"
  decoder_path=${exp_dir}/pti_ft_512/snapshots
fi

f_encoder_path=${exp_dir}/f_space/stitch/encoder/snapshots/20.pth
f_decoder_path=${exp_dir}/f_space/stitch/decoder
to_path=${exp_dir}/f_space/stitch

function main
{
    CUDA_VISIBLE_DEVICES=3 python tools/gen_f_space.py \
                                 --from_path ${gt_path} \
                                 --to_path ${to_path} \
                                 --config_path /data1/chenlong/github/F/expressive_talkinghead_encoding/sh_scripts/bdinv_006/config.yaml \
                                 --model_path ${f_encoder_path}
}

_timestamp=`date +%Y%m%d%H`
main
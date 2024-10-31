set -e

gt_path=$1
exp_dir=$2
gt_path=/data1/chenlong/online_model_set/video/5UWuxkBQ_20240103/video/smooth
exp_dir=/data1/chenlong/0517/video/0822/5UWuxkBQ_20240103/5UWuxkBQ_20240103_exp_set_v4_depoly/exp

f_encoder_path=${exp_dir}/f_space/stitch/encoder/snapshots/best.pth
to_path=${exp_dir}/f_space/stitch/feature
f_space_pt=${exp_dir}/f_space/stitch/f_space.pt

function main
{
    python tools/gen_f_space.py \
                                 --from_path ${gt_path} \
                                 --to_path ${to_path} \
                                 --config_path /data1/chenlong/github/v65/expressive_talkinghead_encoding/sh_scripts/bdinv_012/config.yaml \
                                 --model_path ${f_encoder_path}

    python merge_more2one.py ${to_path} ${f_space_pt}

}

_timestamp=`date +%Y%m%d%H`
main
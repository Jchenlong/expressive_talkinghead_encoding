set -e

gt_path=$1
exp_dir=$2

f_encoder_path=${exp_dir}/f_space/encoder/snapshots/best.pth
to_path=${exp_dir}/f_space/feature
f_space_pt=${exp_dir}/f_space/f_space.pt

function main
{
    python tools/gen_f_space.py \
                                 --from_path ${gt_path} \
                                 --to_path ${to_path} \
                                 --config_path ./sh_scripts/train_set_v7_encoder/config.yaml \
                                 --model_path ${f_encoder_path}

    python merge_more2one.py ${to_path} ${f_space_pt}

}

_timestamp=`date +%Y%m%d%H`
main
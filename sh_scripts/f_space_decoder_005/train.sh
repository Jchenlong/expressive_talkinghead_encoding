set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

exp_dir=/data1/chenlong/0517/video/0822/eR6CMmTu_2/eR6CMmTu_2_exp_set_v4_depoly/exp
decoder_path=${exp_dir}/pti_ft_512/facial_snapshots_ft_512
#mkdir -p results

function main
{
    python tools/f_space_decoder.py \
                                 --config_path ./sh_scripts/${exp_name}/config.yaml \
                                 --save_path /data1/chenlong/0517/video/0822/eR6CMmTu_2/eR6CMmTu_2_exp_set_v4_depoly/exp/f_space_bdinv_012/decoder \
                                 --gpus 4 \
                                 --decoder_path ${decoder_path} 
}

_timestamp=`date +%Y%m%d%H`
export MASTER_PORT=15581
main

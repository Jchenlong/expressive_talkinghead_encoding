"""get style space latent codes script.
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import tqdm
import click
import torch
import time

from tensorboardX import SummaryWriter
from ExpressiveEncoding.train import pivot_finetuning, StyleSpaceDecoder, \
                                 stylegan_path, edict, yaml, \
                                 logger,pipeline_init,get_pose_pipeline_multi,get_facial_pipeline_multi
import torch.multiprocessing as multiprocessing
import time


@click.command()
@click.option('--config_path')
@click.option('--save_path')
@click.option('--gpu_numbers',default = 4)
@click.option('--path', default = None)
def get_attribute_multi(
                config_path: str,
                save_path: str,
                path: str,
                gpu_numbers: int,
              ):
    start_time = time.time()
    print(f'config_path:{config_path}')
    print(f'save_path:{save_path}')
    print(f'face_path:{path}')
    print(f'gpu_numbers:{gpu_numbers}')

    stage_two_path = os.path.join(save_path, "pose")

    gammas, gen_length = pipeline_init(config_path, save_path, path,gpu_numbers)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f'pipeline_init:{total_time}')

    start_time = time.time()
    pose_n_workers = 1 * gpu_numbers
    gpu = 0
    start_index = None
    end_index = None
    torch.cuda.empty_cache()

    multiprocessing.spawn(get_pose_pipeline_multi, nprocs=pose_n_workers,
                          args=(gen_length, config_path, save_path, str(gpu), start_index, end_index, path, gpu_numbers,pose_n_workers))

    torch.cuda.empty_cache()

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f'get_pose:{total_time}')

    start_time = time.time()


    facial_n_workers = 2 * gpu_numbers
    multiprocessing.spawn(get_facial_pipeline_multi, nprocs=facial_n_workers,
                          args=(gen_length, config_path, save_path, str(gpu), start_index, end_index, path, gammas,gpu_numbers,facial_n_workers))
    torch.cuda.empty_cache()

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f'get_facial:{total_time}')



if __name__ == '__main__':
    get_attribute_multi()

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
                                 logger,W_PTI_pipeline_multi
import time

if __name__ == '__main__':
    start_time = time.time()

    args = sys.argv
    print(args)
    if args[-1] == 'PTI':
        print('PTI_MODE!')
        W_PTI_pipeline_multi(args[-4], args[-3], args[-2],args[-5])
    else:
        main()
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f'W_PTI_Train:{total_time}')

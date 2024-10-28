import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import re
import yaml
import click
import cv2
import tqdm

import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from functools import reduce
from easydict import EasyDict as edict

from ExpressiveEncoding.encoder import simpleEncoder, simpleEncoderV2
from ExpressiveEncoding.utils import to_tensor, from_tensor
from ExpressiveEncoding.f_space_train import StyleSpaceDecoder, stylegan_path

@click.command()
@click.option("--from_path")
@click.option("--to_path")
@click.option("--config_path")
@click.option("--model_path")
@click.option("--decoder_path", default = None)
def get_f_space(
                from_path,
                to_path,
                config_path,
                model_path,
                decoder_path
               ):

    device = "cuda:0"
    print(model_path)

    images = [os.path.join(from_path, x) for x in sorted(os.listdir(from_path), key = lambda x: int(x.split('.')[0]))]
    os.makedirs(to_path, exist_ok = True)
    
    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    decoder = StyleSpaceDecoder(stylegan_path = stylegan_path, to_resolution = 512)
    if decoder_path is not None:
        if not decoder_path.endswith('pt') and not decoder_path.endswith('pth'):
            folder = decoder_path 
            decoder_path = os.path.join(folder, sorted(os.listdir(decoder_path), key = lambda x: int(''.join(re.findall('[0-9]+', x))))[-1])
            print(f"latest weight path is {decoder_path}")
        decoder.load_state_dict(torch.load(decoder_path), False)

    net_config = config.pti.net
    encoder = simpleEncoderV2( \
                              base_filter_num = net_config.base_filter_num, \
                              source_size = net_config.source_size, \
                              target_size= net_config.target_size, \
                              target_filter_num = net_config.target_filter_num, \
                              base_code = decoder.get_base_code() \
                             )

    encoder.load_state_dict(torch.load(model_path), False)

    encoder.eval()
    encoder.to(device)
    images = tqdm(images)

    ans = []
    for x in images:
        image = ((to_tensor(np.float32(cv2.imread(x)[..., ::-1] / 255.0)) - 0.5) * 2).to(device)
        with torch.no_grad():
            f = encoder(image)       
        torch.save(
                   f.detach().cpu(), 
                   os.path.join(to_path, os.path.basename(x).split('.')[0] + '.pt')
                  )
    #    ans.append(f.detach().cpu())
    #torch.save(
    #            ans,
    #            to_path
    #          )

if __name__ == "__main__":
    get_f_space()

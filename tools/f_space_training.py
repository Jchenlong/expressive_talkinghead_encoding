"""get style space latent codes script.
"""
import os
import sys
import re
sys.path.insert(0, os.getcwd())
import tqdm
import click
import torch
import time
import hashlib
from tqdm import tqdm


import torch.multiprocessing as mp
#mp.set_start_method('spawn')

from ExpressiveEncoding.f_space_train import bdinv_training, StyleSpaceDecoder, \
                                             stylegan_path, edict, yaml, \
                                             logger

from ExpressiveEncoding.kmeans_dataset import kmeans_data

def kernel(
           rank, 
           world_size,
           config,
           snapshots,
           decoder,
           tensorboard,
           resolution,
           resume_path
          ):
    
    return bdinv_training(
                          config.gt_path, \
                          config.latent_path, \
                          snapshots, \
                          decoder, \
                          config.pti, \
                          tensorboard = tensorboard, \
                          epochs = config.epochs, \
                          resolution = resolution, \
                          resume_path = resume_path, \
                          batchsize = config.batchsize, \
                          lr = config.lr, \
                          rank = rank, \
                          world_size = world_size \
                         )
@click.command()
@click.option('--gt_path')
@click.option('--facial_path')
@click.option('--config_path')
@click.option('--save_path')
@click.option('--decoder_path', default = None)
@click.option('--resume_path', default = None)
@click.option('--gpus', default = 1)
def bdinv_training_invoker(
                    gt_path: str,
                    facial_path: str,
                    config_path: str,
                    save_path: str,
                    decoder_path: str,
                    resume_path: str,
                    gpus: int
                  ):

    start_time = time.time()
    assert gpus >= 1, "expected gpu device."
    tensorboard = os.path.join(save_path, "tensorboard", f"{time.time()}")
    snapshots = os.path.join(save_path, "snapshots")
    os.makedirs(snapshots, exist_ok = True)

    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))
    resolution = config.resolution if hasattr(config, "resolution") else 1024
    decoder = StyleSpaceDecoder(stylegan_path = stylegan_path, to_resolution = resolution)

    config.gt_path = gt_path
    config.latent_path = facial_path
    print(config_path)
    print(config.gt_path)
    print(config.latent_path)

    use_kmeans = config.pti.use_kmeans if hasattr(config.pti, "use_kmeans") else False
    if use_kmeans:
        hash_string = hashlib.md5(config.latent_path.encode(encoding='UTF-8')).hexdigest()
        path = os.path.join("tmp_dir", str(hash_string) + ".pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kmeans_info = kmeans_data(config.latent_path, path)
        setattr(config.pti, "kmeans_info", kmeans_info)

=======
    print(config.gt_path)
    print(config.latent_path)

>>>>>>> origin/feature/train_refactor_162_train_stitch

    if decoder_path is not None:
        if not decoder_path.endswith('pt') and not decoder_path.endswith('pth'):
            folder = decoder_path 
            decoder_path = os.path.join(folder, sorted(os.listdir(decoder_path), key = lambda x: int(''.join(re.findall('[0-9]+', x))))[-1])
            print(f"latest weight path is {decoder_path}")
        decoder.load_state_dict(torch.load(decoder_path), False)


    if gpus <= 1:
        bdinv_training(
                       config.gt_path, \
                       config.latent_path, \
                       snapshots, \
                       decoder, \
                       config.pti, \
                       tensorboard = tensorboard, \
                       epochs = config.epochs, \
                       resolution = resolution, \
                       resume_path = resume_path, \
                       batchsize = config.batchsize, \
                       lr = config.lr \
                      )
    else:
        world_size = gpus
        mp.spawn(
                 kernel,
                 args=(
                        world_size,
                        config,
                        snapshots,
                        decoder,
                        tensorboard,
                        resolution,
                        resume_path
                        #writer
                      ),
                 nprocs=world_size,
                 join=True
                )
                                  
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f'feature_encoder_train:{total_time}')
if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    bdinv_training_invoker()

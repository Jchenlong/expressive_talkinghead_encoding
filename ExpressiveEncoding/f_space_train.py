""" BDInv training code
"""
import os
import sys
sys.path.insert(0, os.getcwd())


import torch
import re
import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from functools import reduce

from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .ImagesDataset import ImagesDataset, ImagesDatasetF, ImagesDatasetHasMask
from .loss import LossRegisterBase
from .decoder import StyleSpaceDecoder
from .encoder import simpleEncoder, simpleEncoderV2
from .train import logger, stylegan_path, edict, yaml
with open(os.path.join("/data1/wanghaoran/Amemori", "template.yaml")) as f:
    config = yaml.load(f, Loader = yaml.CLoader)

regions = eval(config["soft_mask_region"])
output_copy_region = eval(config["output_copy_region"])

def get_soft_mask_by_region():
    soft_mask  = np.zeros((512,512,3), np.float32)
    for region in regions:
        y1,y2,x1,x2 = region
        soft_mask[y1:512,x1:x2,:]=1
    #soft_mask = cv2.GaussianBlur(soft_mask, (101, 101), 11)
    soft_mask = soft_mask.astype(np.float32)
    return soft_mask

def bdinv_training(
                   path_images: str,
                   path_style_latents: str,
                   path_snapshots: str,
                   ss_decoder: object,
                   config: edict,
                   **kwargs
                  ):
    
    resolution = kwargs.get("resolution", 1024)
    batchsize = kwargs.get("batchsize", 1)
    lr = kwargs.get("lr", 3e-4)
    resume_path = kwargs.get("resume_path", None)
    rank = kwargs.get("rank", -1)
    world_size = kwargs.get("world_size", 0)
    device = "cuda:0"

    if rank != -1:
        device = rank
        dist.init_process_group("nccl", rank=rank, world_size=world_size) 
        torch.cuda.set_device(rank)

    def get_dataloader(
                      ):
    
        dataset = ImagesDatasetHasMask(path_images, path_style_latents, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize(size = (resolution, resolution))]),
            )

        if rank != -1:
            batch_size = batchsize // world_size
            return DataLoader(
                              dataset, batch_size = batch_size, \
                              num_workers = min(batchsize, 1),  \
                              #num_workers = 1,  \ 
                              sampler = DistributedSampler(dataset, shuffle = False, rank = rank, num_replicas = world_size, drop_last = True), \
                              pin_memory=True
                             )
        else:
            return DataLoader(
                              dataset, batch_size = batchsize, \
                              shuffle = False, \
                              num_workers = min(batchsize, 1), drop_last = True
                             )
    
    class PivotLossRegister(LossRegisterBase):
        
        def forward(self, 
                    x,
                    y
                   ):
            l2 = self.l2(x,y).mean() * self.l2_weight
            lpips = self.lpips(x,y).mean() * self.lpips_weight
        
            return {
                    "l2": l2,
                    "lpips": lpips
                   }

    loss_register = PivotLossRegister(config) 
    loss_register.lpips.set_device(device)
    dataloader = get_dataloader()
    net = config.net
    name = config.net.name if hasattr(config.net, "name") else "simpleEncoder"
    encoder =   eval(name)(
                            base_filter_num = net.base_filter_num, \
                            source_size = net.source_size, \
                            target_size= net.target_size, \
                            target_filter_num = net.target_filter_num,
                            base_code = ss_decoder.get_base_code().detach() if "V2" in name else None,
                            norm = config.net.norm if hasattr(config.net, "norm") else "BatchNorm2d",
                            res = config.net.res if hasattr(config.net, "res") else False
                          )
    
    logger.info(f"{name}: {encoder}")

    if resume_path is not None:

        if not resume_path.endswith('pt') and not resume_path.endswith('pth'):
            resume_path = int(''.join(re.findall('[0-9]+', os.path.basename(resume_path))))
            logger.info(f"resume from {epoch_from_resume}...")
            start_idx = epoch_from_resume + 1
            total_idx = epoch_from_resume * len(dataloader)
        encoder.load_state_dict(torch.load(resume_path))

    encoder.train()

    for p in encoder.parameters():
        p.requires_grad = True

    for p in ss_decoder.parameters():
        p.requires_grad = False
    
    ss_decoder.to(device)

    encoder.to(device)
    optim = torch.optim.Adam(encoder.parameters(), lr = lr)

    lastest_model_path = None
    start_idx = 1
    if resume_path is not None:
        if not resume_path.endswith('pt') and not resume_path.endswith('pth'):
            resume_path = int(''.join(re.findall('[0-9]+', os.path.basename(resume_path))))
            logger.info(f"resume from {epoch_from_resume}...")
            start_idx = epoch_from_resume + 1
            total_idx = epoch_from_resume * len(dataloader)
        encoder.load_state_dict(torch.load(resume_path))

    if rank != -1:
        encoder = DDP(encoder, device_ids = [rank], find_unused_parameters=True)

    #optim = torch.optim.Adam(parameters, lr = lr)
    total_idx = 0
    epochs = kwargs.get("epochs", 100)
    tensorboard = kwargs.get("tensorboard", None)
    writer = None
    if tensorboard is not None and (rank == 0 or rank == -1):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(tensorboard)

    save_interval = kwargs.get("save_interval", 100)
    epoch_pbar = tqdm(range(start_idx, epochs + 1))

    min_loss = 0xffff # max value.
    internal_size = len(dataloader) // 5
    if internal_size <= 0:
        internal_size = 1
    for epoch in epoch_pbar:
        if rank == 0 or rank == -1:
            logger.info(f"internal_size is {internal_size}.")
            epoch_pbar.update(1)
        sample_loss = 0
        sample_count = 0
        for idx, (image, pivot, _, mask) in enumerate(dataloader):
        
            pivot = [x.to(device) for x in pivot]
            image = image.to(device)  
            mask = mask.to(device)
            f = encoder(image)

            image_gen = ss_decoder(pivot, insert_feature = {"4": f})

            ret = loss_register(image * mask, image_gen * mask, is_gradient = False)
            loss = ret['loss']

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_idx += 1
            if idx % internal_size == 0 and (rank == 0 or rank == -1):
                sample_loss += loss.mean()
                sample_count += 1
                string_to_info = reduce(lambda x, y: x + ', ' + y , [f'{k} {v.mean().item()}' for k, v in ret.items()])
                logger.info(f"{idx+1}/{epoch}/{epochs}: {string_to_info}")

                if writer is not None:
                    image_to_show = torch.cat((image_gen, image),dim = 2)
                    writer.add_image('image', make_grid(image_to_show.detach(), normalize=True, scale_each=True), total_idx)
                    writer.add_scalars('loss', ret, total_idx)

        if (rank == 0 or rank == -1):
            sample_loss /= sample_count
            if sample_loss < min_loss:
                lastest_model_path = os.path.join(path_snapshots, f"{epoch}.pth")
                torch.save(encoder.state_dict() if rank == -1 else encoder.module.state_dict(), lastest_model_path)
                min_loss = sample_loss
                logger.info(f"min_loss: {min_loss}, epoch {epoch}")

    if rank == 0 or rank == -1:
        import shutil
        shutil.copyfile(lastest_model_path, os.path.join(os.path.dirname(lastest_model_path), "best.pth"))
        logger.info(f"training finished; the lastet snapshot saved in {lastest_model_path}")
        writer.close()
        
    return lastest_model_path

def bdinv_detailed_training(
                            path_images: str,
                            path_style_latents: str,
                            path_snapshots: str,
                            ss_decoder: object,
                            config: edict,
                            **kwargs
                           ):
    
    resolution = kwargs.get("resolution", 1024)
    batchsize = kwargs.get("batchsize", 1)
    lr = kwargs.get("lr", 3e-4)
    resume_path = kwargs.get("resume_path", None)
    encoder_resume_path = kwargs.get("encoder_resume_path", None)
    rank = kwargs.get("rank", -1)
    world_size = kwargs.get("world_size", 0)
    device = "cuda:0"

    if rank != -1:
        device = rank
        dist.init_process_group("nccl", rank=rank, world_size=world_size) 
        torch.cuda.set_device(rank)

    def get_dataloader(
                      ):
    
        dataset = ImagesDataset(path_images, path_style_latents, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize(size = (resolution, resolution))]),
            )

        if rank != -1:
            batch_size = batchsize // world_size
            return DataLoader(
                              dataset, batch_size = batch_size, \
                              num_workers = min(batchsize, 1),  \
                              #num_workers = 1,  \
                              sampler = DistributedSampler(dataset, shuffle = False, rank = rank, num_replicas = world_size, drop_last = False), \
                              pin_memory=True
                             )
        else:
            return DataLoader(
                              dataset, batch_size = batchsize, \
                              shuffle = False, \
                              num_workers = min(batchsize, 1), drop_last = True
                             )
    
    class PivotLossRegister(LossRegisterBase):
        
        def forward(self, 
                    x,
                    y,
                    residual
                   ):
            l2 = self.l2(x,y).mean() * self.l2_weight
            lpips = self.lpips(x,y).mean() * self.lpips_weight
            reg_residual = (residual ** 2).mean() * 0.0

            return {
                    "l2": l2,
                    "lpips": lpips,
                    "reg_residual": reg_residual
                   }

    loss_register = PivotLossRegister(config) 
    loss_register.lpips.set_device(device)
    dataloader = get_dataloader()
    net = config.net
    encoder = simpleEncoder(
                            base_filter_num = net.base_filter_num, \
                            source_size = net.source_size, \
                            target_size= net.target_size, \
                            target_filter_num = net.target_filter_num
                           )
    encoder.train()

    for p in encoder.parameters():
        p.requires_grad = False

    for p in ss_decoder.parameters():
        p.requires_grad = False
    
    ss_decoder.to(device)
    #if rank != -1:
    #    ss_decoder = DDP(ss_decoder, device_ids = [rank], find_unused_parameters=True)

    encoder.to(device)
    if encoder_resume_path is not None:
        if not encoder_resume_path.endswith('pt') and not encoder_resume_path.endswith('pth'):
            encoder_resume_path = int(''.join(re.findall('[0-9]+', os.path.basename(encoder_resume_path))))
            logger.info(f"encoder_resume from {encoder_resume_path}...")
            start_idx = epoch_from_encoder_resume + 1
            total_idx = epoch_from_encoder_resume * len(dataloader)
        encoder.load_state_dict(torch.load(encoder_resume_path))

    start_idx = 1
    total_idx = 0
    epochs = kwargs.get("epochs", 100)
    tensorboard = kwargs.get("tensorboard", None)
    writer = None
    if tensorboard is not None and (rank == 0 or rank == -1):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(tensorboard)

    save_interval = kwargs.get("save_interval", 100)
    dataset_pbar = tqdm(dataloader, desc=f'training {rank} ....', leave = False)

    min_loss = 0xffff # max value.
    #internal_size = len(dataloader) // 5
    #if internal_size <= 0:
    #    internal_size = 1
    internal_size = 100
    partial_mask = torch.from_numpy(get_soft_mask_by_region()).permute((2, 0, 1)).unsqueeze(0).to(device)

    for (image, pivot, idx) in dataset_pbar:
        pivot = [x.to(device) for x in pivot]
        image = image.to(device)  
        if resume_path is not None:
            f = torch.load(os.path.join(resume_path, f'{idx.item()}.pt'), map_location = 'cpu').to(device)
        else:
            with torch.no_grad():
                f = encoder(image)
        for x in pivot:
            x.requires_grad = True
        #f.requires_grad = True
        residual = torch.randn_like(f).to(device)
        residual.requires_grad = True
        optim = torch.optim.Adam([residual] + pivot, lr = lr)
        sche = torch.optim.lr_scheduler.StepLR(optim, step_size=250, gamma=0.5)
        for epoch in range(1, epochs + 1):
            sample_loss = 0
            sample_count = 0
            
            image_gen = ss_decoder(pivot, insert_feature = {"4": f + residual})
            n = image_gen.shape[0]
            ret = loss_register(image, image_gen, residual, is_gradient = False)
            loss = ret['loss']
            optim.zero_grad()
            loss.backward()
            optim.step()
            sche.step()
            total_idx += 1
            if epoch % internal_size == 0 and (rank == 0 or rank == -1):
                sample_loss += loss.mean()
                sample_count += 1
                string_to_info = reduce(lambda x, y: x + ', ' + y , [f'{k} {v.mean().item()}' for k, v in ret.items()])
                logger.info(f"{idx.item()+1}/{epoch}/{epochs}: {string_to_info}")

                if writer is not None:
                    image_to_show = torch.cat((image_gen, image, image_gen * partial_mask + image * (1 - partial_mask)),dim = 2)
                    writer.add_image(f'image_{idx.item()}', make_grid(image_to_show.detach(),normalize=True, scale_each=True), total_idx)
                    writer.add_scalars(f'loss_{idx.item()}', ret, total_idx)
        model_path = os.path.join(path_snapshots, f"{idx.item()}.pt")
        torch.save(
                    dict(
                         f = f + (residual).detach(),
                         pivot = [x.detach() for x in pivot]
                        ),
                   model_path
                  )
        """
        if (rank == 0 or rank == -1):
            sample_loss /= sample_count
            if sample_loss < min_loss:
                lastest_model_path = os.path.join(path_snapshots, f"{epoch}.pth")
                torch.save(ss_decoder.state_dict() if rank == -1 else ss_decoder.module.state_dict(), lastest_model_path)
                min_loss = sample_loss
                logger.info(f"min_loss: {min_loss}, epoch {epoch}")
        """

    if rank == 0 or rank == -1:
        writer.close()
    
def f_space_training(
                     path_images: str,
                     path_style_latents: str,
                     path_f_latents: str,
                     path_snapshots: str,
                     ss_decoder: object,
                     config: edict,
                     **kwargs
                    ):
    
    resolution = kwargs.get("resolution", 1024)
    batchsize = kwargs.get("batchsize", 1)
    lr = kwargs.get("lr", 3e-4)
    resume_path = kwargs.get("resume_path", None)
    rank = kwargs.get("rank", -1)
    world_size = kwargs.get("world_size", 0)
    device = "cuda:0"

    if rank != -1:
        device = rank
        dist.init_process_group("nccl", rank=rank, world_size=world_size) 
        torch.cuda.set_device(rank)

    def get_dataloader(
                      ):
    
        dataset = ImagesDatasetF(path_images, path_style_latents, path_f_latents, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize(size = (resolution, resolution))]),
            )

        if rank != -1:
            batch_size = batchsize // world_size
            return DataLoader(
                              dataset, batch_size = batch_size, \
                              num_workers = min(batchsize, 1),  \
                              #num_workers = 1,  \
                              sampler = DistributedSampler(dataset, shuffle = False, rank = rank, num_replicas = world_size, drop_last = False), \
                              pin_memory=True
                             )
        else:
            return DataLoader(
                              dataset, batch_size = batchsize, \
                              shuffle = False, \
                              num_workers = min(batchsize, 1), drop_last = True
                             )
    
    class PivotLossRegister(LossRegisterBase):
        
        def forward(
                    self, 
                    x,
                    y
                   ):
            l2 = self.l2(x,y).mean() * self.l2_weight
            lpips = self.lpips(x,y).mean() * self.lpips_weight
            return {
                    "l2": l2,
                    "lpips": lpips
                   }

    loss_register = PivotLossRegister(config) 
    loss_register.lpips.set_device(device)
    dataloader = get_dataloader()

    for p in ss_decoder.parameters():
        p.requires_grad = True
    
    ss_decoder.to(device)
    if rank != -1:
        ss_decoder = DDP(ss_decoder, device_ids = [rank], find_unused_parameters=True)

    start_idx = 1
    total_idx = 0
    epochs = kwargs.get("epochs", 100)
    tensorboard = kwargs.get("tensorboard", None)
    writer = None
    if tensorboard is not None and (rank == 0 or rank == -1):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(tensorboard)

    save_interval = kwargs.get("save_interval", 100)

    min_loss = 0xffff # max value.
    #internal_size = len(dataloader) // 5
    #if internal_size <= 0:
    #    internal_size = 1
    internal_size = 10
    partial_mask = torch.from_numpy(get_soft_mask_by_region()).permute((2, 0, 1)).unsqueeze(0).to(device)
    optim = torch.optim.Adam(ss_decoder.parameters(), lr = lr)

    pbar = tqdm(range(1, epochs + 1))

    for epoch in pbar:
        sample_loss = 0
        sample_count = 0
            
        for idx, (image, pivot, f) in enumerate(dataloader):
            pivot = [x.to(device) for x in pivot]
            image = image.to(device)  
            image_gen = ss_decoder(pivot, insert_feature = {"4": f})
            n = image_gen.shape[0]
            ret = loss_register(image, image_gen, is_gradient = False)
            loss = ret['loss']
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_idx += 1
            if idx % internal_size == 0 and (rank == 0 or rank == -1):
                sample_loss += loss.mean()
                sample_count += 1
                string_to_info = reduce(lambda x, y: x + ', ' + y , [f'{k} {v.mean().item()}' for k, v in ret.items()])
                logger.info(f"{idx+1}/{epoch}/{epochs}: {string_to_info}")

                if writer is not None:
                    image_to_show = torch.cat((image_gen, image, image_gen * partial_mask + image * (1 - partial_mask)),dim = 2)
                    writer.add_image(f'image', make_grid(image_to_show.detach(),normalize=True, scale_each=True), total_idx)
                    writer.add_scalars(f'loss', ret, total_idx)
        if (rank == 0 or rank == -1):
            sample_loss /= sample_count
            if sample_loss < min_loss:
                lastest_model_path = os.path.join(path_snapshots, f"{epoch}.pth")
                torch.save(ss_decoder.state_dict() if rank == -1 else ss_decoder.module.state_dict(), lastest_model_path)
                min_loss = sample_loss
                logger.info(f"min_loss: {min_loss}, epoch {epoch}")

    if rank == 0 or rank == -1:
        writer.close()

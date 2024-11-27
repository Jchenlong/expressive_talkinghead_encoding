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
from .train import logger, stylegan_path, edict, yaml,BiSeNet
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
                              num_workers = min(batchsize, 8),  \
                              #num_workers = 1,  \ 
                              sampler = DistributedSampler(dataset, shuffle = False, rank = rank, num_replicas = world_size, drop_last = True), \
                              pin_memory=True
                             )
        else:
            return DataLoader(
                              dataset, batch_size = batchsize, \
                              shuffle = False, \
                              num_workers = min(batchsize, 8), drop_last = True
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

import ExpressiveEncoding.seg_model_2
from torch.nn import functional as F
def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)
from typing import List, Optional
def _neight2channels_like_kernel(kernel: torch.Tensor) -> torch.Tensor:
    h, w = kernel.size()
    kernel = torch.eye(h * w, dtype=kernel.dtype, device=kernel.device)
    return kernel.view(h * w, 1, h, w)
def dilation(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = 'unfold',
) -> torch.Tensor:
    r"""Return the dilated image applying the same kernel in each channel.

    .. image:: _static/img/dilation.png

    The kernel must have 2 dimensions.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a non-flat
            structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: convolution is faster and less memory hungry, and unfold is more stable numerically

    Returns:
        Dilated image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> dilated_img = dilation(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == 'geodesic':
        border_value = -max_val
        border_type = 'constant'
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    if engine == 'unfold':
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output, _ = torch.max(output + neighborhood.flip((0, 1)), 4)
        output, _ = torch.max(output, 4)
    elif engine == 'convolution':
        B, C, H, W = tensor.size()
        h_pad, w_pad = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel)
        output, _ = F.conv2d(
            output.view(B * C, 1, h_pad, w_pad), reshape_kernel, padding=0, bias=neighborhood.view(-1).flip(0)
        ).max(dim=1)
        output = output.view(B, C, H, W)
    else:
        raise NotImplementedError(f"engine {engine} is unknown, use 'convolution' or 'unfold'")
    return output.view_as(tensor)

def create_masks(border_pixels, mask, inner_dilation=0, outer_dilation=0, whole_image_border=False):
    image_size = mask.shape[2]
    grid = torch.cartesian_prod(torch.arange(image_size), torch.arange(image_size)).view(image_size, image_size,
                                                                                         2).cuda()
    image_border_mask = logical_or_reduce(
        grid[:, :, 0] < border_pixels,
        grid[:, :, 1] < border_pixels,
        grid[:, :, 0] >= image_size - border_pixels,
        grid[:, :, 1] >= image_size - border_pixels
    )[None, None].expand_as(mask)

    temp = mask
    if inner_dilation != 0:
        temp = dilation(temp, torch.ones(2 * inner_dilation + 1, 2 * inner_dilation + 1, device=mask.device),
                        engine='convolution')

    border_mask = torch.min(image_border_mask, temp)
    full_mask = dilation(temp, torch.ones(2 * outer_dilation + 1, 2 * outer_dilation + 1, device=mask.device),
                         engine='convolution')
    if whole_image_border:
        border_mask_2 = 1 - temp
    else:
        border_mask_2 = full_mask - temp
    border_mask = torch.maximum(border_mask, border_mask_2)

    border_mask = border_mask.clip(0, 1)
    content_mask = (mask - border_mask).clip(0, 1)
    return content_mask, border_mask, full_mask

def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)
def calc_masks(inversion, segmentation_model, border_pixels, inner_mask_dilation, outer_mask_dilation,
               whole_image_border):
    background_classes = [0, 18, 16]
    inversion_resized = torch.cat([F.interpolate(inversion, (512, 512), mode='nearest')])
    inversion_normalized = transforms.functional.normalize(inversion_resized.clip(-1, 1).add(1).div(2),
                                                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    segmentation = segmentation_model(inversion_normalized)[0].argmax(dim=1, keepdim=True)
    is_foreground = logical_and_reduce(*[segmentation != cls for cls in background_classes])
    foreground_mask = is_foreground.float()
    content_mask, border_mask, full_mask = create_masks(border_pixels // 2, foreground_mask, inner_mask_dilation // 2,
                                                        outer_mask_dilation // 2, whole_image_border)
    content_mask = F.interpolate(content_mask, (512, 512), mode='bilinear', align_corners=True)
    border_mask = F.interpolate(border_mask, (512, 512), mode='bilinear', align_corners=True)
    full_mask = F.interpolate(full_mask, (512, 512), mode='bilinear', align_corners=True)
    return content_mask, border_mask, full_mask



def bdinv_training_stitch(
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

        dataset = ImagesDataset(path_images, path_style_latents, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize(size=(resolution, resolution))]),
                                       )

        if rank != -1:
            batch_size = batchsize // world_size
            return DataLoader(
                dataset, batch_size=batch_size, \
                num_workers=min(batchsize, 1), \
                # num_workers = 1,  \
                sampler=DistributedSampler(dataset, shuffle=False, rank=rank, num_replicas=world_size, drop_last=True), \
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset, batch_size=batchsize, \
                shuffle=False, \
                num_workers=min(batchsize, 1), drop_last=True
            )

    class PivotLossRegister(LossRegisterBase):

        def forward(self,
                    x,
                    y
                    ):
            l2 = self.l2(x, y).mean() * self.l2_weight
            lpips = self.lpips(x, y).mean() * self.lpips_weight

            return {
                "l2": l2,
                "lpips": lpips
            }

    loss_register = PivotLossRegister(config)
    loss_register.lpips.set_device(device)
    dataloader = get_dataloader()
    net = config.net
    name = config.net.name if hasattr(config.net, "name") else "simpleEncoder"
    encoder = eval(name)(
        base_filter_num=net.base_filter_num, \
        source_size=net.source_size, \
        target_size=net.target_size, \
        target_filter_num=net.target_filter_num,
        base_code=ss_decoder.get_base_code().detach() if "V2" in name else None,
        norm=config.net.norm if hasattr(config.net, "norm") else "BatchNorm2d",
        res=config.net.res if hasattr(config.net, "res") else False
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
    optim = torch.optim.Adam(encoder.parameters(), lr=lr)

    lastest_model_path = None
    start_idx = 1
    if resume_path is not None:
        if not resume_path.endswith('pt') and not resume_path.endswith('pth'):
            resume_path = int(''.join(re.findall('[0-9]+', os.path.basename(resume_path))))
            logger.info(f"resume from {epoch_from_resume}...")
            start_idx = epoch_from_resume + 1
            total_idx = epoch_from_resume * len(dataloader)
        encoder.load_state_dict(torch.load(resume_path))

    segmentation_model = ExpressiveEncoding.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    segmentation_model.load_state_dict(torch.load(
        '/data1/chenlong/github/v63/expressive_talkinghead_encoding/ExpressiveEncoding/third_party/models/79999_iter.pth'))
    segmentation_model.to(device)
    if rank != -1:
        encoder = DDP(encoder, device_ids=[rank], find_unused_parameters=True)

    # optim = torch.optim.Adam(parameters, lr = lr)
    total_idx = 0
    epochs = kwargs.get("epochs", 100)
    tensorboard = kwargs.get("tensorboard", None)
    writer = None
    if tensorboard is not None and (rank == 0 or rank == -1):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(tensorboard)

    save_interval = kwargs.get("save_interval", 100)
    epoch_pbar = tqdm(range(start_idx, epochs + 1))

    min_loss = 0xffff  # max value.
    internal_size = len(dataloader) // 5
    if internal_size <= 0:
        internal_size = 1
    for epoch in epoch_pbar:
        if rank == 0 or rank == -1:
            logger.info(f"internal_size is {internal_size}.")
            epoch_pbar.update(1)
        sample_loss = 0
        sample_count = 0
        for idx, (image, pivot,index) in enumerate(dataloader):

            pivot = [x.to(device) for x in pivot]
            image = image.to(device)
            f = encoder(image)
            content_mask, border_mask, full_mask = calc_masks(image, segmentation_model, 50, 0, 50, False)

            image_gen = ss_decoder(pivot, insert_feature={"4": f})

            border_loss = masked_l2(image_gen, image, content_mask, True)
            loss = border_loss + 0.1 * masked_l2(image_gen, image, border_mask, True)

            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_dict = {}
            loss_dict['loss'] = loss
            total_idx += 1
            if idx % internal_size == 0 and (rank == 0 or rank == -1):
                sample_loss += loss.mean()
                sample_count += 1
                # string_to_info = reduce(lambda x, y: x + ', ' + y, [f'{k} {v.mean().item()}' for k, v in ret.items()])
                # logger.info(f"{idx + 1}/{epoch}/{epochs}: {string_to_info}")

                if writer is not None:
                    # image_to_show = torch.cat((image_gen, image), dim=2)
                    image_to_show = torch.cat((image_gen, image, (image_gen * border_mask + image_gen * (1 - border_mask)),
                                               border_mask.repeat(1, 3, 1, 1), content_mask.repeat(1, 3, 1, 1)), dim=2)
                    writer.add_image('image', make_grid(image_to_show.detach(), normalize=True, scale_each=True),
                                     total_idx)
                    writer.add_scalars('loss', loss_dict, total_idx)

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
                              num_workers = min(batchsize, 8),  \
                              #num_workers = 1,  \
                              sampler = DistributedSampler(dataset, shuffle = False, rank = rank, num_replicas = world_size, drop_last = False), \
                              pin_memory=True
                             )
        else:
            return DataLoader(
                              dataset, batch_size = batchsize, \
                              shuffle = False, \
                              num_workers = min(batchsize, 8), drop_last = True
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


def bdinv_detailed_training_stitch(
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
            transforms.Resize(size=(resolution, resolution))]),
                                )

        if rank != -1:
            batch_size = batchsize // world_size
            return DataLoader(
                dataset, batch_size=batch_size, \
                num_workers=min(batchsize, 1), \
                # num_workers = 1,  \
                sampler=DistributedSampler(dataset, shuffle=False, rank=rank, num_replicas=world_size, drop_last=False), \
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset, batch_size=batchsize, \
                shuffle=False, \
                num_workers=min(batchsize, 1), drop_last=True
            )


    class PivotLossRegister(LossRegisterBase):

        def forward(self,
                    x,
                    y,
                    residual,
                    content_mask,
                    border_mask,
                    ):
            l2 = self.l2(x, y).mean() * self.l2_weight
            lpips = self.lpips(x, y).mean() * self.lpips_weight

            loss_mouth = (x[..., 274:494, 150:-150] - y[..., 274:494,
                                                                      150:-150]).square().mean() * 10
            loss_lpips_mouth = self.lpips(x[..., 274:494, 150:-150],
                                          y[..., 274:494, 150:-150]).mean() * 4

            return {
                "l2": l2,
                "lpips": lpips,
                "loss_mouth": loss_mouth,
                "loss_lpips_mouth": loss_lpips_mouth,
            }

    loss_register = PivotLossRegister(config)
    loss_register.lpips.set_device(device)
    dataloader = get_dataloader()
    net = config.net
    encoder = simpleEncoder(
        base_filter_num=net.base_filter_num, \
        source_size=net.source_size, \
        target_size=net.target_size, \
        target_filter_num=net.target_filter_num
    )
    encoder.train()

    for p in encoder.parameters():
        p.requires_grad = False

    for p in ss_decoder.parameters():
        p.requires_grad = False

    ss_decoder.to(device)
    # if rank != -1:
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
    dataset_pbar = tqdm(dataloader, desc=f'training {rank} ....', leave=False)

    min_loss = 0xffff  # max value.
    # internal_size = len(dataloader) // 5
    # if internal_size <= 0:
    #    internal_size = 1
    internal_size = 100
    partial_mask = torch.from_numpy(get_soft_mask_by_region()).permute((2, 0, 1)).unsqueeze(0).to(device)

    segmentation_model = ExpressiveEncoding.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    segmentation_model.load_state_dict(torch.load(
        '/data1/chenlong/github/v63/expressive_talkinghead_encoding/ExpressiveEncoding/third_party/models/79999_iter.pth'))
    segmentation_model.to(device)
    residual_pre = None
    index_img = -1
    for (image, pivot, idx) in dataset_pbar:
        index_img += 1
        if index_img < 7:
            continue
        pivot = [x.to(device) for x in pivot]
        image = image.to(device)
        gen_image = image.to(device).clone()
        if resume_path is not None:
            f = torch.load(os.path.join(resume_path, f'{idx.item()}.pt'), map_location='cpu').to(device)
        else:
            with torch.no_grad():
                f = encoder(image)
        for x in pivot:
            x.requires_grad = True
        # f.requires_grad = True
        residual = torch.randn_like(f).to(device)
        if residual_pre is not None:
            residual = residual_pre
            epochs = kwargs.get("epochs", 100)
        else:
            epochs = 1000
        residual.requires_grad = True
        # optim = torch.optim.Adam([residual] + pivot, lr=lr)
        optim = torch.optim.Adam([residual], lr=lr)
        sche = torch.optim.lr_scheduler.StepLR(optim, step_size=250, gamma=0.5)

        content_mask, border_mask, full_mask = calc_masks(image, segmentation_model, 50,  0, 50, False)

        for epoch in range(1, epochs + 1):
            sample_loss = 0
            sample_count = 0

            image_gen = ss_decoder(pivot, insert_feature={"4": f + residual})
            n = image_gen.shape[0]
            ret = loss_register(image, image_gen, residual,content_mask, border_mask,is_gradient=False)
            loss = ret['loss']

            optim.zero_grad()
            loss.backward()
            optim.step()
            sche.step()
            total_idx += 1
            if epoch % internal_size == 0 and (rank == 0 or rank == -1):
                sample_loss += loss.mean()
                sample_count += 1
                string_to_info = reduce(lambda x, y: x + ', ' + y, [f'{k} {v.mean().item()}' for k, v in ret.items()])
                logger.info(f"{idx.item() + 1}/{epoch}/{epochs}: {string_to_info}")

                if writer is not None:
                    image_to_show = torch.cat((image_gen, image, image_gen * partial_mask + image * (1 - partial_mask)),
                                              dim=2)
                    writer.add_image(f'image_{idx.item()}',
                                     make_grid(image_to_show.detach(), normalize=True, scale_each=True), total_idx)
                    writer.add_scalars(f'loss_{idx.item()}', ret, total_idx)
            loss_dict = {}
            loss_dict['loss'] = loss
            if writer is not None:
                image_to_show = torch.cat((image_gen, image, (image_gen * border_mask + image_gen * (1 - border_mask)),
                                           border_mask.repeat(1, 3, 1, 1), content_mask.repeat(1, 3, 1, 1)), dim=2)
                writer.add_image(f'image_{idx.item()}', make_grid(image_to_show.detach(), normalize=True, scale_each=True), total_idx)
                writer.add_scalars(f'loss', ret, total_idx)

        model_path = os.path.join(path_snapshots, f"{idx.item()}.pt")
        torch.save(
            dict(
                f=f + (residual).detach(),
                pivot=[x.detach() for x in pivot]
            ),
            model_path
        )
        residual_pre = residual
        exit()
            # writer.add_scalars(f'loss_{idx.item()}', loss, total_idx)
        """
        if (rank == 0 or rank == -1):
            sample_loss /= sample_count
            if sample_loss < min_loss:
                lastest_model_path = os.path.join(path_snapshots, f"{epoch}.pth")
                torch.save(ss_decoder.state_dict() if rank == -1 else ss_decoder.module.state_dict(), lastest_model_path)
                min_loss = sample_loss
                logger.info(f"min_loss: {min_loss}, epoch {epoch}")
        """
    # if rank == 0 or rank == -1:
    #     writer.close()


def bdinv_detailed_training_previous(
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
            transforms.Resize(size=(resolution, resolution))]),
                                )

        return DataLoader(
                dataset, batch_size=batchsize, \
                shuffle=False, \
                num_workers=min(batchsize, 1), drop_last=True)


    class PivotLossRegister(LossRegisterBase):

        def forward(self,
                    x,
                    y,
                    residual,
                    content_mask,
                    border_mask,
                    ):
            # l2 = self.l2(x, y).mean() * self.l2_weight
            lpips = self.lpips(x*content_mask, y*content_mask).mean() * self.lpips_weight
            border_loss = masked_l2(x, y, content_mask, True)
            loss = border_loss + 0.1 * masked_l2(x, y, border_mask, True)

            return {
                "l2": loss,
                "lpips": lpips,
            }

    loss_register = PivotLossRegister(config)
    loss_register.lpips.set_device(device)
    dataloader = get_dataloader()
    net = config.net

    for p in ss_decoder.parameters():
        p.requires_grad = False

    ss_decoder.to(device)

    start_idx = 1
    total_idx = 0
    epochs = kwargs.get("epochs", 100)
    tensorboard = kwargs.get("tensorboard", None)
    writer = None
    # if tensorboard is not None and (rank == 0 or rank == -1):
    #     from tensorboardX import SummaryWriter
    #     writer = SummaryWriter(tensorboard)

    save_interval = kwargs.get("save_interval", 100)
    dataset_pbar = tqdm(dataloader, desc=f'training {rank} ....', leave=False)

    min_loss = 0xffff  # max value.
    # internal_size = len(dataloader) // 5
    # if internal_size <= 0:
    #    internal_size = 1
    internal_size = 100
    # partial_mask = torch.from_numpy(get_soft_mask_by_region()).permute((2, 0, 1)).unsqueeze(0).to(device)

    segmentation_model = ExpressiveEncoding.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    segmentation_model.load_state_dict(torch.load(
        '/data1/chenlong/github/v63/expressive_talkinghead_encoding/ExpressiveEncoding/third_party/models/79999_iter.pth'))
    segmentation_model.to(device)

    gpu_numbers = 4
    gpu = int(rank)
    gen_length = int(len(os.listdir(path_images)))
    start_index = gpu * (gen_length // gpu_numbers)
    end_index = (gpu + 1) * (gen_length // gpu_numbers)
    if gpu == (gpu_numbers - 1):
        end_index = gen_length
    logger.info(f'{start_index}:{end_index}')

    residual_pre = None
    for (image, pivot, idx) in dataset_pbar:
        ii = int(idx.item())
        if ii > end_index or ii < start_index:
            continue
        pivot = [x.to(device) for x in pivot]
        image = image.to(device)
        gen_image = image.to(device).clone()
        if resume_path is not None:
            f = torch.load(os.path.join(resume_path, f'{idx.item()}.pt'), map_location='cpu').to(device)
        else:
            with torch.no_grad():
                f = encoder(image)
        for x in pivot:
            x.requires_grad = True
        # f.requires_grad = True
        residual = torch.randn_like(f).to(device)
        if residual_pre is not None:
            residual = residual_pre
            epochs = kwargs.get("epochs", 100)
        else:
            epochs = 100
        residual.requires_grad = True
        # optim = torch.optim.Adam([residual] + pivot, lr=lr)
        optim = torch.optim.Adam([residual], lr=lr)
        sche = torch.optim.lr_scheduler.StepLR(optim, step_size=250, gamma=0.5)

        content_mask, border_mask, full_mask = calc_masks(image, segmentation_model, 50,  0, 50, False)

        for epoch in range(1, epochs + 1):
            sample_loss = 0
            sample_count = 0

            image_gen = ss_decoder(pivot, insert_feature={"4": f + residual})
            n = image_gen.shape[0]
            ret = loss_register(image, image_gen, residual,content_mask, border_mask,is_gradient=False)
            loss = ret['loss']

            optim.zero_grad()
            loss.backward()
            optim.step()
            sche.step()
            total_idx += 1
            if epoch % internal_size == 0 and (rank == 0 or rank == -1):
                sample_loss += loss.mean()
                sample_count += 1
                string_to_info = reduce(lambda x, y: x + ', ' + y, [f'{k} {v.mean().item()}' for k, v in ret.items()])
                logger.info(f"{idx.item() + 1}/{epoch}/{epochs}: {string_to_info}")

                if writer is not None:
                    image_to_show = torch.cat((image_gen, image, image_gen * partial_mask + image * (1 - partial_mask)),
                                              dim=2)
                    writer.add_image(f'image_{idx.item()}',
                                     make_grid(image_to_show.detach(), normalize=True, scale_each=True), total_idx)
                    writer.add_scalars(f'loss_{idx.item()}', ret, total_idx)
            # loss_dict = {}
            # loss_dict['loss'] = loss
            # if writer is not None:
            #     image_to_show = torch.cat((image_gen, image, (image_gen * border_mask + image_gen * (1 - border_mask)),
            #                                border_mask.repeat(1, 3, 1, 1), content_mask.repeat(1, 3, 1, 1)), dim=2)
            #     writer.add_image(f'image_{idx.item()}', make_grid(image_to_show.detach(), normalize=True, scale_each=True), total_idx)
            #     writer.add_scalars(f'loss', ret, total_idx)

        model_path = os.path.join(path_snapshots, f"{idx.item()}.pt")
        torch.save(
            dict(
                f=f + (residual).detach(),
                pivot=[x.detach() for x in pivot]
            ),
            model_path
        )
        residual_pre = residual
        # exit()
            # writer.add_scalars(f'loss_{idx.item()}', loss, total_idx)
        """
        if (rank == 0 or rank == -1):
            sample_loss /= sample_count
            if sample_loss < min_loss:
                lastest_model_path = os.path.join(path_snapshots, f"{epoch}.pth")
                torch.save(ss_decoder.state_dict() if rank == -1 else ss_decoder.module.state_dict(), lastest_model_path)
                min_loss = sample_loss
                logger.info(f"min_loss: {min_loss}, epoch {epoch}")
        """
    # if rank == 0 or rank == -1:
    #     writer.close()

def masked_l2(input, target, mask, loss_l2):
    loss = torch.nn.MSELoss if loss_l2 else torch.nn.L1Loss
    criterion = loss(reduction='none')
    masked_input = input * mask
    masked_target = target * mask
    error = criterion(masked_input, masked_target)
    return error.sum() / mask.sum()

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

        def forward(self,
                    x,
                    y,
                    content_mask,
                    border_mask,
                    ):
            l2 = self.l2(x, y).mean() * self.l2_weight
            lpips = self.lpips(x * content_mask, y * content_mask).mean() * self.lpips_weight
            # border_loss = masked_l2(x, y, content_mask, True)
            # loss = border_loss + 0.1 * masked_l2(x, y, border_mask, True)

            return {
                "l2": l2,
                "lpips": lpips,
            }

    loss_register = PivotLossRegister(config) 
    loss_register.lpips.set_device(device)
    dataloader = get_dataloader()

    for p in ss_decoder.parameters():
        p.requires_grad = True

    segmentation_model = ExpressiveEncoding.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    segmentation_model.load_state_dict(torch.load(
        '/data1/chenlong/github/v63/expressive_talkinghead_encoding/ExpressiveEncoding/third_party/models/79999_iter.pth'))
    segmentation_model.to(device)

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
            f = f.to(device)
            image_gen = ss_decoder(pivot, insert_feature = {"4": f})
            n = image_gen.shape[0]
            content_mask, border_mask, full_mask = calc_masks(image, segmentation_model, 50, 0, 50, False)

            ret = loss_register(image, image_gen,content_mask, border_mask, is_gradient = False)

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

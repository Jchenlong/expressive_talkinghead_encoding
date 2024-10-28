"""Expressive total pipeline module.
"""
import os
import sys
sys.path.insert(0, os.getcwd())
from typing import Callable, Union, List
from functools import reduce
import cv2
import torch
import yaml
import imageio
import re
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from DeepLog import logger, Timer
from DeepLog.logger import logging

from ExpressiveEncoding.train import StyleSpaceDecoder, \
                                 stylegan_path, edict, yaml,load_model,from_tensor,Encoder4EditingWrapper,to_tensor


def fast_gaussian_blur(srcImage: np.array,
                       kernel_size: int,
                       scale: int,
                       sigma: float = 0.0
                       ) -> np.array:
    kernel_size = (kernel_size // scale) * 2 + 1
    srcImage_resize = cv2.resize(srcImage, (0,0), fx = 1.0 / scale, fy = 1.0 / scale)
    srcImage_blur = cv2.GaussianBlur(srcImage_resize, (kernel_size, kernel_size), sigma)
    return cv2.resize(srcImage_blur, (srcImage.shape[1], srcImage.shape[0]))


def color_correction(src_image: np.array,
                     guide_image: np.array,
                     radius_scale: float = 0.1) -> np.array:
    """Color correction in order to correct color bias.
        Args:
            src_image: numpy array.
            guide_image: numpy array.
            radius scale: float, use it to limit blur kernel size.
        Returns:
            dst image numpy array
        Raises:
            None
    """
    assert(src_image.shape == guide_image.shape)
    type_checker = lambda x: np.float32(x) # if x.dtype != 'np.float32'
    src_image = type_checker(src_image)
    guide_image = type_checker(guide_image)
    eps = 1e-4
    kernel_size = int(max(*(src_image.shape[:2])) * radius_scale) * 2 + 1
    src_image_blur = fast_gaussian_blur(src_image, kernel_size, 5) #cv2.GaussianBlur(src_image, (kernel_size, kernel_size), 0.0)
    guide_image_blur = fast_gaussian_blur(guide_image, kernel_size,5) #cv2.GaussianBlur(guide_image, (kernel_size, kernel_size), 0.0)
    scale = guide_image_blur / (src_image_blur + 1e-4)
    return scale * src_image

def validate_video_gen(
                        save_video_path:str,
                        state_dict_path: str,
                        latents: Union[str, List[np.ndarray]],
                        ss_decoder: Callable,
                        video_length: int,
                        face_folder_path: str,
                        f_space:List[np.ndarray],
                      ):

    if video_length == -1:
        files = list(filter(lambda x: x.endswith('pt'), os.listdir(latent_folder)))
        assert len(files), "latent_folder has no latent file."
        video_length = len(files)
    if state_dict_path is not None:
        ss_decoder.load_state_dict(torch.load(state_dict_path),strict=False)
    with imageio.get_writer(save_video_path, fps = 25) as writer:
        for index in tqdm(range(video_length)):
            if isinstance(latents, str):
                style_space_latent = torch.load(os.path.join(latents, f"{index+1}.pt"))
            else:
                style_space_latent = latents[index]
            style_space_latent = [s.to("cuda") for s in style_space_latent]
            if not isinstance(style_space_latent, list):
                style_space_latent = ss_decoder.get_style_space(style_space_latent)

            # image = np.uint8(np.clip(from_tensor(ss_decoder(style_space_latent) * 0.5 + 0.5), 0.0, 1.0) * 255.0)
            image = np.uint8(np.clip(from_tensor(ss_decoder(style_space_latent, insert_feature={'4': f_space[index].to('cuda')})  * 0.5 + 0.5), 0.0, 1.0) * 255.0)

            image_gt_path = os.path.join(face_folder_path, f'{index}.png')
            if not os.path.exists(image_gt_path):
                image_gt_path = image_gt_path.replace('png', 'jpg')
            image_gt = cv2.imread(image_gt_path)[...,::-1]
            image_gt = cv2.resize(image_gt, (512,512))
            image_paste = image_gt.copy()
            # image = color_correction(image,image_gt)
            # image = color_correction(image_gt,image)
            image_paste[274:494,80:-80,:] = image[274:494,80:-80,:]
            image_concat = np.concatenate((image, image_paste,image_gt), axis = 0)
            writer.append_data(image_concat)
            if state_dict_path is None:
                workdir = os.path.join(os.path.dirname(save_video_path), "images")
                os.makedirs(workdir,exist_ok = True)
                cv2.imwrite(os.path.join(workdir, f'{index + 1}.jpg'), image[...,::-1])

import click
@click.command()
@click.option('--expname',default='eR6CMmTu_2')
@click.option('--save_path',default='/data1/chenlong/yuyuhang/interface_out/1023')
@click.option('--latest_decoder_path',default='/data1/chenlong/0517/video/0822/eR6CMmTu_2/eR6CMmTu_2_exp_set_v4_depoly/exp/f_space_bdinv_012/decoder/snapshots/20.pth')
@click.option('--f_space_path',default='/data1/chenlong/0517/video/0822/eR6CMmTu_2/eR6CMmTu_2_exp_set_v4_depoly/exp/f_space_bdinv_012/f_space.pt')
def expressive_encoding_pipeline(expname,save_path,latest_decoder_path,f_space_path):
    os.makedirs(save_path, exist_ok=True)
    stage_three_path = f'/data1/chenlong/0517/video/0822/{expname}/{expname}_exp_set_v4_depoly/exp/facial' #torch.load(f'./bdinv_detail_009/s.pt')
    face_folder_path = f'/data1/chenlong/online_model_set/video/{expname}/video_split/smooth'
    # latest_decoder_path = '/data1/chenlong/0517/video/0822/eR6CMmTu_2/eR6CMmTu_2_exp_set_v4_depoly/exp/pti_ft_512/facial_snapshots_ft_512/10.pth'
    if os.path.exists(face_folder_path):
        pass
    else:
        face_folder_path = f'/data1/chenlong/online_model_set/video/{expname}/video/smooth'
    validate_video_path = os.path.join(save_path, f"{expname}_{os.path.basename(latest_decoder_path)}_1023_007.mp4")

    #if latest_decoder_path is not None:
    #    if not latest_decoder_path.endswith('pt') and not latest_decoder_path.endswith('pth'):
    #        folder = latest_decoder_path
    #        latest_decoder_path = os.path.join(folder, sorted(os.listdir(latest_decoder_path), key = lambda x: int(''.join(re.findall('[0-9]+', x))))[-1])
    print(f"latest weight path is {latest_decoder_path}")

    from copy import deepcopy
    G = load_model(stylegan_path,device = 'cuda').synthesis
    for p in G.parameters():
        p.requires_grad = False
    f_space = torch.load(f_space_path)
    length = len(f_space)
    ss_decoder = StyleSpaceDecoder(synthesis=deepcopy(G),to_resolution=512)
    for p in ss_decoder.parameters():
        p.requires_grad = False
    validate_video_gen(
                        validate_video_path,
                        latest_decoder_path,
                        stage_three_path,
                        ss_decoder,
                        length,
                        face_folder_path,
                        f_space
                      )
    logger.info(f"validate video located in {validate_video_path}")

if __name__ == '__main__':
	expressive_encoding_pipeline()

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
from PIL import Image

from ExpressiveEncoding.train import StyleSpaceDecoder, BiSeNet, transforms,\
                                 stylegan_path, edict, yaml,load_model,from_tensor,Encoder4EditingWrapper,to_tensor

with open(os.path.join("/data1/wanghaoran/Amemori", "template.yaml")) as f:
    config = yaml.load(f, Loader = yaml.CLoader)

regions = eval(config["soft_mask_region"])
output_copy_region = eval(config["output_copy_region"])
def get_center_from_mask(mask: np.ndarray):
    """
    """
    m = cv2.moments(mask)
    return int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"])

def get_soft_mask_by_region():
    soft_mask  = np.zeros((512,512,3), np.float32)
    for region in regions:
        y1,y2,x1,x2 = region
        soft_mask[y1:512,x1:x2,:]=1
    #soft_mask = cv2.GaussianBlur(soft_mask, (101, 101), 11)
    soft_mask = soft_mask.astype(np.float32)
    return soft_mask

def merge_from_two_image(
        master: np.ndarray,
        slave: np.ndarray,
        mask: np.ndarray = None,
        blender: object = None
) -> np.ndarray:
    master = master.astype(np.float32)
    slave = slave.astype(np.float32)
    if mask is not None:
        if mask.ndim < 3:
            mask = mask[..., np.newaxis]

        # dilate
        erosion_size = 15
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
        mask_erosion = cv2.erode(np.uint8(mask), element)


        if blender is not None:
            output = blender(master, slave, np.uint8(mask_diff * 255))
            mask_diff = cv2.GaussianBlur(mask_diff, (101, 101), 11)
            if mask_diff.ndim < 3:
                mask_diff = mask_diff[..., np.newaxis]
            slave = output * mask_diff + (1 - mask_diff) * slave
        else:
            mask_blur = mask.astype(np.float32)
            # mask_blur = cv2.boxFilter(mask.astype(np.float32), -1, ksize = (21, 21))
            dilate_size = 15
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_size + 1, 2 * dilate_size + 1),
                                                (dilate_size, dilate_size))
            mask_dilate = cv2.dilate(mask_blur, element)
            if mask_dilate.ndim < 3:
                mask_dilate = mask_dilate[..., np.newaxis]
            mask_diff = mask_dilate - mask_blur
            strength = -5
            h, w, c = mask.shape

            x, y = get_center_from_mask(mask_diff[..., 0])
            x_linspace = np.linspace(0, w - 1, w)
            y_linspace = np.linspace(0, h - 1, h)
            x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)

            offset_strength = (mask[..., 0] * strength).astype(np.float32)
            offset_x = np.sign(x - x_grid) * offset_strength
            offset_y = np.sign(y - y_grid) * offset_strength
            offset_x = cv2.boxFilter(offset_x, -1, ksize=(21, 21))
            offset_y = cv2.boxFilter(offset_y, -1, ksize=(21, 21))
            master = cv2.remap(master, (x_grid + offset_x).astype(np.float32), (y_grid + offset_y).astype(np.float32),
                               cv2.INTER_LINEAR)
            # offset_strength = cv2.boxFilter(offset_strength, -1, ksize = (21, 21))

            # mask_diff = cv2.boxFilter(mask_diff.astype(np.float32), -1, ksize = (21, 21))
            # if mask_diff.ndim < 3:
            #    mask_diff = mask_diff[..., np.newaxis]
            # slave = master * mask_diff + (1 - mask_diff) * slave
            # mask = cv2.GaussianBlur(mask, (101, 101), 11)
            # if mask.ndim < 3:
            #    mask = mask[..., np.newaxis]
        mask = cv2.boxFilter(mask.astype(np.float32), -1, ksize=(21, 21))
        if mask.ndim < 3:
            mask = mask[..., np.newaxis]
        merge_mask = mask
    else:
        merge_mask = np.zeros_like(master)
        dilate_erode_mask = merge_mask.copy()
        for (y1, y2, x1, x2) in output_copy_region:
            merge_mask[y1:y2, x1:x2, ...] = 1.0
        pad_l = 5
        pad_r = 20
        dilate_erode_mask[y1 - pad_l: y2 + pad_r, x1 - pad_l: x2 + pad_r, ...] = 1.0
        mask_diff = dilate_erode_mask - merge_mask
        mask_diff = cv2.boxFilter(mask_diff.astype(np.float32), -1, ksize=(21, 21))
        if mask_diff.ndim < 3:
            mask_diff = mask_diff[..., np.newaxis]
        slave = master * mask_diff + (1 - mask_diff) * slave

    output = merge_mask * slave + (1 - merge_mask) * master
    partial_mask = get_soft_mask_by_region()
    # output = output * partial_mask + master * (1 - partial_mask)


    return np.clip(output, 0.0, 255.0).astype(np.uint8)


class face_parsing:
    def __init__(self, path = os.path.join('/data1/chenlong/github/v63/expressive_talkinghead_encoding/ExpressiveEncoding/third_party', "models", "79999_iter.pth")):

        net = BiSeNet(19)
        state_dict = torch.load(path)
        net.load_state_dict(state_dict)
        net.eval()
        net.to("cuda:0")
        self.net = net
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, x):

        h, w = x.shape[:2]
        x = Image.fromarray(x)
        image = x.resize((512, 512), Image.BILINEAR)
        img = self.to_tensor(image).unsqueeze(0).to("cuda:0")
        out = self.net(img)[0].detach().squeeze(0).cpu().numpy().argmax(0)
        mask = np.zeros_like(out).astype(np.float32)
        for label in list(range(1,  7)) + list(range(10, 14)):
            mask[out == label] = 1
        out = cv2.resize(mask, (w,h))
        return out


from warp_utils import WarpTool
from blend_utils import FaceBlendTool
warp_tool = WarpTool()
face_blend_tool = FaceBlendTool()
face_blend_tool.update_blending_params(copy_region="[[274,494,80,432]]",
                                       soft_mask_region="[[340,494,130,-130],[274,340,130,-130]]")

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
                        face_parse_func:object = None,
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
                style_space_latent = [s.to("cuda") for s in style_space_latent]
            else:
                style_space_latent = latents[index]
            if not isinstance(style_space_latent, list):
                style_space_latent = ss_decoder.get_style_space(style_space_latent)

            # image = np.uint8(np.clip(from_tensor(ss_decoder(style_space_latent) * 0.5 + 0.5), 0.0, 1.0) * 255.0)
            image = np.uint8(np.clip(from_tensor(ss_decoder(style_space_latent, insert_feature={'4': f_space[index].to('cuda')})  * 0.5 + 0.5), 0.0, 1.0) * 255.0)

            image_gt_path = os.path.join(face_folder_path, f'{index}.png')
            if not os.path.exists(image_gt_path):
                image_gt_path = image_gt_path.replace('png', 'jpg')
            image_gt = cv2.imread(image_gt_path)[...,::-1]
            image_gt = cv2.resize(image_gt, (512,512))
            mask = face_parse_func(image_gt)
            image = merge_from_two_image(image_gt, image, mask=mask, blender=None)

            face_mask = np.ones_like(image)
            image = face_blend_tool.blend_with_mask(image_gt, image, image_gt, face_mask)

            image_concat = np.concatenate((image, image_gt), axis = 0)
            writer.append_data(image_concat)
            if state_dict_path is None:
                workdir = os.path.join(os.path.dirname(save_video_path), "images")
                os.makedirs(workdir,exist_ok = True)
                cv2.imwrite(os.path.join(workdir, f'{index + 1}.jpg'), image[...,::-1])

import click
@click.command()
@click.option('--expname',default=' ')
@click.option('--save_path',default=' ')
@click.option('--latest_decoder_path',default=' ')
@click.option('--f_space_path',default=' ')
def expressive_encoding_pipeline(expname,
                                 save_path,
                                 latest_decoder_path,
                                 f_space_path):
    os.makedirs(save_path, exist_ok=True)
    stage_three_path = f'/data1/chenlong/0517/video/0822/{expname}/{expname}_exp_set_v4_depoly/exp/facial_ft'
    face_folder_path = f'/data1/chenlong/online_model_set/video/{expname}/video_split/smooth'

    if os.path.exists(face_folder_path):
        pass
    else:
        face_folder_path = f'/data1/chenlong/online_model_set/video/{expname}/video/smooth'
    validate_video_path = os.path.join(save_path, f"{expname}_{os.path.basename(latest_decoder_path)}_1024_65.mp4")
    length = len(os.listdir(face_folder_path))
    # length = 1000

    if latest_decoder_path is not None:
        if not latest_decoder_path.endswith('pt') and not latest_decoder_path.endswith('pth'):
            folder = latest_decoder_path
            latest_decoder_path = os.path.join(folder, sorted(os.listdir(latest_decoder_path), key = lambda x: int(''.join(re.findall('[0-9]+', x))))[-1])
    print(f"latest weight path is {latest_decoder_path}")

    from copy import deepcopy
    G = load_model(stylegan_path,device = 'cuda').synthesis
    for p in G.parameters():
        p.requires_grad = False
    f_space = torch.load(f_space_path)
    ss_decoder = StyleSpaceDecoder(synthesis=deepcopy(G),to_resolution=512)
    for p in ss_decoder.parameters():
        p.requires_grad = False
    face_parse_func = face_parsing()
    validate_video_gen(
                        validate_video_path,
                        latest_decoder_path,
                        stage_three_path,
                        ss_decoder,
                        length,
                        face_folder_path,
                        f_space,
                        face_parse_func,
                      )
    logger.info(f"validate video located in {validate_video_path}")

if __name__ == '__main__':
	expressive_encoding_pipeline()
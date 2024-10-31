import os
import json
import cv2
import numpy as np
from typing import List

def get_center_from_mask(mask: np.ndarray):
    """
    """
    m = cv2.moments(mask) 
    return int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"])

class FaceBlendTool:
    def __init__(self,
            regions:List[List[int]]=[[340,484,130,-130],[210,340,130,-130]],
            time_recorder=None
        ):
        """
        Blend the output face and the original face.

        Args:
            regions (List[List[int]], optional): 
                [[y1,y2,x1,x2], ...]
            time_recorder (TimeRecoder, optional): 
                TimeRecoder
        """
        self.soft_mask  = np.zeros((512,512,3), np.float32)
        self.soft_mask  = self.get_soft_mask_by_region(regions)
        self.soft_mask_inv = 1.0 - self.soft_mask
        self.copy_region = None

    def get_soft_mask_by_region(self, regions:List[List[int]])->np.ndarray:
        soft_mask  = np.zeros((512,512,3), np.float32)
        for region in regions:
            y1,y2,x1,x2 = region
            soft_mask[y1:y2,x1:x2,:]=1
        soft_mask = cv2.GaussianBlur(soft_mask, (101, 101), 11)
        soft_mask = soft_mask.astype(np.float32)
        return soft_mask
    
    def update_blending_params(self, soft_mask_region:str, copy_region:str):
        if soft_mask_region is None:
            soft_mask_region = "[[340,494,130,-130],[274,340,130,-130]]"
        if soft_mask_region.startswith('['):
            regions = json.loads(soft_mask_region)
        else:
            regions = [[ int(x) for x in soft_mask_region.split(',')]]
        self.soft_mask = self.get_soft_mask_by_region(regions)
        self.soft_mask_inv = 1.0 - self.soft_mask

        self.copy_region = None
        if copy_region is not None:
            self.copy_region:List[list] = json.loads(copy_region)

        if os.environ.get("DEBUG_FACE_VIDEO", None) is not None:
            self.soft_mask_inv *= 0
            self.soft_mask = 1.0 - self.soft_mask_inv
            self.copy_region = [[0,512,0,512]]

    def blend(self, face_ori:np.ndarray, face_gen:np.ndarray, out_dtype=np.uint8) -> np.ndarray:
        """
        Apply soft mask blending on regions.

        Args:
            face_ori (np.ndarray): 
                np.uint8.  Original face.
            face_out (np.ndarray): 
                np.float32.  Generated face.
            out_dtype (_type_, optional): 
                Defaults to np.uint8.

        Returns:
            np.ndarray
        """

        face_copy = face_ori.copy()

        if self.copy_region is not None:
            for (y1,y2,x1,x2) in self.copy_region:
                face_copy[y1:y2,x1:x2] = face_gen[y1:y2, x1:x2] * self.soft_mask[y1:y2, x1:x2] + face_ori[y1:y2, x1:x2] *  self.soft_mask_inv[y1:y2, x1:x2]
            output = face_copy
        else:
            output = face_ori *  self.soft_mask_inv + face_gen * self.soft_mask

        if out_dtype == np.uint8:
            output = output.astype(np.uint8)
        return output
    
    def blend_with_mask(self, face_ori:np.ndarray, face_gen:np.ndarray, face_b:np.ndarray, face_mask:np.ndarray, out_dtype=np.uint8) -> np.ndarray:
        """
        Apply soft mask blending on regions.

        Args:
            face_ori (np.ndarray): 
                np.uint8.  Original face.
            face_out (np.ndarray): 
                np.float32.  Generated face. 0~255
            face_b (np.ndarray): 
                np.float32.  face image processed by AE
            face_mask (np.ndarray): 
                np.float32.  face mask processed by AE
            out_dtype (_type_, optional): 
                Defaults to np.uint8.

        Returns:
            np.ndarray
        """

        face_copy = face_ori.copy()

        if self.copy_region is not None:
            for (y1,y2,x1,x2) in self.copy_region:
                face_copy[y1:y2,x1:x2] = face_gen[y1:y2, x1:x2] * self.soft_mask[y1:y2, x1:x2] + face_ori[y1:y2, x1:x2] *  self.soft_mask_inv[y1:y2, x1:x2]
            output = face_copy
        else:
            output = face_ori *  self.soft_mask_inv + face_gen * self.soft_mask

        if out_dtype == np.uint8:
            output = output.astype(np.uint8)

        assert face_b.dtype == output.dtype
        where = face_mask < 255
        # output[where] = face_b[where]

        return output

    def alpha_blending(self, front:np.ndarray, back:np.ndarray, mask:np.ndarray):
        """
            output = front * mask + back * (1-mask)
        Args:
            front (np.ndarray): int or float array 
            back (np.ndarray): int or float array 
            mask (np.ndarray): np.uint8, 0~255
        """
        front = front.astype(np.float32)
        back = back.astype(np.float32)
        mask = mask.astype(np.float32) / 255.0

        dilate_size = 15
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_size + 1, 2 * dilate_size + 1), (dilate_size, dilate_size))
        mask_dilate = cv2.dilate(mask, element)
        if mask_dilate.ndim < 3:
            mask_dilate = mask_dilate[..., np.newaxis]
        mask_diff = mask_dilate - mask

        strength = -5
        h, w, c = mask.shape

        x,y = get_center_from_mask(mask_diff[...,0])
        x_linspace = np.linspace(0, w - 1, w)
        y_linspace = np.linspace(0, h - 1, h)
        x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)

        offset_strength = (mask[...,0] * strength).astype(np.float32)
        offset_x = np.sign(x - x_grid) * offset_strength
        offset_y = np.sign(y - y_grid) * offset_strength
        offset_x = cv2.boxFilter(offset_x, -1, ksize = (21, 21)) 
        offset_y = cv2.boxFilter(offset_y, -1, ksize = (21, 21)) 
        back = cv2.remap(back, (x_grid + offset_x).astype(np.float32), (y_grid + offset_y).astype(np.float32), cv2.INTER_LINEAR)
        output = front * mask + back * (1 - mask)
        return output

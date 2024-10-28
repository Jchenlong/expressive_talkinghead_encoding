import cv2
import numpy as np

class WarpTool:
    def __init__(self, blending_size=30) -> None:
        self.blending_size = blending_size
        self.init_blending_mask()

    def init_blending_mask(self):

        blending_mask = np.ones((512,512,3), dtype=np.float32)
        blending_padding_size = self.blending_size
        for t in range(blending_padding_size - 1, -1, -1):
            blending_mask[:,t,:] = blending_mask[:,-t,:] = float(t) / float(blending_padding_size)
            blending_mask[t,:,:] = blending_mask[-t,:,:] = float(t) / float(blending_padding_size)
        # return blending_mask
        self.blending_mask = blending_mask
        self.blending_mask_inv = 1.0 - blending_mask
    
    def warp_inverse_ori(self, img, face, M, distortion_face):
        height_origin, width_origin = img.shape[:2]
        _enhance_face = face
        # blend with distortion image to deluminate the transition artifacts
        blending_mask = self.blending_mask
        blending_mask_inv = self.blending_mask_inv
        _enhance_face = np.clip(distortion_face.astype(np.float32) * blending_mask_inv + _enhance_face.astype(np.float32) * (blending_mask), 0.0, 255.0)

        h, w = _enhance_face.shape[:2]
        image_enhance_ones = np.ones((h,w,1), _enhance_face.dtype)
        _enhance_face = np.concatenate([_enhance_face, image_enhance_ones], axis=2)
        image_enhance = cv2.warpPerspective(_enhance_face, M, (width_origin, height_origin), flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        where = image_enhance[..., -1] == 1
        img[where] = image_enhance[..., :3][where]
        return img       

    def warp_inverse(self, img, face, M, distortion_face, should_blend=True):
        _enhance_face = face
        """new warp"""
        m_inv = np.linalg.inv(M)
        dst_quad = np.float32(np.array([[0., 0.],[511.   ,  0.],[511., 511.],[0., 511. ]])+0.5)
        src_quad = cv2.perspectiveTransform(np.array([dst_quad]), m_inv).reshape(4,2)
        src_quad_int = src_quad.astype(np.int_)
        x1, y1, x2, y2 = min(src_quad_int[:,0]), min(src_quad_int[:,1]),max(src_quad_int[:,0]), max(src_quad_int[:,1])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, img.shape[1] - 1)
        y2 = min(y2, img.shape[0] - 1)
        src_quad[:, 0] -= x1
        src_quad[:, 1] -= y1
        new_m = cv2.getPerspectiveTransform(dst_quad, src_quad.astype(np.float32))

        if should_blend:
            blending_mask = self.blending_mask
            blending_mask_inv = self.blending_mask_inv
            size = self.blending_size
            _enhance_face[:size] = np.clip(distortion_face[:size].astype(np.float32) * blending_mask_inv[:size] + _enhance_face[:size].astype(np.float32) * blending_mask[:size], 0.0, 255.0).astype(np.uint8)
            _enhance_face[-size:] = np.clip(distortion_face[-size:].astype(np.float32) * blending_mask_inv[-size:] + _enhance_face[-size:].astype(np.float32) * blending_mask[-size:], 0.0, 255.0).astype(np.uint8)
            _enhance_face[:, :size] = np.clip(distortion_face[:, :size].astype(np.float32) * blending_mask_inv[:, :size] + _enhance_face[:, :size].astype(np.float32) * blending_mask[:, :size], 0.0, 255.0).astype(np.uint8)
            _enhance_face[:, -size:] = np.clip(distortion_face[:, -size:].astype(np.float32) * blending_mask_inv[:, -size:] + _enhance_face[:, -size:].astype(np.float32) * blending_mask[:, -size:], 0.0, 255.0).astype(np.uint8)

        h, w = _enhance_face.shape[:2]
        image_enhance_ones = np.ones((h,w,1), np.uint8)

        image_enhance = cv2.warpPerspective(_enhance_face, new_m,  (int(x2-x1), int(y2-y1)), flags = cv2.INTER_NEAREST)
        mask = cv2.warpPerspective(image_enhance_ones, new_m,  (int(x2-x1), int(y2-y1)), flags = cv2.INTER_NEAREST) * 255

        mask_inv = cv2.bitwise_not(mask)
        im_a = cv2.bitwise_and(img[y1:y2, x1:x2], img[y1:y2, x1:x2], mask=mask_inv)

        im_add = cv2.add(im_a, image_enhance[..., :3].astype(np.uint8))
        img[y1:y2,x1:x2] = im_add

        return img
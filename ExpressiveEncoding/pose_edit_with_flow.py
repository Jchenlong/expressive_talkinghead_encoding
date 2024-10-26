"""the wrapper for StyleFlow used for
   pose editting.
"""
import os
import torch
import numpy as np
from ExpressiveEncoding.StyleFlow.module.flow import cnf
where_am_i = os.path.dirname(os.path.realpath(__file__))

class PoseEdit:
    """Class PoseEdit using StyleFlow.
    """
    _zero_padding = torch.zeros(1, 18, 1)
    def __init__(self,
                 model_path = os.path.join(where_am_i, \
                              '/app/pretrained_models/modellarge10k.pt'),
                 device=0,
                ):
        self.cnf = cnf(512, '512-512-512-512-512', 17, 1)
        self.cnf.to(f"cuda:{device}")
        self.cnf.load_state_dict(torch.load(model_path,map_location=f"cuda:{device}"))
        self.cnf.eval()

        for p in self.cnf.parameters():
            p.requires_grad = False
        self.zflow = None
        self.device = device
        self.reset()

    def reset(self):
        """reset zflow to original
        """
        self.zflow = self._get_attribute_zflow()
        self.zflow.requires_grad = False

    def _get_attribute_zflow(self):
        light_zflow = np.zeros((1,9,1,1), dtype = np.float32)
        attribute_zflow = np.zeros((8,1), dtype = np.float32)
        attribute_zflow[0,0] = 0.0
        attribute_zflow[1,0] = 0.0
        attribute_zflow[2,0] = 0.0
        attribute_zflow[3,0] = 0.0
        attribute_zflow[4,0] = 0.0
        attribute_zflow[5,0] = 0.0
        attribute_zflow[6,0] = 55.0 # set value same as paper
        attribute_zflow[7,0] = 0.0

        zflow_array = np.concatenate([light_zflow, \
                      np.expand_dims(attribute_zflow, axis = (0, -1))], axis = 1)
        return torch.from_numpy(zflow_array).type(\
                torch.FloatTensor).to(f"cuda:{self.device}")

    def _update_zflow(self,
                      yaw,
                      pitch
                      ):
        """Fix other attribute value,
           expose yaw and pitch.
           0-8: light
           9-16: attributes.
                 yaw and pitch index is 11, 12.

        """
        self.zflow[:, 11, 0, 0] = yaw
        self.zflow[:, 12, 0, 0] = pitch

    def __call__(
                self,
                latent,
                yaw,
                pitch,
                is_w_space = False
                ):
        n = latent.shape[0]
        if self.zflow.shape[0] != n:
            self.reset()
            self.zflow = self.zflow.repeat(n,1,1,1)
        self._update_zflow(yaw, pitch)
        return self.cnf(latent, self.zflow, self._zero_padding.to(f'cuda:{self.device}'), is_w_space)[0]

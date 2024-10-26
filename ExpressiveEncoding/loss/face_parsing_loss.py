import os
import sys
where_am_i = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(where_am_i, "FaceParsing"))

import torch
from model import BiSeNet

class FaceParsingLoss(torch.nn.Module):
    def __init__(self, is_reduce = False):
        super().__init__()
        model_path = "/app/pretrained_models/79999_iter.pth"
        model = BiSeNet(19)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.cuda()
        self.net = model
        for p in self.net.parameters():
            p.requires_grad = False

        self.loss = torch.nn.MSELoss() if is_reduce else torch.nn.MSELoss(reduction='none') 
        self.softmax = torch.nn.Softmax(dim = 1)

    def norm(self, tensor_from_image):
        tensor_from_image[:,0, ...] =  (tensor_from_image[:,0, ...] - 0.485) / 0.229
        tensor_from_image[:,1, ...] =  (tensor_from_image[:,1, ...] - 0.456) / 0.224
        tensor_from_image[:,2, ...] =  (tensor_from_image[:,2, ...] - 0.406) / 0.225
        return tensor_from_image

    def forward(self, x, y, mask):
        
        x_resize = torch.nn.functional.interpolate(x, (512,512))
        y_resize = torch.nn.functional.interpolate(y, (512,512))

        mask_resize = torch.nn.functional.interpolate(mask, (512,512))

        x_01 = (x_resize + 1) * 0.5
        y_01 = (y_resize + 1) * 0.5
        
        x_norm = self.norm(x_01)
        y_norm = self.norm(y_01)

        x_scores,_ = torch.max(self.softmax(self.net(x_norm)[0]), dim = 1)
        y_scores,_ = torch.max(self.softmax(self.net(y_norm)[0]), dim = 1)

        x_scores = x_scores.reshape(-1,1,512,512)
        y_scores = y_scores.reshape(-1,1,512,512)
        return self.loss(x_scores * mask_resize, y_scores * mask_resize).mean(dim=(1,2,3))           

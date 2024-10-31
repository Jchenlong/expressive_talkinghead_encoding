"""the dataset module
"""
import os
import torch
import random

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from multiprocessing import Lock

from DeepLog import logger
from .utils import make_dataset,face_parsing

lock = Lock()
DEBUG = os.environ.get("DEBUG", 0)
alpha_indexes = [
                 6, 11, 8, 14, 15, # represent Mouth
                 5, 6, 8, # Chin/Jaw
                 9, 11, 12, 14, 17, # Eyes
                 8, 9 , 11, # Eyebrows
                 9 # Gaze
                ]

alpha_S_indexes = [
                    [113, 202, 214, 259, 378, 501],
                    [6, 41, 78, 86, 313, 361, 365],
                    [17, 387],
                    [12],
                    [45],
                    [50, 505],
                    [131],
                    [390],
                    [63],
                    [257],
                    [82, 414],
                    [239],
                    [28],
                    [6, 28],
                    [30],
                    [320],
                    [409]
                  ]

alphas = [(x, y) for x, y in zip(alpha_indexes, alpha_S_indexes)]
size_of_alpha = 0
for k,v in alphas[:8]:
    size_of_alpha += len(v)

output_copy_region="[[274,494,80,432]]"
output_copy_region = eval(output_copy_region)
def get_mask_by_region():
    _mask  = np.zeros((512,512,3), np.float32)
    for region in output_copy_region:
        y1,y2,x1,x2 = region
        _mask[y1+50+25:y2,x1:x2,:] = 1
    return torch.from_numpy(_mask).permute((2,0,1))

def update_region_offset(
                          dlatents,
                          offset,
                          region_range
                        ):
    dlatents_tmp = [latent.clone() for latent in dlatents]
    count = 0
    #first 5 elements.
    for k, v in alphas[region_range[0]:region_range[1]]:
        for i in v:
            dlatents_tmp[k][i] = dlatents[k][i] + offset[:,count]
            count += 1
    return dlatents_tmp

def update_region_offset_v2(
                            dlatents,
                            offset,
                            region_range
                            ):
    dlatents_tmp = [torch.zeros_like(latent) for latent in dlatents]
    count = 0
    #first 5 elements.
    for k, v in alphas[region_range[0]:region_range[1]]:
        for i in v:
            dlatents_tmp[k][i] = offset[:,count].to(torch.float32)
            count += 1
    return dlatents_tmp

class ImagesDataset(Dataset):
    """ImagesDataset for pivot tuning.
    """
    def __init__(
                 self,
                 source_root,
                 latent_root,
                 source_transform=None,
                 kmeans_info: dict = None,
                 random: bool = False
                ):
        self.source_paths = sorted(make_dataset(source_root), \
                            key = lambda x: int(os.path.basename(x[1]).split('.')[0]))
        self.latent_root = latent_root
        self.source_transform = source_transform
        
        self.remap = None
        if kmeans_info is not None:
            kmeans_labels, centers = kmeans_info["cluster_ids"], kmeans_info["cluster_centers"]
            flags = [False for _ in range(len(centers))]
            indexes = [0 for _ in range(len(centers))]
            for i in range(len(self.source_paths)):
                label = kmeans_labels[i]
                if not flags[label]:
                    indexes[label] = i
                    flags[label] = True
            self.remap = indexes
        if random:
            files_length = len(self.source_paths) // 25
            indexes = np.random.choice(np.array(list(range(len(self.source_paths)))), files_length)
            self.remap = indexes

            
    def __len__(self):
        if DEBUG:
            return 1800
        if self.remap is not None:
            return len(self.remap)
        else:
            return len(self.source_paths)

    def __getitem__(self, index):
        if self.remap is not None:
            index = self.remap[index]
        _, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            from_im = self.source_transform(from_im)
        latent = torch.load(os.path.join(self.latent_root, f'{index + 1}.pt'))
        return from_im, [x[0].detach().cpu() for x in latent], index


class ImagesDatasetW(Dataset):
    """ImagesDataset for pivot tuning.
    """
    def __init__(self,
                 source_root,
                 latent_root,
                 source_transform=None,
                 kmeans_info: dict = None,
                 random: bool = False
                 ):
        self.source_paths = sorted(make_dataset(source_root), \
                            key = lambda x: int(os.path.basename(x[1]).split('.')[0]))
        self.latent_root = latent_root
        self.source_transform = source_transform

        self.remap = None
        if kmeans_info is not None:
            kmeans_labels, centers = kmeans_info["cluster_ids"], kmeans_info["cluster_centers"]
            flags = [False for _ in range(len(centers))]
            indexes = [0 for _ in range(len(centers))]
            for i in range(len(self.source_paths)):
                label = kmeans_labels[i]
                if not flags[label]:
                    indexes[label] = i
                    flags[label] = True
            self.remap = indexes
        if random:
            files_length = len(self.source_paths) // 25
            indexes = np.random.choice(np.array(list(range(len(self.source_paths)))), files_length)
            self.remap = indexes


    def __len__(self):
        if DEBUG:
            return 1800
        if self.remap is not None:
            return len(self.remap)
        else:
            return len(self.source_paths)


    def __getitem__(self, index):
        if self.remap is not None:
            index = self.remap[index]
        _, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            from_im = self.source_transform(from_im)
        latent = torch.load(os.path.join(self.latent_root, f'{index + 1}.pt'))
        return from_im, latent[0], index

class ImagesDatasetV2(ImagesDataset):
    def __init__(self, 
                 source_root,
                 latent_root,
                 expressive_root,
                 source_transform = None
                 ):

        super().__init__(source_root, latent_root, source_transform)
        self.expressive_root = expressive_root
        attr = torch.tensor(torch.load(os.path.join(self.expressive_root, f"{1}.pt"))[1]).reshape(1, -1)
        attributes_min = torch.ones_like(attr) * 0xffff
        attributes_max = torch.zeros_like(attr)
        for i in range(self.__len__()):
            attr = torch.tensor(torch.load(os.path.join(self.expressive_root, f"{i + 1}.pt"))[1]).reshape(1, -1)
            attributes_min = torch.min(attributes_min, attr)
            attributes_max = torch.max(attributes_max, attr)
        
        self.attributes_min = attributes_min
        self.attributes_max = attributes_max

    def __getitem__(self, index):
        _, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            from_im = self.source_transform(from_im)
        latent = torch.load(os.path.join(self.latent_root, f'{index + 1}.pt'))
        """
        random_index = random.choice(list(range(len(self.source_paths))))
        attribute_random = torch.load(os.path.join(self.expressive_root, f'attribute_{random_index + 1}.pt'))
        """
        attribute = torch.tensor(torch.load(os.path.join(self.expressive_root, f'{index + 1}.pt'))[1]).reshape(1, -1)
        attribute_random = attribute.clone()
        mask = torch.tensor([random.choice([0, 1]) for _ in range(21)]).reshape(1, -1)
        mask = torch.ones((1,21)) * random.choice([0, 1])
        attribute_random[0,:21] = self.attributes_min[0,:21] * mask + self.attributes_max[0, :21] * (1 - mask)

        latent = [x[0] for x in latent]
        latent_updated = update_region_offset(latent, attribute, [0, len(alphas)]) 
        latent_updated_random = update_region_offset(latent, attribute_random, [0, len(alphas)])
        mask = get_mask_by_region()
        latent_updated = [torch.cat((x, y), dim = 0) for (x,y) in zip(latent_updated, latent_updated_random)]

        return torch.cat((from_im, mask), dim = 0), latent_updated

class ImagesDatasetV3(ImagesDatasetV2):
    def __init__(self, 
                 source_root,
                 latent_root,
                 expressive_root,
                 ss_root,
                 source_transform = None
                 ):

        super().__init__(source_root, latent_root, expressive_root,  source_transform)
        self.ss_root = ss_root
    
    def __getitem__(self, index):
        _, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            from_im = self.source_transform(from_im)
        latent = torch.load(os.path.join(self.latent_root, f'{index + 1}.pt'))
        ss_latent = torch.load(os.path.join(self.ss_root, f'{index + 1}.pt'))
        attribute = torch.tensor(torch.load(os.path.join(self.expressive_root, f'attribute_{index + 1}.pt'))[1]).reshape(1, -1)
        ss_latent = [x[0] for x in ss_latent]
        latent_residual = update_region_offset_v2(ss_latent, attribute, [0, len(alphas)]) 
        return from_im, [latent[0], latent_residual]


class ImagesDataset_W(Dataset):
    """ImagesDataset for pivot tuning.
    """
    def __init__(self,
                 source_root,
                 latent_root,
                 source_transform=None):
        self.source_paths = sorted(make_dataset(source_root), \
                            key = lambda x: int(os.path.basename(x[1]).split('.')[0]))
        self.latent_root = latent_root
        self.source_transform = source_transform

    def __len__(self):
        if DEBUG:
            return 100
        # return min(len(self.source_paths), 3000)
        return len(self.source_paths)

    def __getitem__(self, index):
        _, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            from_im = self.source_transform(from_im)
        latent = torch.load(os.path.join(self.latent_root, f'{index + 1}.pt'))
        return from_im, latent

class ImagesDatasetF(ImagesDataset):
    """ImagesDataset for pivot tuning.
    """
    def __init__(
                 self,
                 source_root,
                 latent_root,
                 f_latent_root,
                 source_transform=None,
                 kmeans_info: dict = None,
                 random: bool = False
                ):
        super().__init__(
                         source_root,
                         latent_root,
                         source_transform,
                         kmeans_info,
                         random
                        )
        self.f_space = f_latent_root

    def __getitem__(self, index):
        if self.remap is not None:
            index = self.remap[index]
        _, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            from_im = self.source_transform(from_im)
        latent = torch.load(os.path.join(self.latent_root, f'{index + 1}.pt'))
        f_latent = torch.load(os.path.join(self.f_space, f'{index}.pt'), map_location = 'cpu')
        return from_im, [x[0].detach().cpu() for x in latent], f_latent['f'][0]
        # return from_im, [x[0].detach().cpu() for x in latent], f_latent[0]

class ImagesDatasetHasMask(ImagesDataset):
    """ImagesDataset for pivot tuning.
    """
    def __init__(
                 self,
                 source_root,
                 latent_root,
                 source_transform=None,
                 kmeans_info: dict = None,
                 random: bool = False
                ):
        super().__init__(
                         source_root,
                         latent_root,
                         source_transform,
                         kmeans_info,
                         random
                        )
        self.face_parser = face_parsing()
        self.cache = dict()

    def get_facial_mask(
                        self, 
                        x, 
                        index
                       ):
        if index not in self.cache.keys():
            from_cache = Image.fromarray(np.uint8(self.face_parser(x) * 255.0))
            self.cache[index] = from_cache
        else:
            from_cache = self.cache[index]
            
        return from_cache

    def __getitem__(self, index):
        if self.remap is not None:
            index = self.remap[index]
        _, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        mask = self.get_facial_mask(np.array(from_im), index)
        if self.source_transform:
            from_im = self.source_transform(from_im)
            mask = self.source_transform(mask)
        latent = torch.load(os.path.join(self.latent_root, f'{index + 1}.pt'))
        return from_im, [x[0].detach().cpu() for x in latent], index, mask

import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import click
import re
import numpy as np

from DeepLog import logger
from kmeans_pytorch import kmeans, kmeans_predict
from ExpressiveEncoding.ImagesDataset import make_dataset
DEBUG = os.environ.get("DEBUG", "False")
DEBUG = True if DEBUG in ["True", "TRUE", "true"] else False

def kmeans_data(
                 data_dir: str,
                 param: str = None,
                 group_size: int = 25
               ):

    if param is not None and os.path.exists(param):
        logger.info("kmeans file exists.")
        return torch.load(param)
    def check_num(x):
        ans = re.findall('[0-9]+', x)
        return len(ans) > 0
    files = filter(check_num, os.listdir(data_dir))
    data_files = sorted(files, key = lambda x: int(''.join(re.findall('[0-9]+', x))))
    data_files = [os.path.join(data_dir, x) for x in data_files]

    if DEBUG:
        datas = [torch.load(_file) for _file in data_files[:10]]
        group_size = 5
    else:
        datas = [torch.load(_file) for _file in data_files]
    if isinstance(datas[0], torch.Tensor):
        datas = torch.cat(datas, dim = 0)
    elif isinstance(datas[0], list):
        tmp_datas = []
        for data in datas:
            data = torch.cat(data, dim = 1)
            tmp_datas.append(data)
        datas = torch.cat(tmp_datas, dim = 0)
    else:
        raise RuntimeError(f"{type(datas[0])} expected type Tensor or list.")

    n, c = datas.shape[0],datas.shape[1]
    clusters = n // group_size

    cluster_ids, cluster_centers = kmeans(X = datas.reshape(n, -1), num_clusters = clusters, \
                                          distance = 'euclidean', device = torch.device('cuda'))

    if param is not None:
        torch.save(
                    dict(cluster_ids = cluster_ids,
                         cluster_centers = cluster_centers,
                        ),
                    param
                  )
    return dict(
                cluster_ids = cluster_ids,
                cluster_centers = cluster_centers
               )
    
    


if __name__ == "__main__":

    argv = sys.argv

    data_dir = argv[1]
    param = argv[2]
    kmeans_data(data_dir, param)



   


    


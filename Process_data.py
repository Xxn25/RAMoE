# @Time    : 2024/12/30 10:25
# @Author  : Nan Xiao
# @File    : Process_data.py
import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
from scipy.io import loadmat


class ProcessData(data.Dataset):
    def __init__(self, data_dir="", train=True, patch_size=16, scale=4):
        super(ProcessData, self).__init__()
        self.paths = get_image_paths(data_dir)
        self.train = train
        self.scale = scale
        self.patch_size = patch_size

    def __getitem__(self, index):
        img_path = self.paths[index]
        hrhsi, lrhsi, hrmsi = imread_from_mat(img_path)
        HRHSI = float2tensor3(hrhsi)
        LRHSI = float2tensor3(lrhsi)
        HRMSI = float2tensor3(hrmsi)
        return HRHSI, LRHSI, HRMSI, img_path


    def __len__(self):
        return len(self.paths)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.mat']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def imread_from_mat(path):
    #  input: path
    # output: HxWxC
    datas = loadmat(path)
    hrhsi = datas['hrhsi']
    lrhsi = datas['lrhsi']
    hrmsi = datas['hrmsi']
    return hrhsi, lrhsi, hrmsi


# convert float to 3-dimensional torch tensor
def float2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
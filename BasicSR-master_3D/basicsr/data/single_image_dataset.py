"""
Code adapted by Sonia Laguna to work on 3D and niftis.
"""
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from skimage.transform import resize, rescale

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY
import os
import nibabel as nib
import numpy as np
import random
from scipy.ndimage import gaussian_filter

@DATASET_REGISTRY.register()
class SingleImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Loading inference data
        lq_path = self.paths[index]
        # Reading niftis
        img_nib = nib.load(lq_path)
        data = img_nib.get_fdata()

        img_lq = data[..., np.newaxis]
        # Mormalizing data
        img_lq = 2*np.divide(img_lq - np.amin(img_lq), np.amax(img_lq) - np.amin(img_lq)) - 1

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        print(img_lq.type(), 'type of img lq')

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)

"""
Code adapted by Sonia Laguna to work on 3D, niftis, and perform the required data augmentation for low-field MRI super-resolution training.
"""
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from skimage.transform import resize, rescale

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
# from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
import numpy as np
from skimage.transform import resize
import nibabel as nib
import os
import random
from scipy.ndimage import gaussian_filter

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                 self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths_gt, self.paths_lq = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
            self.lr_len = len(os.listdir(self.lq_folder))
            self.lr_shuf = np.arange(self.lr_len)
            np.random.shuffle(self.lr_shuf)
            self.lr_idx = 0

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Loading ground tuth data
        gt_path = self.paths_gt[index]

        #Reading niftis
        img_nib = nib.load(gt_path)
        data = img_nib.get_fdata()
        img_gt = data[..., np.newaxis]
        img_gt = 2*np.divide(img_gt - np.amin(img_gt), np.amax(img_gt) - np.amin(img_gt)) - 1 #Trained 0-1 bc I forgot to do *2 -1 but will do it now!, tenlo en cuenta para inferencia



        img_lq_hcp = 0
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            # Segmentation guided intensity mask
            if self.opt['dataugm']:
                file_name = os.path.split(gt_path)[1]
                seg_path = self.opt['dataroot_seg']
                if os.path.basename(os.path.dirname(self.opt['dataroot_gt'])) == 'T2':
                    seg_nib = nib.load(seg_path + 'subject_'+ file_name[:6] + '.seg.mgz')
                    print('You are in T2')
                else:
                    seg_nib = nib.load(seg_path + file_name[:-7] + 'seg.mgz')
                seg_data = seg_nib.get_fdata().astype(int)
                LUT_path = '../../LUT.npy'
                lut = np.load(LUT_path)

                # how much you want to corrupt / augment, sample this number between 0 and 0.2
                strength = 0.2 * np.random.uniform(size=1)

                # Apply LUT to segmentation
                segm_new = lut[seg_data].astype(int)
                nlab = 11
                factors = np.exp(strength * np.random.normal(size=nlab + 1))
                F = factors[segm_new]
                F_blur =gaussian_filter(F, sigma=1)
                F_blur =F_blur[...,np.newaxis]
                img_gt = img_gt * F_blur
                del F_blur, file_name, seg_nib, seg_data, lut, segm_new, F

            # Only for T2s because they are saved as 0.7mm
            if os.path.basename(os.path.dirname(self.opt['dataroot_gt'])) == 'T2':
                img_gt = rescale(img_gt, [0.7 / 1, 0.7 / 1, 0.7 / 1, 1], anti_aliasing=True)
                print('You are in T2')

            # When finetuning all meta-architecture, loading low-field MRI data
            if self.opt['cut'] == 1 or self.opt['cycle'] == 1 :  # Concatenating the denoiser, SR and DA
                index_lq = self.lr_shuf[self.lr_idx]
                # index_lq = index # If at any point we had paired data for training
                lq_path = self.paths_lq[index_lq]
                self.lr_idx += 1
                if self.lr_idx >= self.lr_len:
                    self.lr_idx = 0
                    np.random.shuffle(self.lr_shuf)
                img_nib = nib.load(lq_path)
                data = img_nib.get_fdata()
                img_lq = data[..., np.newaxis]
                img_lq = 2 * np.divide(img_lq - np.amin(img_lq), np.amax(img_lq) - np.amin(img_lq)) - 1
            else:
                img_lq = img_gt
                lq_path = 'k'
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

            # Augmenting contrast and intensity
            inten = 0.01 * np.random.randint(10)
            contr = 0.01 * np.random.randint(85, high=115)
            img_gt = img_gt * contr + inten + 0.5*(1-contr)

            # Adding bias field
            ratio = 1
            stdev_bias = 0.3 * np.random.uniform(size=1)
            bias_pre = np.random.normal(0, stdev_bias, size=(np.uint8(ratio * 5), np.uint8(ratio * 5), 5))
            bias = np.exp(resize(bias_pre, (img_gt.shape[0], img_gt.shape[1], img_gt.shape[2])))
            img_gt = img_gt * (bias[..., np.newaxis])
            img_gt = np.clip(img_gt, -1, 1)

            # Downsampling
            if not self.opt['cut'] == 1 and not self.opt['cycle'] == 1 or not self.opt['denois']==1: # Training the SR alone, on clean low resolution data
                img_lq = rescale(img_gt, [1 / 1.6, 1 / 1.6, 0.2, 1], anti_aliasing=True)

            # Downsampling and adding noise
            if self.opt['denois'] == 2: # Training the denoiser and SR, on noisy data
                img_lq = rescale(img_gt, [1 / 1.6, 1 / 1.6, 0.2, 1], anti_aliasing=True)
                stdev = 0.1 * np.random.uniform(size=1)
                noise = np.random.normal(0, stdev, size=img_lq.shape)
                img_lq = (img_lq - noise)

        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, 0:img_lq.shape[2] * scale,:]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'lq_hcp': img_lq_hcp}

    def __len__(self):
        return len(self.paths_gt)

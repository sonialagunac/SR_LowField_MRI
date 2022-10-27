"""
Code adapted by Sonia Laguna to load 3D nifti data and work with the augmentations described in the published work
'Super-resolution of portable low-field MRI in real scenarios: integration with denoising and domain adaptation', MIDL 2022
"""
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import nibabel as nib
import random
import numpy as np
from skimage.transform import rescale, resize
import torch
from scipy.ndimage import gaussian_filter


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # Loading nifti files intead of jpegs
        img_nib_A = nib.load(A_path) #A is the low field MRI data
        A_img = np.squeeze(img_nib_A.get_fdata())
        img_nib_B = nib.load(B_path)
        B_img = np.squeeze(img_nib_B.get_fdata())
        A_img = 2*np.divide(A_img - np.amin(A_img), np.amax(A_img) - np.amin(A_img)) - 1
        B_img = 2*np.divide(B_img - np.amin(B_img), np.amax(B_img) - np.amin(B_img)) - 1

        # Data augmentation

        # Segmentation guided intensity mask
        # Uncomment the lines below and fill in the segmentation paths for segmentation guided data augmentation

        # file_name = os.path.split(B_path)[1]
        # if os.path.basename(os.path.dirname(B_path)) == 'T2':
        #     seg_path = '/autofs/cluster/HyperfineSR/sonia/data/3D/train/T2/HCP_seg/' # For T2
        #     seg_nib = nib.load(seg_path + 'subject_' + file_name[:6] + '.seg.mgz') # For T2
        # else:
        #     seg_path = '/autofs/cluster/HyperfineSR/sonia/data/3D/train/T1/HCP_seg1mm/'
        #     seg_nib = nib.load(seg_path + file_name[:-7] + 'seg.mgz')
        # seg_data = seg_nib.get_fdata().astype(int)
        # LUT_path = '../../BasicSR-master_3D/LUT.npy'
        # lut = np.load(LUT_path)
        # strength = 0.2 * np.random.uniform(size=1)
        # segm_new = lut[seg_data].astype(int)
        # nlab = 11
        # factors = np.exp(strength * np.random.normal(size=nlab + 1))
        # F = factors[segm_new]
        # F_blur = gaussian_filter(F, sigma=1)
        # F_blur = F_blur[...]
        # aff = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        # B_img = B_img * F_blur
        # if os.path.basename(os.path.dirname(B_path)) == 'T2':
        #     B_img = rescale(B_img, [0.7 / 1, 0.7 / 1, 0.7 / 1], anti_aliasing=True)

        # Augmenting contrast and intensity
        inten = 0.01 * np.random.randint(10)
        contr = 0.01 * np.random.randint(85, high=115)
        B_img = B_img * contr + inten + 0.5 * (1 - contr)
        B_img = np.clip(B_img, -1, 1)

        # Adding bias field
        ratio = 1
        stdev_bias = 0.3 * np.random.uniform(size=1)
        bias_pre = np.random.normal(0, stdev_bias, size=(np.uint8(ratio * 5), np.uint8(ratio * 5), 5))
        bias = np.exp(resize(bias_pre, (B_img.shape[0], B_img.shape[1], B_img.shape[2])))
        B_img = B_img * (bias)
        B_img = np.clip(B_img, -1, 1)

        # Downsampling and adding noise
        B_img = rescale(B_img, [1 / 1.6, 1 / 1.6, 0.2], anti_aliasing=True)
        stdev = 0.1 * np.random.uniform(size=1)
        noise = np.random.normal(0, stdev, size=B_img.shape)
        # input = np.clip(labels - noise, -1, 1)
        B_img = (B_img - noise)

        # Patches generation
        patch = 32
        patch_f = 80
        top_hr = random.randint(0, A_img.shape[0] - patch_f)
        left_hr = random.randint(0, A_img.shape[1] - patch_f)
        depth_hr = random.randint(0, A_img.shape[2] - patch)
        # At train time:
        A = torch.Tensor(A_img[None, top_hr:top_hr + patch_f, left_hr:left_hr + patch_f, depth_hr:depth_hr + patch])
        # At test time:
        # A = torch.Tensor(A_img[None, ...])

        top_hr_B = random.randint(0, B_img.shape[0] - patch_f)
        left_hr_B = random.randint(0, B_img.shape[1] - patch_f)
        depth_hr_B = random.randint(0, B_img.shape[2] - patch)
        # At train time:
        B = torch.Tensor(B_img[None, top_hr_B:top_hr_B + patch_f, left_hr_B:left_hr_B + patch_f, depth_hr_B:depth_hr_B + patch])
        # At test time:
        # B = torch.Tensor(B_img[None, ...]) #At test time
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

"""
Utils of the denoiser module
Sonia Laguna
"""

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
import random
from skimage.transform import resize, rescale
from scipy.ndimage import gaussian_filter


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".gif", ".nii.gz", ".nii", ".mgz"])


class ImageDataset(Dataset):
    def __init__(self, image_dir_label,seg_dir):
        super(ImageDataset, self).__init__()
        self.image_filenames_label = [os.path.join(image_dir_label, x) for x in os.listdir(image_dir_label) if is_image_file(x)]
        self.image_dir_label = image_dir_label
        self.seg_dir_label = seg_dir

    def __getitem__(self, index):

        # Loading high resolution data
        img = nib.load(self.image_filenames_label[index])
        data = img.get_fdata()
        labels_hr = data[np.newaxis, ...]
        labels_hr = 2*(labels_hr - np.amin(labels_hr))/(np.amax(labels_hr)-np.amin(labels_hr)) -1

        # Data augmentation

        # Segmentation guided intensity mask
        file_name = os.path.split(self.image_filenames_label[index])[1]
        seg_nib = nib.load(self.seg_dir_label + file_name[:-7] + 'seg.mgz')
        seg_data = seg_nib.get_fdata().astype(int)
        LUT_path = '/autofs/cluster/HyperfineSR/sonia/BasicSR-master_3D/LUT.npy' # Load LUT
        lut = np.load(LUT_path)
        strength = 0.2 * np.random.uniform(size=1) # how much you want to corrupt / augment
        segm_new = lut[seg_data].astype(int) # Apply LUT to segmentation
        nlab = 11
        factors = np.exp(strength * np.random.normal(size=nlab + 1))
        F = factors[segm_new]
        F_blur = gaussian_filter(F, sigma=1)
        F_blur = F_blur[np.newaxis, ...]
        labels_hr = labels_hr * F_blur
        if os.path.basename(os.path.dirname(self.image_dir_label)) == 'T2':
            labels_hr = rescale(labels_hr, [1,0.7 / 1, 0.7 / 1, 0.7 / 1], anti_aliasing=True)

        # Augmenting contrast and intensity
        inten = 0.01 * np.random.randint(10)
        contr = 0.01 * np.random.randint(85, high=115)
        labels_hr = labels_hr * contr + inten + 0.5 * (1 - contr)
        labels_hr = np.clip(labels_hr, -1, 1)

        # Adding bias field
        ratio = 1
        stdev_bias = 0.3 * np.random.uniform(size=1)
        bias_pre = np.random.normal(0, stdev_bias, size=(np.uint8(ratio * 5),np.uint8(ratio * 5), 5))
        bias = np.exp(resize(bias_pre, (labels_hr.shape[1], labels_hr.shape[2], labels_hr.shape[3])))
        labels_hr = labels_hr * (bias[np.newaxis, ...])
        labels_hr = np.clip(labels_hr, -1, 1)

        # Random patches generation
        patch = 128
        top_hr = random.randint(0, labels_hr.shape[1] - patch)
        left_hr = random.randint(0, labels_hr.shape[2] - patch)
        depth_hr = random.randint(0, labels_hr.shape[3] - patch)
        labels_hr = labels_hr[:,top_hr:top_hr + patch, left_hr:left_hr + patch, depth_hr:depth_hr + patch]

        # Downsampling to low-field MRI resolution
        labels = rescale(labels_hr, [1, 1/1.6, 1/1.6, 1/5], anti_aliasing=True)
        stdev = 0.1 * np.random.uniform(size=1)
        noise = np.random.normal(0, stdev, size=labels.shape)
        input = (labels - noise)

        # Horizontal flip
        if random.random() < 0.5:
            np.flip(input, 0)
            np.flip(labels, 0)

        # Vertical flip
        if random.random() < 0.5:
            np.flip(input, 1)
            np.flip(labels, 1)

        # Depth flip
        if random.random() < 0.5:
            np.flip(input, 2)
            np.flip(labels, 2)

        return input, labels

    def __len__(self):
        return len(self.image_filenames_label)


def save_checkpoint(epoch, out_path, state):
    model_out_path = "./experiments/" + out_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def image_loader_test(image_name,seg_dir):
    """load image, returns tensor"""

    img_nib = nib.load(image_name)
    data = img_nib.get_fdata()
    data = 2 * np.divide(data - np.amin(data), np.amax(data) - np.amin(data)) - 1
    data = data[None, ...]

    # Data augmentation

    # Segmentation guided intensity mask
    file_name = os.path.split(image_name)[1]
    seg_nib = nib.load(seg_dir + file_name[:-7] + 'seg.mgz')
    seg_data = seg_nib.get_fdata().astype(int)
    LUT_path = '../BasicSR-master_3D/LUT.npy'
    lut = np.load(LUT_path)
    strength = 0.2 * np.random.uniform(size=1)
    segm_new = lut[seg_data].astype(int)
    nlab = 11
    factors = np.exp(strength * np.random.normal(size=nlab + 1))
    F = factors[segm_new]
    F_blur = gaussian_filter(F, sigma=1)
    F_blur = F_blur[None, ...]
    img_gt = data * F_blur

    # Augmenting contrast and intensity
    inten = 0.01 * np.random.randint(10)
    contr = 0.01 * np.random.randint(85, high=115)
    labels_hr = img_gt * contr + inten + 0.5 * (1 - contr)

    # Adding bias field
    ratio = 1
    stdev_bias = 0.3 * np.random.uniform(size=1)
    bias_pre = np.random.normal(0, stdev_bias, size=(5,np.uint8(ratio * 5), 5))
    bias = np.exp(resize(bias_pre, (labels_hr.shape[1], labels_hr.shape[2], labels_hr.shape[3])))
    labels_hr = labels_hr * (bias[np.newaxis, ...])
    labels_hr = np.clip(labels_hr, -1, 1)

    # Downsampling to low-field MRI resolution
    labels = rescale(labels_hr, [1, 1 / 1.6, 1 / 1.6, 1 / 5], anti_aliasing=True)
    stdev = 0.1 * np.random.uniform(size=1)
    noise = np.random.normal(0, stdev, size=labels.shape)
    input = (labels - noise)
    image = torch.tensor(input).cuda().float()
    image = image.unsqueeze(0)

    return image, labels, noise, bias, labels_hr


def remove_noise(model, image, label, labels_hr, aff, out_path, bias):

    pred = model(image)
    image_np = np.squeeze(image.detach().cpu().numpy())
    pred_np = np.squeeze(pred.detach().cpu().numpy())
    out = image_np + pred_np
    img_ni = nib.Nifti1Image(out, affine=aff)
    nib.save(img_ni, './results/' + out_path + '_pred_image.nii')

def validate():
    pass

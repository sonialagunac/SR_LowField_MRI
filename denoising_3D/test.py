"""
Code for inference of the denoiser module
Sonia Laguna
"""

import torch
import os
from archs import DnCNN
from utils import remove_noise, image_loader_test

# Setting experiment parameters
aff = [[1.5, 0., 0., 0.], [0., 1.5, 0., 0.], [0., 0., 5., 0.], [0., 0., 0., 1.]]
# Set directory for test data
test_dir = '/autofs/cluster/HyperfineSR/sonia/data/3D/test/T1/HCP_1mm/'
seg_dir = '/autofs/cluster/HyperfineSR/sonia/data/3D/test/T1/HCP_seg1mm/'

# Set name of experiment to load and checkpoint
out_path = 'denoising_T2_newdataaugm_w10'
ckpt = '1200'
if not os.path.exists("./results/" + out_path):
    os.makedirs("./results/" + out_path)

# Starting inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DnCNN().to(device)
model = torch.load("./experiments/" + out_path + "/model_epoch_" + ckpt + ".pth")

model = model['arch']
files=os.listdir(test_dir)

for index in range(len(files)):
    # Data loader
    image, labels, noise, bias, labels_hr = image_loader_test(test_dir + files[index], seg_dir)

    # Inference
    remove_noise(model, image, labels, labels_hr, aff, out_path + '/' + str(index), bias)

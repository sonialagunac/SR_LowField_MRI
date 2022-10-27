"""
Script used to create the LUT clusters that are used in the data augmentation,
precisely, in the segmentation guided intensity mask.
Sonia Laguna, 2022
"""

import numpy as np
from scipy.ndimage import gaussian_filter

#Clustering labels and creating LUT
groups = []
groups.append((517, 515, 501, 0)) #Air-background
groups.append((16, 46, 7, 2, 41, 28, 60, 85, 530)) #Global white matter
groups.append((8, 3, 42, 53, 54, 18, 17, 11, 50, 26, 58, 77, 80, 47)) #Global gray matter
groups.append((24, 31, 63,  62, 30)) #Gloabl CSF "messy"
groups.append((15, 44, 5, 14, 43, 4, 72, 520, 506)) #Global CSF "clean"
groups.append((514,)) #Veins
groups.append((29, 511,  507, 508, 509, 502, 516)) #Soft tissues
groups.append((52,13)) #Pallidum
groups.append((12,51)) #Putamen
groups.append((512,)) #Spinal cord
groups.append((10,49)) #Thalamus

lut = np.zeros((531))
for g in range(len(groups)):
    for l in range(len(groups[g])):
        lut[groups[g][l]] = g + 1

# Use desired LUT location
LUT_path = '/autofs/cluster/HyperfineSR/sonia/BasicSR-master_3D/LUT.npy'
np.save(LUT_path, lut)

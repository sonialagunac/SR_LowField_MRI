"""
Script used to create the train and test splits coming from subjects of HCP data at 1mm resolution.
precisely, in the segmentation guided intensity mask.
Sonia Laguna, 2022
"""
import os
import numpy as np
import nibabel as nib

# Directory with all HCP data
hcp_data = '/autofs/cluster/HyperfineSR/sonia/HCP_1mm/4sonia'
data=os.listdir(hcp_data)

# Creating training set
# Logging subjects used
textfile = open("/autofs/cluster/HyperfineSR/sonia/data/3D/train/train_data.txt", "w")
sub = []
for i in np.arange(300,600,5): # Using 60 subjects
    for k in [1]:
        # Saving directories
        data_out_3D_im = os.path.join('/autofs/cluster/HyperfineSR/sonia/data/3D/train/T'+ str(k)+'/HCP_1mm/')
        data_out_3D_seg = os.path.join('/autofs/cluster/HyperfineSR/sonia/data/3D/train/T' + str(k) + '/HCP_seg1mm/')

        sub_loc = data[i][8:-8]
        while sub_loc in sub:
                i += 2
                sub_loc = data[i][8:-8]
        sub.append(sub_loc)

        # Loading data
        img=nib.load(hcp_data + '/subject_' + sub_loc + '.ima.mgz')
        seg=nib.load(hcp_data + '/subject_' + sub_loc + '.seg.mgz')
        data_img=img.get_fdata()
        data_seg=seg.get_fdata()
        aff_img = img.affine
        aff_seg = seg.affine
        data_img = np.swapaxes(np.flip(data_img, (1)), 1, 2)
        data_seg = np.swapaxes(np.flip(data_seg, (1)), 1, 2)

        # Cropping
        data_img_new = data_img[37:220, 19:238, :]
        data_seg_new = data_seg[37:220, 19:238, :]

        j=1
        while data_img_new[:,:,-j].any() == False:
                j+=1

        pos = j-1
        pos_init = 74 - pos #256 - 182 so that the total shape is consistent
        if pos_init <= 0:
                print(pos_init, 'init calculated changing to 0', pos, 'elements cut from the bottom')
                pos_init = 0

        data_img_new=data_img_new[:,:,pos_init:-pos]
        data_seg_new = data_seg_new[:, :, pos_init:-pos]

        aff_img_2 = [[1, 0., 0., 0.], [0., 1, 0., 0.], [0., 0., 1, 0.], [0., 0., 0., 1.]]

        # Saving data
        data_img_new_ni = nib.Nifti1Image(data_img_new, affine=aff_img_2)
        data_seg_new_ni = nib.Nifti1Image(data_seg_new, affine=aff_img_2)
        nib.save(data_img_new_ni, data_out_3D_im + 'subject_' + sub_loc + '.ima.mgz')
        nib.save(data_seg_new_ni, data_out_3D_seg + 'subject_' + sub_loc + '.seg.mgz')

    textfile.write(data[-1] + "\n")

textfile.close()

# Creating test set

for j in range(200,206): # Testing on 6 subjects
    for k in [1]:
        # Saving directories
        data_out_3D_im = os.path.join('/autofs/cluster/HyperfineSR/sonia/data/3D/test/T' + str(k) + '/HCP_1mm/')
        data_out_3D_seg = os.path.join('/autofs/cluster/HyperfineSR/sonia/data/3D/test/T' + str(k) + '/HCP_seg1mm/')

        sub_loc = data[i][8:-8]
        while sub_loc in sub:
                i += 2
                sub_loc = data[i][8:-8]
        sub.append(sub_loc)

        # Loading data
        img = nib.load('/autofs/cluster/HyperfineSR/sonia/HCP_1mm/4sonia/subject_' + sub_loc + '.ima.mgz')
        seg = nib.load('/autofs/cluster/HyperfineSR/sonia/HCP_1mm/4sonia/subject_' + sub_loc + '.seg.mgz')
        data_img = img.get_fdata()
        data_seg = seg.get_fdata()
        aff_img = img.affine
        aff_seg = seg.affine
        data_img = np.swapaxes(np.flip(data_img, (1)), 1, 2)
        data_seg = np.swapaxes(np.flip(data_seg, (1)), 1, 2)

        # Crop
        data_img_new = data_img[37:220, 19:238, :]
        data_seg_new = data_seg[37:220, 19:238, :]

        j = 1
        while data_img_new[:, :, -j].any() == False:
                j += 1
        pos = j - 1
        pos_init = 74 - pos  # 256 - 182 so that the total shape is consistent
        if pos_init <= 0:
                print(pos_init, 'init calculated changing to 0', pos, 'elements cut from the bottom')
                pos_init = 0

        data_img_new = data_img_new[:, :, pos_init:-pos]
        data_seg_new = data_seg_new[:, :, pos_init:-pos]

        aff_img_2 = [[1, 0., 0., 0.], [0., 1, 0., 0.], [0., 0., 1, 0.], [0., 0., 0., 1.]]

        # Saving data
        data_img_new_ni = nib.Nifti1Image(data_img_new, affine=aff_img)
        data_seg_new_ni = nib.Nifti1Image(data_seg_new, affine=aff_seg)

        nib.save(data_img_new_ni, data_out_3D_im + 'subject_' + sub_loc + '.ima.mgz')
        nib.save(data_seg_new_ni, data_out_3D_seg + 'subject_' + sub_loc + '.seg.mgz')
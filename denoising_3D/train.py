"""
Code for training of the denoiser module
Sonia Laguna
"""

from archs import DnCNN
from utils import ImageDataset, validate, save_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter

# Setting experiment parameters
# Set directory for training data
train_dir = '/autofs/cluster/HyperfineSR/sonia/data/3D/train/T1/HCP_1mm'
seg_dir = '/autofs/cluster/HyperfineSR/sonia/data/3D/train/T1/HCP_seg1mm'

# Set name of experiment
out_path = 'denoising_T2_newdataaugm'
# Training parameters
batch_size = 1
num_epochs = 300000
lr = 5e-3
if not os.path.exists("./experiments/" + out_path):
    os.makedirs("./experiments/" + out_path)

# Starting training
tb_logger = SummaryWriter(log_dir='./tb_logger/' + out_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DnCNN().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = ImageDataset(train_dir,seg_dir)
training_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

for epoch in range(1, num_epochs + 1):
    epoch_loss = 0
    epoch_loss_1 = 0
    for iteration, (data, label) in enumerate(training_data_loader, 1):
        target = label.float().to(device)
        input = data.float().to(device)
        noise = (target - input).float().to(device)

        output = model(input)
        loss = criterion(output, noise)
        epoch_loss += loss.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    tb_logger.add_scalar('Loss/train', epoch_loss / len(training_data_loader), epoch)

    if epoch % 50 == 0:
        print('Saving model in: ' + out_path)
        save_checkpoint(epoch, out_path, {
            'epoch': epoch + 1,
            'arch': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
    if epoch % 3000 == 0:
        validate()

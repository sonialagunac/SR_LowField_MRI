"""
Architecture of the denoiser module
Sonia Laguna
"""

import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        hidden_layers = []
        for i in range(5):
            hidden_layers.append(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True))
            hidden_layers.append(nn.BatchNorm3d(32))
            hidden_layers.append(nn.ReLU(inplace=True))

        self.mid_layer = nn.Sequential(*hidden_layers)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out1 = self.relu1(self.conv1(x))
        out2 = self.mid_layer(out1)
        out = self.conv3(out2)
        return out

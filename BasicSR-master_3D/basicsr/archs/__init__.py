import importlib
from copy import deepcopy
from os import path as osp
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '/autofs/cluster/HyperfineSR/sonia/BasicSR-master_3D')
import torch.nn as nn
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'basicsr.archs.{file_name}') for file_name in arch_filenames]
class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        hidden_layers = []
        for i in range(5): #Orig was5
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

def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net

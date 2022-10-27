"""
Code adapted by Sonia Laguna to mofidy network architecture. Three modules (denoising, super-resolution and domain adaptation) are loaded and impelmented in this script
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import os

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from basicsr.archs.rrdbnet_arch import combo, DnCNN, combo_GAN, cycle, ResnetGenerator
import functools


@MODEL_REGISTRY.register()
class SRGANModel(SRModel):
    """SRGAN model for single image super-resolution."""

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            # self.net_g_ema = build_network(self.opt['network_g']).to(self.device)

            # define network, original SR architecture
            net_g = build_network(self.opt['network_g']).to(self.device)
            # Loading the denoiser architecture
            if 'train' in self.opt['datasets']:  # If we are training
                if self.opt['datasets']['train']['cut'] == 0 and self.opt['datasets']['train']['cycle'] == 0 and \
                        self.opt['datasets']['train']['denois'] == 1:
                    pretrained = DnCNN().to(torch.cuda.current_device())
                    # Loading previous SR weights
                    resume_state_path_sr = os.path.join(
                        '/autofs/cluster/HyperfineSR/sonia/BasicSR-master_3D/experiments/',
                        self.opt['datasets']['train']['model_sr'], 'models',
                        self.opt['datasets']['train']['epoch_sr'])
                    net_g.load_state_dict(torch.load(resume_state_path_sr)['params'])

                    # Loading denoiser
                    resume_state_path_d = os.path.join('/autofs/cluster/HyperfineSR/sonia/denoising_3D/experiments/',
                                                       self.opt['datasets']['train']['model_d'],
                                                       self.opt['datasets']['train']['epoch_d'])
                    pretrained.load_state_dict(torch.load(resume_state_path_d)['state_dict'])
                    self.net_g = combo(net_g, pretrained)
                elif self.opt['datasets']['train']['cut'] == 0 or self.opt['datasets']['train']['cycle'] == 0:
                    self.net_g_ema = net_g
            else:  # we are testing
                if self.opt['datasets']['test_1']['cut'] == 0 and self.opt['datasets']['test_1']['cycle'] == 0 and \
                        self.opt['datasets']['test_1']['denois'] == 1:
                    pretrained = DnCNN().to(torch.cuda.current_device())
                    self.net_g = combo(net_g, pretrained)
                elif self.opt['datasets']['test_1']['cut'] == 0 or self.opt['datasets']['test_1']['cycle'] == 0:
                    self.net_g_ema = net_g

            # Case of Domain Adaptation
            if 'train' in self.opt['datasets']:  # If we are training
                if self.opt['datasets']['train']['cut'] == 1 or self.opt['datasets']['train']['cycle'] == 1:
                    trained = combo(net_g, DnCNN().to(torch.cuda.current_device())).to(torch.cuda.current_device())
                    resume_state_path_da = os.path.join(
                        '/autofs/cluster/HyperfineSR/sonia/BasicSR-master_3D/experiments/',
                        self.opt['datasets']['train']['model_srd'], 'models',
                        self.opt['datasets']['train']['epoch_srd'])
                    trained.load_state_dict(torch.load(resume_state_path_da)['params'])

                if self.opt['datasets']['train']['cycle']:
                    net = cycle(1, 1, 64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6)
                    resume_state_path_GAN = os.path.join(
                        '/autofs/cluster/HyperfineSR/sonia/pytorch-CycleGAN-and-pix2pix-master_3D/checkpoints/',
                        self.opt['datasets']['train']['model_da'],
                        self.opt['datasets']['train']['epoch_da'])
                    net.load_state_dict(torch.load(resume_state_path_GAN))
                    self.net_g_ema = combo_GAN(trained, net).to(torch.cuda.current_device())

                if self.opt['datasets']['train']['cut']:
                    net = ResnetGenerator(1, 1, 64, norm_layer=functools.partial(nn.BatchNorm3d, affine=True,
                                                                                 track_running_stats=True),
                                          use_dropout=False, no_antialias=False, no_antialias_up=False, n_blocks=6,
                                          opt=None)
                    resume_state_path_fastCUT = os.path.join(
                        '/autofs/cluster/HyperfineSR/sonia/contrastive-unpaired-translation-master_3D/checkpoints/',
                        self.opt['datasets']['train']['model_daf'], self.opt['datasets']['train']['epoch_daf'])
                    net.load_state_dict(torch.load(resume_state_path_fastCUT))
                    self.net_g_ema = combo_GAN(trained, net).to(torch.cuda.current_device())
            else:  # When we are testing
                if self.opt['datasets']['test_1']['cut'] == 1 or self.opt['datasets']['test_1']['cycle'] == 1:
                    trained = combo(net_g, DnCNN().to(torch.cuda.current_device())).to(torch.cuda.current_device())
                if self.opt['datasets']['test_1']['cycle']:
                    net = cycle(1, 1, 64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6)
                    self.net_g_ema = combo_GAN(trained, net).to(torch.cuda.current_device())

                if self.opt['datasets']['test_1']['cut']:
                    net = ResnetGenerator(1, 1, 64, norm_layer=functools.partial(nn.BatchNorm3d, affine=True,
                                                                                 track_running_stats=True),
                                          use_dropout=False, no_antialias=False, no_antialias_up=False, n_blocks=6,
                                          opt=None)
                    self.net_g_ema = combo_GAN(trained, net).to(torch.cuda.current_device())

            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)
        #
        #Loading disc
        # resume_state_path_d = '/autofs/cluster/HyperfineSR/sonia/BasicSR-master/experiments/ESRGAN_denoising_freezing_semi_bias_contr_lessGAN_13/models/net_d_160000.pth'
        # self.net_d.load_state_dict(torch.load(resume_state_path_d)['params'])

        self.net_d_2 = build_network(self.opt['network_d_2'])
        self.net_d_2 = self.model_to_device(self.net_d_2)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)
        self.optimizer_d_2 = self.get_optimizer(optim_type, self.net_d_2.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d_2)

    def optimize_parameters(self, current_iter):
         # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

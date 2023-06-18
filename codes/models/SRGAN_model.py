import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss
import numpy as np
import torchvision

## Initializing the model
import PerceptualSimilarity.models as models
LPIPS_model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)

logger = logging.getLogger('base')


class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.t_num = 13

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        self.netC = networks.define_C(opt).to(self.device)
        if opt['dist']:
            self.netC = DistributedDataParallel(self.netC, device_ids=[torch.cuda.current_device()])
        else:
            self.netC = DataParallel(self.netC)

        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if opt['dist']:
                self.netD = DistributedDataParallel(self.netD,
                                                    device_ids=[torch.cuda.current_device()])
            else:
                self.netD = DataParallel(self.netD)

            # self.netG.train()
            self.netC.train()
            # self.netD.train()

        self.netF_54 = networks.define_F_54(opt, use_bn=False).to(self.device)
        self.netF_44 = networks.define_F_44(opt, use_bn=False).to(self.device)
        self.netF_34 = networks.define_F_34(opt, use_bn=False).to(self.device)
        self.netF_22 = networks.define_F_22(opt, use_bn=False).to(self.device)
        self.netF_12 = networks.define_F_12(opt, use_bn=False).to(self.device)

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'l2+F2':
                    self.cri_pix = (nn.MSELoss().to(self.device))*0.5 + (nn.MSELoss().to(self.device))*0.5
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                if opt['dist']:
                    self.netF_54 = DistributedDataParallel(self.netF_54,
                                                           device_ids=[torch.cuda.current_device()])
                    self.netF_44 = DistributedDataParallel(self.netF_44,
                                                           device_ids=[torch.cuda.current_device()])
                    self.netF_34 = DistributedDataParallel(self.netF_34,
                                                            device_ids=[torch.cuda.current_device()])
                    self.netF_22 = DistributedDataParallel(self.netF_22,
                                                           device_ids=[torch.cuda.current_device()])
                    self.netF_12 = DistributedDataParallel(self.netF_12,
                                                           device_ids=[torch.cuda.current_device()])
                else:
                    self.netF_54 = DataParallel(self.netF_54)
                    self.netF_44 = DataParallel(self.netF_44)
                    self.netF_34 = DataParallel(self.netF_34)
                    self.netF_22 = DataParallel(self.netF_22)
                    self.netF_12 = DataParallel(self.netF_12)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # # G
            # wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            # optim_params = []
            # for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            #     if v.requires_grad:
            #         optim_params.append(v)
            #     else:
            #         if self.rank <= 0:
            #             logger.warning('Params [{:s}] will not optimize.'.format(k))
            # self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
            #                                     weight_decay=wd_G,
            #                                     betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            # self.optimizers.append(self.optimizer_G)

            # C
            wd_C = train_opt['weight_decay_C'] if train_opt['weight_decay_C'] else 0
            optim_params = []
            for k, v in self.netC.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_C = torch.optim.Adam(optim_params, lr=train_opt['lr_C'],
                                                weight_decay=wd_C,
                                                betas=(train_opt['beta1_C'], train_opt['beta2_C']))
            self.optimizers.append(self.optimizer_C)

            # # D
            # wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
            #                                     weight_decay=wd_D,
            #                                     betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            # self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

        self.BTMap_on = 1

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        self.var_LPIPS_maps_best_idx = None


        if self.BTMap_on == 1:
            if self.is_train and data['LPIPS_maps']!=None:
                self.var_LPIPS_maps_best_idx = data['LPIPS_maps'].to(self.device)

        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # C
        self.optimizer_C.zero_grad()

        self.var_H = self.var_H.cuda()
        self.var_L = self.var_L.cuda()

        self.fea_vgg12 = self.netF_12(self.var_L)
        self.fea_vgg22 = self.netF_22(self.var_L)
        self.fea_vgg34 = self.netF_34(self.var_L)
        self.fea_vgg44 = self.netF_44(self.var_L)
        self.fea_vgg54 = self.netF_54(self.var_L)

        size_L = self.var_L.shape
        size = size_L[2:4]
        resize = torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.fea_vgg12 = resize(self.fea_vgg12)
        self.fea_vgg22 = resize(self.fea_vgg22)
        self.fea_vgg34 = resize(self.fea_vgg34)
        self.fea_vgg44 = resize(self.fea_vgg44)
        self.fea_vgg54 = resize(self.fea_vgg54)
        VGG_feats = torch.concat([self.var_L, self.fea_vgg12, self.fea_vgg22, self.fea_vgg34, self.fea_vgg44, self.fea_vgg54], dim=1)
        self.Condi = self.netC(VGG_feats)

        self.fake_H = self.netG((self.var_L, self.Condi))
        self.fake_H = self.fake_H.cuda()

        l_g_total = 0
        l_g_pix = 0
        LPIPS_dist = 0
        self.l_pix_w = 0.001
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                cri_pix_ = self.cri_pix(self.fake_H, self.var_H)
                l_g_pix = cri_pix_ * (self.l_pix_w)
                l_g_total = l_g_total + l_g_pix

                LPIPS_dist_temp = LPIPS_model.forward(self.fake_H, self.var_H)
                LPIPS_dist = LPIPS_dist_temp.mean()
                l_g_total = l_g_total + LPIPS_dist

                if self.BTMap_on == 1:
                    l_g_pix_BTMap = self.cri_pix(self.Condi, torch.unsqueeze(self.var_LPIPS_maps_best_idx[:,0,:,:], dim=1)) * 0.01
                    l_g_total = l_g_total + l_g_pix_BTMap

            l_g_total.backward()
            self.optimizer_C.step()

        # set log
        self.log_dict['l_g_total'] = l_g_total
        self.log_dict['l_g_pix'] = l_g_pix.item()
        self.log_dict['l_g_pix_BTMap'] = l_g_pix_BTMap.item()
        self.log_dict['l_g_LPIPS'] = LPIPS_dist.item()

    def test(self, opt):
        self.netG.eval()
        self.netC.eval()

        with torch.no_grad():
            self.t = opt['T_ctrl']

            self.var_L = self.var_L.cuda()

            self.fea_vgg12 = self.netF_12(self.var_L)
            self.fea_vgg22 = self.netF_22(self.var_L)
            self.fea_vgg34 = self.netF_34(self.var_L)
            self.fea_vgg44 = self.netF_44(self.var_L)
            self.fea_vgg54 = self.netF_54(self.var_L)

            size_L = self.var_L.shape
            size = size_L[2:4]
            resize = torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            self.fea_vgg12 = resize(self.fea_vgg12)
            self.fea_vgg22 = resize(self.fea_vgg22)
            self.fea_vgg34 = resize(self.fea_vgg34)
            self.fea_vgg44 = resize(self.fea_vgg44)
            self.fea_vgg54 = resize(self.fea_vgg54)
            VGG_feats = torch.concat(
                [self.var_L, self.fea_vgg12, self.fea_vgg22, self.fea_vgg34, self.fea_vgg44, self.fea_vgg54], dim=1)
            self.Condi = self.netC(VGG_feats) * self.t

            self.fake_H = self.netG((self.var_L, self.Condi))

        # self.netG.train()
        self.netC.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['CM'] = self.Condi.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

        # Condi
        s, n = self.get_network_description(self.netC)
        if isinstance(self.netC, nn.DataParallel) or isinstance(self.netC, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netC.__class__.__name__,
                                             self.netC.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netC.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network C structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load_G'])

        load_path_C = self.opt['path']['pretrain_model_C']
        if load_path_C is not None:
            logger.info('Loading model for C [{:s}] ...'.format(load_path_C))
            self.load_network(load_path_C, self.netC, self.opt['path']['strict_load_C'])

        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_step):
        # self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netC, 'C', iter_step)
        # self.save_network(self.netD, 'D', iter_step)



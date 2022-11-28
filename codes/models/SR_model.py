import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from .base_model import BaseModel
import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.unet_model as unet_model
import models.modules.discriminator_vgg_arch as VGGFeatureExtractor
import torchvision
from torch.nn.parallel import DataParallel

logger = logging.getLogger('base')

class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        self.rank = -1  # non dist training
        self.SR_model = RRDBNet_arch.RRDBNet(in_nc=opt['network_G']['in_nc'], out_nc=opt['network_G']['out_nc'], nf=opt['network_G']['nf'], nb=opt['network_G']['nb'])
        self.SR_model = self.SR_model.to(self.device)
        self.SR_model = DataParallel(self.SR_model)

        self.OOE_model = unet_model.UNet(n_channels=opt['network_C']['in_nc'], n_classes=opt['network_C']['out_nc'])
        self.OOE_model = self.OOE_model.to(self.device)
        self.OOE_model = DataParallel(self.OOE_model)

        self.netF_12 = VGGFeatureExtractor.define_F_12(opt, use_bn=False).to(self.device)
        self.netF_12 = DataParallel(self.netF_12)
        self.netF_22 = VGGFeatureExtractor.define_F_22(opt, use_bn=False).to(self.device)
        self.netF_22 = DataParallel(self.netF_22)
        self.netF_34 = VGGFeatureExtractor.define_F_34(opt, use_bn=False).to(self.device)
        self.netF_34 = DataParallel(self.netF_34)
        self.netF_44 = VGGFeatureExtractor.define_F_44(opt, use_bn=False).to(self.device)
        self.netF_44 = DataParallel(self.netF_44)
        self.netF_54 = VGGFeatureExtractor.define_F_54(opt, use_bn=False).to(self.device)
        self.netF_54 = DataParallel(self.netF_54)

        # print network
        self.print_network()
        self.load()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def test(self, opt, logger, img_name):
        self.SR_model.eval()
        self.OOE_model.eval()

        with torch.no_grad():
            # self.texture_gain = opt.T_ctrl
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
            fea = torch.concat(
                [self.var_L, self.fea_vgg12, self.fea_vgg22, self.fea_vgg34, self.fea_vgg44, self.fea_vgg54], dim=1)
            self.OOE = self.OOE_model(fea)
            # self.OOE = self.OOE * self.texture_gain

            self.fake_H = self.SR_model((self.var_L, self.OOE))
            self.fake_H = self.fake_H.cuda()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['OOE'] = self.OOE.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.SR_model)
        if isinstance(self.SR_model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.SR_model.__class__.__name__,
                                             self.SR_model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.SR_model.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network SR structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_SR = self.opt['path']['pretrain_model_SR']
        if load_path_SR is not None:
            logger.info('Loading model for SR [{:s}] ...'.format(load_path_SR))
            self.load_network(load_path_SR, self.SR_model, self.opt['path']['strict_load_SR'])

        load_path_OOE = self.opt['path']['pretrain_model_OOE']
        if load_path_OOE is not None:
            logger.info('Loading model for OOE [{:s}] ...'.format(load_path_OOE))
            self.load_network(load_path_OOE, self.OOE_model, self.opt['path']['strict_load_OOE'])
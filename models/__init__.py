# -*- coding: utf-8 -*-
from .semantic import DeepLabv3_plus, PSPNet, UNet, UnetResnetAE

ModelSelector = {'dlv3plus': DeepLabv3_plus,
                 'pspnet': PSPNet,
                 'unet': UNet,
                 'unet_ae': UnetResnetAE}

# -*- coding: utf-8 -*-
from .semantic import DeepLabv3_plus, PSPNet, UNet

ModelSelector = {'deeplabv3+': DeepLabv3_plus,
                 'pspnet': PSPNet,
                 'unet': UNet}

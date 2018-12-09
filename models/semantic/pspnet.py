# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from models import backbone
from modules import PPBlock
from utils import SyncBN2d


class PSPNet(nn.Module):
    def __init__(self, in_channels, num_classes, backend='resnet18', pool_scales=(1, 2, 3, 6), pretrained=True):
        '''
        :param in_channels:
        :param num_classes:
        :param backend: only support resnet, otherwise need to have low_features for aux
                        and high_features methods for out
        '''
        super(PSPNet, self).__init__()
        self.in_channes = in_channels
        self.num_classes = num_classes

        if hasattr(backend, 'low_features') and hasattr(backend, 'high_features') \
                and hasattr(backend, 'lastconv_channel'):
            self.backend = backend
        elif 'resnet' in backend:
            self.backend = ResnetBackend(backend, pretrained)
        else:
            raise NotImplementedError

        self.pp_out_channel = 256
        self.pyramid_pooling = PPBlock(self.backend.lastconv_channel, out_channel=self.pp_out_channel,
                                       pool_scales=pool_scales)

        self.cbr_last = nn.Sequential(nn.Conv2d(self.backend.lastconv_channel + self.pp_out_channel * len(pool_scales),
                                                self.pp_out_channel, kernel_size=3, padding=1, bias=False),
                                      SyncBN2d(self.pp_out_channel),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(self.pp_out_channel, num_classes, kernel_size=1))
        self.cbr_deepsup = nn.Sequential(nn.Conv2d(self.backend.lastconv_channel // 2,
                                                   self.backend.lastconv_channel // 4,
                                                   kernel_size=3, padding=1, bias=False),
                                         SyncBN2d(self.backend.lastconv_channel // 4),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(0.1),
                                         nn.Conv2d(self.backend.lastconv_channel // 4,
                                                   num_classes, kernel_size=1))

    def forward(self, x):
        h, w = x.size()[2:]
        aux = self.backend.low_features(x)
        x = self.backend.high_features(aux)

        x = self.pyramid_pooling(x)
        x = self.cbr_last(x)
        aux = self.cbr_deepsup(aux)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        if self.training:
            return aux, x
        else:
            return x


class ResnetBackend(nn.Module):
    def __init__(self, backend='resnet18', pretrained=True):
        '''
        :param backend: resnet<>
        '''
        super(ResnetBackend, self).__init__()
        _all_resnet_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if backend not in _all_resnet_models:
            raise NotImplementedError(f"{backend} must in {_all_resnet_models}")

        self._backend_model = eval(f"backbone.{backend}(pretrained={pretrained})")

        self.low_features = nn.Sequential(self._backend_model.conv1,
                                          self._backend_model.bn1,
                                          self._backend_model.relu,
                                          self._backend_model.maxpool,
                                          self._backend_model.layer1,
                                          self._backend_model.layer2,
                                          self._backend_model.layer3)

        self.high_features = nn.Sequential(self._backend_model.layer4)

        if backend in ['resnet18', 'resnet34']:
            self.lastconv_channel = 512
        else:
            self.lastconv_channel = 512 * 4


if __name__ == '__main__':
    from torchsummary import summary

    pspnet = PSPNet(in_channels=3, num_classes=21, backend='resnet18', pool_scales=(1, 2, 3, 6)).cuda()
    print(summary(pspnet, [3, 224, 224]))
    x = torch.randn(2, 3, 224, 224)
    out_aux, out = pspnet(x.cuda())
    print(out_aux.size())
    print(out.size())

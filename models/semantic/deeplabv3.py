# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from models import backbone
from modules import ASPPBlock
from utils import SyncBN2d


class DeepLabv3_plus(nn.Module):
    def __init__(self, in_channels, num_classes, backend='resnet18', os=16, pretrained=True):
        '''
        :param in_channels:
        :param num_classes:
        :param backend: only support resnet, otherwise need to have low_features
                        and high_features methods for out
        '''
        super(DeepLabv3_plus, self).__init__()
        self.in_channes = in_channels
        self.num_classes = num_classes
        if hasattr(backend, 'low_features') and hasattr(backend, 'high_features') \
                and hasattr(backend, 'lastconv_channel'):
            self.backend = backend
        elif 'resnet' in backend:
            self.backend = ResnetBackend(backend, pretrained)
        else:
            raise NotImplementedError

        self.aspp_out_channel = self.backend.lastconv_channel // 8
        self.aspp_pooling = ASPPBlock(self.backend.lastconv_channel, self.aspp_out_channel, os)

        self.cbr_low = nn.Sequential(nn.Conv2d(self.aspp_out_channel, self.aspp_out_channel // 4,
                                               kernel_size=1, bias=False),
                                     SyncBN2d(self.aspp_out_channel // 4),
                                     nn.ReLU(inplace=True))
        self.cbr_last = nn.Sequential(nn.Conv2d(self.aspp_out_channel + self.aspp_out_channel // 4,
                                                self.aspp_out_channel, kernel_size=3, padding=1, bias=False),
                                      SyncBN2d(self.aspp_out_channel),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.aspp_out_channel, self.aspp_out_channel,
                                                kernel_size=3, padding=1, bias=False),
                                      SyncBN2d(self.aspp_out_channel),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.aspp_out_channel, self.num_classes, kernel_size=1))

    def forward(self, x):
        h, w = x.size()[2:]
        low_features = self.backend.low_features(x)
        x = self.backend.high_features(low_features)
        x = self.aspp_pooling(x)
        x = F.interpolate(x, size=low_features.size()[2:], mode='bilinear', align_corners=True)
        low_features = self.cbr_low(low_features)

        x = torch.cat([x, low_features], dim=1)
        x = self.cbr_last(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
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
                                          self._backend_model.layer1
                                          )

        self.high_features = nn.Sequential(self._backend_model.layer2,
                                           self._backend_model.layer3,
                                           self._backend_model.layer4)

        if backend in ['resnet18', 'resnet34']:
            self.lastconv_channel = 512
        else:
            self.lastconv_channel = 512 * 4


if __name__ == '__main__':
    from torchsummary import summary

    deeplabv3_ = DeepLabv3_plus(in_channels=3, num_classes=21, backend='resnet18', os=16).cuda()
    print(summary(deeplabv3_, [3, 224, 224]))
    x = torch.randn(2, 3, 224, 224)
    out = deeplabv3_(x.cuda())
    print(out.size())

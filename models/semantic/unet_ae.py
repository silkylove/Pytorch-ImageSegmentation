# -*- coding: utf-8 -*-
import torch
from torch import nn


class UnetResnetAE(nn.Module):
    def __init__(self):
        super(UnetResnetAE, self).__init__()

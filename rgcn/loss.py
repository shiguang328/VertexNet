#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class L2_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_):
        return torch.mean(torch.pow((y - y_), 2))
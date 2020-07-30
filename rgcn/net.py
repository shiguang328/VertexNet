#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn


class RGCN(nn.Module):
    def __init__(self):
        super(RGCN, self).__init__()

        def conv2relu(input, output):

            net = nn.Sequential(nn.Conv2d(input, output, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(output, output, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                                )
            return net

        def conv3relu(input, output):
            net = nn.Sequential(nn.Conv2d(input, output, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(output, output, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(output, output, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                                )
            return net

        self.block1 = conv2relu(3, 64)
        self.block2 = conv2relu(64, 128)
        self.block3 = conv3relu(128, 256)
        self.block4 = conv3relu(256, 512)
        self.fc = nn.Linear(512*14*14, 8)

    def forward(self, x):

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def load():
    model = RGCN()
    return model

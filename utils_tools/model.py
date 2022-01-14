# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 16:43
# @Author  : Scotty
# @FileName: model.py
# @Software: PyCharm
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.out_dim = out_dim
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
        #                        kernel_size=(5, 5), stride=(3, 3),
        #                        bias=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10,
                               kernel_size=(2, 2), stride=(1, 1),
                               bias=True)
        self.actv1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
        #                        kernel_size=(3, 3), stride=(1, 1),
        #                        padding=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20,
                               kernel_size=(2, 2), stride=(1, 1),
                               padding=(1, 1), bias=True)
        self.actv2 = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(nn.Linear(20*25*25, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 128),
                                nn.Linear(128, self.out_dim))

    def forward(self, batch_img):
        out = self.conv1(batch_img)
        out = self.actv1(out)
        out = self.conv2(out)
        out = self.actv2(out)
        out = torch.flatten(out, 1, -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    model = Model(input_shape=(25, 25), out_dim=2)
    img = torch.randn((3, 3, 25, 25))
    out_shape = model(img)
    shape_show = out_shape.shape

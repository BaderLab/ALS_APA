import os, random, torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features):
        super(LinearAttentionBlock, self).__init__()
        self.op = nn.Conv1d(
            in_channels=in_features,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=False,
        )

    def forward(self, l):
        N, C, W = l.size()
        c = self.op(l)  # batch_sizex1xWxH
        a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W)  # attention values
        g2 = torch.mul(a.expand_as(l), l)
        g2 = g2.view(N, C, -1).sum(dim=2)  # batch_sizexC

        return c.view(N, 1, W), g2


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        cnvks=1,
        cnvst=1,
        poolks=1,
        poolst=1,
        pdropout=0,
        activation_t="none",
    ) -> None:
        super(ConvBlock, self).__init__()
        self.op = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, cnvks, cnvst, padding=cnvks // 2),
            # batch normalization before non-linear activation
            nn.BatchNorm1d(out_channel),
            nn.ELU() if activation_t == "ELU" else nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=poolks, stride=poolst),
            nn.Dropout(p=pdropout),
        )

    def forward(self, x):
        return self.op(x)


class FC_block(nn.Module):
    def __init__(self, layer_dims, dropouts, dropout=False):
        super(FC_block, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.BatchNorm1d(num_features=layer_dims[i + 1]))
                layers.append(nn.ReLU())
            if dropout and i < len(layer_dims) - 2:
                layers.append(nn.Dropout(p=dropouts[i]))
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)

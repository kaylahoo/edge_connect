import torch
import torch.nn.functional as F
from torch import nn


class DepConvBNActiv(nn.Module):

    def __init__(self, in_channels, out_channels, sample='none-3',groups=None):
        super(DepConvBNActiv, self).__init__()
        if sample == 'down-31':
            self.Dconv = Depthwise_separable_conv(in_channels, out_channels, kernel_size=31, stride=1,
                                                  padding=15, groups=4)

        elif sample == 'down-29':
            self.Dconv = Depthwise_separable_conv(in_channels, out_channels, kernel_size=29, stride=2, padding=14,
                                                  groups=in_channels)

        elif sample == 'down-27':
            self.Dconv = Depthwise_separable_conv(in_channels, out_channels, kernel_size=27, stride=2, padding=13,
                                                  groups=in_channels)
        elif sample == 'down-13':
            self.Dconv = Depthwise_separable_conv(in_channels, out_channels, kernel_size=13, stride=2, padding=6,
                                                  groups=in_channels)

        else:
            self.Dconv = Depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                                  groups=groups)

    def forward(self, images, masks):
        print('here', images.shape)
        print('here', masks.shape)

        images, masks = self.Dconv(images, masks)




        return images, masks


class Depthwise_separable_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(Depthwise_separable_conv, self).__init__()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            ),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(),
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=1,
            ),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(),
        )

    def forward(self, images, masks):
        images, masks = self.depthwise_conv(images, masks)

        images, masks = self.pointwise_conv(images, masks)
        return images, masks
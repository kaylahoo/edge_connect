import torch
import torch.nn.functional as F
from torch import nn
import timm

# from timm.models.layers import DropPath

from src.drop import drop_path


class DepConvBNActiv(nn.Module):

    def __init__(self, in_channels, out_channels, bn=True, sample='none-3', groups=None, activ='relu'):
        super(DepConvBNActiv, self).__init__()

        if sample == 'down-31':
            self.large_res = self.res_block(in_channels, out_channels, is_large_small='large', kernel_size=31, stride=1,
                                            padding=15, groups=in_channels)

            self.small_res = self.res_block(in_channels, out_channels, is_large_small='small')

            self.decrease_channels = nn.Conv2d(in_channels=in_channels + out_channels * 2, out_channels=out_channels, kernel_size=1)

            self.Dconv = Depthwise_separable_conv(out_channels, out_channels, kernel_size=31, stride=2,
                                                  padding=15, groups=out_channels)

        elif sample == 'down-29':
            self.large_res = self.res_block(in_channels, out_channels, is_large_small='large', kernel_size=29, stride=1,
                                            padding=14, groups=in_channels)

            self.small_res = self.res_block(in_channels, out_channels, is_large_small='small')

            self.decrease_channels = nn.Conv2d(in_channels=in_channels + out_channels * 2, out_channels=out_channels, kernel_size=1)

            self.Dconv = Depthwise_separable_conv(out_channels, out_channels, kernel_size=29, stride=2, padding=14,
                                                  groups=out_channels)

        elif sample == 'down-27':
            self.large_res = self.res_block(in_channels, out_channels, is_large_small='large', kernel_size=27, stride=1,
                                            padding=13, groups=in_channels)

            self.small_res = self.res_block(in_channels, out_channels, is_large_small='small')

            self.decrease_channels = nn.Conv2d(in_channels=in_channels + out_channels * 2, out_channels=out_channels, kernel_size=1)

            self.Dconv = Depthwise_separable_conv(out_channels, out_channels, kernel_size=27, stride=2, padding=13,
                                                  groups=out_channels)
        elif sample == 'down-13':
            self.large_res = self.res_block(in_channels, out_channels, is_large_small='large', kernel_size=13, stride=1,
                                            padding=6, groups=in_channels)

            self.small_res = self.res_block(in_channels, out_channels, is_large_small='small')

            self.decrease_channels = nn.Conv2d(in_channels=in_channels + out_channels * 2, out_channels=out_channels, kernel_size=1)

            self.Dconv = Depthwise_separable_conv(out_channels, out_channels, kernel_size=13, stride=2, padding=6,
                                                  groups=out_channels)

        else:
            self.large_res = None
            self.small_res = None
            self.decrease_channels = None
            self.Dconv = Depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                                  groups=groups)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU6()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def res_block(self, in_channels, out_channels, is_large_small=None, kernel_size=None, stride=None, padding=None,
                  groups=None):
        if is_large_small == 'large':
            return Depthwise_separable_conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, groups=groups)
        if is_large_small == 'small':
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, images, masks):
        if self.small_res is not None:

            images_l, masks_l = self.large_res(images, masks)  # 31,1,15

            images_s = self.small_res(images)  # 3,1,1
            masks_s = self.small_res(masks)  # 3,1,1

            images = torch.concat([images, images_l, images_s], dim=1)  #
            masks = torch.concat([masks, masks_l, masks_s], dim=1)   #

            images = self.decrease_channels(images)
            masks = self.decrease_channels(masks)

        images, masks = self.Dconv(images, masks)

        return images, masks


class Depthwise_separable_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(Depthwise_separable_conv, self).__init__()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            ),
            # nn.SyncBatchNorm(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )
        # def __init__(self,in_channels,out_channels):
        # super(Depthwise_separable_conv)
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
            ),
            # nn.SyncBatchNorm(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, images, masks):
        images1 = self.depthwise_conv(images)
        masks1 = self.depthwise_conv(masks)

        images2 = self.pointwise_conv(images1)
        masks2 = self.pointwise_conv(masks1)

        return images2, masks2
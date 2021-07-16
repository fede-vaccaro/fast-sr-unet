# the unet code is inspired from https://github.com/usuyama/pytorch-unet

import math
from builtins import super
import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleResNet(nn.Module):
    def __init__(self, n_filters, n_blocks):
        super(SimpleResNet, self).__init__()
        self.conv1 = UnetBlock(in_channels=3, out_channels=n_filters, use_residual=True, use_bn=False)
        convblock = [UnetBlock(in_channels=n_filters, out_channels=n_filters, use_residual=True, use_bn=False) for _ in
                     range(n_blocks - 1)]
        self.convblocks = nn.Sequential(*convblock)
        self.sr = sr_espcn(n_filters, scale_factor=2, out_channels=3)
        self.upscale = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.clip = nn.Hardtanh()

    def forward(self, input):
        x = self.conv1(input)
        x = self.convblocks(x)
        x = self.sr(x)

        return self.clip(x + self.upscale(input))

    def reparametrize(self):
        for block in self.convblocks:
            if hasattr(block, 'conv_adapter'):
                block.reparametrize_convs()


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, stride=1, kernel_size=3, use_residual=True):
        super(UnetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_reparametrized = False
        self.use_residual = use_residual

        # if in_channels == out_channels and use_residual:
        self.conv_adapter = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        # self.act = nn.ReLU6()
        # self.act = nn.LeakyReLU(0.2, inplace=True) # used by srgan_128x128_94_10-01-2021_0917_valloss0.30919_mobilenet_flickr.pkl
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        if self.use_residual and not self.is_reparametrized:
            if self.in_channels == self.out_channels:
                x += input + self.conv_adapter(input)
        x = self.bn(x)
        x = self.act(x)
        return x

    def reparametrize_convs(self):
        identity_conv = nn.init.dirac_(torch.empty_like(self.conv1.weight))
        padded_adapter_conv = F.pad(self.conv_adapter.weight, (1, 1, 1, 1), "constant", 0)
        #
        if self.in_channels == self.out_channels:
            new_conv_weights = self.conv1.weight + padded_adapter_conv + identity_conv
            new_conv_bias = self.conv1.bias + self.conv_adapter.bias

            self.conv1.weight.data = new_conv_weights
            self.conv1.bias.data = new_conv_bias

        self.is_reparametrized = True


def layer_generator(in_channels, out_channels, use_batch_norm=False, use_bias=True, residual_block=True, n_blocks=2):
    n_blocks = int(n_blocks)
    first_layer = UnetBlock(in_channels=in_channels, out_channels=out_channels, use_bn=use_batch_norm,
                            use_residual=residual_block)
    return nn.Sequential(*(
            [first_layer] + [
        UnetBlock(in_channels=out_channels, out_channels=out_channels, use_bn=use_batch_norm,
                  use_residual=residual_block)
        for _ in range(n_blocks - 1)]
        # nn.Conv2d(out_channels, out_channels, 3, padding=1),
        # nn.ReLU(inplace=True)
    ))


def sr_espcn(n_filters, scale_factor=2, out_channels=3, kernel_size=1):
    return nn.Sequential(*[
        nn.Conv2d(kernel_size=kernel_size, in_channels=n_filters, out_channels=(scale_factor ** 2) * out_channels,
                  padding=kernel_size // 2),
        nn.PixelShuffle(scale_factor),
    ])


class UNet(nn.Module):

    def __init__(self, in_dim=3, n_class=3, n_filters=32, downsample=None, residual=True, batchnorm=False,
                 scale_factor=2):
        """
        Args
            in_dim (float, optional):
                channel dimension of the input
            n_class (str).
                channel dimension of the output
            n_filters (int, optional)
                number of filters of the first channel. after layer it gets multiplied by 2 during the encoding stage,
                and divided during the decoding
            downsample (None or float, optional):
                can be used for downscaling the output. e.g., if you use downsample=0.5 the output resolution will be halved
            residual (bool):
                if using the residual scheme and adding the input to the final output
            scale_factor (int):
                basic upscale factor. if you want a rational upscale (e.g. 720p to 1080p, which is 1.5) combine it
                with the downsample parameter
        """

        super().__init__()

        self.residual = residual
        self.n_class = n_class
        self.scale_factor = scale_factor

        self.dconv_down1 = layer_generator(in_dim, n_filters, use_batch_norm=False)
        self.dconv_down2 = layer_generator(n_filters, n_filters * 2, use_batch_norm=batchnorm, n_blocks=2)
        self.dconv_down3 = layer_generator(n_filters * 2, n_filters * 4, use_batch_norm=batchnorm, n_blocks=2)
        self.dconv_down4 = layer_generator(n_filters * 4, n_filters * 8, use_batch_norm=batchnorm, n_blocks=2)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.dconv_up3 = layer_generator(n_filters * 8 + n_filters * 4, n_filters * 4, use_batch_norm=batchnorm,
                                         n_blocks=2)
        self.dconv_up2 = layer_generator(n_filters * 4 + n_filters * 2, n_filters * 2, use_batch_norm=batchnorm,
                                         n_blocks=2)
        self.dconv_up1 = layer_generator(n_filters * 2 + n_filters, n_filters, use_batch_norm=False, n_blocks=2)

        sf = (self.scale_factor * (2 if self.use_s2d else 1))

        self.to_rgb = nn.Conv2d(n_filters, 3, kernel_size=1)
        if sf > 1:
            self.conv_last = nn.Conv2d(n_filters, (sf ** 2) * n_class, kernel_size=1, padding=0)
            self.pixel_shuffle = nn.PixelShuffle(sf)
        else:
            self.conv_last = nn.Conv2d(n_filters, 3, kernel_size=1)

        if downsample is not None:
            self.downsample = 'interp'  # nn.UpsamplingBilinear2d(scale_factor=downsample)
        else:
            self.downsample = nn.Identity()
        self.layers = [self.dconv_down1, self.dconv_down2, self.dconv_down3, self.dconv_down4, self.dconv_up3,
                       self.dconv_up2, self.dconv_up1]

    def forward(self, input):
        x = input

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)

        sf = (self.scale_factor * (2 if self.use_s2d else 1))

        if sf > 1:
            x = self.pixel_shuffle(x)
        if self.residual:
            sf = self.scale_factor  # (self.scale_factor // (2 if self.use_s2d and self.scale_factor > 1 else 1))
            x += F.interpolate(input[:, -self.n_class:, :, :],
                               scale_factor=sf,
                               mode='bicubic')
            x = torch.clamp(x, min=-1, max=1)

        return x
        # return torch.clamp(F.interpolate(x, mode='bicubic', scale_factor=0.75, align_corners=True), min=-1, max=1)# self.downsample(x)

    def reparametrize(self):
        for layer in self.layers:
            for block in layer:
                if hasattr(block, 'conv_adapter'):
                    block.reparametrize_convs()


class SRUnet(nn.Module):

    def __init__(self, in_dim=3, n_class=3, downsample=None, residual=False, batchnorm=False, scale_factor=2,
                 n_filters=64, layer_multiplier=1):
        """
        Args:
            in_dim (float, optional):
                channel dimension of the input
            n_class (str):
                channel dimension of the output
            n_filters (int, optional):
                maximum number of filters. the layers start with n_filters / 2,  after each layer this number gets multiplied by 2
                 during the encoding stage and until it reaches n_filters. During the decoding stage the number follows the reverse
                 scheme. Default is 64
            downsample (None or float, optional)
                can be used for downscaling the output. e.g., if you use downsample=0.5 the output resolution will be halved
            residual (bool):
                if using the residual scheme and adding the input to the final output
            scale_factor (int):
                upscale factor. if you want a rational upscale (e.g. 720p to 1080p, which is 1.5) combine it
                 with the downsample parameter
            layer_multiplier (int or float):
                compress or extend the network depth in terms of total layers. configured as a multiplier to the number of the
                basic blocks which composes the layers
            batchnorm (bool, default=False):
                whether use batchnorm or not. If True should decrease quality and performances.
        """

        super().__init__()

        self.residual = residual
        self.n_class = n_class
        self.scale_factor = scale_factor

        self.dconv_down1 = layer_generator(in_dim, n_filters // 2, use_batch_norm=False,
                                           n_blocks=2 * layer_multiplier)
        self.dconv_down2 = layer_generator(n_filters // 2, n_filters, use_batch_norm=batchnorm,
                                           n_blocks=3 * layer_multiplier)
        self.dconv_down3 = layer_generator(n_filters, n_filters, use_batch_norm=batchnorm,
                                           n_blocks=3 * layer_multiplier)
        self.dconv_down4 = layer_generator(n_filters, n_filters, use_batch_norm=batchnorm,
                                           n_blocks=3 * layer_multiplier)

        self.maxpool = nn.MaxPool2d(2)
        if downsample is not None and downsample != scale_factor:
            self.downsample = nn.Upsample(scale_factor=downsample, mode='bicubic', align_corners=True)
        else:
            self.downsample = nn.Identity()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.dconv_up3 = layer_generator(n_filters + n_filters, n_filters, use_batch_norm=batchnorm,
                                         n_blocks=3 * layer_multiplier)
        self.dconv_up2 = layer_generator(n_filters + n_filters, n_filters, use_batch_norm=batchnorm,
                                         n_blocks=3 * layer_multiplier)
        self.dconv_up1 = layer_generator(n_filters + n_filters // 2, n_filters // 2, use_batch_norm=False,
                                         n_blocks=3 * layer_multiplier)

        self.layers = [self.dconv_down1, self.dconv_down2, self.dconv_down3, self.dconv_down4, self.dconv_up3,
                       self.dconv_up2, self.dconv_up1]

        sf = self.scale_factor

        self.to_rgb = nn.Conv2d(n_filters // 2, 3, kernel_size=1)
        if sf > 1:
            self.conv_last = nn.Conv2d(n_filters // 2, (sf ** 2) * n_class, kernel_size=1, padding=0)
            self.pixel_shuffle = nn.PixelShuffle(sf)
        else:
            self.conv_last = nn.Conv2d(n_filters // 2, 3, kernel_size=1)

    def forward(self, input):
        x = input

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)

        sf = self.scale_factor

        if sf > 1:
            x = self.pixel_shuffle(x)

        # x = self.to_rgb(x)
        if self.residual:
            sf = self.scale_factor  # (self.scale_factor // (2 if self.use_s2d and self.scale_factor > 1 else 1))
            x += F.interpolate(input[:, -self.n_class:, :, :],
                               scale_factor=sf,
                               mode='bicubic')
            x = torch.clamp(x, min=-1, max=1)

        return torch.clamp(self.downsample(x), min=-1, max=1)  # self.downsample(x)

    def reparametrize(self):
        for layer in self.layers:
            for block in layer:
                if hasattr(block, 'conv_adapter'):
                    block.reparametrize_convs()

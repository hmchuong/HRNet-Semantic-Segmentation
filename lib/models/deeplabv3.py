"""
Code Adapted from:
https://github.com/sthalles/deeplab_v3

Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import os
import torch
from torch import nn

from .bn_helper import BatchNorm2d as Norm2d, BatchNorm2d_class, relu_inplace
from .resnet import resnet50, resnet101

class get_resnet(nn.Module):
    def __init__(self, trunk_name, output_stride=8):
        super(get_resnet, self).__init__()

        if trunk_name == 'resnet-50':
            resnet = resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                          resnet.relu, resnet.maxpool)
        elif trunk_name == 'resnet-101':
            resnet = resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                          resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif output_stride == 16:
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            raise 'unsupported output_stride {}'.format(output_stride)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        s2_features = x
        x = self.layer2(x)
        s4_features = x
        x = self.layer3(x)
        x = self.layer4(x)
        return s2_features, s4_features, x

def get_trunk(trunk_name, output_stride=8):
    backbone = get_resnet(trunk_name, output_stride=output_stride)
    s2_ch = 256
    s4_ch = -1
    high_level_ch = 2048
    return backbone, s2_ch, s4_ch, high_level_ch

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=(6, 12, 18)):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1,
                                    bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

def get_aspp(high_level_ch, bottleneck_ch, output_stride, dpc=False):
    """
    Create aspp block
    """
    aspp = AtrousSpatialPyramidPoolingModule(high_level_ch, bottleneck_ch,
                                                 output_stride=output_stride)
    aspp_out_ch = 5 * bottleneck_ch
    return aspp, aspp_out_ch

def make_seg_head(in_ch, out_ch):
    bot_ch = 256
    return nn.Sequential(
        nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        Norm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        Norm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False))

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, Norm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class DeepV3Plus(nn.Module):
    """
    DeepLabV3+ with various trunks supported
    Always stride8
    """
    def __init__(self, num_classes, trunk='wrn38', criterion=None,
                 use_dpc=False, init_all=False):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8,
                                          dpc=use_dpc)
        self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        if init_all:
            initialize_weights(self.aspp)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.bot_fine)
            initialize_weights(self.final)
        else:
            initialize_weights(self.final)

    def forward(self, inputs):
        x = inputs

        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)
        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        final = self.final(cat_s4)
        out = Upsample(final, x_size[2:])
        return out


def DeepV3PlusSRNX50(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='seresnext-50', criterion=criterion)


def DeepV3PlusR50(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion)


def DeepV3PlusSRNX101(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='seresnext-101', criterion=criterion)


def DeepV3PlusW38(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='wrn38', criterion=criterion)


def DeepV3PlusW38I(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='wrn38', criterion=criterion,
                      init_all=True)


def DeepV3PlusX71(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='xception71', criterion=criterion)


def DeepV3PlusEffB4(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='efficientnet_b4',
                      criterion=criterion)

def get_seg_model(cfg, **kwargs):
    model = DeepV3PlusR50(cfg.DATASET.NUM_CLASSES, None)
    if os.path.isfile(cfg.MODEL.PRETRAINED):
        state_dict = torch.load(cfg.MODEL.PRETRAINED)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("==> Load pretrained weight", cfg.MODEL.PRETRAINED)
    return model

###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
SimpleNet_v1 network for AI85.
Simplified version of the network proposed in [1].

[1] HasanPour, Seyyed Hossein, et al. "Lets keep it simple, using simple architectures to
    outperform deeper and more complex architectures." arXiv preprint arXiv:1608.06037 (2016).
"""
from torch import nn

import ai8x


class new_model(nn.Module):
    """
    2x Wider SimpleNet v1 Model
    """
    def __init__(
            self,
            num_classes=2,
            num_channels=3,
            dimensions=(64, 64),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()

        self.drop = nn.Dropout2d(p=0.2)
        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 24, 3, stride=1, padding=1, bias=False, #bias=bias
                                            **kwargs)
        self.conv2 = ai8x.FusedConv2dReLU(24, 32, 3, stride=1, padding=1, bias=False, **kwargs)
        self.conv3 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(32, 32, 3, pool_size=4, pool_stride=4,
                                                   stride=1, padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv7 = ai8x.FusedConv2dBNReLU(32, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                   stride=1, padding=1, bias=bias, **kwargs)
        self.conv9 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv10 = ai8x.FusedMaxPoolConv2dBNReLU(64, 128, 3, pool_size=2, pool_stride=2,
                                                    stride=1, padding=1, bias=bias, **kwargs)
        self.conv11 = ai8x.FusedMaxPoolConv2dBNReLU(128, 512, 1, pool_size=2, pool_stride=2,
                                                    padding=0, bias=bias, **kwargs)
        self.conv12 = ai8x.FusedConv2dBNReLU(512, 192, 1, stride=1, padding=0, bias=bias, **kwargs)
        self.conv13 = ai8x.FusedMaxPoolConv2dBNReLU(192, 192, 3, pool_size=2, pool_stride=2,
                                                    stride=1, padding=1, bias=bias, **kwargs)
        self.conv14 = ai8x.Conv2d(192, num_classes, 1, stride=1, padding=0, bias=bias,
                                  wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.drop(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = x.view(x.size(0), -1)
        return x


def new_modelv1(pretrained=False, **kwargs):
    """
    Constructs a SimpleNet v1 model.
    """
    assert not pretrained
    return new_model(**kwargs)


models = [
    {
        'name': 'new_modelv1',
        'min_input': 1,
        'dim': 2,
    },
]

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_upsample_layer


class SherlockUpConvBlock(nn.Module):

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 #upsample_cfg=dict(),
                 dcn=None,
                 plugins=None):
        super().__init__()
        #super(SherlockUpConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=skip_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        # sherlock:
        if skip.size(2) == x.size(2) & skip.size(3) == x.size(3):
            out = torch.cat([skip, x], dim=1)
            out = self.conv_block(out)
        else:
            #x1 = self.up(x1)
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]

            x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))
            # x = F.pad(x, [diffX , 0,
            #                 diffY , 0])
            out = torch.cat([skip, x], dim=1)
            out = self.conv_block(out)
        
        # out = torch.cat([skip, x], dim=1)
        # out = self.conv_block(out)

        return out

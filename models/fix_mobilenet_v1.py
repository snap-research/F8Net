import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from .fix_quant_ops import (
    fix_quant,
    int_op_only_fix_quant,
    FXQAvgPool2d,
    ReLUClipFXQConvBN,
    ReLUClipFXQLinear,
)
from myutils.config import FLAGS


class IntBlock(nn.Module):
    def __init__(self, body):
        super(IntBlock, self).__init__()
        self.body = body
        self.int_op_only = getattr(body, 'int_op_only', False)

    def forward(self, x):
        assert getattr(FLAGS, 'int_infer', False)
        if getattr(self, 'int_op_only', False):
            res = x
            for layer_ in self.body:
                if isinstance(layer_, nn.Conv2d):
                    res = int_op_only_fix_quant(res, 8,
                                                layer_.input_fraclen.item(),
                                                res.output_fraclen,
                                                layer_.input_symmetric)
                    res = layer_(res)
                    output_fraclen = (layer_.weight_fraclen +
                                      layer_.input_fraclen).item()
                    setattr(res, 'output_fraclen', output_fraclen)
                else:
                    res = layer_(res)
        else:
            res = x
            for layer_ in self.body:
                if isinstance(layer_, nn.Conv2d):
                    res = (fix_quant(res, 8, layer_.input_fraclen, 1,
                                     layer_.input_symmetric)[0] *
                           (2**layer_.input_fraclen)).int().float()
                    res = layer_(res)
                    res.div_(2**(layer_.weight_fraclen + layer_.input_fraclen))
                else:
                    res = layer_(res)
        return res


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, outp, stride):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]

        l1 = ReLUClipFXQConvBN(inp, inp, 3, stride, 1, groups=inp)
        l2 = ReLUClipFXQConvBN(inp, outp, 1, 1, 0)
        layers = [
            l1,
            l2,
            nn.ReLU(inplace=True),
        ]
        self.body = nn.Sequential(*layers)
        self.layer_dict = {}

    def forward(self, x):
        return self.body(x)

    def set_following_layer(self, following_layer):
        self.layer_dict['following'] = following_layer
        self.body[0].set_following_layer(self.body[1])
        self.body[1].set_following_layer(following_layer)

    def get_following_layer(self):
        return self.layer_dict['following']

    def master_child(self):
        return self.body[0]

    def int_block(self, avgpool_scale=1.0):
        l1 = self.body[0].int_conv()
        l2 = self.body[1].int_conv(avgpool_scale=avgpool_scale)
        layers = [
            l1,
            nn.ReLU(inplace=True),
            l2,
            nn.ReLU(inplace=True),
        ]
        body = nn.Sequential(*layers)
        return IntBlock(body)


class IntModel(nn.Module):
    def __init__(self, head, body, classifier, block_setting):
        super(IntModel, self).__init__()

        self.block_setting = block_setting

        # head
        self.head = head

        # body
        for idx, [c, n, s] in enumerate(self.block_setting):
            for i in range(n):
                setattr(self, f'stage_{idx}_layer_{i}',
                        body[f'stage_{idx}_layer_{i}'])

        if getattr(FLAGS, 'quant_avgpool', False):
            self.avgpool = FXQAvgPool2d(7)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = classifier

        self.int_op_only = getattr(head, 'int_op_only', False)

    def forward(self, x):
        assert getattr(FLAGS, 'int_infer', False)
        if getattr(self, 'int_op_only', False):
            x = self.head(x)
            output_fraclen = (self.head[0].weight_fraclen +
                              self.head[0].input_fraclen).item()
            setattr(x, 'output_fraclen', output_fraclen)
            for idx, [_, n, _] in enumerate(self.block_setting):
                for i in range(n):
                    blk = getattr(self, f'stage_{idx}_layer_{i}')
                    x = blk(x)
            if getattr(FLAGS, 'quant_avgpool', False):
                x = self.avgpool(x)
                output_fraclen = x.output_fraclen
                x = x.view(x.size(0), -1)
                setattr(x, 'output_fraclen', output_fraclen)
                x = int_op_only_fix_quant(
                    x, 8, self.classifier[0].input_fraclen.item(),
                    x.output_fraclen, self.classifier[0].input_symmetric)
            else:
                output_fraclen = x.output_fraclen
                x = self.avgpool(x.float())
                x = x.view(x.size(0), -1)
                x.div_(2**output_fraclen)
                x = (fix_quant(x, 8, self.classifier[0].input_fraclen.float(),
                               1, self.classifier[0].input_symmetric)[0] *
                     (2**self.classifier[0].input_fraclen.float())).int()
            x = self.classifier(x).float()
        else:
            if getattr(FLAGS, 'normalize', False):
                x = (fix_quant(x, 8, self.head[0].input_fraclen, 1,
                               self.head[0].input_symmetric)[0] *
                     (2**self.head[0].input_fraclen)).int().float()
            else:
                x = (x * (2**self.head[0].input_fraclen)).int().float()
            x = self.head(x)
            x.div_(2**(self.head[0].input_fraclen +
                       self.head[0].weight_fraclen))
            for idx, [_, n, _] in enumerate(self.block_setting):
                for i in range(n):
                    blk = getattr(self, f'stage_{idx}_layer_{i}')
                    x = blk(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = (fix_quant(x, 8, self.classifier[0].input_fraclen, 1,
                           self.classifier[0].input_symmetric)[0] *
                 (2**self.classifier[0].input_fraclen)).int().float()
            x = self.classifier(x)
        return x


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        # head
        channels = 32
        first_stride = 2
        self.head = nn.Sequential(
            ReLUClipFXQConvBN(
                3,
                channels,
                3,
                first_stride,
                1,
                bitw_min=None,
                bita_min=8,
                weight_only=not getattr(FLAGS, 'normalize', False),
                double_side=getattr(FLAGS, 'normalize', False)),
            nn.ReLU(inplace=True),
        )
        prev_layer = self.head[0]

        # body
        for idx, [c, n, s] in enumerate(self.block_setting):
            outp = c
            for i in range(n):
                if i == 0:
                    layer = DepthwiseSeparableConv(channels, outp, s)
                else:
                    layer = DepthwiseSeparableConv(channels, outp, 1)
                setattr(self, 'stage_{}_layer_{}'.format(idx, i), layer)
                channels = outp
                prev_layer.set_following_layer(layer.master_child())
                prev_layer = layer

        if getattr(FLAGS, 'quant_avgpool', False):
            self.avgpool = FXQAvgPool2d(7)
            if getattr(FLAGS, 'pool_fusing', False):
                stage_idx = len(self.block_setting) - 1
                layer_idx = self.block_setting[-1][1] - 1
                last_conv_layer = getattr(
                    self, f'stage_{stage_idx}_layer_{layer_idx}')
                last_conv_layer.avgpool_scale = self.avgpool.scale
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = nn.Sequential(
            ReLUClipFXQLinear(outp, num_classes, bitw_min=None))
        prev_layer.set_following_layer(self.classifier[0])

        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.head(x)
        for idx, [_, n, _] in enumerate(self.block_setting):
            for i in range(n):
                x = getattr(self, 'stage_{}_layer_{}'.format(idx, i))(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def int_model(self):
        if getattr(FLAGS, 'quant_avgpool', False):
            avgpool = FXQAvgPool2d(7)
            avgpool_scale = avgpool.scale
        else:
            avgpool_scale = 1.0
        head = self.head
        head[0] = head[0].int_conv()
        body = {
            f'stage_{idx}_layer_{i}':
            getattr(self, f'stage_{idx}_layer_{i}').int_block(
                avgpool_scale=avgpool_scale if (
                    idx == len(self.block_setting) - 1 and i == n -
                    1) else 1.0)
            for idx, [c, n, s] in enumerate(self.block_setting)
            for i in range(n)
        }
        classifier = self.classifier
        classifier[0] = classifier[0].int_fc()
        return IntModel(head, body, classifier, self.block_setting)

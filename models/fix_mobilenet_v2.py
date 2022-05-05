import math
import numpy as np
import torch
import torch.nn as nn

from .fix_quant_ops import (fix_quant, int_op_only_fix_quant, FXQAvgPool2d,
                            ReLUClipFXQConvBN, ReLUClipFXQLinear)
from myutils.config import FLAGS


class IntBlock(nn.Module):
    def __init__(self, body, residual_connection):
        super(IntBlock, self).__init__()
        self.body = body
        self.residual_connection = residual_connection
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
            if self.residual_connection:
                res_fraclen = res.output_fraclen
                x_fraclen = x.output_fraclen
                if res_fraclen >= x_fraclen:
                    x = x << (res_fraclen - x_fraclen)
                    res += x
                    res.clamp_(max=(1 << 31) - 1, min=-(1 << 31) + 1)
                    output_fraclen = res_fraclen
                    setattr(res, 'output_fraclen', output_fraclen)
                else:
                    res = res << (x_fraclen - res_fraclen)
                    res += x
                    res.clamp_(max=(1 << 31) - 1, min=-(1 << 31) + 1)
                    output_fraclen = x_fraclen
                    setattr(res, 'output_fraclen', output_fraclen)
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
            setattr(res, 'output_fraclen',
                    self.body[-1].weight_fraclen + self.body[-1].input_fraclen)
            if self.residual_connection:
                res_fraclen = res.output_fraclen
                x_fraclen = x.output_fraclen
                output_fraclen = max(res_fraclen, x_fraclen)
                res = res * 2**output_fraclen
                x = x * 2**output_fraclen
                res += x
                res = torch.clamp(res, max=(1 << 31) - 1, min=-(1 << 31) + 1)
                res = res / 2**output_fraclen
                setattr(res, 'output_fraclen', output_fraclen)
        return res


class InvertedResidual(nn.Module):
    def __init__(self,
                 inp,
                 outp,
                 stride,
                 expand_ratio,
                 double_side=False,
                 master_layer=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            # expand
            l0 = ReLUClipFXQConvBN(inp,
                                   expand_inp,
                                   1,
                                   1,
                                   0,
                                   double_side=double_side,
                                   master_layer=master_layer)
            # depthwise + project back
            l1 = ReLUClipFXQConvBN(expand_inp,
                                   expand_inp,
                                   3,
                                   stride,
                                   1,
                                   groups=expand_inp)
            l2 = ReLUClipFXQConvBN(expand_inp, outp, 1, 1, 0)
            layers = [
                l0,
                l1,
                l2,
            ]
        else:
            # depthwise + project back
            l1 = ReLUClipFXQConvBN(expand_inp,
                                   expand_inp,
                                   3,
                                   stride,
                                   1,
                                   groups=expand_inp,
                                   double_side=double_side,
                                   master_layer=master_layer)
            l2 = ReLUClipFXQConvBN(expand_inp, outp, 1, 1, 0)
            layers = [
                l1,
                l2,
            ]
        self.body = nn.Sequential(*layers)
        self.layer_dict = {}

        self.residual_connection = stride == 1 and inp == outp
        if self.residual_connection:
            self.set_master_layer(self.body[0])
        else:
            self.set_master_layer(None)

    def forward(self, x):
        res = self.body(x)
        if self.residual_connection:
            if getattr(FLAGS, 'int_infer', False) and not self.training:
                res_fraclen = res.output_fraclen
                x_fraclen = x.output_fraclen
                output_fraclen = max(res_fraclen, x_fraclen)
                res = res * 2**output_fraclen
                x = x * 2**output_fraclen
                res += x
                res = torch.clamp(res, max=(1 << 31) - 1, min=-(1 << 31) + 1)
                res = res / 2**output_fraclen
                setattr(res, 'output_fraclen', output_fraclen)
            else:
                res += x
        return res

    def set_master_layer(self, master_layer):
        self.layer_dict['master'] = master_layer

    def get_master_layer(self):
        return self.layer_dict['master']

    def set_following_layer(self, following_layer):
        self.layer_dict['following'] = following_layer
        for idx in range(len(self.body) - 1):
            self.body[idx].set_following_layer(self.body[idx + 1])
        self.body[-1].set_following_layer(following_layer)

    def get_following_layer(self):
        return self.layer_dict['following']

    def master_child(self):
        return self.body[0]

    def int_block(self):
        layers = []
        layers.append(self.body[0].int_conv())
        for layer_ in self.body[1:]:
            layers.append(nn.ReLU(inplace=True))
            layers.append(layer_.int_conv())
        body = nn.Sequential(*layers)
        residual_connection = self.residual_connection
        return IntBlock(body, residual_connection)


class IntModel(nn.Module):
    def __init__(self, head, body, tail, classifier, block_setting):
        super(IntModel, self).__init__()

        self.block_setting = block_setting

        # head
        self.head = head

        # body
        for idx, [t, c, n, s] in enumerate(self.block_setting):
            for i in range(n):
                setattr(self, f'stage_{idx}_layer_{i}',
                        body[f'stage_{idx}_layer_{i}'])

        # tail
        self.tail = tail

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
            for idx, [_, _, n, _] in enumerate(self.block_setting):
                for i in range(n):
                    blk = getattr(self, f'stage_{idx}_layer_{i}')
                    x = blk(x)
            x = int_op_only_fix_quant(x, 8, self.tail[0].input_fraclen.item(),
                                      x.output_fraclen,
                                      self.tail[0].input_symmetric)
            x = self.tail(x)
            output_fraclen = (self.tail[0].weight_fraclen +
                              self.tail[0].input_fraclen).item()
            setattr(x, 'output_fraclen', output_fraclen)
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
            for layer_ in self.head:
                if hasattr(x, 'output_fraclen'):
                    output_fraclen = x.output_fraclen
                    x = layer_(x)
                else:
                    x = layer_(x)
                    output_fraclen = layer_.weight_fraclen + layer_.input_fraclen
                setattr(x, 'output_fraclen', output_fraclen)
            x.div_(2**x.output_fraclen)
            for idx, [_, _, n, _] in enumerate(self.block_setting):
                for i in range(n):
                    blk = getattr(self, f'stage_{idx}_layer_{i}')
                    x = blk(x)
            x = (fix_quant(x, 8, self.tail[0].input_fraclen, 1,
                           self.tail[0].input_symmetric)[0] *
                 (2**self.tail[0].input_fraclen)).int().float()
            x = self.tail(x)
            x.div_(2**(self.tail[0].input_fraclen +
                       self.tail[0].weight_fraclen))
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
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
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

        double_side = True
        # body
        master_layer = None
        for idx, [t, c, n, s] in enumerate(self.block_setting):
            outp = c
            for i in range(n):
                if i == 0:
                    layer = InvertedResidual(
                        channels,
                        outp,
                        s,
                        t,
                        double_side=double_side if idx != 0 else False,
                        master_layer=master_layer)
                else:
                    layer = InvertedResidual(channels,
                                             outp,
                                             1,
                                             t,
                                             double_side=double_side,
                                             master_layer=master_layer)
                setattr(self, 'stage_{}_layer_{}'.format(idx, i), layer)
                channels = outp
                master_layer = layer.get_master_layer()
                prev_layer.set_following_layer(layer.master_child())
                prev_layer = layer

        # tail
        outp = 1280
        self.tail = nn.Sequential(
            ReLUClipFXQConvBN(channels,
                              outp,
                              1,
                              1,
                              0,
                              double_side=double_side,
                              master_layer=master_layer),
            nn.ReLU(inplace=True),
        )
        prev_layer.set_following_layer(self.tail[0])
        prev_layer = self.tail[0]

        if getattr(FLAGS, 'quant_avgpool', False):
            self.avgpool = FXQAvgPool2d(7)
            if getattr(FLAGS, 'pool_fusing', False):
                stage_idx = len(self.block_setting) - 1
                layer_idx = self.block_setting[-1][1] - 1
                last_conv_layer = self.tail[0]
                last_conv_layer.avgpool_scale = self.avgpool.scale
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = nn.Sequential(
            ReLUClipFXQLinear(
                outp,
                num_classes,
                bitw_min=None,
                bias=True))
        prev_layer.set_following_layer(self.classifier[0])

        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.head(x)
        for idx, [_, _, n, _] in enumerate(self.block_setting):
            for i in range(n):
                x = getattr(self, 'stage_{}_layer_{}'.format(idx, i))(x)
        x = self.tail(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels
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
                if m.bias is not None:
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
            getattr(self, f'stage_{idx}_layer_{i}').int_block()
            for idx, [t, c, n, s] in enumerate(self.block_setting)
            for i in range(n)
        }
        tail = self.tail
        tail[0] = tail[0].int_conv(avgpool_scale=avgpool_scale)
        classifier = self.classifier
        classifier[0] = classifier[0].int_fc()
        return IntModel(head, body, tail, classifier, self.block_setting)

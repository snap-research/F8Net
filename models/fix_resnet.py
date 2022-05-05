import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from .fix_quant_ops import (fix_quant, int_op_only_fix_quant, FXQMaxPool2d,
                            FXQAvgPool2d, ReLUClipFXQConvBN, ReLUClipFXQLinear)

from myutils.config import FLAGS


class IntBlock(nn.Module):
    def __init__(self, body, shortcut=None):
        super(IntBlock, self).__init__()
        self.body = body

        self.residual_connection = shortcut is None
        if not self.residual_connection:
            self.shortcut = shortcut
        self.post_relu = nn.ReLU(inplace=True)
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
                if res_fraclen > x_fraclen:
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
                x = int_op_only_fix_quant(
                    x, 8, self.shortcut[0].input_fraclen.item(),
                    x.output_fraclen, self.shortcut[0].input_symmetric)
                x = self.shortcut(x)
                output_fraclen = (self.shortcut[-1].weight_fraclen +
                                  self.shortcut[-1].input_fraclen).item()
                setattr(x, 'output_fraclen', output_fraclen)
                res_fraclen = res.output_fraclen
                x_fraclen = x.output_fraclen
                if res_fraclen > x_fraclen:
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
            res = self.post_relu(res)
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
            else:
                x = (fix_quant(x, 8, self.shortcut[0].input_fraclen, 1,
                               self.shortcut[0].input_symmetric)[0] *
                     (2**self.shortcut[0].input_fraclen)).int().float()
                x = self.shortcut(x)
                setattr(
                    x, 'output_fraclen', self.shortcut[-1].weight_fraclen +
                    self.shortcut[-1].input_fraclen)
                x.div_(2**x.output_fraclen)
                res_fraclen = res.output_fraclen
                x_fraclen = x.output_fraclen
                output_fraclen = max(res_fraclen, x_fraclen)
                res = res * 2**output_fraclen
                x = x * 2**output_fraclen
                res += x
                res = torch.clamp(res, max=(1 << 31) - 1, min=-(1 << 31) + 1)
                res = res / 2**output_fraclen
            res = self.post_relu(res)
            setattr(res, 'output_fraclen', output_fraclen)
        return res


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, outp, stride, master_layer=None):
        super(BasicBlock, self).__init__()
        assert stride in [1, 2]

        l1 = ReLUClipFXQConvBN(inp,
                               outp,
                               3,
                               stride,
                               1,
                               master_layer=master_layer)
        l2 = ReLUClipFXQConvBN(outp, outp, 3, 1, 1)
        layers = [
            l1,
            l2,
        ]
        self.body = nn.Sequential(*layers)
        self.layer_dict = {}

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                ReLUClipFXQConvBN(inp,
                                  outp,
                                  1,
                                  stride=stride,
                                  master_layer=master_layer))
            self.set_master_layer(None)
        else:
            self.set_master_layer(self.body[0])
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.body(x)
        if getattr(FLAGS, 'int_infer', False) and not self.training:
            if self.residual_connection:
                res_fraclen = res.output_fraclen
                x_fraclen = x.output_fraclen
                output_fraclen = max(res_fraclen, x_fraclen)
                res = res * 2**output_fraclen
                x = x * 2**output_fraclen
                res += x
                res = torch.clamp(res, max=(1 << 31) - 1, min=-(1 << 31) + 1)
                res = res / 2**output_fraclen
            else:
                x = self.shortcut(x)
                res_fraclen = res.output_fraclen
                x_fraclen = x.output_fraclen
                output_fraclen = max(res_fraclen, x_fraclen)
                res = res * 2**output_fraclen
                x = x * 2**output_fraclen
                res += x
                res = torch.clamp(res, max=(1 << 31) - 1, min=-(1 << 31) + 1)
                res = res / 2**output_fraclen
        else:
            if self.residual_connection:
                res += x
            else:
                res += self.shortcut(x)
        res = self.post_relu(res)
        if getattr(FLAGS, 'int_infer', False) and not self.training:
            setattr(res, 'output_fraclen', output_fraclen)
        return res

    def set_master_layer(self, master_layer):
        self.layer_dict['master'] = master_layer

    def get_master_layer(self):
        return self.layer_dict['master']

    def set_following_layer(self, following_layer):
        self.layer_dict['following'] = following_layer
        self.body[0].set_following_layer(self.body[1])
        self.body[1].set_following_layer(following_layer)
        if not self.residual_connection:
            self.shortcut[0].set_following_layer(following_layer)

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
        ]
        body = nn.Sequential(*layers)

        if not self.residual_connection:
            shortcut = nn.Sequential(self.shortcut[0].int_conv())
        else:
            shortcut = None
        return IntBlock(body, shortcut)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp, outp, stride, master_layer=None):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]

        midp = outp // 4
        l1 = ReLUClipFXQConvBN(inp, midp, 1, 1, 0, master_layer=master_layer)
        l2 = ReLUClipFXQConvBN(midp, midp, 3, stride, 1)
        l3 = ReLUClipFXQConvBN(midp, outp, 1, 1, 0)
        layers = [
            l1,
            l2,
            l3,
        ]
        self.body = nn.Sequential(*layers)
        self.layer_dict = {}

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                ReLUClipFXQConvBN(inp,
                                  outp,
                                  1,
                                  stride=stride,
                                  master_layer=master_layer))
            self.set_master_layer(None)
        else:
            self.set_master_layer(self.body[0])
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.body(x)
        if getattr(FLAGS, 'int_infer', False) and not self.training:
            if self.residual_connection:
                res_fraclen = res.output_fraclen
                x_fraclen = x.output_fraclen
                output_fraclen = max(res_fraclen, x_fraclen)
                res = res * 2**output_fraclen
                x = x * 2**output_fraclen
                res += x
                res = torch.clamp(res, max=(1 << 31) - 1, min=-(1 << 31) + 1)
                res = res / 2**output_fraclen
            else:
                x = self.shortcut(x)
                res_fraclen = res.output_fraclen
                x_fraclen = x.output_fraclen
                output_fraclen = max(res_fraclen, x_fraclen)
                res = res * 2**output_fraclen
                x = x * 2**output_fraclen
                res += x
                res = torch.clamp(res, max=(1 << 31) - 1, min=-(1 << 31) + 1)
                res = res / 2**output_fraclen
        else:
            if self.residual_connection:
                res += x
            else:
                res += self.shortcut(x)
        res = self.post_relu(res)
        if getattr(FLAGS, 'int_infer', False) and not self.training:
            setattr(res, 'output_fraclen', output_fraclen)
        return res

    def set_master_layer(self, master_layer):
        self.layer_dict['master'] = master_layer

    def get_master_layer(self):
        return self.layer_dict['master']

    def set_following_layer(self, following_layer):
        self.layer_dict['following'] = following_layer
        self.body[0].set_following_layer(self.body[1])
        self.body[1].set_following_layer(self.body[2])
        self.body[2].set_following_layer(following_layer)
        if not self.residual_connection:
            self.shortcut[0].set_following_layer(following_layer)

    def get_following_layer(self):
        return self.layer_dict['following']

    def master_child(self):
        return self.body[0]

    def int_block(self, avgpool_scale=1.0):
        l1 = self.body[0].int_conv()
        l2 = self.body[1].int_conv()
        l3 = self.body[2].int_conv(avgpool_scale=avgpool_scale)
        layers = [l1, nn.ReLU(inplace=True), l2, nn.ReLU(inplace=True), l3]
        body = nn.Sequential(*layers)

        if not self.residual_connection:
            shortcut = nn.Sequential(self.shortcut[0].int_conv())
        else:
            shortcut = None
        return IntBlock(body, shortcut)


class IntModel(nn.Module):
    def __init__(self, head, body, classifier, block_setting):
        super(IntModel, self).__init__()

        self.block_setting = block_setting

        # head
        self.head = head

        if getattr(FLAGS, 'quant_maxpool', False):
            self.head[-1] = FXQMaxPool2d(self.head[-1].kernel_size,
                                         self.head[-1].stride,
                                         self.head[-1].padding)

        # body
        for idx, n in enumerate(self.block_setting):
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
            if getattr(FLAGS, 'quant_maxpool', False):
                x = self.head(x)
            else:
                x = self.head[:-1](x)
                x = self.head[-1](x.float()).int()
            output_fraclen = (self.head[0].weight_fraclen +
                              self.head[0].input_fraclen).item()
            setattr(x, 'output_fraclen', output_fraclen)
            for idx, n in enumerate(self.block_setting):
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
            for layer_ in self.head:
                if hasattr(x, 'output_fraclen'):
                    output_fraclen = x.output_fraclen
                    x = layer_(x)
                else:
                    x = layer_(x)
                    output_fraclen = layer_.weight_fraclen + layer_.input_fraclen
                setattr(x, 'output_fraclen', output_fraclen)
            x.div_(2**x.output_fraclen)
            for idx, n in enumerate(self.block_setting):
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

        block_type_dict = {
            18: BasicBlock,
            34: BasicBlock,
            50: Bottleneck,
            101: Bottleneck,
            152: Bottleneck
        }
        block = block_type_dict[FLAGS.depth]

        # head
        channels = 64
        self.head = nn.Sequential(
            ReLUClipFXQConvBN(
                3,
                channels,
                7,
                2,
                3,
                bitw_min=None,
                bita_min=8,
                weight_only=not getattr(FLAGS, 'normalize', False),
                double_side=getattr(FLAGS, 'normalize', False)),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        prev_layer = self.head[0]

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        self.block_setting = self.block_setting_dict[FLAGS.depth]

        feats = [64, 128, 256, 512]

        # body
        master_layer = None
        for idx, n in enumerate(self.block_setting):
            outp = feats[idx] * block.expansion
            for i in range(n):
                if i == 0 and idx != 0:
                    layer = block(channels, outp, 2, master_layer=master_layer)
                else:
                    layer = block(channels, outp, 1, master_layer=master_layer)
                setattr(self, 'stage_{}_layer_{}'.format(idx, i), layer)
                channels = outp
                master_layer = layer.get_master_layer()
                prev_layer.set_following_layer(layer.master_child())
                prev_layer = layer

        if getattr(FLAGS, 'quant_avgpool', False):
            self.avgpool = FXQAvgPool2d(7)
            if getattr(FLAGS, 'pool_fusing', False):
                stage_idx = len(self.block_setting) - 1
                layer_idx = self.block_setting[-1] - 1
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
        if getattr(FLAGS, 'int_infer', False) and not self.training:
            for layer_ in self.head:
                if hasattr(x, 'output_fraclen'):
                    output_fraclen = x.output_fraclen
                    x = layer_(x)
                else:
                    x = layer_(x)
                    output_fraclen = x.output_fraclen
                setattr(x, 'output_fraclen', output_fraclen)
        else:
            x = self.head(x)
        for idx, n in enumerate(self.block_setting):
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
            for idx, n in enumerate(self.block_setting) for i in range(n)
        }
        classifier = self.classifier
        classifier[0] = classifier[0].int_fc()
        return IntModel(head, body, classifier, self.block_setting)

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair


def bn_calib(m):
    if isinstance(m, (ReLUClipFXQConvBN, ReLUClipFXQConvTransposeBN)):
        m.training = True
    if getattr(m, 'track_running_stats', False):
        m.reset_running_stats()
        m.training = True
        m.momentum = None


def fraclen_gridsearch(input, wl, align_dim, signed):
    err_list = []
    for fl in range(wl + 1 - int(signed)):
        res, _ = fix_quant(
            input, wl,
            torch.ones(input.shape[align_dim]).to(input.device) * fl * 1.0,
            align_dim, signed)
        err = torch.mean((input - res)**2)**0.5
        err_list.append(err)
    opt_fl = torch.argmin(torch.tensor(err_list)).to(input.device) * 1.0
    return opt_fl


def metric2fraclen(metric_tensor, metric='std', N=1, signed=True):
    if signed:
        coeff = {'std': 40, 'mae': 30, 'rms': 40}[metric]
    else:
        coeff = {'std': 70, 'mae': 30, 'rms': 50}[metric]
    fl = torch.floor(torch.log2(coeff * N / metric_tensor))
    fl.clamp_(max=8 - int(signed), min=0)
    return fl


class q_k(Function):
    """
        This is the quantization function.
        The input and output should be all on the interval [0, 1].
        bit is only defined on positive integer values.
    """
    @staticmethod
    def forward(ctx,
                input,
                wl=8,
                fl=0,
                align_dim=0,
                signed=True,
                floating=False):
        res, grad_flag = fix_quant(input, wl, fl, align_dim, signed, floating)
        ctx.grad_scale = grad_flag
        return res

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * ctx.grad_scale
        return grad_input, None, None, None, None, None


def fix_quant(input, wl=8, fl=0, align_dim=0, signed=True, floating=False):
    assert wl >= 0
    assert torch.all(fl >= 0)
    if signed:
        assert torch.all(fl <= wl - 1)
    else:
        assert torch.all(fl <= wl)
    assert type(wl) == int
    assert torch.all(torch.round(fl) == fl)
    expand_dim = input.dim() - align_dim - 1
    fl = fl[(..., ) + (None, ) * expand_dim]
    res = input * (2**fl)
    if not floating:
        res.round_()
    if signed:
        bound = 2**(wl - 1) - 1
        grad_scale = torch.abs(res) < bound
        res.clamp_(max=bound, min=-bound)
    else:
        bound = 2**wl - 1
        grad_scale = (res > 0) * (res < bound)
        res.clamp_(max=bound, min=0)
    res.div_(2**fl)
    return res, grad_scale


def int_op_only_fix_quant(input, wl=8, fl=0, input_fl=0, signed=True):
    assert wl >= 0
    assert fl >= 0
    if signed:
        assert fl <= wl - 1
    else:
        assert fl <= wl
    assert type(wl) == int
    assert type(fl) == int
    net_fl = input_fl - fl
    if net_fl > 0:
        ## round half to even
        res = input + (1 << (net_fl - 1))
        res = torch.where((input % (1 << net_fl)) == (1 << (net_fl - 1)),
                          (res >> (net_fl + 1)) << 1, res >> net_fl)
    else:
        res = input << (-net_fl)
    if signed:
        bound = (1 << (wl - 1)) - 1
        res.clamp_(max=bound, min=-bound)
    else:
        bound = (1 << wl) - 1
        res.clamp_(max=bound, min=0)
    setattr(res, 'output_fraclen', fl)
    return res


class FXQAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size):
        super(FXQAvgPool2d, self).__init__(kernel_size=kernel_size)
        self.kernel_size = kernel_size
        self.shiftnum = torch.round(
            torch.log2(torch.tensor(self.kernel_size**2))).int().item()
        self.scale = 2**self.shiftnum / (self.kernel_size**2)
        self.int_op_only = False

    def forward(self, x):
        if self.int_op_only:
            output_fraclen = x.output_fraclen + self.shiftnum
            assert output_fraclen <= 32
            res = x.sum(-1).sum(
                -1)  ## will automatically convert int32 to int64
            assert torch.all(res <= (1 << 32) - 1)
            res = res.int()
            setattr(res, 'output_fraclen', output_fraclen)
        else:
            res = x.sum(-1).sum(-1)
            res.div_(2**self.shiftnum)
        return res


class FXQMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(FXQMaxPool2d, self).__init__(kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

    def forward(self, x):
        output_fraclen = getattr(x, 'output_fraclen', None)
        res = nn.functional.pad(x, (self.padding[1], self.padding[1],
                                    self.padding[0], self.padding[0]))
        res = res.unfold(2, self.kernel_size[0], self.stride[0]).unfold(
            3, self.kernel_size[1], self.stride[1]).max(-1)[0].max(-1)[0]
        setattr(res, 'output_fraclen', output_fraclen)
        return res


class ReLUClipFXQConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bitw_min=None,
                 bita_min=None,
                 weight_only=False,
                 double_side=False,
                 rescale_forward=False,
                 rescale_type='constant',
                 onnx_export=False,
                 master_layer=None,
                 format_type=None,
                 bn_momentum=0.1,
                 bn_eps=1e-5,
                 bn_layer=nn.BatchNorm2d):
        super(ReLUClipFXQConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=False)
        self.bn = bn_layer(out_channels, momentum=bn_momentum, eps=bn_eps)
        self.bitw_min = bitw_min
        self.bita_min = bita_min
        self.weight_only = weight_only
        self.double_side = double_side
        self.rescale_forward = rescale_forward
        self.rescale_type = rescale_type
        self.set_onnx_export(onnx_export)
        self.set_quant()
        self.set_alpha()
        self.weight_format = None
        self.input_format = None
        self.layer_dict = {}
        self.set_master_layer(master_layer)
        self.set_following_layer(None)
        self.floating = False
        self.floating_wo_clip = False
        self.format_type = format_type
        self.format_from_metric = False
        self.metric = None
        self.no_clipping = False
        self.input_fraclen_sharing = False
        self.format_grid_search = False
        self.quant_bias = False
        self.int_infer = False
        self.int_op_only = False
        self.avgpool_scale = 1.0

    def forward(self, input):
        ## Input
        if self.weight_only:
            if self.double_side:
                input_val = input * 1.0
            else:
                input_val = torch.relu(input)
            if (self.int_infer or self.quant_bias) and not self.training:
                input_fraclen = self.get_input_fraclen()
        else:
            if self.floating and self.floating_wo_clip:
                if self.double_side:
                    input_val = input * 1.0
                else:
                    input_val = torch.relu(input)
            else:
                x_wl, x_fl = self.get_input_format()
                if self.floating:
                    input_fraclen = torch.ones(1).to(input.device) * x_fl
                    if self.onnx_export:
                        input_val, _ = self.quant(input, x_wl, input_fraclen,
                                               1, self.double_side,
                                               self.floating)
                    else:
                        input_val = self.quant(input, x_wl, input_fraclen,
                                               1, self.double_side,
                                               self.floating)
                elif self.format_grid_search:
                    if self.training:
                        input_fraclen = fraclen_gridsearch(
                            input, x_wl, 1, self.double_side)
                        if self.onnx_export:
                            input_val, _ = self.quant(input, x_wl,
                                                      input_fraclen, 1,
                                                      self.double_side,
                                                      self.floating)
                        else:
                            input_val = self.quant(input, x_wl, input_fraclen,
                                                   1, self.double_side,
                                                   self.floating)
                        new_input_fraclen = self.momentum * input_fraclen + (
                            1 - self.momentum) * self.get_input_fraclen()
                        self.update_input_fraclen(new_input_fraclen)
                    else:
                        input_fraclen = self.get_input_fraclen()
                        input_fraclen = torch.round(input_fraclen)
                        input_fraclen = torch.clamp(input_fraclen,
                                                    max=x_wl -
                                                    int(self.double_side),
                                                    min=0)
                        if self.onnx_export:
                            input_val, _ = self.quant(input, x_wl,
                                                      input_fraclen, 1,
                                                      self.double_side,
                                                      self.floating)
                        else:
                            input_val = self.quant(input, x_wl, input_fraclen,
                                                   1, self.double_side,
                                                   self.floating)
                elif self.format_from_metric:
                    assert x_wl == 8, 'Word length other than 8bit has not been implemented'
                    if self.training:
                        input_metric = self.input_metric_func(input)
                        input_fraclen = metric2fraclen(input_metric,
                                                       self.metric, 1,
                                                       self.double_side)
                        input_fraclen = torch.clamp(input_fraclen,
                                                    max=x_wl -
                                                    int(self.double_side),
                                                    min=0)
                        if self.onnx_export:
                            input_val, _ = self.quant(input, x_wl,
                                                      input_fraclen, 1,
                                                      self.double_side,
                                                      self.floating)
                        else:
                            input_val = self.quant(input, x_wl, input_fraclen,
                                                   1, self.double_side,
                                                   self.floating)
                        new_input_fraclen = self.momentum * input_fraclen + (
                            1 - self.momentum) * self.get_input_fraclen()
                        self.update_input_fraclen(new_input_fraclen)
                    else:
                        input_fraclen = self.get_input_fraclen()
                        input_fraclen = torch.round(input_fraclen)
                        input_fraclen = torch.clamp(input_fraclen,
                                                    max=x_wl -
                                                    int(self.double_side),
                                                    min=0)
                        if self.onnx_export:
                            input_val, _ = self.quant(input, x_wl,
                                                      input_fraclen, 1,
                                                      self.double_side,
                                                      self.floating)
                        else:
                            input_val = self.quant(input, x_wl, input_fraclen,
                                                   1, self.double_side,
                                                   self.floating)
                else:
                    raise NotImplementedError

        ## Weight
        weight = self.conv.weight
        if self.rescale_forward:
            if self.rescale_type == 'stddev':
                weight_scale = torch.std(self.conv.weight.detach())
            elif self.rescale_type == 'constant':
                weight_scale = 1.0 / (self.out_channels * self.kernel_size[0] * self.kernel_size[1])**0.5
            else:
                raise NotImplementedError
            weight_scale /= torch.std(weight.detach())
        else:
            weight_scale = 1.0
        weight = weight * weight_scale
        avgpool_scale = getattr(self, 'avgpool_scale', 1.0)
        weight = weight * avgpool_scale
        if self.training:
            ## estimate running stats
            if self.floating and self.floating_wo_clip:
                y0 = nn.functional.conv2d(input_val,
                                          weight,
                                          bias=self.conv.bias,
                                          stride=self.conv.stride,
                                          padding=self.conv.padding,
                                          dilation=self.conv.dilation,
                                          groups=self.conv.groups)
            else:
                y0 = nn.functional.conv2d(
                    self.fix_scaling[(..., ) + (None, None)] * input_val,
                    weight,
                    bias=self.conv.bias,
                    stride=self.conv.stride,
                    padding=self.conv.padding,
                    dilation=self.conv.dilation,
                    groups=self.conv.groups)
            self.bn(y0)
            bn_mean = y0.mean([0, 2, 3])
            bn_std = torch.sqrt(
                y0.var([0, 2, 3], unbiased=False) + self.bn.eps)
        else:
            bn_mean = self.bn.running_mean
            bn_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        if self.floating and self.floating_wo_clip:
            weight = (self.bn.weight / bn_std)[:, None, None, None] * weight
            bias = self.bn.bias - self.bn.weight / bn_std * bn_mean

            ## Conv
            y = nn.functional.conv2d(input=input_val,
                                     weight=weight,
                                     bias=bias,
                                     stride=self.conv.stride,
                                     padding=self.conv.padding,
                                     dilation=self.conv.dilation,
                                     groups=self.conv.groups)
        else:
            if self.conv.groups == 1:
                weight = (
                    self.bn.weight /
                    bn_std)[:, None, None, None] * weight * self.fix_scaling[
                        (..., ) +
                        (None, None)] / self.get_following_layer().fix_scaling[
                            (..., ) + (None, None, None)]
            elif self.conv.groups == self.conv.in_channels:
                weight = (self.bn.weight / bn_std
                          )[:, None, None, None] * weight * self.fix_scaling[
                              (..., ) +
                              (None, None, None)] / self.get_following_layer(
                              ).fix_scaling[(..., ) + (None, None, None)]
            else:
                print(
                    'Group-wise conv with groups != in_channels is not supported'
                )
                raise NotImplementedError
            w_wl, w_fl = self.get_weight_format()
            if self.floating:
                weight_fraclen = torch.ones(1).to(weight.device) * w_fl
            elif self.format_grid_search:
                weight_fraclen = fraclen_gridsearch(weight, w_wl, 0, True)
            elif self.format_from_metric:
                assert w_wl == 8, 'Word length other than 8bit has not been implemented'
                weight_metric = self.weight_metric_func(weight)
                weight_fraclen = metric2fraclen(weight_metric, self.metric, 1,
                                                True)
                weight_fraclen = torch.clamp(weight_fraclen,
                                             max=w_wl - 1,
                                             min=0)
            else:
                raise NotImplementedError
            if self.onnx_export:
                weight, _ = self.quant(weight, w_wl, weight_fraclen, 0,
                                       True, self.floating)
            else:
                weight = self.quant(weight, w_wl, weight_fraclen, 0, True,
                                    self.floating)
            bias = (self.bn.bias - self.bn.weight / bn_std *
                    bn_mean) / self.get_following_layer().fix_scaling
            if not self.training and self.quant_bias:
                bias, _ = fix_quant(bias, 32, input_fraclen + weight_fraclen)

            ## Conv
            if self.int_infer and not self.training:
                int_weight = self.int_weight.float()
                int_input = (input_val * (2**input_fraclen)).int().float()
                int_bias = self.int_bias.float()
                y = nn.functional.conv2d(input=int_input,
                                         weight=int_weight,
                                         bias=int_bias,
                                         stride=self.conv.stride,
                                         padding=self.conv.padding,
                                         dilation=self.conv.dilation,
                                         groups=self.conv.groups)
                y = torch.clamp(y, max=(1 << 31) - 1, min=-(1 << 31) + 1)
                y = y / (2**(weight_fraclen + input_fraclen))
                setattr(y, 'output_fraclen', weight_fraclen + input_fraclen)
            else:
                y = nn.functional.conv2d(input=input_val,
                                         weight=weight,
                                         bias=bias,
                                         stride=self.conv.stride,
                                         padding=self.conv.padding,
                                         dilation=self.conv.dilation,
                                         groups=self.conv.groups)
        return y

    def set_weight_format(self, weight_format):
        w_wl, w_fl = weight_format
        if self.bitw_min is not None:
            w_wl = max(w_wl, self.bitw_min)
        assert w_fl <= w_wl
        self.weight_format = (w_wl, w_fl)

    def get_weight_format(self):
        return self.weight_format

    def set_input_format(self, input_format):
        x_wl, x_fl = input_format
        if self.bita_min is not None:
            x_wl = max(x_wl, self.bita_min)
        assert x_fl <= x_wl
        self.input_format = (x_wl, x_fl)

    def get_input_format(self):
        if self.get_master_layer() is not None:
            return self.get_master_layer().get_input_format()
        else:
            return self.input_format

    def set_onnx_export(self, onnx_export):
        self.onnx_export = onnx_export

    def set_quant(self):
        if self.onnx_export:
            self.quant = fix_quant
        else:
            self.quant = q_k.apply

    def set_alpha(self):
        self.alpha = nn.Parameter(
            torch.tensor(8.0).to(self.conv.weight.device))

    def get_alpha(self):
        if self.get_master_layer() is not None:
            return self.get_master_layer().get_alpha()
        elif self.weight_only:
            return torch.ones_like(self.alpha)
        else:
            return self.alpha

    def get_input_fraclen(self):
        if self.weight_only:
            return torch.ones_like(self.input_fraclen) * 8
        elif self.format_from_metric or self.format_grid_search:
            if self.get_master_layer(
            ) is not None and self.input_fraclen_sharing:
                return self.get_master_layer().get_input_fraclen()
            else:
                return self.input_fraclen
        else:
            raise NotImplementedError

    def update_input_fraclen(self, input_fraclen):
        self.get_input_fraclen().data = input_fraclen

    @property
    def fix_scaling(self):
        alpha = torch.abs(self.get_alpha())
        if self.no_clipping:
            return torch.ones_like(alpha)
        if self.weight_only:
            return alpha
        x_wl, x_fl = self.get_input_format()
        if self.floating and not self.floating_wo_clip:
            return 2**x_fl * alpha / (2**(x_wl - int(self.double_side)) - 1)
        if self.format_from_metric or self.format_grid_search:
            ## Note that this is a little different from the input_fraclen used in forward method, especially during training
            input_fraclen = self.get_input_fraclen()
            input_fraclen = torch.round(input_fraclen)
            input_fraclen = torch.clamp(input_fraclen,
                                        max=x_wl - int(self.double_side),
                                        min=0)
            return 2**input_fraclen * alpha / (
                2**(x_wl - int(self.double_side)) - 1)

    def set_master_layer(self, master_layer):
        self.layer_dict['master'] = master_layer

    def get_master_layer(self):
        return self.layer_dict['master']

    def set_following_layer(self, following_layer):
        self.layer_dict['following'] = following_layer

    def get_following_layer(self):
        return self.layer_dict['following']

    @property
    def float_weight(self):
        weight = self.conv.weight
        if self.rescale_forward:
            if self.rescale_type == 'stddev':
                weight_scale = torch.std(self.conv.weight.detach())
            elif self.rescale_type == 'constant':
                weight_scale = 1.0 / (self.out_channels * self.kernel_size[0] * self.kernel_size[1])**0.5
            else:
                raise NotImplementedError
            weight_scale /= torch.std(weight.detach())
        else:
            weight_scale = 1.0
        weight = weight * weight_scale
        bn_mean = self.bn.running_mean
        bn_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        if self.floating and self.floating_wo_clip:
            weight = (self.bn.weight / bn_std)[:, None, None, None] * weight
        else:
            if self.conv.groups == 1:
                weight = (
                    self.bn.weight /
                    bn_std)[:, None, None, None] * weight * self.fix_scaling[
                        (..., ) +
                        (None, None)] / self.get_following_layer().fix_scaling[
                            (..., ) + (None, None, None)]
            elif self.conv.groups == self.conv.in_channels:
                weight = (self.bn.weight / bn_std
                          )[:, None, None, None] * weight * self.fix_scaling[
                              (..., ) +
                              (None, None, None)] / self.get_following_layer(
                              ).fix_scaling[(..., ) + (None, None, None)]
            else:
                print(
                    'Group-wise conv with groups != in_channels is not supported'
                )
                raise NotImplementedError
        return weight

    @property
    def float_bias(self):
        bn_mean = self.bn.running_mean
        bn_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        if self.floating and self.floating_wo_clip:
            bias = self.bn.bias - self.bn.weight / bn_std * bn_mean
        else:
            bias = (self.bn.bias - self.bn.weight / bn_std *
                    bn_mean) / self.get_following_layer().fix_scaling
        return bias

    @property
    def int_weight(self):
        assert not self.floating
        assert not self.floating_wo_clip
        avgpool_scale = getattr(self, 'avgpool_scale', 1.0)
        weight_fraclen = self.get_weight_fraclen()
        w_wl, w_fl = self.get_weight_format()
        if self.onnx_export:
            weight, _ = self.quant(self.float_weight * avgpool_scale, w_wl,
                                   weight_fraclen, 0, True, self.floating)
        else:
            weight = self.quant(self.float_weight * avgpool_scale, w_wl,
                                weight_fraclen, 0, True, self.floating)
        int_weight = (weight * (2**weight_fraclen)).int()
        return int_weight

    @property
    def int_bias(self):
        assert not self.floating
        assert not self.floating_wo_clip
        assert self.quant_bias
        avgpool_scale = getattr(self, 'avgpool_scale', 1.0)
        weight_fraclen = self.get_weight_fraclen()
        x_wl, x_fl = self.get_input_format()
        input_fraclen = self.get_input_fraclen()
        input_fraclen = torch.round(input_fraclen)
        input_fraclen = torch.clamp(input_fraclen,
                                    max=x_wl - int(self.double_side),
                                    min=0)
        bias, _ = fix_quant(self.float_bias * avgpool_scale, 32,
                            input_fraclen + weight_fraclen)
        int_bias = (bias * (2**(input_fraclen + weight_fraclen))).int()
        return int_bias

    def set_metric_func(self):
        if self.format_from_metric:
            if self.format_type == 'per_layer':
                weight_along_axis = (0, 1, 2, 3)
                input_along_axis = (0, 1, 2, 3)
            elif self.format_type == 'per_channel':
                weight_along_axis = (1, 2, 3)
                input_along_axis = (0, 2, 3)
            else:
                raise NotImplementedError
            if self.metric == 'std':
                self.weight_metric_func = lambda x: torch.std(
                    x, axis=weight_along_axis)
                self.input_metric_func = lambda x: torch.std(
                    x, axis=input_along_axis)
            elif self.metric == 'mae':
                self.weight_metric_func = lambda x: torch.mean(
                    torch.abs(x), axis=weight_along_axis)
                self.input_metric_func = lambda x: torch.mean(
                    torch.abs(x), axis=input_along_axis)
            elif self.metric == 'rms':
                self.weight_metric_func = lambda x: torch.mean(
                    x**2, axis=weight_along_axis)**0.5
                self.input_metric_func = lambda x: torch.mean(
                    x**2, axis=input_along_axis)**0.5
            else:
                raise NotImplementedError

    def register_input_format(self, input_format, momentum=0.1):
        x_wl, x_fl = input_format
        if self.format_from_metric or self.format_grid_search:
            if self.format_type == 'per_layer':
                self.register_buffer(
                    'input_fraclen',
                    torch.ones(1).to(self.conv.weight.device) * x_fl)
            elif self.format_type == 'per_channel':
                self.register_buffer(
                    'input_fraclen',
                    torch.ones(self.conv.in_channels).to(
                        self.conv.weight.device) * x_fl)
            else:
                raise NotImplementedError
            self.momentum = momentum

    def get_weight_fraclen(self):
        assert self.format_from_metric or self.format_grid_search
        avgpool_scale = getattr(self, 'avgpool_scale', 1.0)
        weight = self.float_weight * avgpool_scale
        w_wl, w_fl = self.get_weight_format()
        if self.floating:
            weight_fraclen = torch.ones(1).to(weight.device) * w_fl
        elif self.format_grid_search:
            weight_fraclen = fraclen_gridsearch(weight, w_wl, 0, True)
        elif self.format_from_metric:
            assert w_wl == 8, 'Word length other than 8bit has not been implemented'
            weight_metric = self.weight_metric_func(weight)
            weight_fraclen = metric2fraclen(weight_metric, self.metric, 1,
                                            True)
            weight_fraclen = torch.clamp(weight_fraclen, max=w_wl - 1, min=0)
        else:
            raise NotImplementedError
        return weight_fraclen

    def int_conv(self, avgpool_scale=1.0):
        int_conv = nn.Conv2d(self.conv.in_channels,
                             self.conv.out_channels,
                             self.conv.kernel_size,
                             stride=self.conv.stride,
                             padding=self.conv.padding,
                             dilation=self.conv.dilation,
                             groups=self.conv.groups,
                             bias=True)
        self.avgpool_scale = avgpool_scale
        int_weight = self.int_weight
        int_bias = self.int_bias
        double_side = self.double_side
        weight_fraclen = self.get_weight_fraclen().int()
        x_wl, x_fl = self.get_input_format()
        input_fraclen = self.get_input_fraclen()
        input_fraclen = torch.round(input_fraclen)
        input_fraclen = torch.clamp(input_fraclen,
                                    max=x_wl - int(self.double_side),
                                    min=0).int()
        if not self.int_op_only:
            int_weight = int_weight.float()
            int_bias = int_bias.float()
            weight_fraclen = weight_fraclen.float()
            input_fraclen = input_fraclen.float()
        int_conv.weight.data = int_weight
        int_conv.bias.data = int_bias
        if self.int_op_only:
            int_conv.weight.requires_grad = False
            int_conv.bias.requires_grad = False
        setattr(int_conv, 'input_symmetric', double_side)
        int_conv.register_buffer('weight_fraclen', weight_fraclen)
        int_conv.register_buffer('input_fraclen', input_fraclen)
        int_conv.int_op_only = self.int_op_only
        return int_conv


class ReLUClipFXQLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 bitw_min=None,
                 bita_min=None,
                 weight_only=False,
                 double_side=False,
                 rescale_forward=False,
                 rescale_type='constant',
                 onnx_export=False,
                 master_layer=None,
                 format_type=None):
        super(ReLUClipFXQLinear, self).__init__(in_features,
                                                out_features,
                                                bias=bias)
        self.bitw_min = bitw_min
        self.bita_min = bita_min
        self.weight_only = weight_only
        self.double_side = double_side
        self.rescale_forward = rescale_forward
        self.rescale_type = rescale_type
        self.set_onnx_export(onnx_export)
        self.set_quant()
        self.set_alpha()
        self.weight_format = None
        self.input_format = None
        self.layer_dict = {}
        self.set_master_layer(master_layer)
        self.floating = False
        self.floating_wo_clip = False
        self.format_type = format_type
        self.format_from_metric = False
        self.metric = None
        self.no_clipping = False
        self.input_fraclen_sharing = False
        self.format_grid_search = False
        self.quant_bias = False
        self.int_infer = False
        self.int_op_only = False

    def forward(self, input):
        ## Input
        if self.weight_only:
            if self.double_side:
                input_val = input * 1.0
            else:
                input_val = torch.relu(input)
            if (self.int_infer or self.quant_bias) and not self.training:
                input_fraclen = self.get_input_fraclen()
        else:
            if self.floating and self.floating_wo_clip:
                if self.double_side:
                    input_val = input * 1.0
                else:
                    input_val = torch.relu(input)
            else:
                x_wl, x_fl = self.get_input_format()
                if self.floating:
                    input_fraclen = torch.ones(1).to(input.device) * x_fl
                    if self.onnx_export:
                        input_val, _ = self.quant(input, x_wl, input_fraclen,
                                               1, self.double_side,
                                               self.floating)
                    else:
                        input_val = self.quant(input, x_wl, input_fraclen,
                                               1, self.double_side,
                                               self.floating)
                elif self.format_grid_search:
                    if self.training:
                        input_fraclen = fraclen_gridsearch(
                            input, x_wl, 1, self.double_side)
                        if self.onnx_export:
                            input_val, _ = self.quant(input, x_wl,
                                                      input_fraclen, 1,
                                                      self.double_side,
                                                      self.floating)
                        else:
                            input_val = self.quant(input, x_wl, input_fraclen,
                                                   1, self.double_side,
                                                   self.floating)
                        new_input_fraclen = self.momentum * input_fraclen + (
                            1 - self.momentum) * self.get_input_fraclen()
                        self.update_input_fraclen(new_input_fraclen)
                    else:
                        input_fraclen = self.get_input_fraclen()
                        input_fraclen = torch.round(input_fraclen)
                        input_fraclen = torch.clamp(input_fraclen,
                                                    max=x_wl -
                                                    int(self.double_side),
                                                    min=0)
                        if self.onnx_export:
                            input_val, _ = self.quant(input, x_wl,
                                                      input_fraclen, 1,
                                                      self.double_side,
                                                      self.floating)
                        else:
                            input_val = self.quant(input, x_wl, input_fraclen,
                                                   1, self.double_side,
                                                   self.floating)
                elif self.format_from_metric:
                    assert x_wl == 8, 'Word length other than 8bit has not been implemented'
                    if self.training:
                        input_metric = self.input_metric_func(input)
                        input_fraclen = metric2fraclen(input_metric,
                                                       self.metric, 1,
                                                       self.double_side)
                        input_fraclen = torch.clamp(input_fraclen,
                                                    max=x_wl -
                                                    int(self.double_side),
                                                    min=0)
                        if self.onnx_export:
                            input_val, _ = self.quant(input, x_wl,
                                                      input_fraclen, 1,
                                                      self.double_side,
                                                      self.floating)
                        else:
                            input_val = self.quant(input, x_wl, input_fraclen,
                                                   1, self.double_side,
                                                   self.floating)
                        new_input_fraclen = self.momentum * input_fraclen + (
                            1 - self.momentum) * self.get_input_fraclen()
                        self.update_input_fraclen(new_input_fraclen)
                    else:
                        input_fraclen = self.get_input_fraclen()
                        input_fraclen = torch.round(input_fraclen)
                        input_fraclen = torch.clamp(input_fraclen,
                                                    max=x_wl -
                                                    int(self.double_side),
                                                    min=0)
                        if self.onnx_export:
                            input_val, _ = self.quant(input, x_wl,
                                                      input_fraclen, 1,
                                                      self.double_side,
                                                      self.floating)
                        else:
                            input_val = self.quant(input, x_wl, input_fraclen,
                                                   1, self.double_side,
                                                   self.floating)
                else:
                    raise NotImplementedError

        ## Weight
        weight = self.weight * 1.0
        if not (self.floating and self.floating_wo_clip):
            w_wl, w_fl = self.get_weight_format()
            if self.floating:
                weight_fraclen = torch.ones(1).to(weight.device) * w_fl
            elif self.format_grid_search:
                weight_fraclen = fraclen_gridsearch(weight, w_wl, 0, True)
            elif self.format_from_metric:
                assert w_wl == 8, 'Word length other than 8bit has not been implemented'
                weight_metric = self.weight_metric_func(weight)
                weight_fraclen = metric2fraclen(weight_metric, self.metric, 1,
                                                True)
                weight_fraclen = torch.clamp(weight_fraclen,
                                             max=w_wl - 1,
                                             min=0)
            else:
                raise NotImplementedError
            if self.onnx_export:
                weight, _ = self.quant(weight, w_wl, weight_fraclen, 0,
                                       True, self.floating)
            else:
                weight = self.quant(weight, w_wl, weight_fraclen, 0, True,
                                    self.floating)
        if self.rescale_forward:
            if self.rescale_type == 'stddev':
                weight_scale = torch.std(self.weight.detach())
            elif self.rescale_type == 'constant':
                weight_scale = 1.0 / (self.out_features)**0.5
            else:
                raise NotImplementedError
            weight_scale /= torch.std(weight.detach())
        else:
            weight_scale = 1.0

        if self.floating and self.floating_wo_clip:
            weight = weight * weight_scale
            bias = self.bias

            ## FC
            y = nn.functional.linear(input=input_val, weight=weight, bias=bias)
        else:
            if self.training:
                weight = weight * weight_scale
                input_val.mul_(self.fix_scaling)
                bias = self.bias
            else:
                if self.bias is not None:
                    bias = self.bias / self.fix_scaling / weight_scale
                    if self.quant_bias:
                        bias, _ = fix_quant(bias, 32,
                                            input_fraclen + weight_fraclen)
                else:
                    bias = self.bias

            ## FC
            if self.int_infer and not self.training:
                int_weight = self.int_weight.float()
                int_input = (input_val * (2**input_fraclen)).int().float()
                if bias is not None:
                    int_bias = self.int_bias.float()
                else:
                    int_bias = bias
                y = nn.functional.linear(input=int_input,
                                         weight=int_weight,
                                         bias=int_bias)
                y = torch.clamp(y, max=(1 << 31) - 1, min=-(1 << 31) + 1)
                y = y / (2**(weight_fraclen + input_fraclen))
            else:
                y = nn.functional.linear(input=input_val,
                                         weight=weight,
                                         bias=bias)
        return y

    def set_weight_format(self, weight_format):
        w_wl, w_fl = weight_format
        if self.bitw_min is not None:
            w_wl = max(w_wl, self.bitw_min)
        assert w_fl <= w_wl
        self.weight_format = (w_wl, w_fl)

    def get_weight_format(self):
        return self.weight_format

    def set_input_format(self, input_format):
        x_wl, x_fl = input_format
        if self.bita_min is not None:
            x_wl = max(x_wl, self.bita_min)
        assert x_fl <= x_wl
        self.input_format = (x_wl, x_fl)

    def get_input_format(self):
        if self.get_master_layer() is not None:
            return self.get_master_layer().get_input_format()
        else:
            return self.input_format

    def set_onnx_export(self, onnx_export):
        self.onnx_export = onnx_export

    def set_quant(self):
        if self.onnx_export:
            self.quant = fix_quant
        else:
            self.quant = q_k.apply

    def set_alpha(self):
        self.alpha = nn.Parameter(torch.tensor(8.0).to(self.weight.device))

    def get_alpha(self):
        if self.get_master_layer() is not None:
            return self.get_master_layer().get_alpha()
        elif self.weight_only:
            return torch.ones_like(self.alpha)
        else:
            return self.alpha

    def get_input_fraclen(self):
        if self.weight_only:
            return torch.ones_like(self.input_fraclen) * 8
        elif self.format_from_metric or self.format_grid_search:
            if self.get_master_layer(
            ) is not None and self.input_fraclen_sharing:
                return self.get_master_layer().get_input_fraclen()
            else:
                return self.input_fraclen
        else:
            raise NotImplementedError

    def update_input_fraclen(self, input_fraclen):
        self.get_input_fraclen().data = input_fraclen

    @property
    def fix_scaling(self):
        alpha = torch.abs(self.get_alpha())
        if self.no_clipping:
            return torch.ones_like(alpha)
        if self.weight_only:
            return alpha
        x_wl, x_fl = self.get_input_format()
        if self.floating and not self.floating_wo_clip:
            return 2**x_fl * alpha / (2**(x_wl - int(self.double_side)) - 1)
        if self.format_from_metric or self.format_grid_search:
            ## Note that this is a little different from the input_fraclen used in forward method, especially during training or when self.input_adaptive = True
            input_fraclen = self.get_input_fraclen()
            input_fraclen = torch.round(input_fraclen)
            input_fraclen = torch.clamp(input_fraclen,
                                        max=x_wl - int(self.double_side),
                                        min=0)
            return 2**input_fraclen * alpha / (
                2**(x_wl - int(self.double_side)) - 1)

    def set_master_layer(self, master_layer):
        self.layer_dict['master'] = master_layer

    def get_master_layer(self):
        return self.layer_dict['master']

    @property
    def float_weight(self):
        return self.weight

    @property
    def float_bias(self):
        weight = self.weight * 1.0
        if not (self.floating and self.floating_wo_clip):
            w_wl, w_fl = self.get_weight_format()
            if self.floating:
                weight_fraclen = torch.ones(1).to(weight.device) * w_fl
            elif self.format_grid_search:
                weight_fraclen = fraclen_gridsearch(weight, w_wl, 0, True)
            elif self.format_from_metric:
                assert w_wl == 8, 'Word length other than 8bit has not been implemented'
                weight_metric = self.weight_metric_func(weight)
                weight_fraclen = metric2fraclen(weight_metric, self.metric, 1,
                                                True)
                weight_fraclen = torch.clamp(weight_fraclen,
                                             max=w_wl - 1,
                                             min=0)
            else:
                raise NotImplementedError
            if self.onnx_export:
                weight, _ = self.quant(weight, w_wl, weight_fraclen, 0,
                                       True, self.floating)
            else:
                weight = self.quant(weight, w_wl, weight_fraclen, 0, True,
                                    self.floating)
        if self.rescale_forward:
            if self.rescale_type == 'stddev':
                weight_scale = torch.std(self.weight.detach())
            elif self.rescale_type == 'constant':
                weight_scale = 1.0 / (self.out_features)**0.5
            else:
                raise NotImplementedError
            weight_scale /= torch.std(weight.detach())
        else:
            weight_scale = 1.0
        if self.bias is not None:
            bias = self.bias / self.fix_scaling / weight_scale
        else:
            bias = self.bias
        return bias

    @property
    def int_weight(self):
        assert not self.floating
        assert not self.floating_wo_clip
        weight_fraclen = self.get_weight_fraclen()
        w_wl, w_fl = self.get_weight_format()
        if self.onnx_export:
            weight, _ = self.quant(self.float_weight, w_wl, weight_fraclen, 0,
                                   True, self.floating)
        else:
            weight = self.quant(self.float_weight, w_wl, weight_fraclen, 0,
                                True, self.floating)
        int_weight = (weight * (2**weight_fraclen)).int()
        return int_weight

    @property
    def int_bias(self):
        if self.bias is not None:
            assert not self.floating
            assert not self.floating_wo_clip
            assert self.quant_bias
            weight_fraclen = self.get_weight_fraclen()
            x_wl, x_fl = self.get_input_format()
            input_fraclen = self.get_input_fraclen()
            input_fraclen = torch.round(input_fraclen)
            input_fraclen = torch.clamp(input_fraclen,
                                        max=x_wl - int(self.double_side),
                                        min=0)
            bias, _ = fix_quant(self.float_bias, 32,
                                input_fraclen + weight_fraclen)
            int_bias = (bias * (2**(input_fraclen + weight_fraclen))).int()
        else:
            int_bias = self.bias
        return int_bias

    def set_metric_func(self):
        if self.format_from_metric:
            if self.format_type == 'per_layer':
                weight_along_axis = (0, 1)
                input_along_axis = (0, 1)
            elif self.format_type == 'per_channel':
                weight_along_axis = (1, )
                input_along_axis = (0, 1)
                print(
                    'WARNING: Per channel for input quantization format is not supported for linear layers.'
                )
            else:
                raise NotImplementedError
            if self.metric == 'std':
                self.weight_metric_func = lambda x: torch.std(
                    x, axis=weight_along_axis)
                self.input_metric_func = lambda x: torch.std(
                    x, axis=input_along_axis)
            elif self.metric == 'mae':
                self.weight_metric_func = lambda x: torch.mean(
                    torch.abs(x), axis=weight_along_axis)
                self.input_metric_func = lambda x: torch.mean(
                    torch.abs(x), axis=input_along_axis)
            elif self.metric == 'rms':
                self.weight_metric_func = lambda x: torch.mean(
                    x**2, axis=weight_along_axis)**0.5
                self.input_metric_func = lambda x: torch.mean(
                    x**2, axis=input_along_axis)**0.5
            else:
                raise NotImplementedError

    def register_input_format(self, input_format, momentum=0.1):
        x_wl, x_fl = input_format
        if self.format_from_metric or self.format_grid_search:
            if self.format_type == 'per_layer':
                self.register_buffer(
                    'input_fraclen',
                    torch.ones(1).to(self.weight.device) * x_fl)
            elif self.format_type == 'per_channel':
                self.register_buffer(
                    'input_fraclen',
                    torch.ones(1).to(self.weight.device) * x_fl)
                print(
                    'WARNING: Per channel for input quantization format is not supported for linear layers.'
                )
            else:
                raise NotImplementedError
            self.momentum = momentum

    def get_weight_fraclen(self):
        assert self.format_from_metric or self.format_grid_search
        weight = self.float_weight
        w_wl, w_fl = self.get_weight_format()
        if self.floating:
            weight_fraclen = torch.ones(1).to(weight.device) * w_fl
        elif self.format_grid_search:
            weight_fraclen = fraclen_gridsearch(weight, w_wl, 1, True)
        elif self.format_from_metric:
            assert w_wl == 8, 'Word length other than 8bit has not been implemented'
            weight_metric = self.weight_metric_func(weight)
            weight_fraclen = metric2fraclen(weight_metric, self.metric, 1,
                                            True)
            weight_fraclen = torch.clamp(weight_fraclen, max=w_wl - 1, min=0)
        else:
            raise NotImplementedError
        return weight_fraclen

    def int_fc(self):
        int_fc = nn.Linear(self.in_features,
                           self.out_features,
                           bias=self.bias is not None)
        int_weight = self.int_weight
        if self.bias is not None:
            int_bias = self.int_bias
        double_side = self.double_side
        weight_fraclen = self.get_weight_fraclen().int()
        x_wl, x_fl = self.get_input_format()
        input_fraclen = self.get_input_fraclen()
        input_fraclen = torch.round(input_fraclen)
        input_fraclen = torch.clamp(input_fraclen,
                                    max=x_wl - int(self.double_side),
                                    min=0).int()
        if not self.int_op_only:
            int_weight = int_weight.float()
            if self.bias is not None:
                int_bias = int_bias.float()
            weight_fraclen = weight_fraclen.float()
            input_fraclen = input_fraclen.float()
        int_fc.weight.data = int_weight
        int_fc.bias.data = int_bias
        if self.int_op_only:
            int_fc.weight.requires_grad = False
            int_fc.bias.requires_grad = False
        setattr(int_fc, 'input_symmetric', double_side)
        int_fc.register_buffer('weight_fraclen', weight_fraclen)
        int_fc.register_buffer('input_fraclen', input_fraclen)
        int_fc.int_op_only = self.int_op_only
        return int_fc

##import sys
##sys.path.extend([])
##
import importlib
import os
import time
import random
import math
from bisect import bisect_right

from functools import wraps

import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

import torch
import torch.nn as nn
from torch import multiprocessing
from torch.distributed import is_initialized
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.modules.utils import _pair

from myutils.distributed import init_dist, master_only, is_master
from myutils.distributed import get_rank, get_world_size
from myutils.distributed import dist_all_reduce_tensor
from myutils.distributed import master_only_print as mprint
from myutils.distributed import AllReduceDistributedDataParallel, allreduce_grads

from myutils.config import FLAGS
from myutils.meters import ScalarMeter, flush_scalar_meters
from myutils.export import onnx_export

from models.fix_quant_ops import ReLUClipFXQConvBN, ReLUClipFXQLinear, bn_calib, fix_quant

from pytorchcv.model_provider import get_model as ptcv_get_model


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        if is_master():
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            mprint('func:{!r} took: {:2.4f} sec'.format(f.__name__, te - ts))
        else:
            result = f(*args, **kw)
        return result

    return wrap


def ptcv_load(model):
    model_name_dict = {
        'fix_resnet18': 'resnet18',
        'fix_resnet50': 'resnet50b',
        'fix_mobilenet_v1': 'mobilenet_w1',
        'fix_mobilenet_v2': 'mobilenetv2b_w1'
    }
    model_name = FLAGS.model.split('.')[1] + str(getattr(FLAGS, 'depth', ''))
    ptcv_model_name = model_name_dict[model_name]
    net = ptcv_get_model(ptcv_model_name,
                         pretrained=True).to(next(model.parameters()).device)
    if getattr(FLAGS, 'hawq_pretrained', False):
        checkpoint = torch.load(getattr(FLAGS, 'hawq_pretrained_file', ''))
        model_key_list = list(net.state_dict().keys())
        for key in model_key_list:
            if 'num_batches_tracked' in key: model_key_list.remove(key)
        i = 0
        modified_dict = {}
        for key, value in checkpoint.items():
            if 'scaling_factor' in key: continue
            if 'num_batches_tracked' in key: continue
            if 'weight_integer' in key: continue
            if 'min' in key or 'max' in key: continue
            modified_key = model_key_list[i]
            modified_dict[modified_key] = value
            i += 1
        net.load_state_dict(modified_dict, strict=False)
    name_weight_bias_list = []
    for n, m in net.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            name_weight_bias_tuple = (n, m.weight, m.bias)
            name_weight_bias_list.append(name_weight_bias_tuple)
        elif isinstance(m, nn.BatchNorm2d):
            name_weight_bias_tuple = (n, m.weight, m.bias, m.running_mean,
                                      m.running_var)
            name_weight_bias_list.append(name_weight_bias_tuple)
    idx = 0
    for n, m in model.named_modules():
        if isinstance(m, ReLUClipFXQConvBN):
            conv_name, conv_weight, conv_bias = name_weight_bias_list[idx]
            bn_name, bn_weight, bn_bias, bn_running_mean, bn_running_var = name_weight_bias_list[
                idx + 1]
            mprint('model layer name: ', n)
            mprint('ptcv model conv name: ', conv_name)
            mprint('ptcv model bn name: ', bn_name)
            assert m.conv.weight.shape == conv_weight.shape
            assert m.conv.bias is None == conv_bias is None
            if m.conv.bias is not None:
                assert m.conv.bias.shape == conv_bias.shape
            assert m.bn.weight.shape == bn_weight.shape
            assert m.bn.bias.shape == bn_bias.shape
            assert m.bn.running_mean.shape == bn_running_mean.shape
            assert m.bn.running_var.shape == bn_running_var.shape
            m.conv.weight.data = conv_weight.data
            if m.conv.bias is not None:
                m.conv.bias.data = conv_bias.data
            m.bn.weight.data = bn_weight.data
            m.bn.bias.data = bn_bias.data
            m.bn.running_mean.data = bn_running_mean.data
            m.bn.running_var.data = bn_running_var.data
            idx += 2
        if isinstance(m, ReLUClipFXQLinear):
            fc_name, fc_weight, fc_bias = name_weight_bias_list[idx]
            mprint('model layer name: ', n)
            mprint('ptcv model fc name: ', fc_name)
            if len(fc_weight.shape) == 4:
                assert fc_weight.shape[-1] == 1
                assert fc_weight.shape[-2] == 1
                fc_weight = fc_weight[..., 0, 0]
            assert m.weight.shape == fc_weight.shape
            m.weight.data = fc_weight.data
            if m.bias is None:
                assert fc_bias is None
            else:
                assert m.bias.shape == fc_bias.shape
                m.bias.data = fc_bias.data
            idx += 1
    assert idx == len(name_weight_bias_list)
    del net


def nvidia_load(model):
    ckpt = torch.load(getattr(FLAGS, 'nvidia_pretrained_file', ''))
    name_weight_bias_list = []
    name_weight_bias_tuple = []
    for k, v in ckpt.items():
        if 'conv' in k:
            if 'weight' in k:
                n = k.replace('.weight', '')
                name_weight_bias_tuple = (n, v, None)
                name_weight_bias_list.append(name_weight_bias_tuple)
                name_weight_bias_tuple = []
            else:
                raise NotImplementedError
        elif 'bn' in k:
            if 'weight' in k:
                n = k.replace('.weight', '')
                name_weight_bias_tuple.append(n)
                name_weight_bias_tuple.append(v)
            elif 'bias' in k:
                name_weight_bias_tuple.append(v)
            elif 'running_mean' in k:
                name_weight_bias_tuple.append(v)
            elif 'running_var' in k:
                name_weight_bias_tuple.append(v)
            elif 'num_batches_tracked' in k:
                name_weight_bias_tuple = tuple(name_weight_bias_tuple)
                name_weight_bias_list.append(name_weight_bias_tuple)
                name_weight_bias_tuple = []
            else:
                raise NotImplementedError
        elif 'downsample' in k:
            layer_idx = int(k.split('.')[-2])
            if layer_idx == 0:
                if 'weight' in k:
                    n = k.replace('.weight', '')
                    name_weight_bias_tuple = (n, v, None)
                    name_weight_bias_list.append(name_weight_bias_tuple)
                    name_weight_bias_tuple = []
                else:
                    raise NotImplementedError
            elif layer_idx == 1:
                if 'weight' in k:
                    n = k.replace('.weight', '')
                    name_weight_bias_tuple.append(n)
                    name_weight_bias_tuple.append(v)
                elif 'bias' in k:
                    name_weight_bias_tuple.append(v)
                elif 'running_mean' in k:
                    name_weight_bias_tuple.append(v)
                elif 'running_var' in k:
                    name_weight_bias_tuple.append(v)
                elif 'num_batches_tracked' in k:
                    name_weight_bias_tuple = tuple(name_weight_bias_tuple)
                    name_weight_bias_list.append(name_weight_bias_tuple)
                    name_weight_bias_tuple = []
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif 'fc' in k:
            if 'weight' in k:
                n = k.replace('.weight', '')
                name_weight_bias_tuple.append(n)
                name_weight_bias_tuple.append(v)
            elif 'bias' in k:
                name_weight_bias_tuple.append(v)
                name_weight_bias_tuple = tuple(name_weight_bias_tuple)
                name_weight_bias_list.append(name_weight_bias_tuple)
                name_weight_bias_tuple = []
            else:
                raise NotImplementedError
        else:
            mprint('k: ', k)
            raise NotImplementedError
    idx = 0
    for n, m in model.named_modules():
        if isinstance(m, ReLUClipFXQConvBN):
            conv_name, conv_weight, conv_bias = name_weight_bias_list[idx]
            bn_name, bn_weight, bn_bias, bn_running_mean, bn_running_var = name_weight_bias_list[
                idx + 1]
            mprint('model layer name: ', n)
            mprint('nvidia model conv name: ', conv_name)
            mprint('nvidia model bn name: ', bn_name)
            assert m.conv.weight.shape == conv_weight.shape
            assert m.conv.bias is None == conv_bias is None
            if m.conv.bias is not None:
                assert m.conv.bias.shape == conv_weight.shape
            assert m.bn.weight.shape == bn_weight.shape
            assert m.bn.bias.shape == bn_bias.shape
            assert m.bn.running_mean.shape == bn_running_mean.shape
            assert m.bn.running_var.shape == bn_running_var.shape
            m.conv.weight.data = conv_weight.data.to(m.conv.weight.device)
            if m.conv.bias is not None:
                m.conv.bias.data = conv_bias.data.to(m.conv.bias.device)
            m.bn.weight.data = bn_weight.data.to(m.bn.weight.device)
            m.bn.bias.data = bn_bias.data.to(m.bn.bias.device)
            m.bn.running_mean.data = bn_running_mean.data.to(
                m.bn.running_mean.device)
            m.bn.running_var.data = bn_running_var.data.to(
                m.bn.running_var.device)
            idx += 2
        if isinstance(m, ReLUClipFXQLinear):
            fc_name, fc_weight, fc_bias = name_weight_bias_list[idx]
            mprint('model layer name: ', n)
            mprint('nvidia fc name: ', fc_name)
            if len(fc_weight.shape) == 4:
                assert fc_weight.shape[-1] == 1
                assert fc_weight.shape[-2] == 1
                fc_weight = fc_weight[..., 0, 0]
            assert m.weight.shape == fc_weight.shape
            m.weight.data = fc_weight.data.to(m.weight.device)
            if m.bias is None:
                assert fc_bias is None
            else:
                assert m.bias.shape == fc_bias.shape
                m.bias.data = fc_bias.data.to(m.bias.device)
            idx += 1
    assert idx == len(name_weight_bias_list)
    del ckpt


def get_model(gpu_id=None):
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes)
    if getattr(FLAGS, 'distributed', False):
        if getattr(FLAGS, 'distributed_all_reduce', False):
            model_wrapper = AllReduceDistributedDataParallel(model.cuda())
        else:
            model_wrapper = torch.nn.parallel.DistributedDataParallel(
                model.cuda(), [gpu_id], gpu_id, find_unused_parameters=True)
    else:
        model_wrapper = torch.nn.DataParallel(model).cuda()
    for m in model.modules():
        if isinstance(m, (ReLUClipFXQConvBN, ReLUClipFXQLinear)):
            m.set_weight_format(FLAGS.weight_format)
            m.set_input_format(FLAGS.input_format)
            m.rescale_type = getattr(FLAGS, 'rescale_type', 'constant')
            m.set_alpha()
            m.floating = getattr(FLAGS, 'floating_model', False)
            m.floating_wo_clip = getattr(FLAGS, 'floating_wo_clip', False)
            m.format_type = getattr(FLAGS, 'format_type', None)
            m.format_from_metric = getattr(FLAGS, 'format_from_metric', False)
            m.metric = getattr(FLAGS, 'metric', None)
            m.format_grid_search = getattr(FLAGS, 'format_grid_search', False)
            m.set_metric_func()
            m.register_input_format(FLAGS.input_format,
                                    momentum=getattr(FLAGS,
                                                     'momentum_for_metric',
                                                     0.1))
            m.no_clipping = getattr(FLAGS, 'no_clipping', False)
            m.input_fraclen_sharing = getattr(FLAGS, 'input_fraclen_sharing',
                                              False)
            m.quant_bias = getattr(FLAGS, 'quant_bias', False)
            m.int_infer = getattr(FLAGS, 'int_infer', False)
        if isinstance(m, ReLUClipFXQConvBN):
            m.rescale_forward = getattr(FLAGS, 'rescale_forward_conv', False)
        if isinstance(m, ReLUClipFXQLinear):
            m.rescale_forward = getattr(FLAGS, 'rescale_forward', True)
    return model, model_wrapper


def data_transforms():
    """get transform of dataset"""
    if FLAGS.data_transforms in ['imagenet1k']:
        if getattr(FLAGS, 'normalize', False):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    else:
        try:
            transforms_lib = importlib.import_module(FLAGS.data_transforms)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    FLAGS.data_transforms))
    return train_transforms, val_transforms, test_transforms


def dataset(train_transforms, val_transforms, test_transforms):
    """get dataset for classification"""
    if FLAGS.dataset == 'imagenet1k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(os.path.join(
                FLAGS.dataset_dir, 'train'),
                                             transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'val'),
                                       transform=val_transforms)
        test_set = None
    else:
        try:
            dataset_lib = importlib.import_module(FLAGS.dataset)
            return dataset_lib.dataset(train_transforms, val_transforms,
                                       test_transforms)
        except ImportError:
            raise NotImplementedError(
                'Dataset {} is not yet implemented.'.format(FLAGS.dataset))
    return train_set, val_set, test_set


def data_loader(train_set, val_set, test_set):
    """get data loader"""
    train_loader = None
    val_loader = None
    test_loader = None
    if getattr(FLAGS, 'batch_size', False):
        if getattr(FLAGS, 'batch_size_per_gpu', False):
            assert FLAGS.batch_size == (FLAGS.batch_size_per_gpu *
                                        FLAGS.num_gpus_per_job)
        else:
            assert FLAGS.batch_size % FLAGS.num_gpus_per_job == 0
            FLAGS.batch_size_per_gpu = (FLAGS.batch_size //
                                        FLAGS.num_gpus_per_job)
    elif getattr(FLAGS, 'batch_size_per_gpu', False):
        FLAGS.batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job
    else:
        raise ValueError('batch size (per gpu) is not defined')
    batch_size = int(FLAGS.batch_size / get_world_size())
    if FLAGS.data_loader in ['imagenet1k']:
        if getattr(FLAGS, 'distributed', False):
            if FLAGS.test_only:
                train_sampler = None
            else:
                train_sampler = DistributedSampler(train_set)
            val_sampler = DistributedSampler(val_set)
        else:
            train_sampler = None
            val_sampler = None
        if not FLAGS.test_only:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                pin_memory=True,
                num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=FLAGS.data_loader_workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
        test_loader = val_loader
    else:
        try:
            data_loader_lib = importlib.import_module(FLAGS.data_loader)
            return data_loader_lib.data_loader(train_set, val_set, test_set)
        except ImportError:
            raise NotImplementedError(
                'Data loader {} is not yet implemented.'.format(
                    FLAGS.data_loader))
    if train_loader is not None:
        FLAGS.data_size_train = len(train_loader.dataset)
    if val_loader is not None:
        FLAGS.data_size_val = len(val_loader.dataset)
    if test_loader is not None:
        FLAGS.data_size_test = len(test_loader.dataset)
    return train_loader, val_loader, test_loader


def lr_func(x, fun='cos'):
    if fun == 'cos':
        return math.cos(x * math.pi) / 2 + 0.5
    if fun == 'exp':
        return math.exp(-x * 8)


def get_lr_scheduler(optimizer, nBatch=None):
    """get learning rate"""
    if FLAGS.lr_scheduler == 'constant':
        lr_dict = {}
        for i in range(FLAGS.num_epochs + 1):
            lr_dict[i] = 1.
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma)
    elif FLAGS.lr_scheduler == 'multistep_iter':
        FLAGS.num_iters = FLAGS.num_epochs * nBatch
        FLAGS.warmup_iters = FLAGS.warmup_epochs * nBatch
        lr_dict = {}
        for i in range(FLAGS.warmup_iters):
            bs_ratio = 256 / FLAGS.batch_size
            lr_dict[i] = (1 - bs_ratio) / FLAGS.warmup_iters * i + bs_ratio
        for i in range(FLAGS.warmup_iters, FLAGS.num_iters + 1):
            lr_dict[i] = FLAGS.multistep_lr_gamma**bisect_right(
                FLAGS.multistep_lr_milestones, i // nBatch)
        lr_lambda = lambda itr: lr_dict[itr]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'exp_decaying':
        lr_dict = {}
        for i in range(FLAGS.num_epochs + 1):
            if i == 0:
                lr_dict[i] = 1
            elif i % getattr(FLAGS, 'exp_decaying_period', 1) == 0:
                lr_dict[i] = lr_dict[i - 1] * FLAGS.exp_decaying_lr_gamma
            else:
                lr_dict[i] = lr_dict[i - 1]
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'exp_decaying_iter':
        FLAGS.num_iters = FLAGS.num_epochs * nBatch
        FLAGS.warmup_iters = FLAGS.warmup_epochs * nBatch
        lr_dict = {}
        for i in range(FLAGS.warmup_iters):
            bs_ratio = 256 / FLAGS.batch_size
            lr_dict[i] = (1 - bs_ratio) / FLAGS.warmup_iters * i + bs_ratio
        for i in range(FLAGS.warmup_iters, FLAGS.num_iters + 1):
            lr_dict[i] = lr_func((i - FLAGS.warmup_iters) /
                                 (FLAGS.num_iters - FLAGS.warmup_iters), 'exp')
        lr_lambda = lambda itr: lr_dict[itr]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'linear_decaying':
        num_epochs = FLAGS.num_epochs - FLAGS.warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs + 1):
            lr_dict[i] = 1. - (i - FLAGS.warmup_epochs) / FLAGS.num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cos_annealing':
        num_epochs = FLAGS.num_epochs - FLAGS.warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs + 1):
            lr_dict[i] = (1.0 + math.cos(
                (i - FLAGS.warmup_epochs) * math.pi / num_epochs)) / 2
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cos_annealing_iter':
        FLAGS.num_iters = FLAGS.num_epochs * nBatch
        FLAGS.warmup_iters = FLAGS.warmup_epochs * nBatch
        lr_dict = {}
        for i in range(FLAGS.warmup_iters):
            bs_ratio = 256 / FLAGS.batch_size
            lr_dict[i] = (1 - bs_ratio) / FLAGS.warmup_iters * i + bs_ratio
        for i in range(FLAGS.warmup_iters, FLAGS.num_iters + 1):
            lr_dict[i] = (1.0 + math.cos(
                (i - FLAGS.warmup_iters) * math.pi /
                (FLAGS.num_iters - FLAGS.warmup_iters))) / 2
        lr_lambda = lambda itr: lr_dict[itr]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lr_lambda)
    else:
        try:
            lr_scheduler_lib = importlib.import_module(FLAGS.lr_scheduler)
            return lr_scheduler_lib.get_lr_scheduler(optimizer)
        except ImportError:
            raise NotImplementedError(
                'Learning rate scheduler {} is not yet implemented.'.format(
                    FLAGS.lr_scheduler))
    return lr_scheduler


def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == 'sgd':
        model_params = []
        for name, params in model.named_parameters():
            ps = list(params.size())
            weight_decay_scheme = getattr(FLAGS, 'weight_decay_scheme', 'all')
            if weight_decay_scheme == 'all':
                weight_decay = FLAGS.weight_decay
            elif weight_decay_scheme == 'only_no_depthwise':
                # all depthwise convolution (N, 1, x, x) has no weight decay
                # weight decay only on normal conv, bn and fc
                if len(ps) == 4 and ps[1] != 1:
                    weight_decay = FLAGS.weight_decay
                elif len(ps) in [1, 2]:
                    weight_decay = FLAGS.weight_decay
                else:
                    weight_decay = 0
            elif weight_decay_scheme == 'only_no_bn':
                ## only no weight decay on bn
                if len(ps) == 4:
                    weight_decay = FLAGS.weight_decay
                elif len(ps) == 2:
                    weight_decay = FLAGS.weight_decay
                else:
                    weight_decay = 0
            elif weight_decay_scheme == 'no_depthwise_no_bn':
                ## no weight decay on bn and no weight decay on depthwise conv
                if len(ps) == 4 and ps[1] != 1:
                    weight_decay = FLAGS.weight_decay
                elif len(ps) == 2:
                    weight_decay = FLAGS.weight_decay
                else:
                    weight_decay = 0
            else:
                raise NotImplementedError
            lr = FLAGS.lr
            item = {
                'params': params,
                'weight_decay': weight_decay,
                'lr': lr,
                'momentum': FLAGS.momentum,
                'nesterov': FLAGS.nesterov
            }
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))
    return optimizer


def set_random_seed(seed=None):
    """set random seed"""
    if seed is None:
        seed = getattr(FLAGS, 'random_seed', 0)
    mprint('seed for random sampling: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@master_only
def get_meters(phase, single_sample=False):
    """util function for meters"""
    def get_single_meter(phase, suffix=''):
        meters = {}
        meters['loss'] = ScalarMeter('{}_loss/{}'.format(phase, suffix))
        for k in FLAGS.topk:
            meters['top{}_error'.format(k)] = ScalarMeter(
                '{}_top{}_error/{}'.format(phase, k, suffix))
        return meters

    assert phase in ['train', 'val', 'test', 'calib'], 'Invalid phase.'
    if single_sample:
        meters = get_single_meter(phase)
    else:
        meters = get_single_meter(phase)
    if phase == 'val':
        meters['best_val'] = ScalarMeter('best_val')
    return meters


def get_experiment_setting():
    experiment_setting = []
    experiment_setting.append('ptcv_pretrained_{}'.format(
        getattr(FLAGS, 'ptcv_pretrained', False)))
    experiment_setting.append('nvidia_pretrained_{}'.format(
        getattr(FLAGS, 'nvidia_pretrained', False)))
    experiment_setting.append('hawq_pretrained_{}'.format(
        getattr(FLAGS, 'hawq_pretrained', False)))
    experiment_setting.append('finetune_iters_{}'.format(
        getattr(FLAGS, 'finetune_iters', float('inf'))))
    experiment_setting.append('bn_calib_before_test_{}'.format(
        getattr(FLAGS, 'bn_calib_before_test', False)))
    experiment_setting.append('bn_calib_batch_num_{}'.format(
        getattr(FLAGS, 'bn_calib_batch_num', -1)))
    experiment_setting.append('quant_avgpool_{}'.format(
        getattr(FLAGS, 'quant_avgpool', False)))
    experiment_setting.append('pool_fusing_{}'.format(
        getattr(FLAGS, 'pool_fusing', False)))
    experiment_setting.append(
        f'weight_format_wl_{FLAGS.weight_format[0]}_fl_{FLAGS.weight_format[1]}'
    )
    experiment_setting.append(
        f'input_format_wl_{FLAGS.input_format[0]}_fl_{FLAGS.input_format[1]}')
    experiment_setting.append('rescale_forward_{}'.format(
        getattr(FLAGS, 'rescale_forward', True)))
    experiment_setting.append('rescale_forward_conv_{}'.format(
        getattr(FLAGS, 'rescale_forward_conv', False)))
    experiment_setting.append('rescale_type_{}'.format(
        getattr(FLAGS, 'rescale_type', 'constant')))
    experiment_setting.append('input_fraclen_sharing_{}'.format(
        getattr(FLAGS, 'input_fraclen_sharing', False)))
    experiment_setting.append('floating_model_{}'.format(
        getattr(FLAGS, 'floating_model', False)))
    experiment_setting.append('floating_wo_clip_{}'.format(
        getattr(FLAGS, 'floating_wo_clip', False)))
    experiment_setting.append('no_clipping_{}'.format(
        getattr(FLAGS, 'no_clipping', False)))
    if getattr(FLAGS, 'fp_pretrained_file', ''):
        fp_pretrained = True
    else:
        fp_pretrained = False
    experiment_setting.append(f'fp_pretrained_{fp_pretrained}')
    experiment_setting.append('format_type_{}'.format(
        getattr(FLAGS, 'format_type', None)))
    experiment_setting.append('format_from_metric_{}'.format(
        getattr(FLAGS, 'format_from_metric', False)))
    experiment_setting.append('momentum_for_metric_{}'.format(
        getattr(FLAGS, 'momentum_for_metric', 0.1)))
    experiment_setting.append('metric_{}'.format(getattr(
        FLAGS, 'metric', None)))
    experiment_setting.append('format_grid_search_{}'.format(
        getattr(FLAGS, 'format_grid_search', False)))
    experiment_setting.append('bn_momentum_{}'.format(
        getattr(FLAGS, 'bn_momentum', 0.1)))
    experiment_setting.append('bn_eps_{}'.format(getattr(FLAGS, 'bn_eps',
                                                         0.1)))
    experiment_setting.append('lr_{}'.format(FLAGS.lr))
    experiment_setting.append('weight_decay_scheme_{}'.format(
        getattr(FLAGS, 'weight_decay_scheme', 'all')))
    experiment_setting.append('normalize_{}'.format(
        getattr(FLAGS, 'normalize', False)))
    experiment_setting.append('warmup_epochs_{}'.format(FLAGS.warmup_epochs))
    experiment_setting.append('weight_decay_{}'.format(FLAGS.weight_decay))
    experiment_setting = os.path.join(*experiment_setting)
    mprint('Experiment settings: {}'.format(experiment_setting))
    return experiment_setting


def forward_loss(model, criterion, input, target, meter, train=False):
    """forward model and return loss"""
    if getattr(FLAGS, 'normalize', False):
        assert getattr(FLAGS, 'ptcv_pretrained', False) or getattr(
            FLAGS, 'nvidia_pretrained', False) or getattr(
                FLAGS, 'hawq_pretrained', False)
        if getattr(model, 'int_op_only', False):
            input_fraclen = model.head[0].input_fraclen
            input_symmetric = model.head[0].input_symmetric
            input = (fix_quant(input, 8, input_fraclen * 1.0, 1,
                               input_symmetric)[0] * (2**input_fraclen)).int()
            setattr(input, 'output_fraclen', input_fraclen.item())
    else:
        assert torch.all(input >= 0)
        if getattr(model, 'int_op_only', False):
            input = (255 * input).round_().int()
            setattr(input, 'output_fraclen', 8)
        else:
            input = (255 * input).round_() / 256
    output = model(input)
    loss = torch.mean(criterion(output, target))
    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = []
    for k in FLAGS.topk:
        correct_k.append(correct[:k].float().sum(0))
    res = torch.cat([loss.view(1)] + correct_k, dim=0)
    if getattr(FLAGS, 'distributed', False) and getattr(
            FLAGS, 'distributed_all_reduce', False):
        res = dist_all_reduce_tensor(res)
    res = res.cpu().detach().numpy()
    bs = (res.size - 1) // len(FLAGS.topk)
    for i, k in enumerate(FLAGS.topk):
        error_list = list(1. - res[1 + i * bs:1 + (i + 1) * bs])
        if meter is not None:
            meter['top{}_error'.format(k)].cache_list(error_list)
        if getattr(FLAGS, 'print_each_iter', False):
            mprint(f'top{k} err: {sum(error_list) / len(error_list)}.')
    if meter is not None:
        meter['loss'].cache(res[0])
    return loss


@timing
def run_one_epoch(epoch,
                  loader,
                  model,
                  criterion,
                  optimizer,
                  meters,
                  phase='train',
                  scheduler=None):
    """run one epoch for train/val/test/calib"""
    t_start = time.time()
    assert phase in ['train', 'val', 'test',
                     'calib'], "phase not be in train/val/test/calib."
    train = phase == 'train'
    if train:
        model.train()
    else:
        model.eval()
        if phase == 'calib':
            model.apply(bn_calib)

    if getattr(FLAGS, 'distributed', False):
        loader.sampler.set_epoch(epoch)

    skip_print = False
    for batch_idx, (input, target) in enumerate(loader):
        if phase == 'calib':
            if batch_idx == getattr(FLAGS, 'bn_calib_batch_num', -1):
                break
        if train and (batch_idx >= getattr(FLAGS, 'finetune_iters',
                                           float('inf'))):
            if getattr(FLAGS, 'finetune_iters', float('inf')) == 0:
                skip_print = True
            break
        if not getattr(FLAGS, 'int_op_only', False):
            target = target.cuda(non_blocking=True)
        if train:
            if FLAGS.lr_scheduler == 'linear_decaying':
                linear_decaying_per_step = (FLAGS.lr / FLAGS.num_epochs /
                                            len(loader.dataset) *
                                            FLAGS.batch_size)
                for param_group in optimizer.param_groups:
                    param_group['lr'] -= linear_decaying_per_step
            # For PyTorch 1.1+, comment the following two line
            #if FLAGS.lr_scheduler in ['exp_decaying_iter', 'gaussian_iter', 'cos_annealing_iter', 'butterworth_iter', 'mixed_iter', 'multistep_iter']:
            #    scheduler.step()
            optimizer.zero_grad()
            loss = forward_loss(model,
                                criterion,
                                input,
                                target,
                                meters,
                                train=True)
            loss.backward()
            if getattr(FLAGS, 'distributed', False) and getattr(
                    FLAGS, 'distributed_all_reduce', False):
                allreduce_grads(model)
            optimizer.step()
            # For PyTorch 1.0 or earlier, comment the following two lines
            if FLAGS.lr_scheduler in [
                    'exp_decaying_iter', 'cos_annealing_iter', 'multistep_iter'
            ]:
                scheduler.step()
        else:  #not train
            forward_loss(model, criterion, input, target, meters)

    val_top1 = None
    if is_master() and meters is not None and not skip_print:
        results = flush_scalar_meters(meters)
        mprint('{:.1f}s\t{}\t{}/{}: '.format(time.time() - t_start, phase,
                                             epoch, FLAGS.num_epochs) +
               ', '.join('{}: {}'.format(k, v) for k, v in results.items()))
        val_top1 = results['top1_error']
    return val_top1


@timing
def train_val_test():
    """train and val"""
    torch.backends.cudnn.benchmark = True
    # init distributed
    if getattr(FLAGS, 'distributed', False):
        gpu_id = init_dist(init_method=getattr(FLAGS, 'dist_url', None))
    else:
        gpu_id = None
    # seed
    if getattr(FLAGS, 'use_diff_seed',
               False) and not getattr(FLAGS, 'stoch_valid', False):
        mprint('use diff seed is True')
        while not is_initialized():
            mprint('Waiting for initialization ...')
            time.sleep(5)
        mprint('Expected seed: {}'.format(
            getattr(FLAGS, 'random_seed', 0) + get_rank()))
        set_random_seed(getattr(FLAGS, 'random_seed', 0) + get_rank())
    else:
        set_random_seed()

    # experiment setting
    experiment_setting = get_experiment_setting()

    # model
    model, model_wrapper = get_model(gpu_id=gpu_id)
    if getattr(FLAGS, 'int_op_only', False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    # data
    train_transforms, val_transforms, test_transforms = data_transforms()
    train_set, val_set, test_set = dataset(train_transforms, val_transforms,
                                           test_transforms)
    train_loader, val_loader, test_loader = data_loader(
        train_set, val_set, test_set)

    log_dir = FLAGS.log_dir
    log_dir = os.path.join(log_dir, experiment_setting)

    # full precision pretrained
    if getattr(FLAGS, 'fp_pretrained_file', ''):
        checkpoint = torch.load(FLAGS.fp_pretrained_file,
                                map_location=lambda storage, loc: storage)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if getattr(FLAGS, 'pretrained_model_remap_keys', False):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                mprint('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        model_dict = model_wrapper.state_dict()
        for k in list(checkpoint.keys()):
            if k not in model_dict.keys():
                mprint(f'Pop out key {k} from checkpoint keys.')
                checkpoint.pop(k)
        model_dict.update(checkpoint)
        model_wrapper.load_state_dict(model_dict)
        mprint('Loaded full precision model {}.'.format(
            FLAGS.fp_pretrained_file))

    if getattr(FLAGS, 'ptcv_pretrained', False) or getattr(
            FLAGS, 'hawq_pretrained', False):
        ptcv_load(model)
        if getattr(FLAGS, 'ptcv_pretrained', False):
            mprint('Loaded ptcv pretrained model.')
        if getattr(FLAGS, 'hawq_pretrained', False):
            mprint('Loaded hawq pretrained model.')

    if getattr(FLAGS, 'nvidia_pretrained', False):
        nvidia_load(model)
        mprint('Loaded nvidia pretrained model.')

    # check pretrained
    if FLAGS.pretrained_file:
        checkpoint = torch.load(FLAGS.pretrained_file,
                                map_location=lambda storage, loc: storage)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if getattr(FLAGS, 'pretrained_model_remap_keys', False):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                mprint('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        model_wrapper.load_state_dict(checkpoint)
        mprint('Loaded model {}.'.format(FLAGS.pretrained_file))
    optimizer = get_optimizer(model_wrapper)

    if getattr(FLAGS, 'integize', False):
        assert not getattr(FLAGS, 'int_op_only', False)
        model = model.int_model().to(next(model.parameters()).device)
        if getattr(FLAGS, 'distributed', False):
            if getattr(FLAGS, 'distributed_all_reduce', False):
                model_wrapper = AllReduceDistributedDataParallel(model.cuda())
            else:
                model_wrapper = torch.nn.parallel.DistributedDataParallel(
                    model.cuda(), [gpu_id],
                    gpu_id,
                    find_unused_parameters=True)
        else:
            model_wrapper = torch.nn.DataParallel(model).cuda()
        integize_file_path = getattr(FLAGS, 'integize_file_path', '')
        if integize_file_path:
            os.makedirs(os.path.join(integize_file_path, 'checkpoints'),
                        exist_ok=True)
            os.makedirs(os.path.join(integize_file_path, 'onnx_files'),
                        exist_ok=True)
        if integize_file_path:
            integize_checkpoint = os.path.join(integize_file_path,
                                               'checkpoints',
                                               'integized_model.pt')
            torch.save({'model': model_wrapper.state_dict},
                       integize_checkpoint)
            integize_onnx_file = os.path.join(integize_file_path, 'onnx_files',
                                              'integized_model.onnx')
            data_shape = [1, 3, FLAGS.image_size, FLAGS.image_size]
            data_dtype = torch.float
            onnx_export(model, data_shape, data_dtype,
                        next(model.parameters()).device, integize_onnx_file)
            mprint(
                f'Integized model checkpoint and onnx file saved to {integize_file_path}.'
            )

    if getattr(FLAGS, 'int_op_only', False):
        assert not getattr(FLAGS, 'integize', False)
        model.apply(lambda m: setattr(m, 'int_op_only', True))
        model = model.int_model().cpu()
        model.apply(lambda m: setattr(m, 'int_op_only', True))
        model_wrapper = model
        int_op_only_file_path = getattr(FLAGS, 'int_op_only_file_path', '')
        if int_op_only_file_path:
            os.makedirs(os.path.join(int_op_only_file_path, 'checkpoints'),
                        exist_ok=True)
            os.makedirs(os.path.join(int_op_only_file_path, 'onnx_files'),
                        exist_ok=True)
        if int_op_only_file_path:
            int_op_only_checkpoint = os.path.join(int_op_only_file_path,
                                                  'checkpoints',
                                                  'int_op_only_model.pt')
            torch.save({'model': model_wrapper.state_dict},
                       int_op_only_checkpoint)
            int_op_only_onnx_file = os.path.join(int_op_only_file_path,
                                                 'onnx_files',
                                                 'int_op_only_model.onnx')
            data_shape = [1, 3, FLAGS.image_size, FLAGS.image_size]
            data_dtype = torch.int
            onnx_export(model, data_shape, data_dtype,
                        next(model.parameters()).device, int_op_only_onnx_file)
            mprint(
                f'Integer operation only model checkpoint and onnx file saved to {int_op_only_file_path}.'
            )

    if FLAGS.test_only and (test_loader is not None):
        mprint('Start testing.')
        test_meters = get_meters('test')
        with torch.no_grad():
            run_one_epoch(-1,
                          test_loader,
                          model_wrapper,
                          criterion,
                          optimizer,
                          test_meters,
                          phase='test')
        if is_master():
            for n, m in model.named_modules():
                if isinstance(m, (ReLUClipFXQConvBN, ReLUClipFXQLinear)):
                    mprint(f'layer name: {n}.')
                    mprint(f'alpha: {m.alpha}.')
                    if m.get_master_layer() is not None:
                        mprint(
                            f'master layer alpha: {m.get_master_layer().get_alpha()}.'
                        )
                    else:
                        mprint(f'master layer: {m.get_master_layer()}.')
                    mprint(f'alpha in use: {m.get_alpha()}.')
                    mprint(f'weight format: {m.get_weight_format()}.')
                    mprint(f'input format: {m.get_input_format()}.')
                    mprint(f'fix scaling: {m.fix_scaling}.')
                    if isinstance(m, (ReLUClipFXQConvBN, )):
                        mprint(
                            f'following layer fix scaling: {m.get_following_layer().fix_scaling}.'
                        )
                    if m.format_from_metric or m.format_grid_search:
                        mprint(f'input_fraclen: {m.get_input_fraclen()}.')
                        mprint(f'weight_fraclen: {m.get_weight_fraclen()}.')
        return

    # check resume training
    if os.path.exists(os.path.join(log_dir, 'latest_checkpoint.pt')):
        checkpoint = torch.load(os.path.join(log_dir, 'latest_checkpoint.pt'),
                                map_location=lambda storage, loc: storage)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        if FLAGS.lr_scheduler in [
                'exp_decaying_iter', 'cos_annealing_iter', 'multistep_iter'
        ]:
            lr_scheduler = get_lr_scheduler(optimizer, len(train_loader))
            lr_scheduler.last_epoch = last_epoch * len(train_loader)
        else:
            lr_scheduler = get_lr_scheduler(optimizer)
            lr_scheduler.last_epoch = last_epoch
        best_val = checkpoint['best_val']
        train_meters, val_meters = checkpoint['meters']
        mprint('Loaded checkpoint {} at epoch {}.'.format(log_dir, last_epoch))
    else:
        if FLAGS.lr_scheduler in [
                'exp_decaying_iter', 'cos_annealing_iter', 'multistep_iter'
        ]:
            lr_scheduler = get_lr_scheduler(optimizer, len(train_loader))
        else:
            lr_scheduler = get_lr_scheduler(optimizer)
        last_epoch = lr_scheduler.last_epoch
        best_val = 1.
        train_meters = get_meters('train')
        val_meters = get_meters('val')
        # if start from scratch, print model
        mprint(model_wrapper)

    if getattr(FLAGS, 'start_epoch', None) is not None:
        if FLAGS.lr_scheduler in [
                'exp_decaying_iter', 'cos_annealing_iter', 'multistep_iter'
        ]:
            last_epoch = FLAGS.start_epoch
            lr_scheduler.last_epoch = FLAGS.start_epoch * len(train_loader)
        else:
            last_epoch = FLAGS.start_epoch
            lr_scheduler.last_epoch = FLAGS.start_epoch

    if getattr(FLAGS, 'log_dir', None):
        try:
            os.makedirs(log_dir)
        except OSError:
            pass

    mprint('log dir: ', log_dir)

    if getattr(FLAGS, 'bn_calib_before_test', False):
        calib_meters = get_meters('calib')
        mprint('Start calibration.')
        run_one_epoch(-1,
                      train_loader,
                      model_wrapper,
                      criterion,
                      optimizer,
                      calib_meters,
                      phase='calib')
        mprint('Start validation after calibration.')
        if val_meters is not None:
            val_meters['best_val'].cache(best_val)
        with torch.no_grad():
            top1_error = run_one_epoch(-1,
                                       val_loader,
                                       model_wrapper,
                                       criterion,
                                       optimizer,
                                       val_meters,
                                       phase='val')
        if is_master():
            if top1_error < best_val:
                best_val_cal = top1_error
                torch.save({
                    'model': model_wrapper.state_dict(),
                }, os.path.join(log_dir, 'best_model_bn_calibrated.pt'))
                mprint(
                    'New best validation top1 error after bn calibration: {:.3f}'
                    .format(best_val_cal))
        return

    mprint('Start training.')

    for epoch in range(last_epoch, FLAGS.num_epochs):
        if FLAGS.lr_scheduler in [
                'exp_decaying_iter', 'cos_annealing_iter', 'multistep_iter'
        ]:
            lr_sched = lr_scheduler
        else:
            lr_sched = None
            # For PyTorch 1.1+, comment the following line
            #lr_scheduler.step()
        # train
        mprint(' train '.center(40, '*'))
        run_one_epoch(epoch,
                      train_loader,
                      model_wrapper,
                      criterion,
                      optimizer,
                      train_meters,
                      phase='train',
                      scheduler=lr_sched)

        # val
        mprint(' validation '.center(40, '~'))
        if val_meters is not None:
            val_meters['best_val'].cache(best_val)
        with torch.no_grad():
            top1_error = run_one_epoch(epoch,
                                       val_loader,
                                       model_wrapper,
                                       criterion,
                                       optimizer,
                                       val_meters,
                                       phase='val')
        if is_master():
            if top1_error < best_val:
                best_val = top1_error
                torch.save({
                    'model': model_wrapper.state_dict(),
                }, os.path.join(log_dir, 'best_model.pt'))
                mprint(
                    'New best validation top1 error: {:.3f}'.format(best_val))

            # save latest checkpoint
            torch.save(
                {
                    'model': model_wrapper.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'last_epoch': epoch + 1,
                    'best_val': best_val,
                    'meters': (train_meters, val_meters),
                }, os.path.join(log_dir, 'latest_checkpoint.pt'))

        # For PyTorch 1.0 or earlier, comment the following two lines
        if FLAGS.lr_scheduler not in [
                'exp_decaying_iter', 'cos_annealing_iter', 'multistep_iter'
        ] and epoch < FLAGS.num_epochs - 1:
            lr_scheduler.step()

    if is_master():
        for n, m in model.named_modules():
            if isinstance(m, (ReLUClipFXQConvBN, ReLUClipFXQLinear)):
                mprint(f'layer name: {n}.')
                mprint(f'alpha: {m.alpha}.')
                if m.get_master_layer() is not None:
                    mprint(
                        f'master layer alpha: {m.get_master_layer().get_alpha()}.'
                    )
                else:
                    mprint(f'master layer: {m.get_master_layer()}.')
                mprint(f'alpha in use: {m.get_alpha()}.')
                mprint(f'weight format: {m.get_weight_format()}.')
                mprint(f'input format: {m.get_input_format()}.')
                mprint(f'fix scaling: {m.fix_scaling}.')
                if isinstance(m, (ReLUClipFXQConvBN, )):
                    mprint(
                        f'following layer fix scaling: {m.get_following_layer().fix_scaling}.'
                    )
                if m.format_from_metric or m.format_grid_search:
                    mprint(f'input_fraclen: {m.get_input_fraclen()}.')
                    mprint(f'weight_fraclen: {m.get_weight_fraclen()}.')
    return


def init_multiprocessing():
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass


def main():
    """train and eval model"""
    init_multiprocessing()
    train_val_test()


if __name__ == "__main__":
    main()

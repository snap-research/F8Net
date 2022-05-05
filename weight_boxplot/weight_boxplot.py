import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as tv_models
from pytorchcv.model_provider import get_model as ptcv_get_model


def main():
    fig_size = (6, 4)
    axes_label_size = 16
    text_size = 12
    title_size = 16
    legend_size = 8
    font_weight = 'normal'
    for ckpt in ['torchvision', 'pytorchcv']:
        if ckpt == 'torchvision':
            net = tv_models.mobilenetv2.mobilenet_v2(pretrained=True)
        elif ckpt == 'pytorchcv':
            net = ptcv_get_model('mobilenetv2b_w1', pretrained=True)
        eff_weight_list = []
        conv_weight = None
        fc_weight = None
        for n, m in net.named_modules():
            if isinstance(m, nn.Conv2d):
                conv_weight = m.weight
            elif isinstance(m, nn.BatchNorm2d):
                bn_weight = m.weight
                bn_var = m.running_var
                bn_eps = m.eps
                eff_weight = conv_weight * (
                    bn_weight / torch.sqrt(bn_var + bn_eps))[:, None, None,
                                                             None]
                eff_weight_list.append(
                    eff_weight.detach().cpu().flatten().numpy())
                conv_weight = None
            elif isinstance(m, nn.Linear):
                eff_weight = m.weight
                eff_weight_list.append(
                    eff_weight.detach().cpu().flatten().numpy())
                conv_weight = None
            else:
                conv_weight = None
        plt.figure(figsize=fig_size)
        box_edge_color = 'k'
        box_bar_color = 'r'
        plt.boxplot(eff_weight_list,
                    showfliers=False,
                    boxprops={'color': box_edge_color},
                    capprops={'color': box_edge_color},
                    whiskerprops={'color': box_edge_color},
                    flierprops={'markeredgecolor': box_edge_color},
                    medianprops={'color': box_bar_color})
        plt.xticks([])
        plt.xlabel('Layer', fontsize=axes_label_size)
        plt.ylabel('Effective Weight', fontsize=axes_label_size)
        plt.setp(plt.gca().get_xticklabels(),
                 fontsize=axes_label_size,
                 fontweight=font_weight)
        plt.setp(plt.gca().get_yticklabels(),
                 fontsize=axes_label_size,
                 fontweight=font_weight)
        plt.savefig(f'./mobilenetv2_{ckpt}_eff_weight_wo_title.pdf',
                    dpi=300,
                    bbox_inches='tight')
        plt.title('Effective Weight Range (FP MobileNet V2)',
                  fontsize=title_size)
        plt.savefig(f'./mobilenetv2_{ckpt}_eff_weight.pdf',
                    dpi=300,
                    bbox_inches='tight')


if __name__ == '__main__':
    main()

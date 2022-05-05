import os
import numpy as np

import matplotlib
matplotlib.use('Agg')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch


def main():
    work_dir = './'
    log_file_name_dict = {
        'ptcv':
        'res50_fix_quant_ptcv_pretrained.out',
        'nvidia':
        'res50_fix_quant_nvidia_pretrained.out'
    }
    for pretrained_method in log_file_name_dict.keys():
        print(f'pretrained with {pretrained_method}')
        log_file_name = log_file_name_dict[pretrained_method]

        depthwise_layer = []

        layer_fraclen_dict = {}
        with open(os.path.join(work_dir, log_file_name), 'r') as f:
            lines = f.read().splitlines()
            lines = [
                l for l in lines
                if 'fraclen' in l or l.startswith('layer name')
            ]
            lines = [
                l for l in lines
                if 'setting' not in l and 'model' not in l and 'log' not in l
            ]
            lines = [l[:-1] for l in lines]
            assert len(lines) % 3 == 0
        for idx in range(len(lines) // 3):
            assert 'layer name' in lines[idx * 3]
            assert 'input_fraclen' in lines[idx * 3 + 1]
            assert 'weight_fraclen' in lines[idx * 3 + 2]
            layer_name = lines[idx * 3][12:]
            if 'shortcut' in layer_name:
                shortcut = True
            else:
                shortcut = False
            if 'classifier' in layer_name:
                fc = True
            else:
                fc = False
            input_fraclen = int(
                torch.round(eval('torch.' + lines[idx * 3 + 1][15:])).item())
            weight_fraclen = int(np.around(eval(lines[idx * 3 + 2][16:])))
            layer_fraclen_dict[layer_name] = (weight_fraclen, input_fraclen,
                                              shortcut, fc)

        cm = {'b': [80 / 255, 156 / 255, 1.0], 'm': [250 / 255, 0, 101 / 255]}
        model_name = {'resnet50': 'ResNet50'}
        bit_cmap = {
            0: [1.0, 1.0, 1.0],
            1: [155 / 255, 221 / 255, 239 / 255],
            2: [255 / 255, 102 / 255, 153 / 255],
            3: [189 / 255, 146 / 255, 222 / 255],
            4: [75 / 255, 148 / 255, 255 / 255],
            5: [199 / 255, 225 / 255, 181 / 255],
            6: [241 / 255, 245 / 255, 161 / 255],
            7: [255 / 255, 133 / 255, 133 / 255],
            8: [40 / 255, 240 / 255, 128 / 255]
        }
        fig_size = (6, 4)
        axes_label_size = 16
        text_size = 12
        title_size = 16
        legend_size = 8
        font_weight = 'normal'
        for model in ['resnet50']:
            barwidth = 1
            patch_width = 1.0
            arrow_extension = 2.0

            weight_fraclen_list = [v[0] for v in layer_fraclen_dict.values()]
            input_fraclen_list = [-v[1] for v in layer_fraclen_dict.values()]
            ## The first layer does not quantize input
            input_fraclen_list[0] = -8
            fc_list = [v[3] for v in layer_fraclen_dict.values()]
            plt.figure(figsize=fig_size)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().tick_params(axis='both', which='both', length=0)
            ids = np.arange(len(weight_fraclen_list))
            color_w = [cm['b'] for idx in ids]
            color_a = [cm['m'] for idx in ids]
            plt.bar(ids * barwidth,
                    weight_fraclen_list,
                    bottom=0.01,
                    width=barwidth * 0.8,
                    color=color_w,
                    edgecolor='k',
                    linewidth=0.3)
            plt.bar(ids * barwidth,
                    input_fraclen_list,
                    bottom=-0.01,
                    width=barwidth * 0.8,
                    color=color_a,
                    edgecolor='k',
                    linewidth=0.3)
            plt.plot(np.arange(-barwidth / 2,
                               barwidth * len(ids) + barwidth / 2, barwidth),
                     np.ones(len(ids) + 1) * 2,
                     ls='--',
                     lw=0.5,
                     c='k')
            plt.plot(np.arange(-barwidth / 2,
                               barwidth * len(ids) + barwidth / 2, barwidth),
                     np.ones(len(ids) + 1) * 4,
                     ls='--',
                     lw=0.5,
                     c='k')
            plt.plot(np.arange(-barwidth / 2,
                               barwidth * len(ids) + barwidth / 2, barwidth),
                     np.ones(len(ids) + 1) * 6,
                     ls='--',
                     lw=0.5,
                     c='k')
            plt.plot(np.arange(-barwidth / 2,
                               barwidth * len(ids) + barwidth / 2, barwidth),
                     np.ones(len(ids) + 1) * 8,
                     ls='--',
                     lw=0.5,
                     c='k')
            plt.plot(np.arange(-barwidth / 2,
                               barwidth * len(ids) + barwidth / 2, barwidth),
                     -np.ones(len(ids) + 1) * 8,
                     ls='--',
                     lw=0.5,
                     c='k')
            plt.plot(np.arange(-barwidth / 2,
                               barwidth * len(ids) + barwidth / 2, barwidth),
                     -np.ones(len(ids) + 1) * 6,
                     ls='--',
                     lw=0.5,
                     c='k')
            plt.plot(np.arange(-barwidth / 2,
                               barwidth * len(ids) + barwidth / 2, barwidth),
                     -np.ones(len(ids) + 1) * 4,
                     ls='--',
                     lw=0.5,
                     c='k')
            plt.plot(np.arange(-barwidth / 2,
                               barwidth * len(ids) + barwidth / 2, barwidth),
                     -np.ones(len(ids) + 1) * 2,
                     ls='--',
                     lw=0.5,
                     c='k')
            plt.arrow(-barwidth / 2,
                      0,
                      barwidth * len(ids) + barwidth / 2 +
                      barwidth * arrow_extension,
                      0,
                      ls='-',
                      color='k',
                      width=.005,
                      head_width=0.15,
                      head_length=0.1 * arrow_extension)
            rect_w_pw = patches.Rectangle((barwidth * len(ids) / 2 * 0, -10),
                                          2.2 * patch_width,
                                          1.2 * patch_width,
                                          linewidth=.5,
                                          edgecolor='k',
                                          facecolor=cm['b'])
            rect_a_pw = patches.Rectangle((barwidth * len(ids) / 2 * 1.0, -10),
                                          2.2 * patch_width,
                                          1.2 * patch_width,
                                          linewidth=.5,
                                          edgecolor='k',
                                          facecolor=cm['m'])
            plt.gca().add_patch(rect_w_pw)
            plt.gca().add_patch(rect_a_pw)
            if model in ['resnet50']:
                plt.text(barwidth * len(ids) / 2 * 0 + patch_width * 2.6,
                         -9.75,
                         '#Weight FL',
                         fontsize=text_size)
                plt.text(barwidth * len(ids) / 2 * 1 + patch_width * 2.6,
                         -9.75,
                         '#Activation FL',
                         fontsize=text_size)
            plt.xticks([])
            plt.yticks(np.arange(-8, 9, 2), np.abs(np.arange(-8, 9, 2)))
            plt.xlabel('Layer', fontsize=axes_label_size)
            plt.ylabel('Fractional Length', fontsize=axes_label_size)
            plt.setp(plt.gca().get_xticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            plt.setp(plt.gca().get_yticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            plt.gca().xaxis.set_label_coords(1.02, 0.52)
            plt.gca().yaxis.set_label_coords(-0.05, 0.58)
            plt.xlim(-0.6,
                     barwidth * len(ids) + 1.2 * barwidth * arrow_extension)
            plt.ylim(-10, 8.5)
            plt.savefig(f'./{model}_{pretrained_method}_8bit_fraclens_wo_title.pdf',
                        dpi=300,
                        bbox_inches='tight')
            plt.title(
                f'Fractional Length vs Layer (8-bit {model_name[model]})',
                fontsize=title_size)
            plt.savefig(f'./{model}_{pretrained_method}_8bit_fraclens.pdf',
                        dpi=300,
                        bbox_inches='tight')


if __name__ == '__main__':
    main()

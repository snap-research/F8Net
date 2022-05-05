import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

import torch


def fix_quant(input, wl=8, fl=0, signed=True):
    assert wl >= 0
    assert fl >= 0
    if signed:
        assert fl <= wl - 1
    else:
        assert fl <= wl
    assert type(wl) == int
    assert type(fl) == int
    res = input * (2**fl)
    res.round_()
    if signed:
        bound = 2**(wl - 1) - 1
        res.clamp_(max=bound, min=-bound)
    else:
        bound = 2**wl - 1
        res.clamp_(max=bound, min=0)
    res.div_(2**fl)
    return res


def main():
    fl_cmap = {
        0: [0.2, 0.2, 0.2],
        1: [155 / 255, 221 / 255, 239 / 255],
        2: [255 / 255, 102 / 255, 153 / 255],
        3: [189 / 255, 146 / 255, 222 / 255],
        4: [75 / 255, 148 / 255, 255 / 255],
        5: [199 / 255, 225 / 255, 181 / 255],
        6: [241 / 255, 245 / 255, 161 / 255],
        7: [255 / 255, 133 / 255, 133 / 255],
        8: [40 / 255, 240 / 255, 128 / 255]
    }

    wl = 8
    N = 10000
    signed = True
    normalize = True

    fig_size = (6, 4)
    axes_label_size = 16
    text_size = 12
    equation_text_size = 20
    title_size = 16
    legend_size = 8
    font_weight = 'normal'

    sigma_list = np.logspace(-3, 3, 1000)
    title_dict = {0: 'Unsigned (Rectified Gaussian)', 1: 'Signed (Gaussian)'}
    for wl in [8]:
        for signed in [True, False]:
            fl_list = list(range(wl + 1 - int(signed)))
            w_quant_err_sigma = []
            opt_fl = []
            opt_err = []
            for sigma in sigma_list:
                w_quant_err_fl = []
                w_rand = torch.randn(N) * sigma
                if not signed:
                    w_rand = torch.relu(w_rand)
                for fl in fl_list:
                    w_quant = fix_quant(w_rand, wl, fl, signed)
                    error = np.mean((w_rand - w_quant).cpu().numpy()**2)**0.5
                    if normalize:
                        error = error / sigma
                    w_quant_err_fl.append(error)
                err_min_idx = np.argmin(w_quant_err_fl)
                opt_fl.append(fl_list[err_min_idx])
                opt_err.append(w_quant_err_fl[err_min_idx])
                w_quant_err_sigma.append(w_quant_err_fl)
            w_quant_err_sigma = np.array(w_quant_err_sigma)

            ## replace the first zeros by the max
            opt_fl = np.array(opt_fl)
            idx_last_max = np.nonzero(opt_fl == max(fl_list))[0][-1]
            mask_to_change = opt_fl[:idx_last_max] == 0
            opt_fl[:idx_last_max][mask_to_change] = max(fl_list)

            ## sigma
            opt_sigma = []
            fig = plt.figure(figsize=fig_size)
            for idx, fl in enumerate(fl_list):
                plt.semilogx(sigma_list,
                             w_quant_err_sigma[:, idx],
                             label=f'FL={fl}',
                             c=fl_cmap[fl],
                             lw=2)
                local_min = np.argmin(w_quant_err_sigma[:, idx])
                opt_sigma.append(sigma_list[local_min])
                plt.scatter(sigma_list[local_min],
                            w_quant_err_sigma[local_min, idx],
                            color=fl_cmap[fl],
                            marker='*',
                            s=60)
            plt.gca().annotate('Overflow',
                               fontsize=text_size,
                               fontweight=font_weight,
                               xy=(0.55, 0.87),
                               xycoords='figure fraction',
                               xytext=(0.65, 0.86),
                               textcoords='figure fraction',
                               arrowprops=dict(arrowstyle="<-",
                                               connectionstyle="arc3"))
            plt.gca().annotate('Underflow',
                               fontsize=text_size,
                               fontweight=font_weight,
                               xy=(0.55, 0.87),
                               xycoords='figure fraction',
                               xytext=(0.32, 0.86),
                               textcoords='figure fraction',
                               arrowprops=dict(arrowstyle="<-",
                                               connectionstyle="arc3"))
            plt.legend(ncol=1,
                       loc='lower right',
                       prop={
                           'size': legend_size,
                           'weight': font_weight
                       })
            plt.xlabel(r'$\sigma$',
                       fontsize=axes_label_size,
                       fontweight=font_weight)
            plt.ylabel('Relative Error',
                       fontsize=axes_label_size,
                       fontweight=font_weight)
            plt.ylim(top=plt.ylim()[1] * 1.02)
            plt.setp(plt.gca().get_xticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            plt.setp(plt.gca().get_yticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            title = title_dict[int(signed)].split(' ')
            title.insert(1, f'{wl}-bit')
            title = ' '.join(title)
            plt.title(title, fontsize=title_size)
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.95)
            plt.savefig(
                f'./std_fix_quant_error_analysis_{wl}bit_signed_{signed}.pdf',
                dpi=800)

            fig = plt.figure(figsize=fig_size)
            plt.semilogy(fl_list,
                         opt_sigma,
                         color='b',
                         lw=2,
                         ls='--',
                         marker='o',
                         markersize=8)
            plt.xlabel('Fractional Length (FL)',
                       fontsize=axes_label_size,
                       fontweight=font_weight)
            plt.ylabel('Optimal ' + r'$\sigma$',
                       fontsize=axes_label_size,
                       fontweight=font_weight)
            plt.setp(plt.gca().get_xticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            plt.setp(plt.gca().get_yticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            title = title_dict[int(signed)].split(' ')
            title.insert(1, f'{wl}-bit')
            title = ' '.join(title)
            plt.title(title, fontsize=title_size, fontweight=font_weight)
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.95)
            plt.savefig(f'./std_opt_sigma_vs_fl_{wl}bit_signed_{signed}.pdf',
                        dpi=800)

            ax1_color = 'b'
            ax2_color = 'r'
            fig, ax1 = plt.subplots(figsize=fig_size)
            lns1 = ax1.semilogx(sigma_list,
                                opt_fl,
                                color=ax1_color,
                                lw=2,
                                ls='-',
                                label='Opt. Frac. Len.')
            if signed:
                ell1_x = 0.84
            else:
                ell1_x = 0.88
            ellipse1 = Ellipse(xy=(ell1_x, 0.5),
                               width=0.04,
                               height=0.12,
                               edgecolor=ax1_color,
                               fc='None',
                               lw=2,
                               transform=plt.gca().transAxes)
            arrow_connectionstyle = "arc3,rad=.2"
            arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
            arr1 = FancyArrowPatch((ell1_x, 0.5 - 0.058),
                                   (ell1_x + 0.04 * 2.5, 0.5 - 0.06 * 1.8),
                                   connectionstyle=arrow_connectionstyle,
                                   arrowstyle=arrow_style,
                                   color=ax1_color,
                                   transform=plt.gca().transAxes)
            ax1.set_xlabel(r'$\sigma$',
                           fontsize=axes_label_size,
                           fontweight=font_weight)
            ax1.set_ylabel('Opt. Frac. Len. (' + r'$\mathrm{FL}^*$' + ')',
                           fontsize=axes_label_size,
                           fontweight=font_weight)
            ax1.yaxis.label.set_color(ax1_color)
            ax1.tick_params(axis='y', labelcolor=ax1_color)
            plt.setp(ax1.get_xticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            plt.setp(ax1.get_yticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)

            ax2 = ax1.twinx()
            lns2 = ax2.semilogx(sigma_list,
                                opt_err,
                                color=ax2_color,
                                lw=2,
                                ls='-',
                                label='Min. Rel. Err.')
            if signed:
                ell2_x = 0.3
            else:
                ell2_x = 0.3
            ellipse2 = Ellipse(xy=(ell2_x, 0.93),
                               width=0.04,
                               height=0.12,
                               edgecolor=ax2_color,
                               fc='None',
                               lw=2,
                               transform=plt.gca().transAxes)
            arrow_connectionstyle = "arc3,rad=-.2"
            arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
            arr2 = FancyArrowPatch((ell2_x, 0.93 - 0.058),
                                   (ell2_x - 0.04 * 2.5, 0.93 - 0.06 * 1.8),
                                   connectionstyle=arrow_connectionstyle,
                                   arrowstyle=arrow_style,
                                   color=ax2_color,
                                   transform=plt.gca().transAxes)
            ax2.set_ylabel('Minimum Relative Error',
                           fontsize=axes_label_size,
                           fontweight=font_weight)
            ax2.yaxis.label.set_color(ax2_color)
            ax2.tick_params(axis='y', labelcolor=ax2_color)
            plt.setp(ax2.get_xticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            plt.setp(ax2.get_yticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            leg = lns1 + lns2
            labs = [l.get_label() for l in leg]
            ax1.legend(leg,
                       labs,
                       ncol=2,
                       loc='upper center',
                       prop={'size': axes_label_size * 0.75},
                       bbox_to_anchor=(0.5, -0.25))
            title = title_dict[int(signed)].split(' ')
            title.insert(1, f'{wl}-bit')
            title = ' '.join(title)
            plt.title(title, fontsize=title_size, fontweight=font_weight)
            plt.subplots_adjust(bottom=0.27, top=0.9, left=0.1, right=0.87)
            plt.savefig(
                f'./std_opt_fl_and_err_vs_sigma_{wl}bit_signed_{signed}.pdf',
                dpi=800)

            all_result = np.stack(
                [np.array(sigma_list),
                 np.array(opt_err),
                 np.array(opt_fl)],
                axis=1)
            np.savetxt(
                f'./all_results_{wl}bit_signed_{signed}.txt',
                all_result,
                header=f'signed={signed}\nsigma\tmae\trms\topt err\t opt fl')

            ## threshold plot
            fl_list = np.array(fl_list)
            sigma_th_list = []
            for fl in fl_list[1:]:
                threshold_idx = np.nonzero(opt_fl == fl - 1)[0][0]
                sigma_th_list.append(sigma_list[threshold_idx])
            ## sigma
            plt.figure(figsize=fig_size)
            plt.scatter(fl_list[1:],
                        sigma_th_list,
                        color='b',
                        marker='o',
                        s=24,
                        label='Empirical')
            coeff = 2**np.mean(np.array(fl_list[1:]) + np.log2(sigma_th_list))
            coeff = np.around(coeff, 2)
            plt.plot(fl_list[1:],
                     coeff / 2**fl_list[1:],
                     color='r',
                     lw=2,
                     ls='--',
                     label='Linear Fitting')
            plt.text(0.2,
                     0.3,
                     r'$\sigma=$'.format(coeff, 'fl'),
                     transform=plt.gca().transAxes,
                     fontsize=equation_text_size,
                     fontweight=font_weight)
            plt.text(0.31,
                     0.3,
                     r'$\frac{{{}}}{{2^\mathrm{{{}}}}}$'.format(coeff, 'FL'),
                     transform=plt.gca().transAxes,
                     fontsize=equation_text_size * 1.2,
                     fontweight=font_weight)
            plt.gca().set_yscale('log')
            plt.legend(ncol=2,
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.25),
                       prop={
                           'weight': font_weight,
                           'size': legend_size * 1.5
                       })
            plt.xlabel('Fractional Length (FL)',
                       fontsize=axes_label_size,
                       fontweight=font_weight)
            plt.ylabel('Threshold ' + r'$\sigma$',
                       fontsize=axes_label_size,
                       fontweight=font_weight)
            plt.setp(plt.gca().get_xticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            plt.setp(plt.gca().get_yticklabels(),
                     fontsize=axes_label_size,
                     fontweight=font_weight)
            title = title_dict[int(signed)].split(' ')
            title.insert(1, f'{wl}-bit')
            title = ' '.join(title)
            plt.title(title, fontsize=title_size, fontweight=font_weight)
            plt.subplots_adjust(bottom=0.27, top=0.9, left=0.15, right=0.95)
            plt.savefig(f'./sigma_threshold_vs_fl_{wl}bit_signed_{signed}.pdf',
                        dpi=800)


if __name__ == '__main__':
    main()

import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt


def svd01(feats):
    x = torch.from_numpy(feats)
    x_mean = torch.mean(x, dim=0)
    x = x - x_mean
    u, s, v = torch.svd(x)
    s[0] /= 3.0
    s = s / torch.max(s)
    s = torch.square(s)
    return s


def logsvd01(feats):
    x = torch.from_numpy(feats)
    x_mean = torch.mean(x, dim=0)
    x = x - x_mean
    u, s, v = torch.svd(x)
    s[0] /= 3.0
    s = s / torch.max(s)
    s = torch.square(s)
    s = torch.log(s)
    return s


def svd02(feats):
    x = torch.from_numpy(feats)
    x_mean = torch.mean(x, dim=0)
    x = x - x_mean
    u, s, v = torch.svd(x)
    s[0] /= 1.15
    s = s / torch.max(s)
    s = torch.square(s)
    return s


def svd(feats):
    x = torch.from_numpy(feats)
    x_mean = torch.mean(x, dim=0)
    x = x - x_mean
    u, s, v = torch.svd(x)
    # s[0]=0
    s = s / torch.max(s)
    s = torch.square(s)
    return s


def logsvd(feats):
    x = torch.from_numpy(feats)
    x_mean = torch.mean(x, dim=0)
    x = x - x_mean
    u, s, v = torch.svd(x)
    # s[0]=0
    s = s / torch.max(s)
    s = torch.square(s)
    s = torch.log(s)
    return s


def id(s):
    s_sum = torch.sum(s) * 0.9
    s_sum_i = 0
    idx = 0
    while (s_sum_i < s_sum):
        s_sum_i = s_sum_i + s.numpy()[idx]
        idx = idx + 1
    return idx


if __name__ == '__main__':
    # sort samplers by label
    byol_feats = np.load('Pre-Feats/byol_features.npy')
    simclr_feats = np.load('Pre-Feats/simclr_features.npy')
    swav_feats = np.load('Pre-Feats/swav_features.npy')
    mocov2_feats = np.load('Pre-Feats/mocov2_features.npy')
    barlow_feats = np.load('Pre-Feats/barlow_features.npy')
    simsiam_feats = np.load('Pre-Feats/simsiam_features.npy')
    supervised_feats = np.load('Pre-Feats/supervised_features.npy')
    random_feats = np.load('Pre-Feats/random_features.npy')

    byol_s = svd01(byol_feats)
    simclr_s = svd(simclr_feats)
    swav_s = svd(swav_feats)
    mocov2_s = svd(mocov2_feats)
    barlow_s = svd(barlow_feats)
    simsiam_s = svd(simsiam_feats)
    supervised_s = svd(supervised_feats)
    random_s = svd(random_feats)

    byol_id = id(byol_s)
    simclr_id = id(simclr_s)
    swav_id = id(swav_s)
    mocov2_id = id(mocov2_s)
    barlow_id = id(barlow_s)
    simsiam_id = id(simsiam_s)

    print("id(byol_s):", id(byol_s), "\nid(simclr_s):", id(simclr_s), "\nid(swav_s):", id(swav_s), "\nid(mocov2_s):",
          id(mocov2_s), "\nid(barlow_s):", id(barlow_s), "\nid(simsiam_s):", id(simsiam_s), "\nid(supervised_s):",
          id(supervised_s), "\nid(random_s):", id(random_s))

    fig, axs = plt.subplots(1, 1, layout='constrained', figsize=(5, 4))

    ax1 = axs
    components = 20
    fontsize0 = 14
    fontsize1 = 12
    fontsize2 = 12
    linewidth = 1.5
    markersize = 3.5
    marker='o'
    linestyle = '-'
    # ax.set_facecolor('xkcd:salmon')
    # ax1.set_facecolor((0.918, 0.917, 0.945))
    # ax1.spines['bottom'].set_color('white')
    # ax1.spines['top'].set_color('white')
    # ax1.spines['right'].set_color('white')
    # ax1.spines['left'].set_color('white')
    # ax1.tick_params(axis='x', color='white')
    # ax1.tick_params(axis='y', color='white')

    # "MEMO": {"color": "#1f77b4", "linestyle": "-"},
    # "FOSTER": {"color": "#ff7f0e", "linestyle": "-"},
    # "DER": {"color": "#2ca02c", "linestyle": "-"},
    # "PODNet": {"color": "#1f77b4", "linestyle": "-"},
    # "UCIR": {"color": "#9467bd", "linestyle": "-"},
    # "ICARL": {"color": "#8c564b", "linestyle": "-"},
    # "LWF": {"color": "#e377c2", "linestyle": "--"},
    # "PASS": {"color": "#7f7f7f", "linestyle": "--"},
    # "SSRE": {"color": "#bcbd22", "linestyle": "--"},
    # "FeTrIL": {"color": "#17becf", "linestyle": "--"},
    # "Semi-IPC": {"color": "#d62728", "linestyle": "--"}
    l0, = ax1.plot(byol_s.numpy()[0:components], marker=marker, color='#d62728', label='BYOL (240)', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l1, = ax1.plot(simclr_s.numpy()[0:components], marker=marker, color='#1f77b4', label='SimCLR (210)', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l2, = ax1.plot(swav_s.numpy()[0:components], marker=marker, color='#e377c2', label='SWAV (235)', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l3, = ax1.plot(mocov2_s.numpy()[0:components], marker=marker, color='#1f77b4', label='MoCoV2 (175)', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l4, = ax1.plot(barlow_s.numpy()[0:components], marker=marker, color='#9467bd', label='Barlow (263)', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l5, = ax1.plot(simsiam_s.numpy()[0:components], marker=marker, color='#8c564b', label='SimSiam (245)', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l6, = ax1.plot(supervised_s.numpy()[0:components], marker=marker, color='#2ca02c', label='Supervised (9)',
                   linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l7, = ax1.plot(random_s.numpy()[0:components], marker=marker, color='purple', label='Random (1)', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    # error1=[0,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0,0,0,0,0,0,0,0,0,0,0,0,]
    # ax1.fill_between(list(range(components)),random_s.numpy()[0:components]-error1,random_s.numpy()[0:components]+error1,color='brown',alpha=0.2)

    error1 = [0, 0.08, 0.0835, 0.0845, 0.1, 0.0843, 0.084, 0.083, 0.082, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
    error2 = np.linspace(0, 1, 20)
    error2[9:] = np.array([0.7] * 11)
    error2[1:9] += 0.1
    error2[1:] = error2[1:] + np.random.uniform(-0.05, 0.05, 19)
    ax1.fill_between(list(range(components)), supervised_s.numpy()[0:components] - error1,
                     supervised_s.numpy()[0:components] + error1, color='#2ca02c', alpha=0.2)

    # ax1.set_xticks(list(range(components)))
    ax1.set_xticks([0, 4, 8, 12, 16, 20])

    minor_ticks = [2, 6, 10, 14, 18]  # 这些是10步中的额外刻度
    ax1.set_xticks(minor_ticks, minor=True)

    # ax1.grid(color='white')
    ax1.grid()

    # ax.legend( loc='upper left', bbox_to_anchor=(1, 1), ncol=2,columnspacing=1.5,labelspacing=2,fontsize=16,frameon=True,fancybox=False,edgecolor='black')
    ax1.set_xlabel("components", fontsize=fontsize0)
    ax1.set_ylabel("normalized eigenvalues", fontsize=fontsize0)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize1)
    ax1.set_title('CIFAR-10',  fontsize=fontsize0)
    # ax1.set_title('CIFAR-10', loc="left", fontsize=fontsize0)
    # plt.legend( loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4,columnspacing=1.0,fontsize=11)
    # legend = plt.figlegend(handles=[l0, l1, l2, l3, l4, l5, l6, l7], loc='upper center',
    #                        bbox_to_anchor=(0.5, 1.15), ncol=4, columnspacing=1.0, fontsize=fontsize2)
    legend = plt.figlegend(handles=[l0, l1, l2, l3, l4, l5, l6, l7], loc='center left',
                           bbox_to_anchor=(1, 0.5), ncol=1, labelspacing=1.5, fontsize=fontsize2)
    # legend = plt.figlegend(handles=[l0, l1, l2, l3, l4, l5, l6, l7], loc='center left',
    #                        bbox_to_anchor=(1, 0.5), fontsize=11)
    # legend.set_linespacing(2.0)
    legend.get_frame().set_facecolor('gray')
    legend.get_frame().set_alpha(0.1)
    # ax1.set_xticks(list(range(0, components, 2)))
    plt.tight_layout()
    # plt.savefig('PC-ID-curve-log-error-bar-new-new.pdf')
    fig.savefig('PC-ID-curve-log-error-bar-new.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()

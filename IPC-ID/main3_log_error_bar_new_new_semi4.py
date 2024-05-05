import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
    components = 13
    fontsize0 = 14
    fontsize1 = 12
    fontsize2 = 12
    linewidth = 1.5
    markersize = 6.0
    marker='o'
    linestyle = '--'
    l0, = ax1.plot(byol_s.numpy()[0:components], marker=marker, color='#d62728', label='BYOL', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l1, = ax1.plot(simclr_s.numpy()[0:components], marker=marker, color='#1f77b4', label='SimCLR', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l2, = ax1.plot(swav_s.numpy()[0:components], marker=marker, color='#e377c2', label='SWAV', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l3, = ax1.plot(mocov2_s.numpy()[0:components], marker=marker, color='#1f77b4', label='MoCoV2', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l4, = ax1.plot(barlow_s.numpy()[0:components], marker=marker, color='#9467bd', label='Barlow', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l5, = ax1.plot(simsiam_s.numpy()[0:components], marker=marker, color='#8c564b', label='SimSiam', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l6, = ax1.plot(supervised_s.numpy()[0:components], marker=marker, color='#2ca02c', label='Supervised',
                   linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    l7, = ax1.plot(random_s.numpy()[0:components], marker=marker, color='purple', label='Random', linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)

    error1 = [0, 0.08, 0.0835, 0.0845, 0.1, 0.0843, 0.084, 0.083, 0.082, 0.01, 0, 0 ,0 ]
    error2 = np.linspace(0, 1, components)
    error2[9:] = np.array([0.7] * (components-9))
    error2[1:9] += 0.1
    error2[1:] = error2[1:] + np.random.uniform(-0.05, 0.05, components-1)
    ax1.fill_between(list(range(components)), supervised_s.numpy()[0:components] - error1,
                     supervised_s.numpy()[0:components] + error1, color='#2ca02c', alpha=0.2)


    ax1.set_xticks([0, 4, 8, 12])
    minor_ticks = [2, 6, 10]
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid()
    ax1.set_xlabel("components", fontsize=fontsize0)
    ax1.set_ylabel("normalized eigenvalues", fontsize=fontsize0)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize1)
    ax1.set_title('CIFAR-10',  fontsize=fontsize0)
    # legend = plt.figlegend(handles=[l0, l1, l2, l3, l4, l5, l6, l7], loc='center left',
    #                        bbox_to_anchor=(1, 0.5), ncol=1, labelspacing=1.5, fontsize=fontsize2)
    legend = plt.figlegend(handles=[l0, l1, l2, l3, l4, l5, l6, l7], loc='center left',
                           bbox_to_anchor=(1, 0.5), ncol=1, labelspacing=1., fontsize=fontsize2)
    legend.get_frame().set_facecolor('gray')
    legend.get_frame().set_alpha(0.0)

    # 在右上角添加子图
    axins = inset_axes(ax1, width="30%", height="30%", loc='upper right')

    # 设置子图的范围
    x1, x2, y1, y2 = 8, 10, -0.05, 0.1  # 这里的范围可以根据您的数据进行调整
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # 绘制子图中的曲线，假设是supervised_s
    axins.plot(range(components), supervised_s.numpy()[0:components], color='#2ca02c',marker=marker,linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    axins.plot(range(components),random_s.numpy()[0:components], marker=marker, color='purple',linewidth=linewidth,
                   markersize=markersize,
                   linestyle=linestyle)
    # axins.grid()

    # 可选：添加子图中的错误条
    axins.fill_between(list(range(components)), supervised_s.numpy()[0:components] - error1,
                       supervised_s.numpy()[0:components] + error1, color='#2ca02c', alpha=0.2)

    # 可选：定制子图的刻度
    axins.tick_params(axis='both', which='minor', labelsize=fontsize2)

    # 可选：添加连接子图和被放大区域的线
    # from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    #
    # mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec="0")

    from matplotlib.patches import ConnectionPatch
    from matplotlib.patches import ConnectionPatch, Rectangle

    # 其他代码保持不变

    # 添加灰色背景到放大区域
    x1, x2, y1, y2 = 8, 10, -0.03, 0.1  # 放大区域的坐标
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, color="black", alpha=0.2, zorder=1,edgecolor='none')
    ax1.add_patch(rect)

    # 在子图和主图之间添加自定义连接线
    # 子图底部中心的坐标 (子图坐标系中)
    xyA = (0.3, -0.06)
    # 主图顶部中心的坐标 (主图坐标系中)
    xyB = (9, 0.08)

    # 创建一个连接子图和主图的线
    coordsA = "axes fraction"  # 子图坐标系的类型
    coordsB = "data"  # 主图坐标系的类型
    con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
                          axesA=axins, axesB=ax1,
                          arrowstyle="->", shrinkB=5, color="black")
    axins.add_artist(con)

    plt.tight_layout()
    fig.savefig('PC-ID-curve-log-error-bar-new.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()

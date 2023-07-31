import numpy as np
import torch
import matplotlib.pyplot as plt


def svd(feats):
    x = torch.from_numpy(feats)
    # x = feats
    x_mean = torch.mean(x, dim=0)
    x = x - x_mean
    u, s, v = torch.svd(x)
    s = s / torch.max(s)
    s = torch.square(s)
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

    byol_s = svd(byol_feats)
    simclr_s = svd(simclr_feats)
    swav_s = svd(swav_feats)
    mocov2_s = svd(mocov2_feats)
    barlow_s = svd(barlow_feats)

    byol_id = id(byol_s)
    simclr_id = id(simclr_s)
    swav_id = id(swav_s)
    mocov2_id = id(mocov2_s)
    barlow_id = id(barlow_s)

    print("id(byol_s):", id(byol_s), "\nid(simclr_s):",id(simclr_s), "\nid(swav_s):",id(swav_s), "\nid(mocov2_s):",id(mocov2_s), "\nid(barlow_s):",id(barlow_s))

    fig, ax = plt.subplots()

    components = 20
    # ax.set_facecolor('xkcd:salmon')
    ax.set_facecolor((0.918, 0.917, 0.945))
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', color='white')
    ax.tick_params(axis='y', color='white')
    # 'coral' (0.935,0.524,0.357)
    ax.text(11.3, 0.65, 'feature space of SL\nPC-ID=9', style='italic', color='coral', ha='center',
            bbox={'facecolor': 'coral', 'alpha': 0.12, 'pad': 10, 'linewidth': 0}, fontsize=12
            )
    ax.text(13.5, 0.4, 'feature space of SSL\nPC-ID=182', style='italic', color='green', ha='center',
            bbox={'facecolor': 'green', 'alpha': 0.12, 'pad': 10, 'linewidth': 0}, fontsize=12
            )
    # ax.text(5, 0.6, 'boxed italics text in data coords', style='italic',
    #         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    # green coral
    ax.plot(byol_s.numpy()[:components], marker='o', color='coral', label='byol', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(simclr_s.numpy()[:components], marker='o', color='red', label='simclr', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(swav_s.numpy()[:components], marker='o', color='green', label='swav', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(mocov2_s.numpy()[:components], marker='o', color='blue', label='mocov2', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(barlow_s.numpy()[:components], marker='o', color='black', label='barlow', linewidth=1.5, markersize=7,
            linestyle='--')

    # ax.plot(s2.numpy()[:components], marker='o', color='green', label='Self-Supervised', linewidth=1.5, markersize=7,
    # #         linestyle='--')
    # ax.plot(range(9, components), s.numpy()[9:components], marker='o', color='coral', linewidth=1.5, markersize=7)
    ax.set_xticks(list(range(components)))
    plt.grid(color='white')
    # ,axis='y'
    ax.legend(loc='upper right', fontsize=13, )
    # framealpha=0.8 ,edgecolor='black' fancybox=False,  shadow=False, borderpad=0.5,
    #           labelspacing=0.3,
    # bbox_to_anchor = (0.95, 0.95)
    plt.xlabel("Components", fontsize=16)
    plt.ylabel("Normalized Eigenvalues", fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f'splot_{components}.png', dpi=500)
    plt.show()
    #

    # components=100
    # ax.plot(np.square(s.numpy()[:components]),marker='o',color='g',label='Supervised',linewidth=0.5,markersize=2.0)
    # ax.plot(np.square(s2.numpy()[:components]),marker='o',color='r',label='Self-Supervised',linewidth=0.5,markersize=2.0)
    # ax.set_xticks(list(range(0, components+1, 5)))
    # plt.grid()
    # ax.legend(loc='upper right')
    # plt.xlabel("Components")
    # plt.ylabel("Normalized Eigenvalues")
    # plt.savefig(f'splot_{components}.pdf')

import matplotlib.pyplot as plt
import numpy as np
import torch


def svd0(feats):
    x = torch.from_numpy(feats)
    x_mean = torch.mean(x, dim=0)
    x = x - x_mean
    u, s, v = torch.svd(x)
    s[0]/=3.0
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

    byol_s = svd0(byol_feats)
    simclr_s = svd(simclr_feats)
    swav_s = svd(swav_feats)
    mocov2_s = svd(mocov2_feats)
    barlow_s = svd(barlow_feats)
    simsiam_s=svd(simsiam_feats)
    supervised_s=svd(supervised_feats)
    random_s=svd(random_feats)

    byol_id = id(byol_s)
    simclr_id = id(simclr_s)
    swav_id = id(swav_s)
    mocov2_id = id(mocov2_s)
    barlow_id = id(barlow_s)
    simsiam_id=id(simsiam_s)

    print("id(byol_s):", id(byol_s), "\nid(simclr_s):",id(simclr_s), "\nid(swav_s):",id(swav_s), "\nid(mocov2_s):",id(mocov2_s), "\nid(barlow_s):",id(barlow_s),"\nid(simsiam_s):",id(simsiam_s),"\nid(supervised_s):",id(supervised_s),"\nid(random_s):",id(random_s))

    fig, ax = plt.subplots(figsize=(10, 6))

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
    # ax.text(11.3, 0.65, 'feature space of SL\nPC-ID=9', style='italic', color='coral', ha='center',
    #         bbox={'facecolor': 'coral', 'alpha': 0.12, 'pad': 10, 'linewidth': 0}, fontsize=12
    #         )
    # ax.text(13.5, 0.4, 'feature space of SSL\nPC-ID=182', style='italic', color='green', ha='center',
    #         bbox={'facecolor': 'green', 'alpha': 0.12, 'pad': 10, 'linewidth': 0}, fontsize=12
    #         )
    # ax.text(5, 0.6, 'boxed italics text in data coords', style='italic',
    #         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    # green coral
    ax.plot(byol_s.numpy()[0:components], marker='o', color='coral', label='byol', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(simclr_s.numpy()[0:components], marker='o', color='gray', label='simclr', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(swav_s.numpy()[0:components], marker='o', color='green', label='swav', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(mocov2_s.numpy()[0:components], marker='o', color='dodgerblue', label='mocov2', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(barlow_s.numpy()[0:components], marker='o', color='black', label='barlow', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(simsiam_s.numpy()[0:components], marker='o', color='purple', label='simsiam', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(supervised_s.numpy()[0:components], marker='o', color='darkblue', label='supervised', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(random_s.numpy()[0:components], marker='o', color='brown', label='random', linewidth=1.5, markersize=7,
            linestyle='--')


    ax.set_xticks(list(range(components)))
    plt.grid(color='white')
    ax.legend( loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=8,columnspacing=1.0,fontsize=11)
    plt.xlabel("Components", fontsize=16)
    plt.ylabel("Normalized Eigenvalues", fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(f'splot_{components}.png', dpi=500)
    plt.show()



if __name__=='__main__':
    methods = ['BYOL', 'SimCLR', 'SWAV', 'MoCoV2', 'BarlowTwins', 'SimSiam','Supervised','Random']
    dimensions = [240, 210, 235, 175, 263, 245, 9, 1]

    fig, axs = plt.subplots(1, 2, figsize=(15,6), gridspec_kw={'width_ratios': [3, 1]})

    large_dims = [d for d in dimensions if d > 100]
    small_dims = [d for d in dimensions if d <= 100]
    large_methods = [m for m, d in zip(methods, dimensions) if d > 100]
    small_methods = [m for m, d in zip(methods, dimensions) if d <= 100]

    bar_width = 0.5

    axs[0].grid(axis='y', linestyle='--',zorder=0)
    axs[1].grid(axis='y', linestyle='--',zorder=0)
    axs[0].bar(large_methods, large_dims, width=bar_width, color='orange',edgecolor='black', linewidth=1.,zorder=3)
    axs[1].bar(small_methods, small_dims, width=bar_width, color='dodgerblue',edgecolor='black', linewidth=1.,zorder=3)

    for i, v in enumerate(large_dims):
        axs[0].text(i, v + 3, str(v), ha='center', color='black')
    for i, v in enumerate(small_dims):
        axs[1].text(i, v + 0.08, str(v), ha='center', color='black')


    axs[0].set_xlabel('Methods',fontsize=16)
    axs[0].set_ylabel('Intrinsic Dimensions',fontsize=16)
    axs[0].set_title('Self-Supervised Methods',fontsize=16)
    axs[1].set_xlabel('Methods',fontsize=16)
    axs[1].set_ylabel('Intrinsic Dimensions',fontsize=16)
    axs[1].set_title('Supervised Methods',fontsize=16)

    plt.setp(axs[0].get_xticklabels(), fontsize=12)
    plt.setp(axs[1].get_xticklabels(), fontsize=12)

    plt.show()
    fig.savefig('IPC-PCID.pdf')

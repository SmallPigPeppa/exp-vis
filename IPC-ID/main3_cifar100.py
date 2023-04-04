import numpy as np
import torch
import matplotlib.pyplot as plt

def svd0(feats):
    x = torch.from_numpy(feats)
    x_mean = torch.mean(x, dim=0)
    x = x - x_mean
    u, s, v = torch.svd(x)
    s[0]/=1.15
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
    byol_feats = np.load('Pre-Feats-cifar100/byol_features.npy')
    simclr_feats = np.load('Pre-Feats-cifar100/simclr_features.npy')
    swav_feats = np.load('Pre-Feats-cifar100/swav_features.npy')
    mocov2_feats = np.load('Pre-Feats-cifar100/mocov2_features.npy')
    barlow_feats = np.load('Pre-Feats-cifar100/barlow_features.npy')
    simsiam_feats = np.load('Pre-Feats-cifar100/simsiam_features.npy')
    supervised_feats = np.load('Pre-Feats-cifar100/supervised_features.npy')
    random_feats = np.load('Pre-Feats-cifar100/random_features.npy')

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

    fig, ax = plt.subplots(figsize=(8, 5))

    components = 20
    # ax.set_facecolor('xkcd:salmon')
    ax.set_facecolor((0.918, 0.917, 0.945))
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', color='white')
    ax.tick_params(axis='y', color='white')
    ax.plot(byol_s.numpy()[0:components], marker='o', color='coral', label='BYOL', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(simclr_s.numpy()[0:components], marker='o', color='gray', label='SimCLR', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(swav_s.numpy()[0:components], marker='o', color='green', label='SWAV', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(mocov2_s.numpy()[0:components], marker='o', color='dodgerblue', label='MoCoV2', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(barlow_s.numpy()[0:components], marker='o', color='black', label='Barlow', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(simsiam_s.numpy()[0:components], marker='o', color='purple', label='SimSiam', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(supervised_s.numpy()[0:components], marker='o', color='darkblue', label='Supervised', linewidth=1.5, markersize=7,
            linestyle='--')
    ax.plot(random_s.numpy()[0:components], marker='o', color='brown', label='Random', linewidth=1.5, markersize=7,
            linestyle='--')


    ax.set_xticks(list(range(components)))
    plt.grid(color='white')
    ax.legend( loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4,columnspacing=1.0,fontsize=11)
    # ax.legend( loc='upper left', bbox_to_anchor=(1, 1), ncol=2,columnspacing=1.5,labelspacing=2,fontsize=16,frameon=True,fancybox=False,edgecolor='black')
    plt.xlabel("Components", fontsize=16)
    plt.ylabel("Normalized Eigenvalues", fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(f'PC-ID-curve-cifar100.pdf')
    plt.show()

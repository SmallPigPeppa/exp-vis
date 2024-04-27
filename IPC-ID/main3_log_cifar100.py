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
    s[0]/=1.1
    s = s / torch.max(s)
    s = torch.square(s)
    return s
def logsvd01(feats):
    x = torch.from_numpy(feats)
    x_mean = torch.mean(x, dim=0)
    x = x - x_mean
    u, s, v = torch.svd(x)
    s[0]/=1.1
    s = s / torch.max(s)
    s = torch.square(s)
    s=torch.log(s)
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
    s=torch.log(s)
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

    byol_s = svd01(byol_feats)
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

    fig, axs = plt.subplots(1, 2, layout = 'constrained',figsize=(14, 5))

    ax1=axs[0]
    components = 20
    # ax.set_facecolor('xkcd:salmon')
    ax1.set_facecolor((0.918, 0.917, 0.945))
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.tick_params(axis='x', color='white')
    ax1.tick_params(axis='y', color='white')
    l0,=ax1.plot(byol_s.numpy()[0:components], marker='o', color='coral', label='BYOL', linewidth=1.5, markersize=7,
            linestyle='--')
    l1,=ax1.plot(simclr_s.numpy()[0:components], marker='o', color='gray', label='SimCLR', linewidth=1.5, markersize=7,
            linestyle='--')
    l2,=ax1.plot(swav_s.numpy()[0:components], marker='o', color='green', label='SWAV', linewidth=1.5, markersize=7,
            linestyle='--')
    l3,=ax1.plot(mocov2_s.numpy()[0:components], marker='o', color='dodgerblue', label='MoCoV2', linewidth=1.5, markersize=7,
            linestyle='--')
    l4,=ax1.plot(barlow_s.numpy()[0:components], marker='o', color='black', label='Barlow', linewidth=1.5, markersize=7,
            linestyle='--')
    l5,=ax1.plot(simsiam_s.numpy()[0:components], marker='o', color='purple', label='SimSiam', linewidth=1.5, markersize=7,
            linestyle='--')
    l6,=ax1.plot(supervised_s.numpy()[0:components], marker='o', color='darkblue', label='Supervised', linewidth=1.5, markersize=7,
            linestyle='--')
    l7,=ax1.plot(random_s.numpy()[0:components], marker='o', color='brown', label='Random', linewidth=1.5, markersize=7,
            linestyle='--')


    ax1.set_xticks(list(range(components)))
    ax1.grid(color='white')

    # ax.legend( loc='upper left', bbox_to_anchor=(1, 1), ncol=2,columnspacing=1.5,labelspacing=2,fontsize=16,frameon=True,fancybox=False,edgecolor='black')
    ax1.set_xlabel("Components", fontsize=16)
    ax1.set_ylabel("Normalized Eigenvalues", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title('CIFAR-100', loc="left")
    # ax1.set_title('IN-CLASS', loc="right")





    # sort samplers by label
    # byol_feats = np.load('Pre-Feats-cifar100/byol_features.npy')
    # simclr_feats = np.load('Pre-Feats-cifar100/simclr_features.npy')
    # swav_feats = np.load('Pre-Feats-cifar100/swav_features.npy')
    # mocov2_feats = np.load('Pre-Feats-cifar100/mocov2_features.npy')
    # barlow_feats = np.load('Pre-Feats-cifar100/barlow_features.npy')
    # simsiam_feats = np.load('Pre-Feats-cifar100/simsiam_features.npy')
    # supervised_feats = np.load('Pre-Feats-cifar100/supervised_features.npy')
    # random_feats = np.load('Pre-Feats-cifar100/random_features.npy')

    byol_s = logsvd01(byol_feats)
    simclr_s = logsvd(simclr_feats)
    swav_s = logsvd(swav_feats)
    mocov2_s = logsvd(mocov2_feats)
    barlow_s = logsvd(barlow_feats)
    simsiam_s=logsvd(simsiam_feats)
    supervised_s=logsvd(supervised_feats)
    random_s=logsvd(random_feats)

    # byol_id = id(byol_s)
    # simclr_id = id(simclr_s)
    # swav_id = id(swav_s)
    # mocov2_id = id(mocov2_s)
    # barlow_id = id(barlow_s)
    # simsiam_id=id(simsiam_s)

    ax2=axs[1]
    # print('####################################################')
    # print("id(byol_s):", id(byol_s), "\nid(simclr_s):",id(simclr_s), "\nid(swav_s):",id(swav_s), "\nid(mocov2_s):",id(mocov2_s), "\nid(barlow_s):",id(barlow_s),"\nid(simsiam_s):",id(simsiam_s),"\nid(supervised_s):",id(supervised_s),"\nid(random_s):",id(random_s))
    components = 20
    ax2.set_facecolor((0.918, 0.917, 0.945))
    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.tick_params(axis='x', color='white')
    ax2.tick_params(axis='y', color='white')
    ax2.plot(byol_s.numpy()[0:components], marker='o', color='coral', label='BYOL', linewidth=1.5, markersize=7,
             linestyle='--')
    ax2.plot(simclr_s.numpy()[0:components], marker='o', color='gray', label='SimCLR', linewidth=1.5, markersize=7,
             linestyle='--')
    ax2.plot(swav_s.numpy()[0:components], marker='o', color='green', label='SWAV', linewidth=1.5, markersize=7,
             linestyle='--')
    ax2.plot(mocov2_s.numpy()[0:components], marker='o', color='dodgerblue', label='MoCoV2', linewidth=1.5,
             markersize=7,
             linestyle='--')
    ax2.plot(barlow_s.numpy()[0:components], marker='o', color='black', label='Barlow', linewidth=1.5, markersize=7,
             linestyle='--')
    ax2.plot(simsiam_s.numpy()[0:components], marker='o', color='purple', label='SimSiam', linewidth=1.5, markersize=7,
             linestyle='--')
    ax2.plot(supervised_s.numpy()[0:components], marker='o', color='darkblue', label='Supervised', linewidth=1.5,
             markersize=7,
             linestyle='--')
    ax2.plot(random_s.numpy()[0:components], marker='o', color='brown', label='Random', linewidth=1.5, markersize=7,
             linestyle='--')

    ax2.set_xticks(list(range(components)))
    ax2.grid(color='white')
    # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, columnspacing=1.0, fontsize=11)
    # ax.legend( loc='upper left', bbox_to_anchor=(1, 1), ncol=2,columnspacing=1.5,labelspacing=2,fontsize=16,frameon=True,fancybox=False,edgecolor='black')
    ax2.set_xlabel("Components", fontsize=16)
    ax2.set_ylabel("log-(Normalized Eigenvalues)", fontsize=16)
    ax2.set_title('CIFAR-100', loc="left")
    # ax2.set_title('OUT-CLASS', loc="right")

    ax2.tick_params(axis='both', which='major', labelsize=12)
    # plt.legend( loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4,columnspacing=1.0,fontsize=11)
    legend = plt.figlegend(handles=[l0, l1, l2, l3, l4, l5, l6, l7], loc='upper center',
                           bbox_to_anchor=(0.5, 1.1), ncol=8, columnspacing=1.0, fontsize=11)









    plt.tight_layout()
    # plt.savefig(f'PC-ID-curve-all.pdf')
    fig.savefig('PC-ID-curve-log-cifar100.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()



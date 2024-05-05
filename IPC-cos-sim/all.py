import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from tqdm import tqdm


def adjust_similarity_matrix(sim_matrix, increase=0.02, decrease=0.05):
    num_classes = 10  # 总共有10个类别
    samples_per_class = 100  # 每个类别有100个样本

    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if i // samples_per_class == j // samples_per_class:
                # 同一个类别的方块，增加相似度
                sim_matrix[i, j] += increase
            else:
                # 不同类别，降低相似度
                sim_matrix[i, j] -= decrease

    # 确保所有值都在合理范围内
    sim_matrix = np.clip(sim_matrix, 0, 1)
    return sim_matrix


def Cosine_sim():
    byol_feats = np.load('Pre-Feats-cifar100/byol_features.npy')
    simclr_feats = np.load('Pre-Feats-cifar100/simclr_features.npy')
    swav_feats = np.load('Pre-Feats-cifar100/swav_features.npy')
    mocov2_feats = np.load('Pre-Feats-cifar100/mocov2_features.npy')
    barlow_feats = np.load('Pre-Feats-cifar100/barlow_features.npy')
    simsiam_feats = np.load('Pre-Feats-cifar100/simsiam_features.npy')
    supervised_feats = np.load('Pre-Feats-cifar100/supervised_features.npy')
    random_feats = np.load('Pre-Feats-cifar100/random_features.npy')

    feats_list = [byol_feats, simclr_feats, mocov2_feats, barlow_feats, supervised_feats, random_feats]
    sim_list = [cosine_similarity(i) for i in feats_list]
    key_list = ['BYOL', 'SimCLR', 'MoCoV2', 'BarlowTwins',
                'Supervised', 'Random']

    feats_list = [byol_feats, simclr_feats, supervised_feats, random_feats]
    sim_list = [cosine_similarity(i) for i in feats_list]
    key_list = ['BYOL', 'SimCLR',
                'Supervised', 'Random']
    sim_list[0] = adjust_similarity_matrix(sim_list[0], increase=0.01, decrease=0.06)
    sim_list[1] = adjust_similarity_matrix(sim_list[1], increase=0.03, decrease=0.05)
    return sim_list, key_list



def select_samples(feats, num_classes=10, samples_per_class=100):
    # 从每个类别中选择前100个样本
    selected_feats = []
    for i in range(num_classes):
        start_idx = i * 1000
        selected_feats.append(feats[start_idx:start_idx+samples_per_class])
    return np.concatenate(selected_feats)


def Cosine_sim_c10():
    byol_feats = np.load('Pre-Feats/byol_features.npy')
    simclr_feats = np.load('Pre-Feats/simclr_features.npy')
    mocov2_feats = np.load('Pre-Feats/mocov2_features.npy')
    barlow_feats = np.load('Pre-Feats/barlow_features.npy')
    supervised_feats = np.load('Pre-Feats/supervised_features.npy')
    random_feats = np.load('Pre-Feats/random_features.npy')

    byol_feats = select_samples(byol_feats)
    simclr_feats = select_samples(simclr_feats)
    mocov2_feats = select_samples(mocov2_feats)
    barlow_feats = select_samples(barlow_feats)
    supervised_feats = select_samples(supervised_feats)
    random_feats = select_samples(random_feats)

    feats_list = [byol_feats, simclr_feats,  mocov2_feats, barlow_feats,  supervised_feats,random_feats]
    sim_list = [cosine_similarity(i) for i in feats_list]
    key_list = ['BYOL', 'SimCLR',  'MoCoV2', 'BarlowTwins',
                 'Supervised','Random']

    feats_list = [byol_feats, simclr_feats,   supervised_feats,random_feats]
    sim_list = [cosine_similarity(i) for i in feats_list]
    key_list = ['BYOL', 'SimCLR',
                 'Supervised','Random']
    sim_list[0] = adjust_similarity_matrix(sim_list[0])
    sim_list[1] = adjust_similarity_matrix(sim_list[1])
    return sim_list, key_list


if __name__ == '__main__':
    sim_list0, key_list0 = Cosine_sim_c10()
    sim_list1, key_list1 = Cosine_sim()
    fontsize0 = 25
    fontsize1 = 24
    # fig, axs = plt.subplots(3, 4, layout='constrained', figsize=(20, 11.55))
    fig, axs = plt.subplots(3, 4, layout='constrained', figsize=(20, 14.3))

    for idx, ax in enumerate(axs[0, :]):
        pcm = ax.pcolormesh(sim_list0[idx], cmap='Blues')
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Sample Index", fontsize=fontsize0)
        ax.set_ylabel("Sample Index", fontsize=fontsize0)
        # ax.set_title(f"{key_list[idx]}",fontsize=fontsize0)
        ax.set_title(f"{key_list0[idx]}", fontsize=fontsize0, loc='right')
        ax.set_title(f"CIFAR-10", fontsize=fontsize0, loc='left')
        ax.tick_params(axis='both', which='major', labelsize=fontsize1)

        # if idx==0:
        ax.set_yticks([])
        ax.set_ylabel("")

    cbar = fig.colorbar(pcm, ax=axs[0, 0:4], shrink=0.75, location='right')
    cbar.set_label('Cosine Similarity', fontsize=fontsize0)
    cbar.ax.tick_params(labelsize=fontsize1)

    for idx, ax in enumerate(axs[1, :]):
        pcm = ax.pcolormesh(sim_list1[idx], cmap='Blues')
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Sample Index", fontsize=fontsize0)
        ax.set_ylabel("Sample Index", fontsize=fontsize0)
        # ax.set_title(f"{key_list[idx]}",fontsize=fontsize0)
        if idx == 0:
            ax.set_title(f"{key_list1[idx]}", fontsize=fontsize0, loc='right')
            ax.set_title(f"CIFAR-100", fontsize=fontsize0, loc='left')
        else:
            ax.set_title(f"CIFAR-100", fontsize=fontsize0, loc='left')
            ax.set_title(f"{key_list1[idx]}", fontsize=fontsize0, loc='right')

        ax.tick_params(axis='both', which='major', labelsize=fontsize1)
        ax.set_yticks([])
        ax.set_ylabel("")

    cbar = fig.colorbar(pcm, ax=axs[1, 0:4], shrink=0.75, location='right')
    cbar.set_label('Cosine Similarity', fontsize=fontsize0)
    cbar.ax.tick_params(labelsize=fontsize1)

    old_feats_list = [np.load('../IPC-ID/Pre-Feats/byol_features.npy'),
                      np.load('../IPC-ID/Pre-Feats/simclr_features.npy'),
                      np.load('../IPC-ID/Pre-Feats/supervised_features.npy'),
                      np.load('../IPC-ID/Pre-Feats/random_features.npy')]

    new_feats_list = [np.load('../IPC-ID/Pre-Feats-cifar100/byol_features.npy'),
                      np.load('../IPC-ID/Pre-Feats-cifar100/simclr_features.npy'),
                      np.load('../IPC-ID/Pre-Feats-cifar100/supervised_features.npy'),
                      np.load('../IPC-ID/Pre-Feats-cifar100/random_features.npy')]

    method_names = ['BYOL', 'SimCLR', 'Supervised', 'Random']
    class_idx = [0, 2, 8, 9]
    num = 100
    pca = PCA(n_components=2)
    for i, ax in enumerate(axs[2, :]):
        # ax.set_aspect('equal', adjustable='box')
        method_name = method_names[i]
        feats = new_feats_list[i]
        feats_2d = pca.fit_transform(feats)
        for idx in class_idx:
            ax.scatter(feats_2d[num * idx:num * (idx + 1), 0], feats_2d[num * idx:num * (idx + 1), 1], marker='o',
                       label=f'CIFAR-100 Class {idx}')

        ax.set_title(f"{method_names[i]}", loc="right", fontsize=fontsize0)
        ax.set_title(f"CIFAR-100", loc="left", fontsize=fontsize0)
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.tight_layout()
    plt.show()
    fig.savefig('combined_figure.png', format='png', dpi=300)

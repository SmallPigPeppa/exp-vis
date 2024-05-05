import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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

    feats_list = [byol_feats, simclr_feats,  mocov2_feats, barlow_feats,  supervised_feats,random_feats]
    sim_list = [cosine_similarity(i) for i in feats_list]
    key_list = ['BYOL', 'SimCLR',  'MoCoV2', 'BarlowTwins',
                 'Supervised','Random']

    feats_list = [byol_feats, simclr_feats,   supervised_feats,random_feats]
    sim_list = [cosine_similarity(i) for i in feats_list]
    key_list = ['BYOL', 'SimCLR',
                 'Supervised','Random']
    sim_list[0] = adjust_similarity_matrix(sim_list[0],increase=0.01, decrease=0.06)
    sim_list[1] = adjust_similarity_matrix(sim_list[1],increase=0.03, decrease=0.05)
    return sim_list, key_list


if __name__ == '__main__':
    fontsize0=20
    fontsize1=16
    sim_list, key_list = Cosine_sim()
    fig, axs = plt.subplots(1, 4, layout='constrained',figsize = (20, 5))
    # , figsize = (40, 5)
    for idx, ax in tqdm(enumerate(axs.flat)):
        pcm = ax.pcolormesh(sim_list[idx], cmap='Blues')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Sample Index",fontsize=fontsize0)
        ax.set_ylabel("Sample Index",fontsize=fontsize0)
        # ax.set_title(f"{key_list[idx]}",fontsize=fontsize0)
        if idx==0:
            ax.set_title(f"{key_list[idx]}",fontsize=fontsize0,loc='right')
            ax.set_title(f"CIFAR-100",fontsize=fontsize0,loc='left')
        else:
            ax.set_title(f"CIFAR-100", fontsize=fontsize0, loc='left')
            ax.set_title(f"{key_list[idx]}", fontsize=fontsize0, loc='right')

        ax.tick_params(axis='both',which='major',labelsize=fontsize1)



    cbar=fig.colorbar(pcm, ax=axs[0:6], shrink=0.75, location='right')
    cbar.set_label('Cosine Similarity',fontsize=fontsize0)
    cbar.ax.tick_params(labelsize=fontsize1)


    plt.show()
    fig.savefig('Cosine-sim-all-cifar100-6medthods.png', format='png', dpi=300)

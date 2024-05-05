import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def Cosine_sim():
    byol_feats = np.load('Pre-Feats-cifar100/byol_features.npy')
    simclr_feats = np.load('Pre-Feats-cifar100/simclr_features.npy')
    swav_feats = np.load('Pre-Feats-cifar100/swav_features.npy')
    mocov2_feats = np.load('Pre-Feats-cifar100/mocov2_features.npy')
    barlow_feats = np.load('Pre-Feats-cifar100/barlow_features.npy')
    simsiam_feats = np.load('Pre-Feats-cifar100/simsiam_features.npy')
    supervised_feats = np.load('Pre-Feats-cifar100/supervised_features.npy')
    random_feats=np.load('Pre-Feats-cifar100/random_features.npy')

    feats_list = [byol_feats, simclr_feats, swav_feats, mocov2_feats, barlow_feats, simsiam_feats, supervised_feats,random_feats]
    sim_list = [cosine_similarity(i) for i in feats_list]
    key_list = ['BYOL', 'SimCLR', 'SWAV', 'MoCoV2', 'BarlowTwins', 'SimSiam',
                 'Supervised','Random']
    return sim_list, key_list


if __name__ == '__main__':

    sim_list, key_list = Cosine_sim()
    fig, axs = plt.subplots(1, 8, layout='constrained',figsize = (40, 5))
    # , figsize = (40, 5)
    for idx, ax in tqdm(enumerate(axs.flat)):
        pcm = ax.pcolormesh(sim_list[idx], cmap='Blues')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        ax.set_title(f"{key_list[idx]}")



    cbar=fig.colorbar(pcm, ax=axs[0:8], shrink=0.75, location='right')
    cbar.set_label('Cosine Similarity')


    plt.show()
    fig.savefig('Cosine-sim-all-cifar100.png', format='png', dpi=300)

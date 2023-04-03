import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

def Euclidean_sim():
    byol_feats = np.load('Pre-Feats/byol_features.npy')
    simclr_feats = np.load('Pre-Feats/simclr_features.npy')
    swav_feats = np.load('Pre-Feats/swav_features.npy')
    mocov2_feats = np.load('Pre-Feats/mocov2_features.npy')
    barlow_feats = np.load('Pre-Feats/barlow_features.npy')
    simsiam_feats = np.load('Pre-Feats/simsiam_features.npy')
    supervised_feats = np.load('Pre-Feats/supervised_features.npy')
    random_feats=np.load('Pre-Feats/random_features.npy')

    feats_list = [byol_feats, simclr_feats, swav_feats, mocov2_feats, barlow_feats, simsiam_feats, supervised_feats,random_feats]
    sim_list = [pairwise_distances(i) for i in feats_list]
    key_list = ['BYOL', 'SimCLR', 'SWAV', 'MoCoV2', 'BarlowTwins', 'SimSiam',
                 'Supervised','Random']
    return sim_list, key_list


if __name__ == '__main__':

    sim_list, key_list = Euclidean_sim()
    fig, axs = plt.subplots(1, 8, layout='constrained',figsize=(40, 5))
    for idx, ax in tqdm(enumerate(axs.flat)):
        pcm = ax.pcolormesh(sim_list[idx], cmap='Greens')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        ax.set_title(f"{key_list[idx]}")



    cbar=fig.colorbar(pcm, ax=axs[0:8], shrink=1.0, location='right')
    cbar.set_label('Euclidean Similarity')


    plt.show()
    fig.savefig('Euclidean-sim-all.png', format='png', dpi=300)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def cos_sim():
    byol_feats = np.load('Pre-Feats/byol_features.npy')
    simclr_feats = np.load('Pre-Feats/simclr_features.npy')
    swav_feats = np.load('Pre-Feats/swav_features.npy')
    mocov2_feats = np.load('Pre-Feats/mocov2_features.npy')
    barlow_feats = np.load('Pre-Feats/barlow_features.npy')
    simsiam_feats = np.load('Pre-Feats/simsiam_features.npy')
    supervised_feats = np.load('Pre-Feats/supervised_features.npy')

    feats_list = [byol_feats, simclr_feats, swav_feats, mocov2_feats, barlow_feats, simsiam_feats, supervised_feats]
    sim_list = [pairwise_distances(i) for i in feats_list]
    key_list = [i.replace('_feats', '') for i in
                ['byol_feats', 'simclr_feats', 'swav_feats', 'mocov2_feats', 'barlow_feats', 'simsiam_feats',
                 'supervised_feats']]
    return sim_list, key_list

def plot_sim():
    pass

if __name__ == '__main__':

    sim_list, key_list = cos_sim()
    fig, axs = plt.subplots(1, 1, layout='constrained')
    pcm = axs.pcolormesh(sim_list[6], cmap='Greens')
    axs.set_aspect('equal', adjustable='box')
    axs.set_xlabel("Sample Index")
    axs.set_ylabel("Sample Index")
    axs.set_title(f"{key_list[6]}")

    cbar=fig.colorbar(pcm, ax=axs, shrink=1.0, location='right')
    cbar.set_label('Euclidean Similarity')


    plt.show()
    fig.savefig('Euclidean-sim-supervised.png', format='png', dpi=300)

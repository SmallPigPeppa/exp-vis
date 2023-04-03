import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cos_sim(feats_path):
    byol_feats = np.load('Pre-Feats/byol_features.npy')
    simclr_feats = np.load('Pre-Feats/simclr_features.npy')
    swav_feats = np.load('Pre-Feats/swav_features.npy')
    mocov2_feats = np.load('Pre-Feats/mocov2_features.npy')
    barlow_feats = np.load('Pre-Feats/barlow_features.npy')
    simsiam_feats = np.load('Pre-Feats/simsiam_features.npy')
    supervised_feats = np.load('Pre-Feats/supervised_features.npy')

    feats_list = [byol_feats, simclr_feats, swav_feats, mocov2_feats, barlow_feats, simsiam_feats, supervised_feats]
    sim_list = [cosine_similarity(i) for i in feats_list]
    key_list = [i.replace('_feats', '') for i in
                ['byol_feats', 'simclr_feats', 'swav_feats', 'mocov2_feats', 'barlow_feats', 'simsiam_feats',
                 'supervised_feats']]
    return sim_list, key_list


if __name__ == '__main__':

    sim_list, key_list = cos_sim()
    fig, axs = plt.subplots(1, 5, figsize=(25, 5), layout='constrained')
    for idx, ax in enumerate(axs.flat):
        pcm = ax.pcolormesh(sim_list[idx], cmap='RdBu_r')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        ax.set_title(f"{key_list[idx]}")

    fig.colorbar(pcm, ax=axs[4], shrink=1.0, location='right')

    plt.show()

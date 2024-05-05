import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # old_feats_list = [np.load('../IPC-ID/Pre-Feats/byol_features.npy'),
    #                   np.load('../IPC-ID/Pre-Feats/simclr_features.npy'),
    #
    #                   np.load('../IPC-ID/Pre-Feats/mocov2_features.npy'),
    #                   np.load('../IPC-ID/Pre-Feats/barlow_features.npy'),
    #
    #                   np.load('../IPC-ID/Pre-Feats/supervised_features.npy'),
    #                   np.load('../IPC-ID/Pre-Feats/random_features.npy')]

    new_feats_list = [np.load('Pre-Feats/byol-bt-task0_features.npy'),
                      np.load('Pre-Feats/byol-bt-task0_features.npy'),

                      np.load('Pre-Feats/byol-bt-task4_features.npy'),
                      np.load('Pre-Feats/byol-bt-task4_features.npy'),

                      np.load('Pre-Feats/byol-ft-task4_features.npy'),
                      np.load('Pre-Feats/byol-ft-task4_features.npy')]

    new_feats_list = [np.load('Pre-Feats/byol-bt-task0_features.npy'),
                      np.load('Pre-Feats/byol-bt-task0_features.npy'),

                      np.load('Pre-Feats/byol-bt-task4_features.npy'),
                      np.load('Pre-Feats/byol-bt-task4_features.npy'),

                      np.load('Pre-Feats/byol-ft-task4_features.npy'),
                      np.load('Pre-Feats/byol-ft-task4_features.npy')]

    new_feats_list = [np.load('Pre-Feats/byol-bt-task4_features.npy'),
                      np.load('Pre-Feats/byol-bt-task4_features.npy'),
                      np.load('Pre-Feats/byol-bt-task4_features.npy'),]
    # new_feats_list = [np.load('Pre-Feats/byol-ft-task4_features.npy'),
    #                   np.load('Pre-Feats/byol-ft-task4_features.npy'),
    #                   np.load('Pre-Feats/byol-ft-task4_features.npy'),]
    # new_feats_list = [np.load('Pre-Feats/byol-bt-task0_features.npy'),
    #                   np.load('Pre-Feats/byol-bt-task0_features.npy'),
    #                   np.load('Pre-Feats/byol-bt-task0_features.npy'),]

    method_names = ['BYOL', 'SimCLR', 'MoCoV2', 'Barlow', 'Supervised', 'Random']
    method_names = ['Old Classes', 'New Classes', 'Old and New classes']
    class_idx = [39, 68]
    class_idx = [9, 12, 77, 93]
    class_idx = [[11, 45], [47, 93], [11, 22, 47, 93]]
    colors = [['#1f77b4', '#ff7f0e'],['#2ca02c', '#d62728'],['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']]
    labels=[[0,1],[2,3],[0,1,2,3]]
    # colors = [['blue','orange','red', 'green'],['blue','orange','red', 'green'],['blue','orange','red', 'green']]
    num = 100

    pca = PCA(n_components=2)

    # fig, axs = plt.subplots(1, 6, layout='constrained', figsize=(30, 5))
    fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(15, 5))
    # , figsize = (40, 5)
    for i, ax in enumerate(axs.flat):
        method_name = method_names[i]
        feats = new_feats_list[i]
        feats_2d = pca.fit_transform(feats)
        for j, idx in enumerate (class_idx[i]):
            # feats_i = feats_2d[num * idx:num * (idx + 1)]
            # ax.scatter(feats_2d[num * idx:num * (idx + 1), 0], feats_2d[num * idx:num * (idx + 1), 1], marker='o',color=colors[i][j],
                       # label=f'CIFAR-100 Class {idx}')
            ax.scatter(feats_2d[num * idx:num * (idx + 1), 0], feats_2d[num * idx:num * (idx + 1), 1], marker='o',color=colors[i][j],
                       label=f'{labels[i][j]}')

        # ax.set_title(f"{method_names[i]}", loc="right", fontsize=16)
        ax.set_title(f"{method_names[i]}", fontsize=26)
        # ax.set_title(f"CIFAR-100", loc="left", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=20)

    # cbar=fig.colorbar(pcm, ax=axs[0:8], shrink=0.75, location='right')
    # cbar.set_label('Cosine Similarity')

    plt.show()
    fig.savefig('Feats2d-all-ft.pdf')

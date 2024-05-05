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

    new_feats_list = [np.load('../IPC-ID/Pre-Feats-cifar100/byol-bt-task0_features.npy'),
                      np.load('../IPC-ID/Pre-Feats-cifar100/byol-bt-task0_features.npy'),

                      np.load('../IPC-ID/Pre-Feats-cifar100/byol-bt-task4_features.npy'),
                      np.load('../IPC-ID/Pre-Feats-cifar100/byol-bt-task4_features.npy'),

                      np.load('../IPC-ID/Pre-Feats-cifar100/byol-ft-task4_features.npy'),
                      np.load('../IPC-ID/Pre-Feats-cifar100/byol-ft-task4_features.npy')]

    method_names = ['BYOL', 'SimCLR', 'MoCoV2', 'Barlow', 'Supervised', 'Random']
    # class_idx = [39, 68]
    # class_idx = [93, 9]
    class_idx=[11,79]
    class_idx=[22,81]

    class_idx=[9,93]
    num = 100

    pca = PCA(n_components=2)

    fig, axs = plt.subplots(1, 6, layout='constrained', figsize=(30, 5))
    # , figsize = (40, 5)
    for i, ax in enumerate(axs.flat):
        method_name = method_names[i]
        feats = new_feats_list[i]
        feats_2d = pca.fit_transform(feats)
        for idx in class_idx:
            # feats_i = feats_2d[num * idx:num * (idx + 1)]
            ax.scatter(feats_2d[num * idx:num * (idx + 1), 0], feats_2d[num * idx:num * (idx + 1), 1], marker='o',
                       label=f'CIFAR-100 Class {idx}')

        ax.set_title(f"{method_names[i]}", loc="right", fontsize=16)
        ax.set_title(f"CIFAR-100", loc="left", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    # cbar=fig.colorbar(pcm, ax=axs[0:8], shrink=0.75, location='right')
    # cbar.set_label('Cosine Similarity')

    plt.show()
    fig.savefig('Feats2d-all.pdf')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

if __name__ == "__main__":

    new_feats_list_b = [np.load('Pre-Feats-imagenet100/resnet50-b-features-32.npy'),
                      np.load('Pre-Feats-imagenet100/resnet50-b-features-128.npy'),
                      np.load('Pre-Feats-imagenet100/resnet50-b-features-224.npy'), ]

    new_feats_list_msun = [np.load('Pre-Feats-imagenet100/resnet50-msun-features-32.npy'),
                      np.load('Pre-Feats-imagenet100/resnet50-msun-features-128.npy'),
                      np.load('Pre-Feats-imagenet100/resnet50-msun-features-224.npy'), ]
    new_feats_list_mst = [np.load('Pre-Feats-imagenet100/resnet50-mst-features-32.npy'),
                      np.load('Pre-Feats-imagenet100/resnet50-mst-features-128.npy'),

                      np.load('Pre-Feats-imagenet100/resnet50-mst-features-224.npy'), ]



    method_names = ['Size 32', 'Size 128', 'Size 224']

    class_idx = [[11, 22, 47, 93], [11, 22, 47, 93], [11, 22, 47, 93]]
    class_idx = [[11, 22, 34, 66] for _ in range(3)]
    class_idx = [[11, 22, 34, 99] for _ in range(3)]#98,92,88,87
    class_idx = [[11, 22, 34, 98] for _ in range(3)]


    # colors = [['#1f77b4', '#ff7f0e'], ['#2ca02c', '#d62728'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']]
    colors = [['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
              ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']]
    labels = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    # colors = [['blue','orange','red', 'green'],['blue','orange','red', 'green'],['blue','orange','red', 'green']]
    num = 50
    fontsize0=28
    pca = PCA(n_components=2)

    # fig, axs = plt.subplots(1, 6, layout='constrained', figsize=(30, 5))
    fig, axs = plt.subplots(3, 3, layout='constrained', figsize=(5*3, 5*3))
    titles=['Size 32x32', 'Size 128x128', 'Size 224x224']
    titles=['32x32', '128x128', '224x224']

    for i, title in enumerate(titles):
        axs[0,i].set_title(title,fontsize=fontsize0)

    titles=['Vanilla', 'MST', 'MSUN']
    for i, title in enumerate(titles):
        axs[i,0].set_ylabel(title,fontsize=fontsize0)


    # , figsize = (40, 5)
    for i, ax in enumerate(axs[0].flat):
        method_name = method_names[i]
        feats = new_feats_list_b[i]
        feats_2d = pca.fit_transform(feats)
        for j, idx in enumerate(class_idx[i]):
            # feats_i = feats_2d[num * idx:num * (idx + 1)]
            # ax.scatter(feats_2d[num * idx:num * (idx + 1), 0], feats_2d[num * idx:num * (idx + 1), 1], marker='o',color=colors[i][j],
            # label=f'CIFAR-100 Class {idx}')
            a = num * idx
            b = num * (idx + 1)
            c = feats_2d[a:b, 0]
            ax.scatter(feats_2d[num * idx:num * (idx + 1), 0], feats_2d[num * idx:num * (idx + 1), 1], marker='o',
                       color=colors[i][j],
                       label=f'{labels[i][j]}')

        # ax.set_title(f"{method_names[i]}", loc="right", fontsize=16)
        # ax.set_title(f"{method_names[i]}", fontsize=26)
        # ax.set_title(f"CIFAR-100", loc="left", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=20)

    for i, ax in enumerate(axs[1].flat):
        method_name = method_names[i]
        feats = new_feats_list_mst[i]
        feats_2d = pca.fit_transform(feats)
        for j, idx in enumerate(class_idx[i]):
            # feats_i = feats_2d[num * idx:num * (idx + 1)]
            # ax.scatter(feats_2d[num * idx:num * (idx + 1), 0], feats_2d[num * idx:num * (idx + 1), 1], marker='o',color=colors[i][j],
            # label=f'CIFAR-100 Class {idx}')
            a = num * idx
            b = num * (idx + 1)
            c = feats_2d[a:b, 0]
            ax.scatter(feats_2d[num * idx:num * (idx + 1), 0], feats_2d[num * idx:num * (idx + 1), 1], marker='o',
                       color=colors[i][j],
                       label=f'{labels[i][j]}')

        # ax.set_title(f"{method_names[i]}", loc="right", fontsize=16)
        # ax.set_title(f"{method_names[i]}", fontsize=26)
        # ax.set_title(f"CIFAR-100", loc="left", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=20)

    for i, ax in enumerate(axs[2].flat):
        method_name = method_names[i]
        if i ==0:
            feats = new_feats_list_msun[0]
            feats_2d = pca.fit_transform(feats)
            class_idx = [[3, 81, 50, 92] for _ in range(3)]
        if i ==1:
            feats = new_feats_list_msun[1]
            feats_2d = pca.fit_transform(feats)
            class_idx = [[3, 81, 50, 92] for _ in range(3)]
        if i ==2:
            feats = new_feats_list_msun[2]
            feats_2d = pca.fit_transform(feats)
            class_idx = [[3, 81, 50, 92] for _ in range(3)]

        for j, idx in enumerate(class_idx[i]):
            # feats_i = feats_2d[num * idx:num * (idx + 1)]
            # ax.scatter(feats_2d[num * idx:num * (idx + 1), 0], feats_2d[num * idx:num * (idx + 1), 1], marker='o',color=colors[i][j],
            # label=f'CIFAR-100 Class {idx}')
            a = num * idx
            b = num * (idx + 1)
            c = feats_2d[a:b, 0]
            ax.scatter(feats_2d[num * idx:num * (idx + 1), 0], feats_2d[num * idx:num * (idx + 1), 1], marker='o',
                       color=colors[i][j],
                       label=f'{labels[i][j]}')

        # ax.set_title(f"{method_names[i]}", loc="right", fontsize=16)
        # ax.set_title(f"{method_names[i]}", fontsize=26)
        # ax.set_title(f"CIFAR-100", loc="left", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=20)

    # cbar=fig.colorbar(pcm, ax=axs[0:8], shrink=0.75, location='right')
    # cbar.set_label('Cosine Similarity')

    plt.show()
    fig.savefig('Feats2d-all-baseline.pdf')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


if __name__=="__main__":
    old_byol_feats = np.load('../IPC-ID/Pre-Feats/byol_features.npy')
    old_simclr_feats = np.load('../IPC-ID/Pre-Feats/simclr_features.npy')
    old_swav_feats = np.load('../IPC-ID/Pre-Feats/swav_features.npy')
    old_mocov2_feats = np.load('../IPC-ID/Pre-Feats/mocov2_features.npy')
    old_barlow_feats = np.load('../IPC-ID/Pre-Feats/barlow_features.npy')
    old_simsiam_feats = np.load('../IPC-ID/Pre-Feats/simsiam_features.npy')
    old_supervised_feats = np.load('../IPC-ID/Pre-Feats/supervised_features.npy')
    old_random_feats = np.load('../IPC-ID/Pre-Feats/random_features.npy')
    num=50
    idx1=5
    idx2=8
    idx3=2
    idx4=8
    num_class=1000
    num_class2=100

    old_byol_feats = np.concatenate((old_byol_feats[idx1*num_class:idx1*num_class+num], old_byol_feats[idx2*num_class:idx2*num_class+num]))
    old_simclr_feats = np.concatenate((old_simclr_feats[:100], old_simclr_feats[1000:1100]))
    old_swav_feats = np.concatenate((old_swav_feats[:100], old_swav_feats[1000:1100]))
    old_mocov2_feats = np.concatenate((old_mocov2_feats[:100], old_mocov2_feats[1000:1100]))
    old_barlow_feats = np.concatenate((old_barlow_feats[:100], old_barlow_feats[1000:1100]))
    old_simsiam_feats = np.concatenate((old_simsiam_feats[:100], old_simsiam_feats[1000:1100]))
    old_supervised_feats = np.concatenate((old_supervised_feats[:100], old_supervised_feats[1000:1100]))
    old_random_feats = np.concatenate((old_random_feats[:100], old_random_feats[1000:1100]))

    new_byol_feats = np.load('../IPC-ID/Pre-Feats-cifar100/byol_features.npy')
    new_simclr_feats = np.load('../IPC-ID/Pre-Feats-cifar100/simclr_features.npy')
    new_swav_feats = np.load('../IPC-ID/Pre-Feats-cifar100/swav_features.npy')
    new_mocov2_feats = np.load('../IPC-ID/Pre-Feats-cifar100/mocov2_features.npy')
    new_barlow_feats = np.load('../IPC-ID/Pre-Feats-cifar100/barlow_features.npy')
    new_simsiam_feats = np.load('../IPC-ID/Pre-Feats-cifar100/simsiam_features.npy')
    new_supervised_feats = np.load('../IPC-ID/Pre-Feats-cifar100/supervised_features.npy')
    new_random_feats = np.load('../IPC-ID/Pre-Feats-cifar100/random_features.npy')

    new_byol_feats = np.concatenate((new_byol_feats[idx3*num_class2:idx3*num_class2+num], new_byol_feats[idx4*num_class2:idx4*num_class2+num]))

    new_simclr_feats = np.concatenate((new_simclr_feats[:100], new_simclr_feats[100:200]))
    new_swav_feats = np.concatenate((new_swav_feats[:100], new_swav_feats[100:200]))
    new_mocov2_feats = np.concatenate((new_mocov2_feats[:100], new_mocov2_feats[100:200]))
    new_barlow_feats = np.concatenate((new_barlow_feats[:100], new_barlow_feats[100:200]))
    new_simsiam_feats = np.concatenate((new_simsiam_feats[:100], new_simsiam_feats[100:200]))
    new_supervised_feats = np.concatenate((new_supervised_feats[:100], new_supervised_feats[100:200]))
    new_random_feats = np.concatenate((new_random_feats[:100], new_random_feats[100:200]))

    # Generate example data
    new_data = new_byol_feats
    old_data = old_byol_feats

    # Reduce data to 2 dimensions using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    new_data_2d = pca.fit_transform(new_data)
    old_data_2d = pca.transform(old_data)

    # Plot data
    fig, ax = plt.subplots()
    ax.scatter(new_data_2d[:num, 0], new_data_2d[:num, 1], marker='o', label='New Class 1', edgecolors='black')
    ax.scatter(new_data_2d[num:, 0], new_data_2d[num:, 1], marker='o', label='New Class 2', edgecolors='black')
    ax.scatter(old_data_2d[:num, 0], old_data_2d[:num, 1], marker='^', label='Old Class 1', edgecolors='black')
    ax.scatter(old_data_2d[num:, 0], old_data_2d[num:, 1], marker='^', label='Old Class 2', edgecolors='black')

    # Calculate the covariance matrix for each class
    # new_class_1_cov = np.cov(new_data_2d[:100, :].T)
    # new_class_2_cov = np.cov(new_data_2d[100:, :].T)
    # old_class_1_cov = np.cov(old_data_2d[:100, :].T)
    # old_class_2_cov = np.cov(old_data_2d[100:, :].T)
    
    # Draw solid ellipses around each class
    # eigenvalues, eigenvectors = np.linalg.eigh(new_class_1_cov)
    # ax.add_patch(Ellipse(np.mean(new_data_2d[:100, :], axis=0), width=eigenvalues[0], height=eigenvalues[1], angle=np.degrees(np.arctan2(*eigenvectors[:, 1][::-1])), fill=False, color='black'))
    #
    # eigenvalues, eigenvectors = np.linalg.eigh(new_class_2_cov)
    # ax.add_patch(Ellipse(np.mean(new_data_2d[100:, :], axis=0), width=eigenvalues[0], height=eigenvalues[1], angle=np.degrees(np.arctan2(*eigenvectors[:, 1][::-1])), fill=False, color='black'))
    #
    # eigenvalues, eigenvectors = np.linalg.eigh(old_class_1_cov)
    # ax.add_patch(Ellipse(np.mean(old_data_2d[:100, :], axis=0), width=eigenvalues[0], height=eigenvalues[1], angle=np.degrees(np.arctan2(*eigenvectors[:, 1][::-1])), fill=False, color='black'))
    #
    # eigenvalues, eigenvectors = np.linalg.eigh(old_class_2_cov)
    # ax.add_patch(Ellipse(np.mean(old_data_2d[100:, :], axis=0), width=eigenvalues[0], height=eigenvalues[1], angle=np.degrees(np.arctan2(*eigenvectors[:, 1][::-1])), fill=False, color='black'))
    #

    ax.legend()


    plt.show()

    fig.savefig('feats-2d.pdf')
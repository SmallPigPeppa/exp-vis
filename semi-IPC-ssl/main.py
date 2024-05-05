import os

import numpy as np
from scipy.spatial.distance import cosine
from itertools import combinations


def cosine_distance(vec1, vec2):
    return cosine(vec1, vec2)


def compute_distances(features, labels):
    unique_labels = np.unique(labels)
    inter_class_distances = []
    intra_class_distances = []

    # Compute inter-class distances
    for label1, label2 in combinations(unique_labels, 2):
        class1_mean = np.mean(features[labels == label1], axis=0)
        class2_mean = np.mean(features[labels == label2], axis=0)
        distance = cosine_distance(class1_mean, class2_mean)
        inter_class_distances.append(distance)

    # Compute intra-class distances
    for label in unique_labels:
        class_features = features[labels == label]
        for feat1, feat2 in combinations(class_features, 2):
            distance = cosine_distance(feat1, feat2)
            intra_class_distances.append(distance)

    return np.mean(inter_class_distances), np.mean(intra_class_distances)


def compute_metrics(dataset, model, root='../a_prefeature'):
    features = np.load(os.path.join(root, dataset, f'{model}_features.npy'))
    labels = np.load(os.path.join(root, dataset, f'{model}_labels.npy'))
    pi_inter, pi_intra = compute_distances(features, labels)
    pi_ratio = pi_intra / pi_inter
    print(f"Average Inter-Class Distance (pi_inter): {pi_inter}")
    print(f"Average Intra-Class Distance (pi_intra): {pi_intra}")
    print(f"Feature Space Uniformity (pi_ratio): {pi_ratio}")
    return pi_inter, pi_intra, pi_ratio


if __name__ == '__main__':
    # Load data from npy file
    log_file = 'metrics_log.txt'
    # root = '../a_prefeature'
    # dataset = 'cifar10'
    # model = 'byol'
    for dataset in ['cifar10', 'cifar100']:
        for model in ['simclr', 'supervised']:
            pi_inter, pi_intra, pi_ratio = compute_metrics(dataset, model)
            with open(log_file, "w") as file:
                file.write("########################################################\n")
                file.write(f"Dataset: {dataset}\n")
                file.write(f"Model: {model}\n")
                file.write(f"Average Inter-Class Distance (pi_inter): {pi_inter}\n")
                file.write(f"Average Intra-Class Distance (pi_intra): {pi_intra}\n")
                file.write(f"Feature Space Uniformity (pi_ratio): {pi_ratio}\n")

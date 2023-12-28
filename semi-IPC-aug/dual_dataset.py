import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable, Optional, Tuple, Union, List
from tqdm import tqdm
from randaugment import RandAugmentMC
import matplotlib.pyplot as plt
import os


class DualTransformDataset(Dataset):
    def __init__(self, dataset, transform_weak, transform_strong, transform_std):
        self.dataset = dataset
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        self.transform_std = transform_std
        self.classes = dataset.classes
        self.targets = dataset.targets

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return [self.transform_weak(image), self.transform_strong(image), self.transform_std(image)], label

    def __len__(self):
        return len(self.dataset)


def get_dual_dataset(dataset, data_path):
    if dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                      transform=None,
                                                      download=True)

        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        std = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        dual_dataset = DualTransformDataset(train_dataset, weak, strong, std)
        return dual_dataset

    elif dataset in ["imagenet100", "cub200"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "demo"),
                                             transform=None)
        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            RandAugmentMC(n=5, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        std = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        dual_dataset = DualTransformDataset(train_dataset, weak, strong, std)
        return dual_dataset

    elif dataset in ["mini"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),
                                             transform=None)
        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(84),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        std = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        dual_dataset = DualTransformDataset(train_dataset, weak, strong, std)
        return dual_dataset


def unnormalize(img, mean, std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)  # 反归一化
    return img


def imshow(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    npimg = unnormalize(img, mean, std).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')


if __name__ == '__main__':
    mini = get_dual_dataset('mini', '/Users/lwz/torch_ds/imagenet-mini')
    mini = get_dual_dataset('imagenet100', '/Users/lwz/torch_ds/imagenet100')
    fig = plt.figure(figsize=(9, 9))

    for i in range(3):
        # 获取图片和标签
        [weak, strong, std], _ = mini[i]

        # 为每种变换添加子图

        ax = fig.add_subplot(3, 3, i * 3 + 1)
        imshow(std)
        # if i == 0:
        #     ax.set_title("Standard")
        ax = fig.add_subplot(3, 3, i * 3 + 2)
        imshow(weak)
        # if i == 0:
        #     ax.set_title("Weak")

        ax = fig.add_subplot(3, 3, i * 3 + 3)
        imshow(strong)
        # if i == 0:
        #     ax.set_title("Strong")

    plt.tight_layout()
    plt.show()
    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fig.savefig(f'{current_time}-demo.pdf')

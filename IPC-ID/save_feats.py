import numpy as np
import os
from tqdm import tqdm
from torchvision.models import resnet18
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import argparse


def laod_args():
    parser = argparse.ArgumentParser(description='Extract features and labels from a pretrained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/byol-cifar10-32brzx9a-ep=999.ckpt', help='')
    parser.add_argument('--data_dir', type=str, default='./data', help='')
    parser.add_argument('--method_name', type=str, default='byol', help='')
    parser.add_argument('--save_dir', type=str, default='Pre-Feats', help='')
    args = parser.parse_args()
    return args


def load_weight(ckpt_path):
    encoder = resnet18()
    encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    encoder.maxpool = torch.nn.Identity()
    encoder.fc = torch.nn.Identity()
    print(f"load pretrained model from {ckpt_path}")
    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]

    encoder.load_state_dict(state, strict=False)
    return encoder


if __name__ == '__main__':
    args = laod_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_weight(args.ckpt_path)
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 获取特征和标签
    features = []
    labels = []
    with torch.no_grad():
        for images, labels_batch in tqdm(test_loader):
            images = images.to(device)
            features_batch = model(images)
            features.append(features_batch.cpu().numpy())
            labels.append(labels_batch)

    # 保存特征和标签
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    features=np.concatenate(features)
    labels=np.concatenate(labels)
    sorted_indices = np.argsort(labels)
    sorted_features = features[sorted_indices]
    sorted_labels = labels[sorted_indices]

    np.save(os.path.join(args.save_dir, f'{args.method_name}_features.npy'), sorted_features)
    np.save(os.path.join(args.save_dir, f'{args.method_name}_labels.npy'), sorted_labels)

import numpy as np
import os
from tqdm import tqdm
from resnet50_baseline import ResNet50 as resnet50_b
from resnet50_l2 import ResNet50 as resnet50_msun
from resnet50_mst import ResNet50 as resnet50_mst
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import argparse
import torch.nn as nn


def laod_args():
    parser = argparse.ArgumentParser(description='Extract features and labels from a pretrained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/resnet50-mstrain/last.ckpt', help='')
    parser.add_argument('--data_dir', type=str, default='data/imagenet-100-val', help='')
    parser.add_argument('--method_name', type=str, default='resnet50-mst', help='')
    parser.add_argument('--save_dir', type=str, default='Pre-Feats-imagenet100', help='')
    args = parser.parse_args()
    return args


def load_weight(ckpt_path):
    pl_model = resnet50_mst()
    print(f"load pretrained model from {ckpt_path}")
    state = torch.load(ckpt_path)["state_dict"]

    # for k in list(state.keys()):
    #     if "encoder" in k and "momentum" not in k:
    #         state[k.replace("encoder.", "")] = state[k]
    #     # if "backbone" in k:
    #     #     state[k.replace("backbone.", "")] = state[k]
    #     del state[k]

    pl_model.load_state_dict(state, strict=True)
    pl_model.model.fc = nn.Identity()

    return pl_model


if __name__ == '__main__':
    args = laod_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_weight(args.ckpt_path)

    model.to(device)
    model.eval()
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(args.data_dir, val_transforms)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 获取特征和标签
    # features = []
    features_32 = []
    features_128 = []
    features_224 = []

    labels = []
    with torch.no_grad():
        for images, labels_batch in tqdm(val_loader):
            images = images.to(device)
            # features_batch = model(images)
            features_batch_32 = model.forward_32(images)
            features_batch_128 = model.forward_128(images)
            features_batch_224=model.forward_224(images)
            # features.append(features_batch.cpu().numpy())
            features_32.append(features_batch_32.cpu().numpy())
            features_128.append(features_batch_128.cpu().numpy())
            features_224.append(features_batch_224.cpu().numpy())
            labels.append(labels_batch)

    # 保存特征和标签
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    features_32 = np.concatenate(features_32)
    features_128 = np.concatenate(features_128)
    features_224 = np.concatenate(features_224)


    labels = np.concatenate(labels)
    sorted_indices = np.argsort(labels)
    sorted_features_32 = features_32[sorted_indices]
    sorted_features_128 = features_128[sorted_indices]
    sorted_features_224 = features_224[sorted_indices]
    sorted_labels = labels[sorted_indices]

    np.save(os.path.join(args.save_dir, f'{args.method_name}-features-32.npy'), sorted_features_32[:5000])
    np.save(os.path.join(args.save_dir, f'{args.method_name}-features-128.npy'), sorted_features_128[:5000])
    np.save(os.path.join(args.save_dir, f'{args.method_name}-features-224.npy'), sorted_features_224[:5000])
    np.save(os.path.join(args.save_dir, f'{args.method_name}-labels.npy'), sorted_labels[:5000])

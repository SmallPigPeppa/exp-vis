import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import os
import torch
from torch import nn
import torchvision
from torchvision.transforms.functional import InterpolationMode
from torchvision.models import resnet50
from torchvision import transforms, datasets


def print_info(model):
    # Load means and vars
    bn_means = []
    bn_vars = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_means.append(module.running_mean.cpu().detach().numpy())
            bn_vars.append(module.running_var.cpu().detach().numpy())

    # Print
    mean_str = np.array2string(np.array([mean.mean() for mean in bn_means]), separator=', ')
    print("Means:")
    print(mean_str)

    var_str = np.array2string(np.array([var.mean() for var in bn_vars]), separator=', ')
    print("\nVars:")
    print(var_str)


def reset_running_stats(module):
    if isinstance(module, nn.BatchNorm2d):
        module.running_mean.zero_()
        module.running_var.fill_(1)


def get_loader(hflip_prob=0.5, dataset_path='/mnt/mmtech01/dataset/lzy/ILSVRC2012', batch_size=128):
    interpolation = InterpolationMode.BILINEAR
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=interpolation),
        transforms.RandomHorizontalFlip(hflip_prob),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, "val"), transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    return loader


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPU found. Please make sure you have GPUs installed."
    # 32 resolution forward
    imagesize = 32
    batch_size_per_gpu = 128

    # Load pretrained ResNet50 model
    model = resnet50(pretrained=True)

    # if multi gpu
    if num_gpus > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    # Set the model to training mode
    model.train()
    # Reset running mean and running variance
    model.apply(reset_running_stats)


    total_batch_size = batch_size_per_gpu * num_gpus
    loader = get_loader(batch_size=total_batch_size)

    # Forward pass with your dataloader
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.cuda(), y.cuda()
            x = F.interpolate(x, size=imagesize, mode='bilinear')
            x = F.interpolate(x, size=224, mode='bilinear')
            outputs = model(x)

    print_info(model)
    save_path = "resnet50_32.pth"
    # Save the model parameters
    torch.save(model.state_dict(), save_path)

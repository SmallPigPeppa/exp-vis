from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import numpy as np
import os
from tqdm import tqdm
from resnet50_baseline import ResNet50 as resnet50_b
from resnet50_l2 import ResNet50 as resnet50_msun
from resnet50_l2 import ResNet50_32 as resnet50_msun_32
from resnet50_l2 import ResNet50_128 as resnet50_msun_128
from resnet50_mst import ResNet50 as resnet50_mst
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import argparse
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


def load_weight_msun():
    ckpt_path = 'ckpt/resnet50-l2-last/last.ckpt'
    pl_model = resnet50_msun()
    print(f"load pretrained model from {ckpt_path}")
    state = torch.load(ckpt_path)["state_dict"]
    pl_model.load_state_dict(state, strict=True)
    pl_model.model.fc = nn.Identity()
    pl_model.train()
    return pl_model

def load_weight_baseline():
    model = resnet50(pretrained=True)
    # pl_model = resnet50_msun()
    # print(f"load pretrained model from {ckpt_path}")
    # state = torch.load(ckpt_path)["state_dict"]
    # pl_model.load_state_dict(state, strict=True)
    # pl_model.model.fc = nn.Identity()
    model.train()
    return model

def load_weight_mst():
    ckpt_path = 'ckpt/resnet50-mstrain/last.ckpt'
    pl_model = resnet50_mst()
    print(f"load pretrained model from {ckpt_path}")
    state = torch.load(ckpt_path)["state_dict"]
    pl_model.load_state_dict(state, strict=True)
    pl_model.model.fc = nn.Identity()
    pl_model.train()
    return pl_model


def load_weight_msun_32():
    ckpt_path = 'ckpt/resnet50-l2-last/last.ckpt'
    pl_model = resnet50_msun_32()
    print(f"load pretrained model from {ckpt_path}")
    state = torch.load(ckpt_path)["state_dict"]
    pl_model.load_state_dict(state, strict=True)
    pl_model.model.fc = nn.Identity()
    pl_model.train()

    return pl_model


def load_weight_msun_128():
    ckpt_path = 'ckpt/resnet50-l2-last/last.ckpt'
    pl_model = resnet50_msun_128()
    print(f"load pretrained model from {ckpt_path}")
    state = torch.load(ckpt_path)["state_dict"]
    pl_model.load_state_dict(state, strict=True)
    pl_model.model.fc = nn.Identity()
    pl_model.train()

    return pl_model


def get_grad_vis(input_tensor, model, target_layers,img=None):
    # target_layers = [model.model.unified_net.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    # 从张量中提取一张RGB图像，并转换为NumPy数组
    rgb_img = input_tensor[0].cpu().numpy()
    rgb_img = np.transpose(rgb_img, (1, 2, 0))  # 从(C, H, W)转换为(H, W, C)

    # 反标准化（如果需要）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1)

    if img is not None:
        rgb_img = img[0].cpu().numpy()
        rgb_img = np.transpose(rgb_img, (1, 2, 0))  # 从(C, H, W)转换为(H, W, C)

        # 反标准化（如果需要）

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1)

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = load_weight(args.ckpt_path)
    #
    # model.to(device)
    # model.eval()
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms_32 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(32),
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms_128 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(128),
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_path = 'ILSVRC2012_val_00005567.JPEG'  # 指定图片路径
    image_path = 'ILSVRC2012_val_00004893.JPEG'
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = val_transforms(input_image).unsqueeze(0)  # 添加batch维度
    input_tensor = input_tensor.to(device)

    input_tensor_32 = val_transforms_32(input_image).unsqueeze(0)  # 添加batch维度
    input_tensor_32 = input_tensor_32.to(device)

    input_tensor_128 = val_transforms_128(input_image).unsqueeze(0)  # 添加batch维度
    input_tensor_128 = input_tensor_128.to(device)

    model = load_weight_msun()
    model_32 = load_weight_msun_32()
    model_128 = load_weight_msun_128()
    msun = get_grad_vis(input_tensor, model,[model.model.unified_net.layer4[-1]])
    msun_32 = get_grad_vis(input_tensor_32, model_32,[model_32.model.unified_net.layer4[-1]])
    msun_128 = get_grad_vis(input_tensor_128, model_128,[model_128.model.unified_net.layer4[-1]])












    # baseline

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms_32 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(32),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms_128 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(64),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms_128_128 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(128),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_path = 'ILSVRC2012_val_00005567.JPEG'  # 指定图片路径
    image_path = 'ILSVRC2012_val_00004893.JPEG'
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = val_transforms(input_image).unsqueeze(0)  # 添加batch维度
    input_tensor = input_tensor.to(device)

    input_tensor_32 = val_transforms_32(input_image).unsqueeze(0)  # 添加batch维度
    input_tensor_32 = input_tensor_32.to(device)

    input_tensor_128 = val_transforms_128(input_image).unsqueeze(0)  # 添加batch维度
    input_tensor_128 = input_tensor_128.to(device)

    input_tensor_128_128 = val_transforms_128_128(input_image).unsqueeze(0)  # 添加batch维度
    input_tensor_128_128 = input_tensor_128_128.to(device)

    model = resnet50(pretrained=True)
    model.train()

    baseline = get_grad_vis(input_tensor, model,[model.layer4[-1]])
    baseline_32 = get_grad_vis(input_tensor_32, model,[model.layer4[-1]])
    baseline_128 = get_grad_vis(input_tensor_128, model,[model.layer4[-1]], img=input_tensor_128_128)

    model = load_weight_mst()
    mst = get_grad_vis(input_tensor, model,[model.model.unified_net.layer4[-1]])
    mst_32 = get_grad_vis(input_tensor_32, model,[model.model.unified_net.layer4[-1]])
    mst_128 = get_grad_vis(input_tensor_128, model,[model.model.unified_net.layer4[-1]])







    #
    # fig, axs = plt.subplots(1, 4, figsize=(4 * 5, 5))  # 1 row, 3 columns, and figure size of (15,5)
    #
    # # Display the first visualization
    #
    # rgb_img = input_tensor[0].cpu().numpy()
    # rgb_img = np.transpose(rgb_img, (1, 2, 0))  # 从(C, H, W)转换为(H, W, C)
    #
    # # 反标准化（如果需要）
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # rgb_img = std * rgb_img + mean
    # rgb_img = np.clip(rgb_img, 0, 1)
    #
    # fontsize0 = 24
    # axs[0].imshow(rgb_img)
    # axs[0].axis('off')
    # # axs[0].set_title('Image',fontsize=fontsize0,fontweight='bold')
    #
    # axs[1].imshow(baseline)
    # axs[1].imshow(mst)
    # axs[1].imshow(msun)
    # axs[1].axis('off')
    # # axs[1].set_title('Size 224',fontsize=fontsize0,fontweight='bold')
    #
    # # Display the second visualization
    # axs[2].imshow(baseline_128)
    # axs[2].imshow(mst_128)
    # axs[2].imshow(msun_128)
    # axs[2].axis('off')
    # # axs[2].set_title('Size 128',fontsize=fontsize0,fontweight='bold')
    #
    # # Display the third visualization
    # axs[3].imshow(baseline_32)
    # axs[3].imshow(mst_32)
    # axs[2].imshow(msun_32)
    # axs[3].axis('off')
    # # axs[3].set_title('Size 32',fontsize=fontsize0,fontweight='bold')
    #
    # # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # # plt.tight_layout(pad=0.2)
    # plt.tight_layout()
    # plt.savefig('demo_msun.jpg')
    # plt.show()
    fig, axs = plt.subplots(3, 4, figsize=(4 * 5.2, 3 * 5))

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)


    # Original image
    rgb_img = input_tensor[0].cpu().numpy()
    rgb_img = np.transpose(rgb_img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1)
    fontsize0=24

    titles = ['Image', 'Size 224', 'Size 128', 'Size 32']
    for i, title in enumerate(titles):
        axs[0, i].set_title(title,fontsize=fontsize0)

    # Add labels on the side
    labels = ['Vanilla', 'MST', 'MSUN']
    for i, label in enumerate(labels):
        axs[i, 0].set_ylabel(label,fontsize=fontsize0)

    # Display the original image in the first column of each row
    for i in range(3):
        axs[i, 0].imshow(rgb_img)
        # axs[i, 0].axis('off')



    # Display baseline, mst, msun results for different sizes
    results = [
        [baseline, baseline_128, baseline_32],
        [mst, mst_128, mst_32],
        [msun, msun_128, msun_32]
    ]

    for i, row in enumerate(results):
        for j, img in enumerate(row):
            axs[i, j + 1].imshow(img)
            # axs[i, j + 1].axis('off')


    plt.tight_layout()
    plt.savefig('demo_msun.jpg')
    plt.show()

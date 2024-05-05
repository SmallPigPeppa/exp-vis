from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
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
from PIL import Image
import matplotlib.pyplot as plt


def laod_args():
    parser = argparse.ArgumentParser(description='Extract features and labels from a pretrained model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 32)')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/resnet50-l2-last/last.ckpt', help='')
    parser.add_argument('--data_dir', type=str, default='data/imagenet-100-val', help='')
    parser.add_argument('--method_name', type=str, default='resnet50-msun', help='')
    parser.add_argument('--save_dir', type=str, default='Pre-Feats-imagenet100', help='')
    args = parser.parse_args()
    return args


def load_weight(ckpt_path):
    pl_model = resnet50_msun()
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

def get_grad_vis(input_tensor,model,img=None):
    target_layers = [model.layer4[-1]]
    args.use_cuda = True
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)
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
    args = laod_args()
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
    image_path= 'ILSVRC2012_val_00004893.JPEG'
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


    visualization = get_grad_vis(input_tensor,model)
    visualization_32 = get_grad_vis(input_tensor_32,model)
    visualization_128 = get_grad_vis(input_tensor_128,model,img=input_tensor_128_128)

    fig, axs = plt.subplots(1, 4, figsize=(4*5, 5))  # 1 row, 3 columns, and figure size of (15,5)

    # Display the first visualization

    rgb_img = input_tensor[0].cpu().numpy()
    rgb_img = np.transpose(rgb_img, (1, 2, 0))  # 从(C, H, W)转换为(H, W, C)

    # 反标准化（如果需要）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1)


    fontsize0=24
    axs[0].imshow(rgb_img)
    axs[0].axis('off')
    # axs[0].set_title('Image',fontsize=fontsize0,fontweight='bold')

    axs[1].imshow(visualization)
    axs[1].axis('off')
    # axs[1].set_title('Size 224',fontsize=fontsize0,fontweight='bold')


    # Display the second visualization
    axs[2].imshow(visualization_128)
    axs[2].axis('off')
    # axs[2].set_title('Size 128',fontsize=fontsize0,fontweight='bold')


    # Display the third visualization
    axs[3].imshow(visualization_32)
    axs[3].axis('off')
    # axs[3].set_title('Size 32',fontsize=fontsize0,fontweight='bold')


    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.tight_layout(pad=0.2)
    plt.tight_layout()
    plt.savefig('demo.jpg')
    plt.show()

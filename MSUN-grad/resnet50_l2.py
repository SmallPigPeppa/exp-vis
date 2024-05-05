from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchvision.models import resnet50


def unified_net():
    u_net = resnet50(pretrained=False)
    u_net.conv1 = nn.Identity()
    u_net.bn1 = nn.Identity()
    u_net.relu = nn.Identity()
    u_net.maxpool = nn.Identity()
    u_net.layer1 = nn.Identity()
    return u_net


class ResNet50_L2(LightningModule):
    def __init__(self):
        super().__init__()
        self.large_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet50(pretrained=False).layer1
        )
        self.mid_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet50(pretrained=False).layer1
        )
        self.small_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), resnet50(pretrained=False).layer1
        )
        self.unified_net = unified_net()
        self.small_size = (32, 32)
        self.mid_size = (128, 128)
        self.large_size = (224, 224)
        self.unified_size = (56, 56)

    def forward(self, imgs):
        small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')

        z1 = self.small_net(small_imgs)
        z2 = self.mid_net(mid_imgs)
        z3 = self.large_net(large_imgs)

        z1 = F.interpolate(z1, size=self.unified_size, mode='bilinear')
        z2 = F.interpolate(z2, size=self.unified_size, mode='bilinear')

        y1 = self.unified_net(z1)
        y2 = self.unified_net(z2)
        y3 = self.unified_net(z3)

        return z1, z2, z3, y1, y2, y3

    def forward_32(self, imgs):
        small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        z1 = self.small_net(small_imgs)
        z1 = F.interpolate(z1, size=self.unified_size, mode='bilinear')
        y1 = self.unified_net(z1)

        return y1

    def forward_128(self, imgs):
        mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        z2 = self.mid_net(mid_imgs)
        z2 = F.interpolate(z2, size=self.unified_size, mode='bilinear')
        y2 = self.unified_net(z2)

        return y2

    def forward_224(self, imgs):
        large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')
        z3 = self.large_net(large_imgs)
        y3 = self.unified_net(z3)

        return y3


class ResNet50(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ResNet50_L2()

    def forward(self, x):
        return self.model.forward_224(x)

    def forward_32(self, x):
        return self.model.forward_32(x)

    def forward_128(self, x):
        return self.model.forward_128(x)

    def forward_224(self, x):
        return self.model.forward_224(x)


class ResNet50_128(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ResNet50_L2()

    def forward(self, x):
        return self.model.forward_128(x)

    def forward_32(self, x):
        return self.model.forward_32(x)

    def forward_128(self, x):
        return self.model.forward_128(x)

    def forward_224(self, x):
        return self.model.forward_224(x)


class ResNet50_32(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ResNet50_L2()

    def forward(self, x):
        return self.model.forward_32(x)

    def forward_32(self, x):
        return self.model.forward_32(x)

    def forward_128(self, x):
        return self.model.forward_128(x)

    def forward_224(self, x):
        return self.model.forward_224(x)

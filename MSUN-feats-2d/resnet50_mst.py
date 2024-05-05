import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchvision.models import resnet50


class ResNet50_L2(LightningModule):
    def __init__(self):
        super().__init__()
        self.unified_net = resnet50(pretrained=False)
        self.small_size = (32, 32)
        self.mid_size = (128, 128)
        self.large_size = (224, 224)

    # def forward(self, imgs):
    #     small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
    #     mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
    #     large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')
    #
    #     small_imgs = F.interpolate(small_imgs, size=self.large_size, mode='bilinear')
    #     mid_imgs = F.interpolate(mid_imgs, size=self.large_size, mode='bilinear')
    #
    #     y1 = self.unified_net(small_imgs)
    #     y2 = self.unified_net(mid_imgs)
    #     y3 = self.unified_net(large_imgs)
    #
    #     return y1, y2, y3

    def forward(self, imgs):
        y = self.unified_net(imgs)
        return y


class ResNet50(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ResNet50_L2()

    def forward(self, x):
        return self.model(x)

    def forward_32(self, x):
        x = F.interpolate(x, size=32, mode='bilinear')
        x = F.interpolate(x, size=224, mode='bilinear')
        return self.model(x)

    def forward_128(self, x):
        x = F.interpolate(x, size=128, mode='bilinear')
        x = F.interpolate(x, size=224, mode='bilinear')
        return self.model(x)

    def forward_224(self, x):
        return self.model(x)

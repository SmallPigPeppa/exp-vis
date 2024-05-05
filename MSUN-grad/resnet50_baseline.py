import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule


class ResNet50(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False)

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

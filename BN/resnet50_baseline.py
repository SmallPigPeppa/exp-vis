import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 加载预训练的ResNet-50模型
model = models.resnet50(pretrained=True)

# 从ResNet-50模型中提取所有BN层的均值和方差
bn_means = []
bn_vars = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        bn_means.append(module.running_mean.detach().numpy())
        bn_vars.append(module.running_var.detach().numpy())

# 计算每个BN层的均值和方差
mean_values = [mean.mean() for mean in bn_means]
var_values = [var.mean() for var in bn_vars]

# 绘制均值和方差的误差条形图
x = list(range(len(mean_values)))
plt.bar(x, mean_values, yerr=var_values, capsize=5, alpha=0.5)
plt.xlabel("Batch Normalization Layers")
plt.ylabel("Mean and Variance")
plt.title("Mean and Variance of BN Layers in ResNet-50")
plt.show()

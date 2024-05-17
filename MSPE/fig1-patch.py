import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# 定义文件路径
base_path = '/Users/lwz/Downloads/AAA-MSPE-vis/'

# 读取CSV文件
vanilla_df = pd.read_csv(base_path + 'vanilla.csv')
flexivit_df = pd.read_csv(base_path + 'flexivit.csv')
msp_df = pd.read_csv(base_path + 'msp.csv')

# 绘图
# Set the overall figure size
fig, ax = plt.subplots(figsize=(10, 7))  # Width, Height in inches

# Set font sizes
title_fontsize = 30
axis_label_fontsize = 28
tick_label_fontsize = 26
legend_fontsize = 24



# 数据集合


datasets = [
    (vanilla_df, '#1f77b4', 'Vanilla'),
    (flexivit_df, '#ff7f0e', 'FlexiViT'),
    (msp_df, '#2ca02c', 'Our')
]

# 计算全局最小值和最大值
data_min = min(df['Sim Patches'].min() for df, _, _ in datasets)
data_max = max(df['Sim Patches'].max() for df, _, _ in datasets)
# data_max = 0.95
# data_min = 0.45

# 定义bins
num_bins = 30
bin_width = (data_max - data_min) / num_bins
base_bins = np.arange(data_min, data_max + bin_width, bin_width)

# 绘制直方图和高斯曲线
for i, (df, color, label) in enumerate(datasets):
    # 计算每个数据集特定的bins偏移，以创建空隙
    offset = (i - 1) * bin_width / 4  # 调整偏移量
    bins = base_bins + offset

    # 绘制直方图
    ax.hist(df['Sim Patches'], bins=bins, alpha=0.4, color=color, label=f'{label}', density=True, rwidth=0.8)

    # 高斯分布曲线的参数
    mean_val = df['Sim Patches'].mean()
    std_val = df['Sim Patches'].std()
    x = np.linspace(data_min, data_max, 1000)
    gaussian_curve = norm.pdf(x, mean_val, std_val)

    # 绘制高斯分布曲线
    ax.plot(x, gaussian_curve, color=color, linestyle='-', label=None)
    # ax.plot(x, gaussian_curve, color=color, linestyle='-', label=f'{label} Gaussian Fit')

# 图例和标签

# Adjust font size for title, labels, and ticks
ax.set_title('(a) Patch Embedding', fontsize=title_fontsize)
ax.set_xlabel('Cosine Similarity', fontsize=axis_label_fontsize)
ax.set_ylabel('Density', fontsize=axis_label_fontsize)

# Adjust tick label size
ax.tick_params(axis='both', labelsize=tick_label_fontsize)

# Adjust legend font size
ax.legend(fontsize=legend_fontsize)
ax.grid(True)


plt.tight_layout()
plt.show()

fig.savefig('fig1-patch-embedding.pdf')

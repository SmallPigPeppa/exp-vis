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
data_min = min(df['Sim Classes'].min() for df, _, _ in datasets)
data_max = max(df['Sim Classes'].max() for df, _, _ in datasets)

# 定义bins
num_bins = 30
bin_width = (data_max - data_min) / num_bins
base_bins = np.arange(data_min, data_max + bin_width, bin_width)

# 绘制直方图和高斯曲线
for i, (df, color, label) in enumerate(datasets):
    pass

# 设置轴线和标签颜色
# ax.spines['bottom'].set_color('white')
# ax.spines['left'].set_color('white')
# ax.spines['top'].set_color('white')
# ax.spines['right'].set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# 取消刻度线，只显示边框
ax.tick_params(axis='both', length=0)

# Adjust font size for title, labels, and ticks
ax.set_title('(a) Illustration of Motivation', fontsize=title_fontsize)
ax.set_xlabel('Cosine Similarity', fontsize=axis_label_fontsize, color='white')
ax.set_ylabel('Density', fontsize=axis_label_fontsize, color='white')

# Adjust tick label size
ax.tick_params(axis='both', labelsize=tick_label_fontsize)

# Adjust legend font size
# ax.legend(fontsize=legend_fontsize)
# ax.grid(True)

plt.tight_layout()
plt.show()

fig.savefig('fig1-demo.pdf')

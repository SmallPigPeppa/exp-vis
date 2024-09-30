import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 数据集名称
datasets = ['Flowers', 'DTD', 'Food101', 'StanfordCars']
methods = ['EWC', 'ZSCL', 'Mod-X', 'MOE-CL', 'C-CLIP', 'CLIP']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
font1 = 16  # 标题字体大小
font3 = 16  # 图例字体大小
font2 = 15  # 坐标轴刻度和标签的字体大小

# 各方法在每个数据集上的性能值
EWC = [38.36, 29.18, 65.62, 40.19]
ZSCL = [42.39, 30.08, 66.92, 44.92]
Mod_X = [39.07, 30.66, 64.61, 42.90]
MOE_CL = [40.41, 31.22, 66.68, 40.54]
C_CLIP = [54.32, 38.30, 73.70, 50.35]
CLIP = [70.39, 41.22, 83.58, 61.26]


# 将数据组合成列表
data = [EWC, ZSCL, Mod_X, MOE_CL, C_CLIP, CLIP]

# 柱子的宽度和间距参数
bar_width = 2.0
bar_spacing = 4.0  # 柱子之间的间距
opacity = 0.4  # 透明度

# 创建子图
fig, axes = plt.subplots(1, 4, figsize=(12, 3))

# 绘制图表
for i, ax in enumerate(axes):
    # 当前数据集的值
    values = [method_data[i] for method_data in data]

    # 调整柱子的位置，使第一个柱子的左边缘与 y 轴的距离为 bar_spacing / 2
    x = bar_spacing / 2 + bar_width / 2 + np.arange(len(methods)) * (bar_width + bar_spacing)

    # 绘制柱状图
    bars = ax.bar(x, values, color=colors, alpha=opacity, width=bar_width)

    # 设置x轴标签和标题
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, fontsize=font2)
    ax.set_title(datasets[i], fontsize=font1)
    ax.tick_params(axis='y', labelsize=font2)

    # 设置y轴范围
    ax.set_ylim(0, max(CLIP[i] + 10, max(values) + 10))

    # 调整x轴范围
    xlim_left = x[0] - bar_width / 2 - bar_spacing / 2
    xlim_right = x[-1] + bar_width / 2 + bar_spacing / 2
    ax.set_xlim(xlim_left, xlim_right)

# 自定义图例，调整透明度
legend_elements = [Patch(facecolor=colors[j], label=methods[j], alpha=opacity) for j in range(len(methods))]

# 调整布局并添加图例
legend = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=font3, ncol=6)

plt.tight_layout()
# 保存图表为PDF文件
plt.savefig('fig3.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')

plt.show()

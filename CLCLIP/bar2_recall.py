import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 设置风格
sns.set(style="whitegrid")

methods = ['Fine-tune', 'ZSCL', 'Mod-X', 'Our']
datasets = ['COCO (5K test)', 'Flick30K (1K test)']

data = {
    'Fine-tune': [54.09, 52.2],
    'ZSCL': [56.35, 50.66],
    'Mod-X': [55.62, 49.72],
    'Our': [55.62, 49.72],
}

# Origin 数据
origin = [11.3, 16.9]

# 设置参数
mpl.rcParams['hatch.linewidth'] = 5
bar_width = 0.15  # 柱子宽度
opacity = 0.55    # 柱子透明度
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
spacing = 0.2
# 柱子间隔

fontsize0 = 24
fontsize1 = 18

fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1行2列的子图

for model_idx, dataset in enumerate(datasets):
    for i, method in enumerate(methods):
        axs[model_idx].bar(i * (bar_width + spacing), data[method][model_idx], bar_width, alpha=opacity,
                           edgecolor='black', color=colors[i])

    # 设置标题和y轴范围
    axs[model_idx].set_xlabel(dataset, fontsize=fontsize0)
    axs[model_idx].set_ylim(45, 62)
    axs[model_idx].set_xticks([])  # 去掉x轴刻度
    axs[model_idx].grid(True, linestyle='--', linewidth=0.5)
    axs[model_idx].yaxis.set_tick_params(labelsize=fontsize1)

    # 画 origin 的红色虚线
    axs[model_idx].axhline(origin[model_idx], color='red', linestyle='--', linewidth=2)
    axs[model_idx].yaxis.set_major_locator(MaxNLocator(integer=True))

# 只为第一个子图添加y轴标签
axs[0].set_ylabel('Image-text R@1 (%)', fontsize=fontsize0)

# 创建柱状图的 handles 用于图例
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=opacity) for i in range(len(methods))]

# 添加 origin 的 handle 到图例中
origin_handle = mpl.lines.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Origin')

# 将 origin 的 handle 和柱状图的 handles 一起加入到 legend 中
handles.append(origin_handle)

# 添加图例，包含methods和origin
fig.legend(handles, methods + ['Origin CLIP'], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=fontsize0)

# 展示图像
plt.show()

# 保存图像
fig.savefig('bar_recall.pdf', bbox_inches='tight')

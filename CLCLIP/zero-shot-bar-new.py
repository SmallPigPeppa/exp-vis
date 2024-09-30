import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 数据
datasets = ['StanfordCars', 'STL-10', 'Flowers', 'Pets', 'DTD', 'Aircraft']
methods = ['EWC', 'ZSCL', 'Mod-X', 'MOE-CL', 'C-CLIP', 'CLIP']
colors = ['#2ca02c', '#9467bd', '#ff7f0e', '#e377c2', '#d62728', '#8c564b']
font1 = 20  # 标题和图例的字体大小
font2 = 16  # 坐标轴刻度和标签的字体大小

# 各方法在每个数据集上的性能值
EWC = [38.65, 40.77, 43.06, 41.55, 39.07, 38.65]
ZSCL = [42.43, 46.08, 45.04, 44.93, 42.27, 42.43]
Mod_X = [39.63, 42.88, 42.26, 41.49, 41.18, 39.63]
MOE_CL = [40.87, 44.77, 44.89, 43.31, 42.70, 40.87]
C_CLIP = [51.23, 55.74, 54.41, 53.85, 52.42, 51.23]
CLIP = [61.09, 65, 60, 55, 65, 70]

# 将数据组合成列表
data = [EWC, ZSCL, Mod_X, MOE_CL, C_CLIP, CLIP]

# 柱子的宽度和间距参数
bar_width = 2.0
bar_spacing = 4.0  # 柱子之间的间距
opacity = 0.6  # 透明度

# 创建子图
fig, axes = plt.subplots(1, 6, figsize=(24, 6))

# 绘制图表
for i, ax in enumerate(axes):
    # 当前数据集的值
    values = [method_data[i] for method_data in data]
    clip_value = CLIP[i]

    # 柱子的位置
    x = np.arange(len(methods)) * (bar_width + bar_spacing)

    # 绘制柱状图
    bars = ax.bar(x, values, color=colors, alpha=opacity, width=bar_width)

    # 在每个柱子上添加黑色短线，表示与CLIP的差距
    for j, bar in enumerate(bars):
        if methods[j] != 'CLIP':
            # 差距
            gap = clip_value - values[j]
            # 线的长度与差距成比例，可以根据需要调整比例系数
            line_length = gap * 0.1  # 比例系数为0.1

            # 获取柱子的顶部位置
            bar_top = bar.get_height()
            bar_x = bar.get_x() + bar.get_width() / 2

            # 绘制短线
            ax.plot([bar_x, bar_x], [bar_top, bar_top + line_length], color='black', linewidth=1.0)

    # 设置x轴标签和标题
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, fontsize=font2)
    ax.set_title(datasets[i], fontsize=font1)
    ax.tick_params(axis='y', labelsize=font2)

    # 设置y轴范围
    ax.set_ylim(0, max(CLIP[i] + 10, max(values) + 10))

    # 调整x轴范围以适应所有柱子
    ax.set_xlim(-bar_width, x[-1] + bar_width + bar_spacing)

# 自定义图例，调整透明度
legend_elements = [Patch(facecolor=colors[j], label=methods[j], alpha=opacity) for j in range(len(methods))]

# 调整布局并添加图例
fig.subplots_adjust(top=0.85)
fig.legend(handles=legend_elements, loc='upper center', ncol=6, fontsize=font1, bbox_to_anchor=(0.5, 0.95))

plt.tight_layout()

# 保存图表为PDF文件
plt.savefig('bar_charts_with_lines.pdf', format='pdf')

plt.show()

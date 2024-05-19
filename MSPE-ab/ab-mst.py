import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 定义格式化函数
def custom_formatter(x, pos):
    return f'{int(x)}%'

# 数据
resolutions = ['56x56', '112x112', '224x224']
with_mst = [77.944, 83.746, 83.5]
without_mst = [45.68, 80.12, 85.10]

# 数据组织，按分辨率
data = [with_mst, without_mst]
labels = ['w/ MST', 'w/o MST']
colors = ['#C76426', '#8F7E1E']

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6))

# 柱状图的位置
x = range(len(resolutions))
width = 0.3  # 柱子的宽度
spacing = 0.05  # 间距

# 绘制柱状图
for i in range(len(data)):
    ax.bar([x_pos + (width + spacing) * i for x_pos in x], data[i], width=width, color=colors[i], label=labels[i], alpha=0.7)

# 设置x轴
ax.set_xticks([x_pos + (width + spacing) / 2 for x_pos in x])  # Center ticks between the bars
ax.set_xticklabels(resolutions)

# 设置字体大小
title_fontsize = 26
tick_label_fontsize = 23
legend_fontsize = 21
line_width = 4  # Adjust line width
minor_tick_length = 10

ax.tick_params(axis='both', labelsize=tick_label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize, width=line_width, length=minor_tick_length)

# 设置y轴
ax.set_ylabel('ImageNet-1K Acc@1', fontsize=title_fontsize)
ax.set_xlabel('Resolution', fontsize=title_fontsize)
ax.set_title(r'Impact of $\mathbf{M}$ulti-$\mathbf{S}$cale $\mathbf{T}$raining', fontsize=title_fontsize)


# 设置y轴范围
ax.set_ylim(40, 90)

# 设置y轴刻度格式
ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))

# 移除不必要的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(line_width)
ax.spines['left'].set_linewidth(line_width)

# 添加图例并设置透明度
ax.legend(fontsize=legend_fontsize, framealpha=0.5)

# 设置主副刻度的网格线
ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.75)
ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5)

# 设置y轴主刻度和副刻度
ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))

# 调整布局
plt.tight_layout()

# 展示图表
plt.show()

# 保存图表
fig.savefig('ab-mst.pdf')

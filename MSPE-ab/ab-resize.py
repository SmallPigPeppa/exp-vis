import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter


def custom_formatter(x, pos):
    return f'{int(x)}%'

#
# "MEMO": {"color": "#1f77b4", "linestyle": "-"},
# "FOSTER": {"color": "#ff7f0e", "linestyle": "-"},
# "DER": {"color": "#2ca02c", "linestyle": "-"},
# "PODNet": {"color": "#1f77b4", "linestyle": "-"},
# "UCIR": {"color": "#9467bd", "linestyle": "-"},
# "ICARL": {"color": "#8c564b", "linestyle": "-"},
# "LWF": {"color": "#e377c2", "linestyle": "--"},
# "PASS": {"color": "#7f7f7f", "linestyle": "--"},
# "SSRE": {"color": "#bcbd22", "linestyle": "--"},
# "FeTrIL": {"color": "#17becf", "linestyle": "--"},
# "Semi-IPC": {"color": "#d62728", "linestyle": "--"}
title_fontsize = 26
tick_label_fontsize = 23
legend_fontsize = 21
line_width = 2  # Adjust line width
minor_tick_length = 6
l1 = 2
marker_size = 8  # Adjust marker size

# 分辨率
resolutions = [28, 42, 56, 70, 84, 98, 112, 126, 140, 168, 224, 238, 252, 448]
resolution_labels = [str(res) for res in resolutions]

# 各种方法的准确率
area_accuracy = [53.068, 59.942, 75.808, 74.16, 78.312, 78.342, 81.188, 79.948, 81.02, 82.558, 85.1, 81.962, 82.104,
                 74.248]
bilinear_accuracy = [51.94, 38.838, 77.022, 67.09, 79.022, 66.692, 82.312, 72.232, 76.38, 83.522, 85.1, 79.81, 82.946,
                     76.584]
pi_resize_accuracy = [56.408, 71.016, 77.944, 79.544, 81.632, 82.508, 83.746, 83.814, 83.936, 84.7, 85.1, 85.13, 85.13,
                      85.1]

# 创建绘图
fig, ax = plt.subplots(figsize=(6 / 0.7, 5.2))

# 绘制不同方法的准确率曲线
ax.plot(resolution_labels, area_accuracy, marker='s', markersize=marker_size, label='Area', linestyle='--', linewidth=l1,
        color='#1f77b4')
ax.plot(resolution_labels, bilinear_accuracy, marker='o', markersize=marker_size, label='Bilinear', linestyle='--',
        linewidth=l1, color='#ff7f0e')
ax.plot(resolution_labels, pi_resize_accuracy, marker=None, markersize=marker_size, label='PI-Resize', linewidth=l1,
        color='r')
# ax.plot(resolution_labels, pi_resize_accuracy, marker=None, markersize=marker_size, label='PI-Resize', linewidth=l1,
#         color='#2ca02c')

# 设置图形标题和轴标签
ax.set_xlabel(f'test resolution $r_i$', fontsize=title_fontsize)
ax.set_ylabel('ImageNet-1K Acc@1', fontsize=title_fontsize)
ax.set_title(f'(b) $width=height$', fontsize=title_fontsize)
# ax.set_title(r'Impact of Resize Method $\mathrm{B_{r}^{r*}}$', fontsize=title_fontsize)
ax.set_title(r'Impact of Resize Method', fontsize=title_fontsize)

# 设置x轴标签每隔一个res显示一个
ax.set_xticks(resolution_labels[::2])
ax.set_xticklabels(resolution_labels[::2])

# 设置小刻度
ax.xaxis.set_minor_locator(MultipleLocator(1))

# 设置轴的可见性和线宽
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(line_width)
ax.spines['left'].set_linewidth(line_width)
ax.tick_params(axis='both', labelsize=tick_label_fontsize)
ax.tick_params(axis='x', which='minor', length=minor_tick_length)
ax.tick_params(axis='both', which='major', width=line_width, length=minor_tick_length)

# 添加图例
ax.legend(fontsize=legend_fontsize)

# 显示网格
ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.75)
ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5)

# 设置y轴的major和minor ticks
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))

ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

plt.tight_layout()

# 显示图形
plt.show()

fig.savefig('ab-resize.pdf')

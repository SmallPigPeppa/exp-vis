import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

def custom_formatter(x, pos):
    return f'{int(x)}%'

title_fontsize = 33
tick_label_fontsize = 26
legend_fontsize = 28
line_width = 3  # Adjust line width
minor_tick_length = 8
l1 = 1.5
marker_size = 7
# 分辨率
resolutions = [28, 42, 56, 70, 84, 98, 112, 126, 140, 168, 224, 448, 896]
resolution_labels = [str(res) for res in resolutions]

# 各种方法的准确率
vanilla_accuracy = [8.52, 37.86, 54.67, 65.94, 72.58, 76.29, 78.75, 80.62, 82.00, 83.66, 85.10, 53.81, 0.675999978557229]
flexivit_accuracy = [14.86, 48.52, 63.44, 70.53, 74.47, 78.03, 80.24, 82.28, 83.10, 84.70, 85.10, 85.11, 85.10]
mspe_accuracy = [56.41, 71.02, 77.94, 79.54, 81.63, 82.51, 83.75, 83.81, 83.94, 84.70, 85.10, 85.11, 85.10]

# 创建绘图
fig, ax = plt.subplots(figsize=(6 / 0.7, 6))

# 绘制不同方法的准确率曲线
ax.plot(resolution_labels, vanilla_accuracy, label='Vanilla', linestyle='--', linewidth=l1, marker='*', markersize=marker_size)
ax.plot(resolution_labels, flexivit_accuracy, label='FlexiViT', linestyle='--', linewidth=l1, marker='^',
        markersize=marker_size)
# ax.plot(resolution_labels, vanilla_accuracy,  label='Vanilla',linewidth=l1, linestyle='--', marker='*', markersize=marker_size)
# ax.plot(resolution_labels, flexivit_accuracy, label='FlexiViT', linestyle='--',linewidth=l1, marker='*', markersize=marker_size)

ax.plot(resolution_labels, mspe_accuracy, marker=None, label='MSPE',linewidth=l1,color='r')
# ax.plot(resolution_labels, mspe_accuracy, marker=None, label='MSPE',linewidth=l1,color='#2ca02c')

# 设置图形标题和轴标签
ax.set_xlabel(f'test width $w_i$', fontsize=title_fontsize)
ax.set_ylabel('ImageNet-1K Acc@1', fontsize=title_fontsize)
ax.set_title(f'(a) $height=width$', fontsize=title_fontsize)

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
ax.tick_params(axis='x', which='minor', width=line_width, length=minor_tick_length)
ax.tick_params(axis='both', which='major', width=line_width, length=minor_tick_length)

# 添加图例
ax.legend(fontsize=legend_fontsize)

# 显示网格
ax.grid(True, which='major', axis='y', linestyle='-', linewidth=1.0)
ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.7)

# 设置y轴的major和minor ticks
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_minor_locator(MultipleLocator(10))

ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

plt.tight_layout()

# 显示图形
plt.show()

fig.savefig('intro-fig1.pdf')

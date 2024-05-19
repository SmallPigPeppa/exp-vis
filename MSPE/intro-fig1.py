import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mticker


def custom_formatter(x, pos):
    return f'{int(x)}%'


title_fontsize = 26
tick_label_fontsize = 23
legend_fontsize = 21
line_width = 2  # Adjust line width
minor_tick_length = 6

# 分辨率
resolutions = [28, 42, 56, 70, 84, 98, 112, 126, 140, 168, 224, 448, 896]
resolution_labels = [str(res) for res in resolutions]

# 各种方法的准确率
vanilla_accuracy = [8.52, 37.86, 54.67, 65.94, 72.58, 76.29, 78.75, 80.62, 82.00, 83.66, 85.10, 53.81, 24.31]
flexivit_accuracy = [14.86, 48.52, 63.44, 70.53, 74.47, 78.03, 80.24, 82.28, 83.10, 84.70, 85.10, 85.11, 85.10]
mspe_accuracy = [56.41, 71.02, 77.94, 79.54, 81.63, 82.51, 83.75, 83.81, 83.94, 84.70, 85.10, 85.11, 85.10]

# 创建绘图
fig, ax = plt.subplots(figsize=(5.5 / 0.618, 5.5))

# 绘制不同方法的准确率曲线
ax.plot(resolution_labels, vanilla_accuracy, marker=None, label='Vanilla')
ax.plot(resolution_labels, flexivit_accuracy, marker=None, label='FlexiViT', linestyle='--')
ax.plot(resolution_labels, mspe_accuracy, marker=None, label='MSPE')

# 设置图形标题和轴标签
# ax.set_title('Accuracy vs Resolution for Different Methods')
ax.set_xlabel('Resolution', fontsize=title_fontsize)
ax.set_ylabel('ImageNet-1K Acc@1', fontsize=title_fontsize)

# 设置x轴标签每隔一个res显示一个
ax.set_xticks(resolution_labels[::2])
ax.set_xticklabels(resolution_labels[::2])

# 设置小刻度
ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(line_width)
ax.spines['left'].set_linewidth(line_width)
ax.tick_params(axis='both', labelsize=tick_label_fontsize)
ax.tick_params(axis='x', which='minor', length=minor_tick_length)
ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize, width=line_width, length=minor_tick_length)

# 添加图例
ax.legend(fontsize=legend_fontsize)

# 显示网格
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))

plt.tight_layout()

# 显示图形
plt.show()

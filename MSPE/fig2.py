import matplotlib.pyplot as plt
import numpy as np

# 数据准备
x1 = [8, 7, 6, 5, 4, 3, 2]
y1_vanilla = [78.76, 77.58, 75.81, 73.05, 68.52, 59.86, 40.63]
y1_flexivit = [80.25, 78.88, 76.65, 74.42, 71.09, 63.70, 44.66]
y1_mspe = [83.75, 83.20, 82.74, 81.77, 80.19, 75.72, 69.27]

x2 = [12, 11, 9, 7, 5, 3]
y2_vanilla = [83.64, 83.26, 82.15, 80.01, 75.25, 62.03]
y2_flexivit = [84.70, 84.27, 83.14, 81.24, 76.56, 65.20]
y2_mspe = [84.70, 84.27, 84.13, 83.43, 81.69, 75.67]

x3 = [16, 14, 12, 10, 8, 6, 4]
y3_vanilla = [85.11, 84.92, 84.71, 83.86, 82.17, 78.87, 72.18]
y3_flexivit = [85.11, 85.04, 85.05, 84.16, 82.72, 79.57, 72.32]
y3_mspe = [85.11, 85.04, 85.05, 84.74, 83.89, 82.71, 79.84]

x4 = [24, 21, 18, 15, 12, 9, 6]
y4_vanilla = [78.93, 81.02, 82.73, 83.77, 84.48, 83.12, 77.90]
y4_flexivit = [85.11, 85.13, 85.10, 84.82, 84.56, 83.46, 79.25]
y4_mspe = [85.11, 85.13, 85.10, 85.12, 85.05, 84.13, 82.81]

# 绘图
fig, axs = plt.subplots(1, 4, figsize=(32 * 0.95, 6))

# Font sizes setup
title_fontsize = 35
axis_label_fontsize = 33
tick_label_fontsize = 30
legend_fontsize = 30
marker_size = 10  # Adjust marker size
line_width = 2  # Adjust line width


# Define a common function to plot the data
def plot_data(ax, x, y_vanilla, y_flexivit, y_mspe, bit_depth, yticks):
    ax.plot(x, y_vanilla, label='Vanilla', marker='*', color='#ff7f0e', markersize=marker_size, linewidth=line_width)
    ax.plot(x, y_flexivit, label='FlexiViT', marker='*', color='#1f77b4', markersize=marker_size, linewidth=line_width)
    ax.plot(x, y_mspe, label='MSPE', marker='*', color='red', alpha=0.5, markersize=marker_size, linewidth=line_width)
    ax.set_title(f'$R_{{\\max}} = {bit_depth * 16}$', fontsize=title_fontsize)
    ax.invert_xaxis()
    ax.legend(fontsize=legend_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    ax.set_xlabel(r'$R_{{\min}}$', fontsize=axis_label_fontsize)
    ax.set_ylabel('Accuracy', fontsize=axis_label_fontsize)
    ax.set_yticks(yticks)
    ax.set_xticks(np.linspace(min(x), max(x), len(x)))
    # ax.set_xticklabels([f"{int(xi * 16)}" for xi in np.linspace(min(x), max(x), len(x))], fontsize=tick_label_fontsize)
    ax.set_xticks(x[::2])  # 设置x轴ticks, 每隔一个显示一个
    ax.set_xticklabels([f"{int(xi * 16)}" for xi in x][::2], fontsize=tick_label_fontsize)  # 设置x轴ticks的标签
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', axis='both', linestyle='--')


# Manually specified y-ticks for each subplot
yticks1 = [40, 50, 60, 70, 80]
yticks2 = [60, 65, 70, 75, 80, 85]
# yticks2 = [60, 70, 75, 85]
yticks3 = [70, 75, 80, 85]
# yticks4 = [75, 80, 85]
yticks4 = [76, 79, 82, 85]

# Map the plots to the new subplot structure
plot_data(axs[0], x1, y1_vanilla, y1_flexivit, y1_mspe, 8, yticks1)
plot_data(axs[1], x2, y2_vanilla, y2_flexivit, y2_mspe, 12, yticks2)
plot_data(axs[2], x3, y3_vanilla, y3_flexivit, y3_mspe, 16, yticks3)
plot_data(axs[3], x4, y4_vanilla, y4_flexivit, y4_mspe, 24, yticks4)

# 调整子图间的水平间距
plt.subplots_adjust(left=None, right=None, wspace=3, hspace=None)

plt.tight_layout()
plt.show()
fig.savefig('fig2.pdf')  # Save the figure in a suitable format

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# 定义格式化函数
def custom_formatter(x, pos):
    return f'{int(x)}%'


# 应用自定义格式化函数


# 数据
resolutions = [f'$r_i = 64$', f'$r_i = 160$', f'$r_i = 224$']
vanilla = [60.71, 80.62, 85.1]
navit = [67.5, 82.5, 85.9]
mspe = [78.18, 83.92, 85.104]

# 数据组织，按分辨率
data = [vanilla, navit, mspe]
labels = ['Vanilla', 'NaViT', 'MSPE']
colors = ['#5F9E6E', '#CC8963', '#5975A4']

# 创建图表
fig, ax = plt.subplots(1, 1, figsize=(8 , 6))

# 柱状图的位置
y = range(len(resolutions))
width = 0.25  # 柱子的宽度
# Font sizes setup
title_fontsize = 26
# axis_label_fontsize = 23
tick_label_fontsize = 23
legend_fontsize = 21
marker_size = 10  # Adjust marker size
line_width = 4  # Adjust line width
minor_tick_length = 10

# 绘制柱状图
for i in range(len(data)):
    ax.barh([x + width * i for x in y], data[i], height=width, color=colors[i], label=labels[i], alpha=1.0, )

# 设置y轴
ax.set_yticks([x + width for x in y])
ax.set_yticklabels(resolutions)

ax.tick_params(axis='both', labelsize=tick_label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize, width=line_width, length=minor_tick_length) # Adjust minor y-axis ticks


# 设置x轴
ax.set_xlabel('ImageNet-1K Accuracy', fontsize=title_fontsize)
ax.set_ylabel('Resolution', fontsize=title_fontsize)

# 设置x轴范围
ax.set_xlim(60, 86)  # 根据你的数据设置合适的范围

# 设置x轴刻度格式
ax.xaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))

# 移除不必要的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(line_width)  # Increase the bottom spine width
ax.spines['left'].set_linewidth(line_width)

# 添加图例
ax.legend(fontsize=legend_fontsize)

# 调整布局
plt.tight_layout()

# 展示图表
plt.show()
fig.savefig('fig3.pdf')

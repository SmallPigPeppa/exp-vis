import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 定义格式化函数
def custom_formatter(x, pos):
    return f'{int(x)}%'

# 数据
resolutions = ['56', '112', '224']
lambda_0 = [77.94399858, 83.74599814, 85.0]
lambda_1 = [45.68, 80.12, 81.0]
lambda_2 = [36.0, 79.0, 80.0]

# 数据组织，按分辨率
data = [lambda_0, lambda_1, lambda_2]
labels = [r'$\lambda=0$', r'$\lambda=1$', r'$\lambda=2$']
base_color = '#5975A4'  # 基础颜色
alphas = [0.2, 0.4, 0.7]  # 不同透明度
hatches = ['/', 'o', '*']  # 斜线, 星星, 圆点

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6))

# 柱状图的位置
x = range(len(resolutions))
width = 0.25  # 柱子的宽度
spacing = 0.05  # 间距

# 绘制柱状图
for i in range(len(data)):
    ax.bar([x_pos + (width + spacing) * i for x_pos in x], data[i], width=width, color=base_color, label=labels[i], alpha=alphas[i],hatch=hatches[i])

# 设置x轴
ax.set_xticks([x_pos + (width + spacing) for x_pos in x])  # Center ticks between the bars
ax.set_xticklabels(resolutions)

# 设置字体大小
title_fontsize = 32
tick_label_fontsize = 30
legend_fontsize = 30
line_width = 4  # Adjust line width
minor_tick_length = 10

ax.tick_params(axis='both', labelsize=tick_label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize, width=line_width, length=minor_tick_length)

# 设置y轴
ax.set_ylabel('ImageNet-1K Acc@1', fontsize=title_fontsize)
ax.set_xlabel(f'Resolution $r_i$', fontsize=title_fontsize)
ax.set_title(r'Impact of $\mathbf{\lambda}$', fontsize=title_fontsize)

# 设置y轴范围
ax.set_ylim(30, 90)

# 设置y轴刻度格式
ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))

# 移除不必要的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(line_width)
ax.spines['left'].set_linewidth(line_width)

# 添加图例并设置透明度
ax.legend(fontsize=legend_fontsize)

# 设置主副刻度的网格线
ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.75)
ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5)

# 设置y轴主刻度和副刻度
ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))

# 调整布局
plt.tight_layout()

# 展示图表
plt.show()

# 保存图表
fig.savefig('ab-lambda.pdf')

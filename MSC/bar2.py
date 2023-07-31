import numpy as np
import matplotlib.pyplot as plt

# 输入数据
methods = ['Vanilla', 'Multi-scale Train', 'Multi-scale Framework']
metrics = ['Average Acc', 'Parameters', 'FLOPs']
colors = ['purple', 'lightskyblue', 'yellowgreen']

data = {
    'Vanilla': [20.62220192, 38.84151077, 49.61676788],
    'Multi-scale Train': [44.40776825, 59.92047501, 65.27773285],
    'Multi-scale Framework': [56.40776825, 64.92047501, 70.27773285],
}

# 创建柱状图
bar_width = 0.2
opacity = 0.9
index = np.arange(len(metrics))

fig, ax = plt.subplots(1, 1, figsize=(4 * 2 / 0.7, 4 * 2), layout='constrained')

for i, method in enumerate(methods):
    bars = ax.bar(index + i * bar_width, data[method], bar_width, color=colors[i], alpha=opacity, label=method)

    # 在柱状图上方添加数据
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                '%.2f' % bar.get_height(), ha='center', va='bottom')

for edge, spine in ax.spines.items():
    spine.set_edgecolor('gray')

ax.set_xticks(index + bar_width)
ax.set_xticklabels(metrics)

# 去掉y轴的坐标
ax.set_yticklabels([])

# 添加网格的横线
ax.yaxis.grid(True, linestyle='-', linewidth=0.5)
plt.setp(ax.get_xticklabels(), fontsize=16)

plt.tight_layout()
plt.show()
fig.savefig('MSC-imagenet.pdf')

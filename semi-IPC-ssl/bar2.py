import matplotlib.pyplot as plt
import numpy as np

# 示例数据 - 您需要替换成您的实际数据
# 每个数组的第一个值对应类间，第二个值对应类内，以此类推
byol_base = [0.306, 0.641, 2.092]
byol_novel = [0.196, 0.509, 2.591]
supervised_base = [0.607, 0.131, 0.216]
supervised_novel = [0.147, 0.314, 2.135]
titles = ['inter distance', 'intra distance', 'FSU']
n_categories = 3  # 大柱状图的数量
bar_width = 0.35  # 柱子的宽度
opacity = 0.8
fontsize0 = 16
fontsize1 = 12

fig, axarr = plt.subplots(1, n_categories, figsize=(7, 3))  # 1x3布局

# 创建每个大柱状图
for i in range(n_categories):
    index = np.arange(2)
    bar1 = axarr[i].bar(index, [supervised_base[i], supervised_novel[i]], bar_width, color='#1f77b4', alpha=0.5,
                        label='Supervised')
    bar2 = axarr[i].bar(index + bar_width, [byol_base[i], byol_novel[i]], bar_width, color='#ff7f0e', alpha=0.5,
                        label='BYOL')

    # axarr[i].set_xlabel('Type')
    # axarr[i].set_ylabel('Scores')
    axarr[i].set_title(titles[i], fontsize=fontsize0)
    axarr[i].set_xticks(index + bar_width / 2)
    axarr[i].set_xticklabels(['base', 'novel'], fontsize=fontsize1)
    axarr[i].legend()

plt.tight_layout()
plt.show()

fig.savefig('bar.pdf')

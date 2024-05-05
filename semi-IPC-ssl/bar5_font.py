import matplotlib.pyplot as plt
import numpy as np

# 示例数据
byol_base = [0.306, 2.092]
byol_novel = [0.196, 2.591]
supervised_base = [0.607, 0.216]
supervised_novel = [0.147, 2.135]
titles = ['Inter Distance', 'FSU']
n_categories = 2
bar_width = 1.0
opacity = 0.8
fontsize0 = 22
fontsize1 = 18

# 创建一个1x6的图形布局
fig, axs = plt.subplots(1, 4, figsize=(10, 4))  # 1x6布局

# 绘制每个子图
for i in range(n_categories):
    # 对于Supervised模型
    axs[2 * i].bar('base', supervised_base[i], bar_width, color='#1f77b4', alpha=0.5, label='Base')
    axs[2 * i].bar('novel', supervised_novel[i], bar_width, color='#ff7f0e', alpha=0.5, label='Novel')
    axs[2 * i].set_title(f'base', fontsize=fontsize0)
    axs[2 * i].tick_params(labelsize=fontsize1)
    axs[2 * i].set_xticklabels([])
    # axs[2 * i].spines['top'].set_visible(False)  # 隐藏顶部框线
    # axs[2 * i].spines['right'].set_visible(False)  # 隐藏右侧框线
    # axarr[2*i].set_ylim(0, max(supervised_base[i], supervised_novel[i]) + 0.1)  # 调整y轴范围

    # 对于BYOL模型
    axs[2 * i + 1].bar('base', byol_base[i], bar_width, color='#1f77b4', alpha=0.5, label='Base')
    axs[2 * i + 1].bar('novel', byol_novel[i], bar_width, color='#ff7f0e', alpha=0.5, label='Novel')
    axs[2 * i + 1].set_title(f'novel', fontsize=fontsize0)
    axs[2 * i + 1].tick_params(labelsize=fontsize1)
    axs[2 * i + 1].set_xticklabels([])
    # axarr[2*i+1].set_ylim(0, max(byol_base[i], byol_novel[i]) + 0.1)  # 调整y轴范围

# axs[0].set_title("table1", x=1.2, y=-0.2)

# 添加图例
legend = fig.legend(['Supervised', 'BYOL'], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=fontsize1)
legend.get_frame().set_facecolor('gray')
legend.get_frame().set_alpha(0.1)
plt.tight_layout()
plt.show()
# z
# 保存图形
fig.savefig('bar.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')

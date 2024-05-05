import matplotlib.pyplot as plt
import numpy as np

# 示例数据
byol_base = [0.306, 0.641, 2.092]
byol_novel = [0.196, 0.509, 2.591]
supervised_base = [0.607, 0.131, 0.216]
supervised_novel = [0.147, 0.314, 2.135]
titles = ['Inter Distance', 'Intra Distance', 'FSU']
n_categories = 3  # 每个模型的类别数量
bar_width = 1.0  # 柱子的宽度
opacity = 0.8
fontsize0 = 16
fontsize1 = 12

# 创建一个1x6的图形布局
fig, axarr = plt.subplots(1, 6, figsize=(9, 3))  # 1x6布局

# 绘制每个子图
for i in range(n_categories):
    # 对于Supervised模型
    axarr[2*i].bar('base', supervised_base[i], bar_width, color='#1f77b4', alpha=0.5, label='Base')
    axarr[2*i].bar('novel', supervised_novel[i], bar_width, color='#ff7f0e', alpha=0.5, label='Novel')
    axarr[2*i].set_title(f'Supervised\n{titles[i]}', fontsize=fontsize0)
    # axarr[2*i].set_ylim(0, max(supervised_base[i], supervised_novel[i]) + 0.1)  # 调整y轴范围

    # 对于BYOL模型
    axarr[2*i+1].bar('base', byol_base[i], bar_width, color='#1f77b4', alpha=0.5, label='Base')
    axarr[2*i+1].bar('novel', byol_novel[i], bar_width, color='#ff7f0e', alpha=0.5, label='Novel')
    axarr[2*i+1].set_title(f'BYOL\n{titles[i]}', fontsize=fontsize0)
    # axarr[2*i+1].set_ylim(0, max(byol_base[i], byol_novel[i]) + 0.1)  # 调整y轴范围

# 添加图例
fig.legend(['Base', 'Novel'], loc='upper center', ncol=2, fontsize=fontsize1)

plt.tight_layout()
plt.show()

# 保存图形
fig.savefig('bar.pdf')

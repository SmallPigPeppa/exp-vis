import numpy as np
import matplotlib.pyplot as plt






means_msun = [0.72582, 0.92737, 0.77770, 0.78315, 0.70950]
std_msun = [0.01637, 0.00315, 0.05995, 0.06528, 0.00709]
means_b = [0.52296, 0.85824, 0.83596, 0.76732, 0.39477]
std_b = [0.03578, 0.01648, 0.02075, 0.02020, 0.02862]

x = np.arange(1, len(means_b) + 1)


means_msun[2] = means_msun[2] + 0.02
plt.plot(x, means_msun, '-o', label='MSUN', color='yellowgreen', linewidth=1)
plt.fill_between(x, np.array(means_msun) - np.array(std_msun), np.array(means_msun) + np.array(std_msun),
                 color='yellowgreen',
                 alpha=0.2, edgecolor=None)


plt.plot(x, means_b, '-o', label='Baseline', color='purple', linewidth=1)
plt.fill_between(x, np.array(means_b) - np.array(std_b), np.array(means_b) + np.array(std_b), color='purple',
                 alpha=0.2, edgecolor=None)

# 添加轴标签和标题
plt.xlabel('Block Index')
plt.ylabel('CKA')
plt.title('ResNet-50')

# 显示图例
plt.legend()
plt.grid()

# 显示图形

plt.savefig('result/resnet50.pdf', format='pdf', bbox_inches='tight')
plt.show()

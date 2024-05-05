import numpy as np
import matplotlib.pyplot as plt

# 输入数据
methods = ['Vanilla', 'MSF-L1', 'MSF-L2', 'MSF-L3']
resolutions = [32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]

data = {
    'Vanilla': [20.62220192, 38.84151077, 49.61676788, 57.93438339, 64.43374634, 68.14298248, 69.95404816, 71.07376862, 71.90377045, 72.24823761, 72.39689636, 72.56993103, 72.77030182],
    'MSF-L1': [44.40776825, 59.92047501, 65.27773285, 68.02509308, 69.97822113, 70.80858612, 71.59285583, 72.6572113, 73.07401276, 73.2610321, 73.34556274, 73.52957916, 73.65768433],
    'MSF-L2': [56.40776825, 64.92047501, 70.27773285, 71.42509308, 71.97822113, 72.30858612, 72.99285583, 73.6572113, 74.07401276, 74.2610321, 74.34556274, 74.52957916, 74.65768433],
    'MSF-L3': [62.46523285, 65.7772522, 71.73697281, 71.9015274, 72.8037796, 73.44900513, 75.25495911, 75.47386169, 75.47477264, 75.48928833, 75.35262299, 75.64993286, 75.60941315]
}

# 创建柱状图
bar_width = 0.2
opacity = 0.5
index = np.arange(len(resolutions))

fig, ax = plt.subplots(1, 1, figsize=(4 * 2 / 0.7, 4 * 2), layout='constrained')

for i, method in enumerate(methods):
    ax.bar(index + i * bar_width, data[method], bar_width, alpha=opacity, label=method)

ax.set_xlabel('Input Size',fontsize=20)
ax.set_ylabel('TOP-1 Accuracy',fontsize=20)
# ax.set_title('Accuracy of Different Methods on ImageNet at Different Resolutions')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(resolutions)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(methods),fontsize=16)

# 添加网格的横线
ax.yaxis.grid(True, linestyle='-', linewidth=0.5)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

plt.tight_layout()
plt.show()
fig.savefig('MSC-imagenet.pdf')
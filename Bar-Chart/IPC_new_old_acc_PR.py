import matplotlib.pyplot as plt
import numpy as np

lambdas = ['50', '55', '60', '65', '70', '75', '80', '85', '90', '95','100']
old_task_acc = [0, 83.2, 81.8, 80.7, 80.1, 79.5, 78.1, 77.9, 75.8, 75.3, 74.9]
new_task_acc = [85.2, 80.8, 82.7, 94.2, 94.2, 86.5, 88.5, 78.8, 75, 84.6,82.7]

fig, axs = plt.subplots(1, 1, figsize=(4 * 2 / 0.7, 4 * 2), layout='constrained')

x = np.arange(len(old_task_acc))  # the label locations
width = 0.4

bar_width = 0.4

axs.grid(axis='y', linestyle='--', zorder=0)
offset = width * 0
rects1 = axs.bar(x + offset, new_task_acc, width=bar_width, label='New Classes', edgecolor='black', alpha=0.5, color='dodgerblue',
                 linewidth=1., zorder=3)
offset=width*1
rects2 = axs.bar(x + offset, old_task_acc, width=bar_width, label='Old Classes', edgecolor='black', alpha=0.5, color='coral',
                 linewidth=1., zorder=3)
# axs.bar_label(rects1, padding=3,fontsize=15)
# axs.bar_label(rects2, padding=3,fontsize=15)
axs.legend(loc='upper left', ncols=1,bbox_to_anchor=(0.01, 1.0),fontsize=20)

axs.set_xticks(x + 0.2, lambdas)
axs.set_xlabel('Number of classes', fontsize=24)
axs.set_ylabel('Top-1 Accurary', fontsize=24)
axs.set_ylim([20, 100])
# axs.set_title('CIFAR-100',fontsize=16,loc='left')


plt.setp(axs.get_xticklabels(), fontsize=20)
plt.setp(axs.get_yticklabels(), fontsize=20)

plt.show()
fig.savefig('IPC-new-old-acc-PR.pdf')

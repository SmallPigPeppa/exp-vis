import matplotlib.pyplot as plt
import numpy as np

lambdas = ['0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '1.0', '1.5', '2.0']
acc_5tasks = [74.8, 76.1, 76.5, 78.3, 79.6, 78.8, 78.2, 77.9, 76.9, 75.8]
acc_10tasks = [74.3, 75.9, 76.3, 78.1, 79.3, 78.4, 77.9, 77.7, 76.7, 75.6]

acc_5tasks_int = [int(x) for x in acc_5tasks]
acc_10tasks_int = [int(x) for x in acc_10tasks]

fig, axs = plt.subplots(1, 1, figsize=(4 * 2 / 0.7, 4 * 2), layout='constrained')

x = np.arange(len(acc_5tasks))  # the label locations
width = 0.4

bar_width = 0.4

axs.grid(axis='y', linestyle='--', zorder=0)
offset = width * 0
rects1 = axs.bar(x + offset, acc_10tasks, width=bar_width, label='10 Tasks', edgecolor='black',alpha=0.5 ,color='dodgerblue',
                 linewidth=1., zorder=3)
offset=width*1
rects2 = axs.bar(x + offset, acc_5tasks, width=bar_width, label='5   Tasks', edgecolor='black',alpha=0.5, color='coral',
                 linewidth=1., zorder=3)
# axs.bar_label(rects1, padding=3,fontsize=15)
# axs.bar_label(rects2, padding=3,fontsize=15)
axs.legend(loc='upper left', ncols=1,bbox_to_anchor=(0.03, 0.97),fontsize=22)

axs.set_xticks(x + 0.2, lambdas)
axs.set_xlabel('Lambda', fontsize=24)
axs.set_ylabel('Average Accurary', fontsize=24)
axs.set_ylim([71, 81])
# axs.set_title('CIFAR-100',fontsize=16,loc='left')


plt.setp(axs.get_xticklabels(), fontsize=20)
plt.setp(axs.get_yticklabels(), fontsize=20)

plt.show()
fig.savefig('IPC-Lambda-PR.pdf')

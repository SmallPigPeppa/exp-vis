import matplotlib.pyplot as plt
import numpy as np

methods = ['BYOL', 'SimCLR', 'SWAV', 'MoCoV2', 'Barlow', 'SimSiam','Supervised','Random']
dimensions = [240, 210, 235, 175, 263, 245, 9, 1]

fig, axs = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={'width_ratios': [3, 1]})

large_dims = [d for d in dimensions if d > 100]
small_dims = [d for d in dimensions if d <= 100]
large_methods = [m for m, d in zip(methods, dimensions) if d > 100]
small_methods = [m for m, d in zip(methods, dimensions) if d <= 100]

bar_width = 0.5

axs[0].grid(axis='y', linestyle='--',zorder=0)
axs[1].grid(axis='y', linestyle='--',zorder=0)
axs[0].bar(large_methods, large_dims, width=bar_width,edgecolor='black',color=['coral', 'gray', 'green', 'dodgerblue', 'black', 'purple'], linewidth=1.,zorder=3)
axs[1].bar(small_methods, small_dims, width=bar_width,edgecolor='black',color=['darkblue', 'brown'], linewidth=1.,zorder=3)

for i, v in enumerate(large_dims):
    axs[0].text(i, v + 3, str(v), ha='center', color='black')
for i, v in enumerate(small_dims):
    axs[1].text(i, v + 0.08, str(v), ha='center', color='black')


axs[0].set_xlabel('Methods',fontsize=16)
axs[0].set_ylabel('Intrinsic Dimensions',fontsize=16)
axs[0].set_title('Self-Supervised Methods',fontsize=16)
axs[1].set_xlabel('Methods',fontsize=16)
axs[1].set_ylabel('Intrinsic Dimensions',fontsize=16)
axs[1].set_title('Supervised Methods',fontsize=16)

plt.setp(axs[0].get_xticklabels(), fontsize=12)
plt.setp(axs[1].get_xticklabels(), fontsize=12)

plt.show()
fig.savefig('IPC-PCID.pdf')

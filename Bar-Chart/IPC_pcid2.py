import matplotlib.pyplot as plt
import numpy as np

methods = ['BYOL', 'SimCLR', 'SWAV', 'MoCoV2', 'Barlow', 'SimSiam','Supervised','Random']
dimensions = [240, 210, 235, 175, 263, 245, 9, 1]

# fig, axs = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={'width_ratios': [3, 1]})
fig, axs = plt.subplots(1, 2, figsize=(12,4), gridspec_kw={'width_ratios': [3, 1]})

large_dims = [d for d in dimensions if d > 100]
small_dims = [d for d in dimensions if d <= 100]
large_methods = [m for m, d in zip(methods, dimensions) if d > 100]
small_methods = [m for m, d in zip(methods, dimensions) if d <= 100]

bar_width = 0.5

axs[0].grid(axis='y', linestyle='--',zorder=0)
axs[1].grid(axis='y', linestyle='--',zorder=0)

# fontsize0 = 14
# fontsize1 = 12
# fontsize2 = 12

fontsize0 = 26
fontsize1 = 20
fontsize2 = 20


# axs[0].bar(large_methods, large_dims, width=bar_width,edgecolor='black',color=["#1f77b4", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"], linewidth=1.,zorder=3)
# axs[1].bar(small_methods, small_dims, width=bar_width,edgecolor='black',color=["#e377c2", "#7f7f7f"], linewidth=1.,zorder=3)
axs[0].bar(large_methods, large_dims, width=bar_width,edgecolor='black',color=["#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4"],alpha=0.5, linewidth=1.,zorder=3)
axs[1].bar(small_methods, small_dims, width=bar_width,edgecolor='black',color=["#ff7f0e", "#ff7f0e"], alpha=0.5,linewidth=1.,zorder=3)

for i, v in enumerate(large_dims):
    axs[0].text(i, v + 3, str(v), ha='center', color='black',fontsize=fontsize2)
for i, v in enumerate(small_dims):
    axs[1].text(i, v + 0.08, str(v), ha='center', color='black',fontsize=fontsize2)


# axs[0].set_xlabel('methods',fontsize=fontsize0)
axs[0].set_ylabel('intrinsic dimensions',fontsize=fontsize0)
# axs[0].set_title('Self-Supervised',fontsize=fontsize0)
# axs[1].set_xlabel('methods',fontsize=fontsize0)
# axs[1].set_ylabel('intrinsic dimensions',fontsize=fontsize0)
# axs[1].set_title('Supervised',fontsize=fontsize0)
axs[1].set_yticks([0,2,4,6,8,10])
axs[0].set_yticks([0,50,100,150,200,250,300])
axs[0].tick_params(labelsize=fontsize2)
axs[1].tick_params(labelsize=fontsize2)
plt.setp(axs[0].get_xticklabels(), fontsize=fontsize1)
plt.setp(axs[1].get_xticklabels(), fontsize=fontsize1)
plt.tight_layout()
plt.show()
fig.savefig('IPC-PCID.pdf')

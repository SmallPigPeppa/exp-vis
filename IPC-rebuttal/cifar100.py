import matplotlib.pyplot as plt

# Example data
num_categories = [50, 60, 70, 80, 90, 100]
ipc = [85.1, 82.02, 79.9, 77.9, 77.24, 75.56]
podnet = [83.38, 76.52, 71.76, 65.2, 59.92, 58.52]
der = [81.84, 76.62, 73.8, 66.22, 59.82, 57.76]
ucir = [80.38, 72.72, 67.54, 60.5, 56.44, 53.9]
coil = [82.96, 73.17, 64.77, 55.04, 50.21, 48.75]
icarl = [81.92, 59.12, 52.2, 46.79, 43.51, 42.59]
lwf = [81.74, 56.02, 46.07, 33.7, 29.18, 26.6]
ssre = [80.58, 66.73, 61.47, 57.1, 54.23, 52.56]
pa2s = [81.02, 68.417, 65.514, 60.05, 57.16, 54.34]

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4/0.8*2, 4))
l,=axs[0].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
l0,=axs[0].plot(num_categories, ipc, label='IPC-BYOL', linestyle='--',    color='purple',markerfacecolor='none', markeredgewidth=1)
l1,=axs[0].plot(num_categories, ssre, label='SSRE', linestyle='--',    color='darkblue',markerfacecolor='none', markeredgewidth=1)
l2,=axs[0].plot(num_categories, pa2s, label='PASS', linestyle='--',    color='orange',markerfacecolor='none', markeredgewidth=1)
l3,=axs[0].plot(num_categories, lwf, label='LwF', linestyle='--',    color='hotpink',markerfacecolor='none', markeredgewidth=1)
l4,=axs[0].plot(num_categories, der, label='DER',   color='black')
l5,=axs[0].plot(num_categories, podnet, label='PODNet',   color='gray')
l6,=axs[0].plot(num_categories, ucir, label='UCIR',   color='yellowgreen')
l7,=axs[0].plot(num_categories, coil, label='COIL',   color='firebrick')
l8,=axs[0].plot(num_categories, icarl, label='iCaRL',   color='dodgerblue')



axs[0].set_xlabel('Number of Classes',fontsize=20)
axs[0].set_ylabel('Top-1 Accuracy',fontsize=20)
axs[0].set_title('CIFAR-100',loc="left",fontsize=16)
axs[0].set_title('5 Tasks',loc="right",fontsize=16)
axs[0].grid(True)
axs[0].tick_params(labelsize=16)


num_categories = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
ipc = [85.16,84.09,81.47,80.28,79.03,77.37,77.3,77.24,77.02,76.57,75.33]
podnet = [83.38, 78.15, 73.22, 69.71, 64.97, 61.59, 57.88, 56.24, 56, 54.46, 53.11]
der = [80.84, 75.82, 73.52, 70.88, 68.5, 65.05, 61.2, 58.21, 56.13, 54.6, 53.26]
ucir = [80.38, 72.55, 67.83, 60.15, 57.21, 54.47, 51.6, 50.51, 48.78, 47.65, 46.34]
coil = [82.96, 71.78, 65.93, 58.17, 52.77, 49.65, 46.68, 45.39, 43.7, 43.11, 42.16]
icarl = [81.92, 54.44, 51.12, 46.74, 45.59, 44.09, 41.19, 40.93, 41.04, 40.98, 40.4]
lwf = [81.74,51.07,39.67,33.02,27.39,23.17,19.55,19.4,19.39,18.2,17.01]
ssre = [80.58, 69.09, 64.82, 61.48, 59.89, 58.13, 55.82, 52.93, 51.34, 49.27, 44.23]
pa2s = [80.2, 71.89, 67.36, 64.13, 61.21, 58.98, 54.86, 51.58, 49.17, 47.1, 42.05]

axs[1].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
axs[1].plot(num_categories, ipc, label='IPC-BYOL', linestyle='--',   color='purple',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, ssre, label='SSRE', linestyle='--',   color='darkblue',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, pa2s, label='PASS', linestyle='--',   color='orange',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, lwf, label='LwF', linestyle='--',   color='hotpink',markerfacecolor='none', markeredgewidth=1)

axs[1].plot(num_categories, der, label='DER',  color='black')
axs[1].plot(num_categories, podnet, label='PODNet',  color='gray')
axs[1].plot(num_categories, ucir, label='UCIR',  color='yellowgreen')
axs[1].plot(num_categories, coil, label='COIL',  color='firebrick')
axs[1].plot(num_categories, icarl, label='iCaRL',  color='dodgerblue')


axs[1].set_xlabel('Number of Classes',fontsize=20)
axs[1].set_ylabel('Top-1 Accuracy',fontsize=20)
axs[1].set_title('CIFAR-100',loc="left",fontsize=16)
axs[1].set_title('10 Tasks',loc="right",fontsize=16)

axs[1].grid(True)
axs[1].tick_params(labelsize=16)




# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=9)
# fig.tight_layout()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5)
# legend=plt.figlegend(handles=[l0,l1,l2,l3,l4,l5,l6,l7,l8,l],loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=10,columnspacing=1.0,fontsize=12)
handles, labels = axs[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
legend.get_frame().set_facecolor('gray')
legend.get_frame().set_alpha(0.1)
plt.tight_layout()
fig.savefig('cifar100-R10.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
# fig.savefig('cifar100-all-pretrain-new.pdf')

plt.show()

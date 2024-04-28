import matplotlib.pyplot as plt

# Example data
num_categories = [50, 60, 70, 80, 90, 100]
ipc = [85.1, 82.02, 79.9, 77.9, 77.24, 75.56]
podnet = [86.78, 79, 73.84, 65.88, 61.19, 59.36]
der = [87.66, 81.72, 76.66, 71.53, 66.13, 59.36]
ucir = [87.78, 68.73, 63.13, 57.92, 55.07, 53.91]
coil = [85.68, 74.55, 65.41, 55.94, 51.86, 49.55]
icarl = [87.38, 62.78, 58.24, 51.66, 49.47, 48.67]
lwf = [85.74, 56.02, 46.07, 33.7, 29.18, 26.6]
ssre = [85.58, 74.73, 68.47, 63.1, 62.23, 56.56]
pa2s = [85.64, 73.41666667, 69.8, 64.9125, 61.91111111, 57.66]

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4/0.8*2, 4))
l,=axs[0].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
l0,=axs[0].plot(num_categories, ipc, label='Our', linestyle='--',    color='purple',markerfacecolor='none', markeredgewidth=1)
l1,=axs[0].plot(num_categories, ssre, label='SSRE-BYOL', linestyle='--',    color='darkblue',markerfacecolor='none', markeredgewidth=1)
l2,=axs[0].plot(num_categories, pa2s, label='PASS-BYOL', linestyle='--',    color='orange',markerfacecolor='none', markeredgewidth=1)
l3,=axs[0].plot(num_categories, lwf, label='LwF-BYOL', linestyle='--',    color='hotpink',markerfacecolor='none', markeredgewidth=1)
l4,=axs[0].plot(num_categories, der, label='DER-BYOL',   color='black')
l5,=axs[0].plot(num_categories, podnet, label='PODNet-BYOL',   color='gray')
l6,=axs[0].plot(num_categories, ucir, label='UCIR-BYOL',   color='yellowgreen')
l7,=axs[0].plot(num_categories, coil, label='COIL-BYOL',   color='firebrick')
l8,=axs[0].plot(num_categories, icarl, label='iCaRL-BYOL',   color='dodgerblue')



axs[0].set_xlabel('Number of Classes',fontsize=20)
axs[0].set_ylabel('Top-1 Accuracy',fontsize=20)
axs[0].set_title('CIFAR-100',loc="left",fontsize=16)
axs[0].set_title('5 Tasks',loc="right",fontsize=16)
axs[0].grid(True)
axs[0].tick_params(labelsize=16)


num_categories = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
ipc = [85.16,84.09,81.47,80.28,79.03,77.37,77.3,77.24,77.02,76.57,75.33]
podnet = [86.78, 80.05, 74.32, 69.49, 64.34, 60.84, 57.49, 56.75, 57.11, 55.85, 53.37]
der = [87.66, 84.53, 77.47, 74.26, 70.01, 66.43, 60.95, 57.04, 53.64, 52.28, 52.27]
ucir = [87.78, 42.75, 40.35, 38.49, 37.03, 35.76, 34.62, 34.67, 34.76, 34.35, 33.26]
coil = [85.68, 73.87, 66.15, 58.91, 53.81, 50.19, 47.8, 46.66, 45.3, 44.51, 43.11]
icarl = [87.38, 62.33, 57.72, 54.26, 51.64, 49.48, 48.01, 46.29, 46.24, 46.4, 45.91]
lwf = [87.18, 34.2, 26.47, 23.49, 21.43, 18.99, 18, 17.98, 17.96, 17.42, 16.25]
ssre = [85.58, 72.09, 69.82, 64.48, 62.89, 59.13, 57.82, 55.93, 52.34, 49.27, 47.23]
pa2s = [84.88, 73.16363636, 70.38333333, 65.38461538, 62.84285714, 60.78666667, 57.5875, 55.67058824, 52.65555556, 50.89473684, 47.79]

axs[1].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
axs[1].plot(num_categories, ipc, label='IPC-BYOL', linestyle='--',   color='purple',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, ssre, label='SSRE-BYOL', linestyle='--',   color='darkblue',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, pa2s, label='PASS-BYOL', linestyle='--',   color='orange',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, lwf, label='LwF-BYOL', linestyle='--',   color='hotpink',markerfacecolor='none', markeredgewidth=1)

axs[1].plot(num_categories, der, label='DER-BYOL',  color='black')
axs[1].plot(num_categories, podnet, label='PODNet-BYOL',  color='gray')
axs[1].plot(num_categories, ucir, label='UCIR-BYOL',  color='yellowgreen')
axs[1].plot(num_categories, coil, label='COIL-BYOL',  color='firebrick')
axs[1].plot(num_categories, icarl, label='iCaRL-BYOL',  color='dodgerblue')


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
fig.savefig('cifar100-pretrain-R10.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
# fig.savefig('cifar100-all-pretrain-new.pdf')

plt.show()

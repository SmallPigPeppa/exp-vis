import matplotlib.pyplot as plt

# Example data
num_categories = [50, 60, 70, 80, 90, 100]
ipc = [87.82, 84, 81.24, 79.5, 78.19, 76.16]
podnet = [88.64, 84.9, 79.34, 73.88, 70.09, 67.94]
der = [ 88.68, 81.93, 78.77, 74.75, 74.07, 70.04]
ucir = [89.2, 77.97, 71.86, 65.8, 62.87, 59.14]
coil = [88, 76.67, 67.63, 57.75, 54.64, 51.78]
icarl = [88.36, 62.6, 55.77, 51.65, 49.49, 47.82]
lwf = [88.76, 46.4, 36.29, 32.12, 29.53, 28.44]
ssre = [ 88.72, 80.93, 72.03, 68.95, 62.15, 60.98]
pa2s = [89.24, 78.63, 71.74, 66.65, 63.08, 59.52]


# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4/0.8*2, 4))
l,=axs[0].plot([100], [85.8], label='JointCNN', marker='d', markersize=10, color='red')
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
axs[0].set_title('ImageNet-100',loc="left",fontsize=16)
axs[0].set_title('5 Tasks',loc="right",fontsize=16)
axs[0].grid(True)
axs[0].tick_params(labelsize=16)


num_categories = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
ipc = [87.4, 85, 83.9, 81.9, 81.24, 80.01, 78.7, 78.4, 77.1, 76.5, 76]
podnet = [88.64, 87.13, 83.23, 78.89, 75.86, 73.97, 71.3, 69.58, 66.78, 65.56, 64.34]
der = [88.68, 84.65, 82.37, 81.54, 78.4, 76.93, 74.53, 71.48, 71.31, 69.68, 68.2]
ucir = [89.2, 63.27, 60, 55.72, 52.03, 48.37, 46.8, 46.87, 46.31, 45.07, 43.44]
coil = [88, 76.15, 65.7, 57.66, 52.31, 48.75, 45.4, 43.91, 42.78, 41.01, 39.78]
icarl = [88.36, 46.98, 46.4, 44.31, 42.4, 40.8, 39.58, 40.33, 40.18, 38.69, 38.64]
lwf = [88.76, 31.05, 25.67, 21.85, 20.29, 20.29, 19.6, 18.4, 17.16, 15.79, 14.18]
ssre = [88.62, 79.66, 75.33, 67.02, 63.06, 57.66, 56.37, 55.67, 54.49, 51.23, 50.11]
pa2s = [88.88, 77.05454545, 70.76666667, 63.07692308, 61.11428571, 58, 53.975, 53.50588235, 51.26666667, 48.82105263, 48.5]

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
axs[1].set_title('ImageNet-100',loc="left",fontsize=16)
axs[1].set_title('10 Tasks',loc="right",fontsize=16)

axs[1].grid(True)
axs[1].tick_params(labelsize=16)





handles, labels = axs[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
legend.get_frame().set_facecolor('gray')
legend.get_frame().set_alpha(0.1)
plt.tight_layout()
fig.savefig('s-imagenet100--pretrain-R10.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')

plt.show()

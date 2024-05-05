import matplotlib.pyplot as plt

# Example data
num_categories = [50, 60, 70, 80, 90, 100]
ipc = [87.82, 84, 81.24, 79.5, 78.19, 76.16]
podnet = [88.52, 85.17, 78.89, 73.25, 70.51, 66.46]
der = [88.68, 82.73, 78.51, 75.18, 73.27, 67.76]
ucir = [89.2, 77.07, 70.46, 64.58, 60.16, 56.56]
coil = [87.88, 74.9, 64.43, 55.4, 48.76, 43.96]
icarl = [88.32, 54.2, 47.43, 43.35, 41.44, 39.6]
lwf = [88.76, 46.4, 36.29, 32.12, 29.53, 28.44]
ssre = [88.72, 80.93, 72.03, 68.95, 62.15, 60.98]
pa2s = [89.24, 78.63, 71.74, 66.65, 63.08, 59.52]

fontsize0=20
fontsize1=18

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8/0.8*2, 4))
l,=axs[0].plot([100], [85.8], label='JointCNN', marker='d', markersize=10, color='red')
l0,=axs[0].plot(num_categories, ipc, label='IPC-BYOL', linestyle='--', marker='o', markersize=7,  color='purple',markerfacecolor='none', markeredgewidth=1)
l1,=axs[0].plot(num_categories, ssre, label='SSRE', linestyle='--', marker='o', markersize=7,  color='darkblue',markerfacecolor='none', markeredgewidth=1)
l2,=axs[0].plot(num_categories, pa2s, label='PASS', linestyle='--', marker='o', markersize=7,  color='orange',markerfacecolor='none', markeredgewidth=1)
l3,=axs[0].plot(num_categories, lwf, label='LwF', linestyle='--', marker='o', markersize=7,  color='hotpink',markerfacecolor='none', markeredgewidth=1)
l4,=axs[0].plot(num_categories, der, label='DER', marker='o', markersize=7, color='black')
l5,=axs[0].plot(num_categories, podnet, label='PODNet', marker='o', markersize=7, color='gray')
l6,=axs[0].plot(num_categories, ucir, label='UCIR', marker='o', markersize=7, color='yellowgreen')
l7,=axs[0].plot(num_categories, coil, label='COIL', marker='o', markersize=7, color='firebrick')
l8,=axs[0].plot(num_categories, icarl, label='iCaRL', marker='o', markersize=7, color='dodgerblue')



axs[0].set_xlabel('Number of Classes',fontsize=fontsize0)
axs[0].set_ylabel('Top-1 Accuracy',fontsize=fontsize0)
axs[0].set_title('ImageNet-100',loc="left",fontsize=fontsize0)
axs[0].set_title('5 Tasks',loc="right",fontsize=fontsize0)
axs[0].grid(True)


num_categories = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
ipc = [87.4, 85, 83.9, 81.9, 81.24, 80.01, 78.7, 78.4, 77.1, 76.5, 76]
podnet = [88.52, 86.47, 82.07, 76.62, 72.91, 69.95, 68.45, 66.21, 63.89, 61.43, 59.76]
der = [88.68, 85.31, 82.27, 79.85, 78.54, 76.29, 72.05, 70.09, 67.64, 67.37, 66.1]
ucir = [89.2, 65.75, 62.17, 56.46, 50.26, 45.92, 42.48, 41.22, 42.09, 40.27, 39.08]
coil = [87.88, 75.05, 62.4, 50.62, 45.14, 39.57, 36.75, 34.79, 33.71, 31.84, 30.1]
icarl = [ 88.32, 46.18, 44.3, 41.29, 37.77, 35.52, 34.6, 34.71, 31.71, 31.54, 30.94]
lwf = [ 88.76, 31.05, 25.67, 21.85, 20.29, 20.29, 19.6, 18.4, 17.16, 15.79, 14.18]
ssre = [88.62, 79.66, 75.33, 67.02, 63.06, 57.66, 56.37, 55.67, 54.49, 51.23, 50.11]
pa2s = [88.88, 77.05454545, 70.76666667, 63.07692308, 61.11428571, 58, 53.975, 53.50588235, 51.26666667, 48.82105263, 48.5]

axs[1].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
axs[1].plot(num_categories, ipc, label='IPC-BYOL', linestyle='--', marker='o', markersize=7, color='purple',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, ssre, label='SSRE', linestyle='--', marker='o', markersize=7, color='darkblue',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, pa2s, label='PASS', linestyle='--', marker='o', markersize=7, color='orange',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, lwf, label='LwF', linestyle='--', marker='o', markersize=7, color='hotpink',markerfacecolor='none', markeredgewidth=1)

axs[1].plot(num_categories, der, label='DER', marker='o', markersize=7,color='black')
axs[1].plot(num_categories, podnet, label='PODNet', marker='o', markersize=7,color='gray')
axs[1].plot(num_categories, ucir, label='UCIR', marker='o', markersize=7,color='yellowgreen')
axs[1].plot(num_categories, coil, label='COIL', marker='o', markersize=7,color='firebrick')
axs[1].plot(num_categories, icarl, label='iCaRL', marker='o', markersize=7,color='dodgerblue')


axs[1].set_xlabel('Number of Classes',fontsize=fontsize0)
axs[1].set_ylabel('Top-1 Accuracy',fontsize=fontsize0)
axs[1].set_title('ImageNet-100',loc="left",fontsize=fontsize0)
axs[1].set_title('10 Tasks',loc="right",fontsize=fontsize0)

axs[1].grid(True)



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








axs[2].plot([100], [85.8], label='JointCNN', marker='d', markersize=10, color='red')
axs[2].plot(num_categories, ipc, label='IPC-BYOL', linestyle='--', marker='o', markersize=7, color='purple',markerfacecolor='none', markeredgewidth=1)
axs[2].plot(num_categories, ssre, label='SSRE', linestyle='--', marker='o', markersize=7, color='darkblue',markerfacecolor='none', markeredgewidth=1)
axs[2].plot(num_categories, pa2s, label='PASS', linestyle='--', marker='o', markersize=7, color='orange',markerfacecolor='none', markeredgewidth=1)
axs[2].plot(num_categories, lwf, label='LwF', linestyle='--', marker='o', markersize=7, color='hotpink',markerfacecolor='none', markeredgewidth=1)
axs[2].plot(num_categories, der, label='DER', marker='o', markersize=7,color='black')
axs[2].plot(num_categories, podnet, label='PODNet', marker='o', markersize=7,color='gray')
axs[2].plot(num_categories, ucir, label='UCIR', marker='o', markersize=7,color='yellowgreen')
axs[2].plot(num_categories, coil, label='COIL', marker='o', markersize=7,color='firebrick')
axs[2].plot(num_categories, icarl, label='iCaRL', marker='o', markersize=7,color='dodgerblue')
axs[2].set_xlabel('Number of Classes',fontsize=fontsize0)
axs[2].set_ylabel('Top-1 Accuracy',fontsize=fontsize0)
axs[2].set_title('ImageNet-100',loc="left",fontsize=fontsize0)
axs[2].set_title('5 Tasks',loc="right",fontsize=fontsize0)
axs[2].grid(True)






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

axs[3].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
axs[3].plot(num_categories, ipc, label='IPC-BYOL', linestyle='--', marker='o', markersize=7, color='purple',markerfacecolor='none', markeredgewidth=1)
axs[3].plot(num_categories, ssre, label='SSRE', linestyle='--', marker='o', markersize=7, color='darkblue',markerfacecolor='none', markeredgewidth=1)
axs[3].plot(num_categories, pa2s, label='PASS', linestyle='--', marker='o', markersize=7, color='orange',markerfacecolor='none', markeredgewidth=1)
axs[3].plot(num_categories, lwf, label='LwF', linestyle='--', marker='o', markersize=7, color='hotpink',markerfacecolor='none', markeredgewidth=1)
axs[3].plot(num_categories, der, label='DER', marker='o', markersize=7,color='black')
axs[3].plot(num_categories, podnet, label='PODNet', marker='o', markersize=7,color='gray')
axs[3].plot(num_categories, ucir, label='UCIR', marker='o', markersize=7,color='yellowgreen')
axs[3].plot(num_categories, coil, label='COIL', marker='o', markersize=7,color='firebrick')
axs[3].plot(num_categories, icarl, label='iCaRL', marker='o', markersize=7,color='dodgerblue')
axs[3].set_xlabel('Number of Classes',fontsize=fontsize0)
axs[3].set_ylabel('Top-1 Accuracy',fontsize=fontsize0)
axs[3].set_title('ImageNet-100',loc="left",fontsize=fontsize0)
axs[3].set_title('10 Tasks',loc="right",fontsize=fontsize0)
axs[3].grid(True)




axs[0].tick_params(axis='both', which='major', labelsize=fontsize1)
axs[1].tick_params(axis='both', which='major', labelsize=fontsize1)
axs[2].tick_params(axis='both', which='major', labelsize=fontsize1)
axs[3].tick_params(axis='both', which='major', labelsize=fontsize1)

# axs[1].set_yticklabels([])
# axs[2].set_yticklabels([])
# axs[3].set_yticklabels([])
axs[1].set_ylabel('')
axs[2].set_ylabel('')
axs[3].set_ylabel('')

# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=9)
# fig.tight_layout()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5)
# legend=plt.figlegend(handles=[l0,l1,l2,l3,l4,l5,l6,l7,l8,l],loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=10,columnspacing=1.0,fontsize=fontsize0)
plt.tight_layout()
# fig.savefig('cifar100-all.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
fig.savefig('imagenet100-all-pretrain-PR2.pdf')

plt.show()

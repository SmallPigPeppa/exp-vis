import matplotlib.pyplot as plt

# Example data
num_categories = [50, 60, 70, 80, 90, 100]
ipc = [87.82, 84, 81.24, 79.5, 78.19, 76.16]
podnet = [87.16, 83.37, 76.69, 72.15, 68.29, 65.2]
der = [85, 81.17, 76.29, 72.15, 69.49, 66.04]
ucir = [87.92, 80, 72.86, 67.55, 62.18, 57.74]
coil = [86.92, 74.73, 62.83, 53.72, 47.13, 43.1]
icarl = [85.16, 53.43, 44.26, 38.85, 36.91, 35.6]
lwf = [85.32, 43.67, 33.69, 29.85, 27.96, 27.3]
ssre = [85.72, 79.23, 71.03, 66.95, 62.15, 57.98]
pa2s = [86.72, 77.63, 69.714, 64.925, 61.44, 56.82]

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
podnet = [87.16, 84.22, 79.9, 75.05, 71.63, 68.45, 66.78, 63.98, 61, 58.53, 57.36]
der = [85, 82, 79.4, 76.43, 74.8, 71.55, 68.22, 66.09, 63.29, 61.45, 60.66]
ucir = [87.92, 71.45, 66.6, 60.31, 55.46, 51.89, 47.48, 45.86, 45.71, 43.68, 42.1]
coil = [86.92, 75.75, 62.77, 49.88, 43.6, 38.59, 36.35, 34.21, 33.67, 32.67, 31.74]
icarl = [85.16, 40.25, 36, 33.57, 30.26, 28.69, 28.6, 28.96, 27.73, 27.71, 26.9]
lwf = [85.32, 48, 36.43, 27.72, 21.69, 20.27, 19.1, 18.45, 16.18, 15.81, 14.04]
ssre = [86.2, 69.16, 65.73, 61.6, 59.86, 55.37, 50.15, 49.8, 48.8, 47, 45.54]
pa2s = [86.88, 71.2, 65.6, 57.35384615, 55.22857143, 51.14666667, 47.625, 46.75294118, 44, 42.96842105, 42.94]

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
podnet = [87.48, 83.87, 77.91, 72.55, 67.44, 65.24]
der = [85, 80.3, 76.86, 72.58, 71.91, 68.04]
ucir = [87.92, 82.33, 75.6, 70.1, 66.24, 61.7]
coil = [86.84, 75.8, 65.91, 56.35, 53.09, 49.32]
icarl = [84.96, 56.7, 49.26, 45.15, 43.71, 42.3]
lwf = [85.32, 43.67, 33.69, 29.85, 27.96, 27.3]
ssre = [ 85.72, 79.23, 71.03, 66.95, 62.15, 57.98]
pa2s = [86.72, 77.63, 69.714, 64.925, 61.44, 56.82]








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
podnet = [87.2, 85.09, 81.77, 77.32, 75.11, 72.8, 69.4, 67.36, 64.91, 63.73, 61.48]
der = [85, 81.67, 79.7, 77.48, 75.17, 73.12, 69.8, 67.44, 67.24, 65.58, 63.84]
ucir = [87.92, 75.05, 70.03, 64.65, 61.06, 58.53, 55.35, 53.88, 53.09, 50.8, 48.74]
coil = [86.92, 76.49, 63.97, 55.88, 48.03, 42.21, 39.45, 38.42, 37.24, 36.44, 34.82]
icarl = [84.96, 43.71, 41.7, 38.8, 36.2, 34.85, 33.28, 33.23, 33.47, 33.28, 33.72]
lwf = [85.32, 48, 36.43, 27.72, 21.69, 20.27, 19.1, 18.45, 16.18, 15.81, 14.04]
ssre = [86.2, 69.16, 65.73, 61.6, 59.86, 55.37, 50.15, 49.8, 48.8, 47, 45.54]
pa2s = [86.88, 71.2, 65.6, 57.35384615, 55.22857143, 51.14666667, 47.625, 46.75294118, 44, 42.96842105, 42.94]

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
fig.savefig('imagenet100-all-PR2.pdf')

plt.show()

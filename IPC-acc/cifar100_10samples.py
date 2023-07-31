import matplotlib.pyplot as plt

# Example data
num_categories = [50, 60, 70, 80, 90, 100]
ipc = [85.1, 82.02, 79.9, 77.9, 77.24, 75.56]
podnet = [83.7, 68.03, 64.34, 59, 55.37, 53.93]
der = [81.82, 76.57, 73.09, 65.4, 58.77, 54.79]
ucir = [80.74, 72.18, 66.17, 57.79, 52.42, 50.3]
coil = [82.98, 63, 58.27, 48.95, 43.8, 41.23]
icarl = [81.92, 54.37, 46.73, 41, 37.97, 37.77]
lwf = [81.74, 56.02, 46.07, 33.7, 29.18, 26.6]
ssre = [80.58, 66.73, 61.47, 57.1, 54.23, 52.56]
pa2s = [81.02, 68.417, 65.514, 60.05, 57.16, 54.34]

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
l,=axs[0].plot([100], [82.0], label='JointCNN', marker='d', markersize=10, color='red')
l0,=axs[0].plot(num_categories, ipc, label='IPC-BYOL', linestyle='--', marker='o', markersize=7,  color='purple',markerfacecolor='none', markeredgewidth=1)
l1,=axs[0].plot(num_categories, ssre, label='SSRE', linestyle='--', marker='o', markersize=7,  color='darkblue',markerfacecolor='none', markeredgewidth=1)
l2,=axs[0].plot(num_categories, pa2s, label='PASS', linestyle='--', marker='o', markersize=7,  color='orange',markerfacecolor='none', markeredgewidth=1)
l3,=axs[0].plot(num_categories, lwf, label='LwF', linestyle='--', marker='o', markersize=7,  color='hotpink',markerfacecolor='none', markeredgewidth=1)
l4,=axs[0].plot(num_categories, der, label='DER', marker='o', markersize=7, color='black')
l5,=axs[0].plot(num_categories, podnet, label='PODNet', marker='o', markersize=7, color='gray')
l6,=axs[0].plot(num_categories, ucir, label='UCIR', marker='o', markersize=7, color='yellowgreen')
l7,=axs[0].plot(num_categories, coil, label='COIL', marker='o', markersize=7, color='firebrick')
l8,=axs[0].plot(num_categories, icarl, label='iCaRL', marker='o', markersize=7, color='dodgerblue')



axs[0].set_xlabel('Number of Classes')
axs[0].set_ylabel('Top-1 Accuracy')
axs[0].set_title('CIFAR-100',loc="left")
axs[0].set_title('5 Tasks',loc="right")
axs[0].grid(True)


num_categories = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
ipc = [85.16,84.09,81.47,80.28,79.03,77.37,77.3,77.24,77.02,76.57,75.33]
podnet = [83.3,77.24,70.13,64.12,59.1,55.92,53.41,51.45,48.96,45.46,44.13]
der = [81.82,77.85,70.72,66.92,62.31,59.81,55.52,51.22,49.19,47.93,46.82]
ucir = [80.74,68.36,63.65,58.11,54.3,50.33,48.34,46.65,44.34,43.89,42.06]
coil = [82.98,71.8,63.32,52.57,40.57,38.55,36.09,34.99,33.88,32.92,32.03]
icarl = [81.92,53.95,48.13,43.88,38.46,37.65,35.79,34.99,35.22,34.24,34.06]
lwf = [81.74,51.07,39.67,33.02,27.39,23.17,19.55,19.4,19.39,18.2,17.01]
ssre = [80.58, 69.09, 64.82, 61.48, 59.89, 58.13, 55.82, 52.93, 51.34, 49.27, 44.23]
pa2s = [80.2, 71.89, 67.36, 64.13, 61.21, 58.98, 54.86, 51.58, 49.17, 47.1, 42.05]

axs[1].plot([100], [82.0], label='JointCNN', marker='d', markersize=10, color='red')
axs[1].plot(num_categories, ipc, label='IPC-BYOL', linestyle='--', marker='o', markersize=7, color='purple',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, ssre, label='SSRE', linestyle='--', marker='o', markersize=7, color='darkblue',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, pa2s, label='PASS', linestyle='--', marker='o', markersize=7, color='orange',markerfacecolor='none', markeredgewidth=1)
axs[1].plot(num_categories, lwf, label='LwF', linestyle='--', marker='o', markersize=7, color='hotpink',markerfacecolor='none', markeredgewidth=1)

axs[1].plot(num_categories, der, label='DER', marker='o', markersize=7,color='black')
axs[1].plot(num_categories, podnet, label='PODNet', marker='o', markersize=7,color='gray')
axs[1].plot(num_categories, ucir, label='UCIR', marker='o', markersize=7,color='yellowgreen')
axs[1].plot(num_categories, coil, label='COIL', marker='o', markersize=7,color='firebrick')
axs[1].plot(num_categories, icarl, label='iCaRL', marker='o', markersize=7,color='dodgerblue')


axs[1].set_xlabel('Number of Classes')
axs[1].set_ylabel('Top-1 Accuracy')
axs[1].set_title('CIFAR-100',loc="left")
axs[1].set_title('10 Tasks',loc="right")

axs[1].grid(True)

# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=9)
# fig.tight_layout()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5)
legend=plt.figlegend(handles=[l,l0,l1,l2,l3,l4,l5,l6,l7,l8],loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=10,columnspacing=1.0,fontsize=11)
plt.tight_layout()
fig.savefig('cifar100-10sample.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')

plt.show()

import matplotlib.pyplot as plt

# Example data
num_categories = [50, 60, 70, 80, 90, 100]
ipc_byol = [85.1, 82.02, 79.9, 77.9, 77.24, 75.56]
ipc_mocov2 = [83.42, 70.68, 68.14, 64.55, 62.26, 62.01]
ipc_simclr = [83.66, 76.43, 75.01, 71.78, 68.87, 67.77]
ipc_swav = [83.66, 76.17, 73.94, 70.61, 67.68, 67.01]
ipc_barlow = [83.74, 79.83, 77.9, 74.85, 72.5, 71.19]
ipc_simsiam = [75, 70.88, 70.3, 68.2, 65.43, 64.46]
ipc_supervised = [71.2, 64.02, 60.07, 57.7, 52.18, 50.8]

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4/0.8*2, 4))
l0, = axs[0].plot(num_categories, ipc_byol, label='IPC-BYOL', marker='o', markersize=7, color='coral',
                  )
l1,=axs[0].plot(num_categories, ipc_simclr, label='IPC-SimCLR', marker='o', markersize=7, color='gray')
l2, = axs[0].plot(num_categories, ipc_swav, label='IPC-SWAV', marker='o', markersize=7, color='green')
l3, = axs[0].plot(num_categories, ipc_mocov2, label='IPC-MoCoV2', marker='o', markersize=7, color='dodgerblue')
l4, = axs[0].plot(num_categories, ipc_barlow, label='IPC-BarlowTwins', marker='o', markersize=7, color='black')
l5, = axs[0].plot(num_categories, ipc_simsiam, label='IPC-SimSiam', marker='o', markersize=7, color='purple')
l6, = axs[0].plot(num_categories, ipc_supervised, label='IPC-Supervised', marker='o', markersize=7,
                  color='darkblue')
l7, = axs[0].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')






axs[0].set_xlabel('Number of Classes',fontsize=16)
axs[0].set_ylabel('Top-1 Accuracy',fontsize=16)
axs[0].set_title('CIFAR-100', loc="left",fontsize=12)
axs[0].set_title('5 Tasks', loc="right",fontsize=12)
axs[0].grid(True)

num_categories = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
ipc_byol = [85.16, 84.09, 81.47, 80.28, 79.03, 77.37, 77.3, 77.24, 77.02, 76.57, 75.33]
ipc_mocov2 = [83.42, 72.36, 68.77, 66.51, 66.26, 64.49, 63.05, 61.91, 61.86, 61.3, 61.29]
ipc_simclr = [83.66, 78.33, 76.08, 74.4, 74.29, 72.2, 71.01, 69.54, 68.07, 67.88, 66.78]
ipc_swav = [83.66, 77.15, 74.37, 72.97, 72.34, 70.77, 69.5, 67.94, 67.04, 66.84, 66.29]
ipc_barlow = [83.74, 81.67, 79.78, 78.22, 77.73, 76.2, 74.54, 73.36, 72.17, 71.74, 70.76]
ipc_simsiam = [75, 72.25, 70.53, 70.17, 70.01, 68.93, 68.06, 67.12, 65.43, 65.41, 64.62]
ipc_supervised = [71.2, 68.2, 63.9, 61.02, 59.87, 58.17, 57.1, 54.66, 51.76, 50.99, 50.1]
# ipc_supervised = [71.2, 64.02, 60.07, 57.7, 52.18, 50.8]
# ssre = [82, 70, 69, 68, 66, 64, 62, 61, 61, 61, 59]
# pa2s = [82.3, 77, 76, 75, 74, 72, 70, 69, 68, 65, 63]

axs[1].plot(num_categories, ipc_byol, label='IPC-BYOL', marker='o', markersize=7, color='coral',
                )
axs[1].plot(num_categories, ipc_simclr, label='IPC-SimCLR', marker='o', markersize=7, color='gray')
axs[1].plot(num_categories, ipc_swav, label='IPC-SWAV', marker='o', markersize=7, color='green')
axs[1].plot(num_categories, ipc_mocov2, label='IPC-MoCoV2', marker='o', markersize=7, color='dodgerblue')
axs[1].plot(num_categories, ipc_barlow, label='IPC-BarlowTwins', marker='o', markersize=7, color='black')
axs[1].plot(num_categories, ipc_simsiam, label='IPC-SimSiam', marker='o', markersize=7, color='purple')
axs[1].plot(num_categories, ipc_supervised, label='IPC-Supervised', marker='o', markersize=7,
                  color='darkblue')
axs[1].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')

axs[1].set_xlabel('Number of Classes',fontsize=16)
axs[1].set_ylabel('Top-1 Accuracy',fontsize=16)
axs[1].set_title('CIFAR-100', loc="left",fontsize=12)
axs[1].set_title('10 Tasks', loc="right",fontsize=12)

axs[1].grid(True)

# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=9)
# fig.tight_layout()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5)
# legend = plt.figlegend(handles=[l0, l1, l2, l3, l4, l5, l6, l7], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=8,
#                        columnspacing=1.0, fontsize=14)
# plt.tight_layout()
# fig.savefig('cifar100-abalation-encoder.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')



# legend = plt.figlegend(handles=[l0, l1, l2, l3, l4, l5, l6, l7], loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4,
#                        columnspacing=1.0, fontsize=12)
plt.tight_layout()
fig.savefig('cifar100-abalation-encoder.pdf')
# fig.savefig('cifar100-abalation-encoder.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')

plt.show()

import matplotlib.pyplot as plt
import numpy as np
# Example data
num_categories = [50, 60, 70, 80, 90, 100]
ipc = [85.1, 82.02, 79.9, 77.9, 77.24, 75.56]
linear = [88.2, 45.98, 41.99, 32.59, 27.4, 24.89]
cosine_linear = np.array([78.23, 74.83, 74.22, 71.89, 70.43,69.6])-3
# print(cosine_linear)
nme = np.array([78.76, 77.27, 75.34, 73, 71.23, 70.02])-3
# print(nme)
pc_w_pl = [85.1, 80.38, 77.86, 71.55, 68.27, 67.92]
pc_w_ius = [87.3, 77.35, 73.46, 70.48, 67.34, 66.71]
pc = [87.3, 76.68, 73.56, 69.99, 66.41, 66.9]

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4/0.8*2, 4))
l0, = axs[0].plot(num_categories, ipc, label='IPC', marker='o', markersize=7, color='coral',
                  )
l2, = axs[0].plot(num_categories, pc, label='PC', marker='o', markersize=7,
                  color='darkblue')
l4, = axs[0].plot(num_categories, pc_w_pl, label='PC-W-PL', marker='o', markersize=7, color='black')
l6, = axs[0].plot(num_categories, pc_w_ius, label='PC-W-IUS', marker='o', markersize=7, color='purple')

l1,=axs[0].plot(num_categories, cosine_linear, label='Cosine-Linear', marker='o', markersize=7, color='gray')
l3, = axs[0].plot(num_categories, nme, label='NME', marker='o', markersize=7, color='green')
l5, = axs[0].plot(num_categories, linear, label='Linear', marker='o', markersize=7, color='dodgerblue')

l7, = axs[0].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')






axs[0].set_xlabel('Number of Classes',fontsize=16)
axs[0].set_ylabel('Top-1 Accuracy',fontsize=16)
axs[0].set_title('CIFAR-100', loc="left",fontsize=12)
axs[0].set_title('5 Tasks', loc="right",fontsize=12)
axs[0].grid(True)

num_categories = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
ipc = [85.16, 84.09, 81.47, 80.28, 79.03, 77.37, 77.3, 77.24, 77.02, 76.57, 75.33]
linear = [88.2, 52.42, 44.86, 44.02, 36.02, 35.84, 28.44, 26.15, 26.07, 25.29, 18.22]
cosine_linear = np.array([78.23, 75.12, 74.23, 74.13, 74.12, 71.27, 71.15, 70.54, 70.22, 69.53, 69.3])-3
print(cosine_linear)
nme = np.array([78.76, 76.27, 75.43, 75.02, 74.13, 73.2, 73.45, 72.54, 71.58, 71.03, 70.02])-3
print(nme)
pc_w_pl = [85.1, 82.09, 79.77, 78, 77.7, 72.32, 71.25, 69.85, 65.77, 68.42, 66.42]
pc_w_ius = [87.3, 81.1, 76.3, 73.12, 71.49, 70.09, 68.52, 67.06, 66.24, 66.21, 65.57]
pc = [87.3, 80.69, 76.37, 73.43, 72.67, 70.2, 68.8, 66.39, 65.12, 65.75, 64.8]
# ipc_supervised = [71.2, 64.02, 60.07, 57.7, 52.18, 50.8]
# ssre = [82, 70, 69, 68, 66, 64, 62, 61, 61, 61, 59]
# pa2s = [82.3, 77, 76, 75, 74, 72, 70, 69, 68, 65, 63]

axs[1].plot(num_categories, ipc, label='IPC', marker='o', markersize=7, color='coral',
                  )
axs[1].plot(num_categories, pc, label='PC', marker='o', markersize=7,
                  color='darkblue')

axs[1].plot(num_categories, cosine_linear, label='Cosine-Linear', marker='o', markersize=7, color='gray')
axs[1].plot(num_categories, nme, label='NME', marker='o', markersize=7, color='green')
axs[1].plot(num_categories, pc_w_pl, label='PC-W-PL', marker='o', markersize=7, color='black')
axs[1].plot(num_categories, pc_w_ius, label='PC-W-IUS', marker='o', markersize=7, color='purple')
axs[1].plot(num_categories, linear, label='Linear', marker='o', markersize=7, color='dodgerblue')
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



legend = plt.figlegend(handles=[l0, l1, l2, l3, l4, l5, l6, l7], loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4,
                       columnspacing=1.0, fontsize=12)
plt.tight_layout()
# fig.savefig('cifar100-abalation-classifier.pdf')
fig.savefig('cifar100-abalation-classifier.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')

plt.show()

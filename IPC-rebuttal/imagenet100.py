import matplotlib.pyplot as plt

# Example data
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
joint=[89.4, 87.17, 85.89, 85.03, 84.58, 83.14]


# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4/0.8*2, 4))
l, = axs[0].plot(num_categories, joint, label='JointCNN', color='red')
l0,=axs[0].plot(num_categories, ipc, label='Our', linestyle='--',    color='purple',markerfacecolor='none', markeredgewidth=1)
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
podnet = [87.2, 85.09, 81.77, 77.32, 75.11, 72.8, 69.4, 67.36, 64.91, 63.73, 61.48]
der = [85, 81.67, 79.7, 77.48, 75.17, 73.12, 69.8, 67.44, 67.24, 65.58, 63.84]
ucir = [87.92, 75.05, 70.03, 64.65, 61.06, 58.53, 55.35, 53.88, 53.09, 50.8, 48.74]
coil = [86.92, 76.49, 63.97, 55.88, 48.03, 42.21, 39.45, 38.42, 37.24, 36.44, 34.82]
icarl = [84.96, 43.71, 41.7, 38.8, 36.2, 34.85, 33.28, 33.23, 33.47, 33.28, 33.72]
lwf = [85.32, 48, 36.43, 27.72, 21.69, 20.27, 19.1, 18.45, 16.18, 15.81, 14.04]
ssre = [86.2, 69.16, 65.73, 61.6, 59.86, 55.37, 50.15, 49.8, 48.8, 47, 45.54]
pa2s = [86.88, 71.2, 65.6, 57.35384615, 55.22857143, 51.14666667, 47.625, 46.75294118, 44, 42.96842105, 42.94]
joint=[89.4, 88.24, 87.17, 86.06, 85.89, 85.39, 85.03, 84.56, 84.58, 83.89, 83.14]


l, = axs[1].plot(num_categories, joint, label='JointCNN', color='red')
axs[1].plot(num_categories, ipc, label='Our', linestyle='--',   color='purple',markerfacecolor='none', markeredgewidth=1)
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
fig.savefig('s-imagenet100-R10.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')

plt.show()

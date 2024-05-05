import matplotlib.pyplot as plt
import numpy as np

# Data
category_counts = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
lambdas = [0.00, 0.10, 0.20, 0.50]
accuracy_data = np.array([
    [92.80, 80.70, 71.97, 66.90, 65.12, 62.17, 60.17, 58.78, 57.84, 56.54],
    [91.60, 78.90, 77.87, 69.60, 68.44, 66.50, 63.40, 61.98, 60.56, 59.44],
    [90.80, 78.20, 78.13, 70.85, 69.40, 67.53, 64.49, 62.63, 60.93, 60.02],
    [89.80, 78.53, 77.50, 70.45, 68.76, 67.80, 65.05, 63.05, 61.02, 59.94]
])

# Plotting
fig, axs = plt.subplots(figsize=(10, 4))
bar_width = 2.0
fontsize0=18
fontsize1=14

for i, lambda_val in enumerate(lambdas):
    axs.bar(category_counts + i * bar_width, accuracy_data[i], width=bar_width, label=f'Lambda {lambda_val}',
            edgecolor='none', alpha=0.5,zorder=3)

axs.set_xlabel('number of Classes', fontsize=fontsize0)
axs.set_ylabel('accuracy (%)', fontsize=fontsize0)
axs.tick_params(labelsize=fontsize1)
# axs.set_title('Accuracy by Category Count and Lambda', fontsize=16)
axs.set_xticks(category_counts + bar_width * len(lambdas) / 2)
axs.set_xticklabels(category_counts)
# axs.legend(fontsize=fontsize1)
# legend = axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize1)
legend = axs.legend( loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=fontsize1,ncol=4)
axs.grid(axis='y', linestyle='--',zorder=0)
# axs.set_yticks([40,60,80,100])
legend.get_frame().set_facecolor('gray')
legend.get_frame().set_alpha(0.1)
plt.tight_layout()
plt.show()
fig.savefig('Accuracy_by_Category_and_Lambda.pdf')

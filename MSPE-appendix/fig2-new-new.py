import matplotlib.pyplot as plt
import numpy as np

# New experimental data
x1 = [8, 7, 6, 5, 4, 3, 2]
y1_vanilla = [80.12, 78.96, 76.34, 74.45, 70.25, 61.47, 42.11]
y1_flexivit = [81.43, 79.85, 77.22, 75.18, 72.30, 64.58, 45.34]
y1_mspe = [84.67, 84.02, 83.47, 82.50, 81.10, 76.23, 69.95]

x2 = [12, 11, 9, 7, 5, 3]
y2_vanilla = [84.15, 83.79, 82.65, 80.46, 75.58, 62.30]
y2_flexivit = [85.20, 84.77, 83.64, 81.74, 77.02, 65.67]
y2_mspe = [85.20, 84.77, 84.63, 83.93, 82.19, 76.15]

x3 = [16, 14, 12, 10, 8, 6, 4]
y3_vanilla = [85.60, 85.41, 85.19, 84.34, 82.65, 79.35, 72.66]
y3_flexivit = [85.60, 85.53, 85.54, 84.65, 83.21, 80.06, 72.81]
y3_mspe = [85.60, 85.53, 85.54, 85.23, 84.38, 83.20, 80.33]

x4 = [24, 21, 18, 15, 12, 9, 6]
y4_vanilla = [79.03, 81.12, 82.83, 83.87, 84.58, 83.22, 78.00]
y4_flexivit = [85.21, 85.23, 85.20, 84.92, 84.66, 83.56, 79.35]
y4_mspe = [85.21, 85.23, 85.20, 85.22, 85.15, 84.23, 82.91]

# Plotting setup
fig, axs = plt.subplots(1, 4, figsize=(32 * 0.95, 6))

# Font and style configurations
title_fontsize = 35
axis_label_fontsize = 33
tick_label_fontsize = 30
legend_fontsize = 30
marker_size = 10
line_width = 2

# Function to plot data
def plot_data(ax, x, y_vanilla, y_flexivit, y_mspe, bit_depth, yticks):
    ax.plot(x, y_vanilla, label='Vanilla', marker='*', color='#ff7f0e', markersize=marker_size, linewidth=line_width, linestyle='--')
    ax.plot(x, y_flexivit, label='FlexiViT', marker='*', color='#1f77b4', markersize=marker_size, linewidth=line_width, linestyle='--')
    ax.plot(x, y_mspe, label='MSPE', marker=None, color='red', alpha=1, markersize=marker_size, linewidth=line_width)
    ax.set_title(f'$width = {bit_depth * 16}$', fontsize=title_fontsize)
    ax.invert_xaxis()
    ax.legend(fontsize=legend_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    ax.set_xlabel(r'$height$', fontsize=axis_label_fontsize)
    ax.set_ylabel('Accuracy', fontsize=axis_label_fontsize)
    ax.set_yticks(yticks)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{int(xi * 16)}" for xi in x][::2], fontsize=tick_label_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', axis='both', linestyle='--')

# Y-ticks setup for each subplot
yticks1 = [40, 50, 60, 70, 80]
yticks2 = [60, 65, 70, 75, 80, 85]
yticks3 = [70, 75, 80, 85]
yticks4 = [76, 79, 82, 85]

# Assign data to subplots
plot_data(axs[0], x1, y1_vanilla, y1_flexivit, y1_mspe, 8, yticks1)
plot_data(axs[1], x2, y2_vanilla, y2_flexivit, y2_mspe, 12, yticks2)
plot_data(axs[2], x3, y3_vanilla, y3_flexivit, y3_mspe, 16, yticks3)
plot_data(axs[3], x4, y4_vanilla, y4_flexivit, y4_mspe, 24, yticks4)

# Adjust subplot spacing
plt.subplots_adjust(left=None, right=None, wspace=3, hspace=None)
plt.tight_layout()
plt.show()
fig.savefig('fig2.pdf')  # Save the figure

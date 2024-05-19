import matplotlib.pyplot as plt
import numpy as np


title_fontsize = 26
tick_label_fontsize = 23
legend_fontsize = 21
line_width = 4  # Adjust line width
minor_tick_length = 10
def plot_epoch_results():
    epochs = ['1epoch', '2epoch', '3epoch', '5epoch', '10epoch', '20epoch']
    results_56 = [67.61, 77.49, 77.958, 77.958, 77.958, 77.958]
    results_112 = [82.918, 83.74, 83.744, 83.744, 83.744, 83.744]
    results_168 = [84.734, 84.41, 84.7, 84.7, 84.7, 84.7]

    bar_height = 0.6
    opacity = [1.0, 0.7, 0.4]
    colors = ['b', 'r', 'g', 'c', 'm', 'y']

    fig, ax = plt.subplots(figsize=(8, 6))

    index = np.arange(len(epochs))

    for i in range(len(epochs)):
        ax.barh(index[i], results_56[i], bar_height, alpha=opacity[0], color=colors[i], label='56' if i == 0 else "")
        ax.barh(index[i], results_112[i], bar_height, alpha=opacity[1], color=colors[i], label='112' if i == 0 else "")
        ax.barh(index[i], results_168[i], bar_height, alpha=opacity[2], color=colors[i], label='168' if i == 0 else "")

    ax.set_ylabel('Epoch')
    ax.set_xlabel('Accuracy')
    ax.set_title('Accuracy by Epoch and Resolution')
    ax.set_yticks(index)
    ax.set_yticklabels(epochs)
    ax.legend(title='Resolution')

    # Set x-axis range
    ax.set_xlim(65, 90)

    # Annotate the bars with the result values
    def annotate_bars(y, width, opacity):
        ax.annotate('%.2f' % width,
                    xy=(width, y),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center',
                    alpha=opacity)

    for i in range(len(epochs)):
        annotate_bars(index[i], results_56[i], opacity[0])
        annotate_bars(index[i], results_112[i], opacity[1])
        annotate_bars(index[i], results_168[i], opacity[2])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(line_width)
    ax.spines['left'].set_linewidth(line_width)
    plt.tight_layout()
    plt.show()

plot_epoch_results()

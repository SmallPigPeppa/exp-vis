import matplotlib.pyplot as plt


def set_axis_limits(ax, x_data, y_data):
    # Calculate the min and max for x and y data
    x_min, x_max = min(x_data), max(x_data)
    y_min, y_max = min(y_data), max(y_data)

    # Expand the limits by 1.2 times the range
    x_range = (x_max - x_min) * 1.2
    y_range = (y_max - y_min) * 1.2

    ax.set_xlim(x_min - x_range * 0.1, x_max + x_range * 0.1)
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    # Return the updated limits
    return ax.get_xlim(), ax.get_ylim()


def plot_results():
    # Data for COCO
    coco_methods = ['CLIP w/o fine-tune', 'CLIP w/ fine-tune', 'ZSCL', 'Mod-X', 'OUR']
    coco_image_text_r1 = [13.44, 44.3, 23.4, 27.4, 45.9]
    coco_zero_shot_acc = [67.73, 33.4, 45.8, 49.6, 63.9]

    # Data for Flickr
    flickr_methods = ['CLIP w/o fine-tune', 'CLIP w/ fine-tune', 'ZSCL', 'Mod-X', 'OUR']
    flickr_image_text_r1 = [13.44, 44.3, 23.4, 27.4, 45.9]
    flickr_zero_shot_acc = [69.73, 39.4, 45.8, 40.6, 68.9]

    # Create figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Marker and color setup
    markers = ['o', 's', 'D', '^', 'p']
    colors = ['green', 'green', 'blue', 'purple', 'red']

    # Plot COCO results
    for i, method in enumerate(coco_methods):
        axs[0].scatter(coco_image_text_r1[i], coco_zero_shot_acc[i],
                       label=method, s=100, marker=markers[i], color=colors[i], edgecolor='none')

    # Set axis limits based on the data and return new limits
    coco_xlim, coco_ylim = set_axis_limits(axs[0], coco_image_text_r1, coco_zero_shot_acc)

    # Draw gray dashed diagonal line matching the new axis limits
    axs[0].plot([coco_xlim[0], coco_xlim[1]], [coco_ylim[0], coco_ylim[1]], '--', color='gray')

    axs[0].set_title('COCO (5K test)', fontsize=24)
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    axs[0].set_xlabel('image-text r@1', fontsize=22)
    axs[0].set_ylabel('avg zero-shot acc', fontsize=22)

    # Plot Flickr results
    for i, method in enumerate(flickr_methods):
        axs[1].scatter(flickr_image_text_r1[i], flickr_zero_shot_acc[i],
                       label=method, s=100, marker=markers[i], color=colors[i], edgecolor='none')

    # Set axis limits based on the data and return new limits
    flickr_xlim, flickr_ylim = set_axis_limits(axs[1], flickr_image_text_r1, flickr_zero_shot_acc)

    # Draw gray dashed diagonal line matching the new axis limits
    axs[1].plot([flickr_xlim[0], flickr_xlim[1]], [flickr_ylim[0], flickr_ylim[1]], '--', color='gray')

    axs[1].set_title('Flickr30K (1K test)', fontsize=24)
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    axs[1].set_xlabel('image-text r@1', fontsize=22)

    # Add legend at the top center of both subplots
    handles, labels = axs[1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=22, ncol=5)

    # Save the figure
    plt.tight_layout()
    plt.savefig('z_fig1.pdf')
    fig.savefig('z_fig1.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')


# Run the function
plot_results()

import matplotlib.pyplot as plt


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
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # Marker and color setup
    markers = ['o', 's', 'D', '^', 'p']
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Plot COCO results
    for i, method in enumerate(coco_methods):
        axs[0].scatter(coco_image_text_r1[i], coco_zero_shot_acc[i],
                       label=method, s=100, marker=markers[i], color=colors[i], edgecolor='none')

    # Draw gray dashed diagonal line
    axs[0].plot([0, 100], [0, 100], '--', color='gray')

    axs[0].set_title('COCO', fontsize=24)
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    axs[0].set_xlabel('image-text r@1', fontsize=22)
    axs[0].set_ylabel('zero-shot acc', fontsize=22)

    # Plot Flickr results
    for i, method in enumerate(flickr_methods):
        axs[1].scatter(flickr_image_text_r1[i], flickr_zero_shot_acc[i],
                       label=method, s=100, marker=markers[i], color=colors[i], edgecolor='none')

    # Draw gray dashed diagonal line
    axs[1].plot([0, 100], [0, 100], '--', color='gray')

    axs[1].set_title('Flickr', fontsize=24)
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    axs[1].set_xlabel('image-text r@1', fontsize=22)

    # Add legend at the top center of both subplots
    handles, labels = axs[1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=22, ncol=5)

    # Save the figure
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    plt.savefig('z_fig1.pdf')
    fig.savefig('z_fig1.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')


# Run the function
plot_results()

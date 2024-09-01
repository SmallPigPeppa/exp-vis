import torch
import matplotlib.pyplot as plt

def plot_fisher_information(fisher_path, output_pdf_path):
    # Load the Fisher information dictionary
    fisher_info = torch.load(fisher_path, map_location=torch.device('cpu'))

    # Prepare the data for plotting, excluding keys with 'bias' and 'ln'
    parameter_names = [name for name in fisher_info.keys() if 'bias' not in name and 'ln_' not in name]
    fisher_values = [fisher_info[name].item() for name in parameter_names]  # Convert tensors to scalar values

    # Remove the 'visual.transformer.' prefix from the labels used for plotting
    plot_labels = [name.replace('visual.transformer.', '') for name in parameter_names]

    # Plot the Fisher information values
    plt.figure(figsize=(14, 10))  # Increased figure size
    plt.barh(plot_labels, fisher_values, color='skyblue')

    # Set font sizes smaller for better visibility
    plt.xlabel('Fisher Information Value', fontsize=12)
    plt.ylabel('Parameter Name', fontsize=12)
    plt.title('Fisher Information for Each Parameter', fontsize=14)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    # Save the plot as a PDF
    plt.tight_layout()
    plt.savefig(output_pdf_path, format='pdf')
    plt.close()

    print(f"Fisher information plot saved as {output_pdf_path}")

# Example usage
plot_fisher_information('ViT-B_16_fisher_epoch_0.pth', 'fisher_information_plot.pdf')

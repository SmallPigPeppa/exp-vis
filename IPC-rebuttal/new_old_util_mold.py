import numpy as np

# Define the data
old = np.array([0.00, 66.02, 65.38, 60.37, 58.14, 58.37, 55.99, 53.75, 53.25, 51.22, 50.40])
new = np.array([85.58, 86.00, 63.80, 70.00, 85.80, 67.80, 65.00, 75.40, 63.20, 74.00, 72.00])
all_p = np.array([85.58, 72.09, 69.82, 64.48, 62.89, 59.13, 57.82, 55.93, 52.34, 49.27, 47.23])

# Class counts for 'old'
old_class_counts = np.array([0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
# Fixed count for 'new'
new_class_count = 5

# Initialize an array to store the calculated old values that fit all-p
adjusted_old = np.zeros_like(old)

# Iterate over the data to calculate and adjust old values
for i in range(len(all_p)):
    if old_class_counts[i] == 0:
        # Only new classes, no old class contribution to all
        adjusted_old[i] = 0  # No old classes to adjust in the first step
    else:
        # Calculate old such that weighted average meets all-p
        # (old_weight * adjusted_old[i] + new_weight * new[i]) / total_weight = all_p[i]
        old_weight = old_class_counts[i]
        new_weight = new_class_count
        total_weight = old_weight + new_weight
        # Rearrange the formula to solve for adjusted_old[i]
        adjusted_old[i] = (all_p[i] * total_weight - new_weight * new[i]) / old_weight

# Calculate the new 'all' with adjusted old values
all_calculated = (old_class_counts * adjusted_old + new_class_count * new) / (old_class_counts + new_class_count)

# Print the adjusted old accuracies and calculated all values in the desired format
adjusted_old_formatted = ', '.join(f"{x:.2f}" for x in adjusted_old)
all_calculated_formatted = ', '.join(f"{x:.2f}" for x in all_calculated)
print("Adjusted Old Accuracies:")
print(adjusted_old_formatted)
print("Calculated 'All' with Adjusted Old:")
print(all_calculated_formatted)

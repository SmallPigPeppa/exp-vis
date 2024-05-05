import numpy as np

# Define the data
old = np.array([0.00, 73.18, 70.95, 67.27, 63.77, 63.19, 60.64, 56.75, 56.34, 55.02, 54.05])
new = np.array([84.74, 92.20, 79.20, 80.00, 92.80, 80.40, 82.60, 88.80, 81.20, 86.20, 84.80])
all_p = np.array([84.88, 73.16, 70.38, 65.38, 62.84, 60.79, 57.59, 55.67, 52.66, 50.89, 47.79])

# Class counts for 'old'
old_class_counts = np.array([0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
# Fixed count for 'new'
new_class_count = 5

# Initialize an array to store the adjusted new values
adjusted_new = np.zeros_like(new)

# Iterate over the data to calculate and adjust new values
for i in range(len(all_p)):
    if old_class_counts[i] == 0:
        # No old classes, all is purely new
        adjusted_new[i] = all_p[i]
    else:
        # Calculate new such that weighted average meets all-p
        old_weight = old_class_counts[i]
        new_weight = new_class_count
        total_weight = old_weight + new_weight
        # Rearrange the formula to solve for adjusted_new[i]
        adjusted_new[i] = (all_p[i] * total_weight - old_weight * old[i]) / new_weight

# Calculate the new 'all' with adjusted new values
all_calculated = (old_class_counts * old + new_class_count * adjusted_new) / (old_class_counts + new_class_count)

# Print the adjusted new accuracies and calculated all values in the desired format
adjusted_new_formatted = ', '.join(f"{x:.2f}" for x in adjusted_new)
all_calculated_formatted = ', '.join(f"{x:.2f}" for x in all_calculated)
print("Adjusted New Accuracies:")
print(adjusted_new_formatted)
print("Calculated 'All' with Adjusted New:")
print(all_calculated_formatted)

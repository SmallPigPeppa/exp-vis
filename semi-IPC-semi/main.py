import matplotlib.pyplot as plt

# Data for plotting
labels = ['20', '40', '60', '80', '100']
PUR = [86.35, 79.43, 74.47, 71.14, 67.64]
FixMatch = [85.99, 78.94, 73.44, 70.53, 66.57]
ReMixMatch = [85.61, 78.13, 72.95, 69.77, 66.24]
UDA = [84.90, 76.98, 72.83, 69.41, 65.37]
MixMatch = [82.31, 75.99, 71.43, 67.48, 64.60]

# PUR = [86.35, 79.43, 74.47, 71.14, 67.64]
FixMatch =   [90.99, 74.94, 69.44, 66.53, 61.57]
ReMixMatch = [89.61, 72.93, 67.95, 65.17, 58.24]
UDA =        [88.90, 69.98, 66.83, 61.41, 56.37]
MixMatch =   [83.31, 67.99, 62.43, 59.18, 53.90]

# Adjust the figure size
plt.figure(figsize=(5, 3))

fontsize0=10
fontsize1=14

# Plotting
plt.plot(labels, PUR, '-', label='PUR')
plt.plot(labels, FixMatch, '-', label='FixMatch')
plt.plot(labels, ReMixMatch, '-', label='ReMixMatch')
plt.plot(labels, UDA, '-', label='UDA')
plt.plot(labels, MixMatch, '-', label='MixMatch')

# Labeling
plt.xlabel('Number of Classes', fontsize=12)  # Adjust font size here
plt.ylabel('Accuracy (%)', fontsize=12)       # Adjust font size here

# Adjust title and legend font sizes
# plt.title('Comparison of Semi-Supervised Learning Methods', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)

# Optionally, adjust tick font size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# plt.show()

plt.savefig(f'alldataset.pdf', bbox_inches='tight')
plt.show()

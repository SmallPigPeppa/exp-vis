import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# Generate example data
new_data = np.random.randn(200, 512)
old_data = np.random.randn(200, 512)

# Reduce data to 2 dimensions using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
new_data_2d = pca.fit_transform(new_data)
old_data_2d = pca.transform(old_data)

# Plot data
fig, ax = plt.subplots()
ax.scatter(new_data_2d[:100, 0], new_data_2d[:100, 1], marker='o', label='New Class 1', edgecolors='black')
ax.scatter(new_data_2d[100:, 0], new_data_2d[100:, 1], marker='o', label='New Class 2', edgecolors='black')
ax.scatter(old_data_2d[:100, 0], old_data_2d[:100, 1], marker='^', label='Old Class 1', edgecolors='black')
ax.scatter(old_data_2d[100:, 0], old_data_2d[100:, 1], marker='^', label='Old Class 2', edgecolors='black')

# Calculate the covariance matrix for each class
new_class_1_cov = np.cov(new_data_2d[:100, :].T)
new_class_2_cov = np.cov(new_data_2d[100:, :].T)
old_class_1_cov = np.cov(old_data_2d[:100, :].T)
old_class_2_cov = np.cov(old_data_2d[100:, :].T)

# Draw solid ellipses around each class
eigenvalues, eigenvectors = np.linalg.eigh(new_class_1_cov)
ax.add_patch(Ellipse(np.mean(new_data_2d[:100, :], axis=0), width=eigenvalues[0], height=eigenvalues[1], angle=np.degrees(np.arctan2(*eigenvectors[:, 1][::-1])), fill=False, color='black'))

eigenvalues, eigenvectors = np.linalg.eigh(new_class_2_cov)
ax.add_patch(Ellipse(np.mean(new_data_2d[100:, :], axis=0), width=eigenvalues[0], height=eigenvalues[1], angle=np.degrees(np.arctan2(*eigenvectors[:, 1][::-1])), fill=False, color='black'))

eigenvalues, eigenvectors = np.linalg.eigh(old_class_1_cov)
ax.add_patch(Ellipse(np.mean(old_data_2d[:100, :], axis=0), width=eigenvalues[0], height=eigenvalues[1], angle=np.degrees(np.arctan2(*eigenvectors[:, 1][::-1])), fill=False, color='black'))

eigenvalues, eigenvectors = np.linalg.eigh(old_class_2_cov)
ax.add_patch(Ellipse(np.mean(old_data_2d[100:, :], axis=0), width=eigenvalues[0], height=eigenvalues[1], angle=np.degrees(np.arctan2(*eigenvectors[:, 1][::-1])), fill=False, color='black'))


ax.legend()


plt.show()

fig.savefig('feats-2d.pdf')
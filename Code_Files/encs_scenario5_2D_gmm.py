from scipy.stats import multivariate_normal
import cv2
import numpy as np
from fitfunction import fit_gmm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from fitfunction import fit_gmm
np.random.seed(0)

# Define means and covariances of the Gaussians
mu1 = [1, -2.5]
sigma1 = [[2, 0], [0, 0.4]]
mu2 = [2, 1]
sigma2 = [[0.5, 0], [0, 1.5]]
mu3 = [-2, 1]
sigma3 = [[1, -0.5], [-0.5, 1]]
mu4 = [-3, 0]
sigma4 = [[0.09, 0], [0, 0.09]]

# Generate the 2D dataset
x1 = np.random.multivariate_normal(mu1, sigma1, 200)
x2 = np.random.multivariate_normal(mu2, sigma2, 200)
x3 = np.random.multivariate_normal(mu3, sigma3, 200)
x4 = np.random.multivariate_normal(mu4, sigma4, 80)
x = np.vstack((x1, x2, x3, x4))

# Fit the GMM to the data
gmm = fit_gmm(x, n_components=4)

# Assign cluster labels to pixels in the image
y = gmm.predict(x)

# Plot the original signal distribution
plt.figure(figsize=(8, 6))
plt.scatter(x[:, 0], x[:, 1], color='gray', alpha=0.5)
plt.title('Original Signal Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot the segmented signal distribution
colors = ['r', 'g', 'b', 'c']
plt.figure(figsize=(8, 6))
for i in range(x.shape[0]):
    plt.scatter(x[i, 0], x[i, 1], color=colors[y[i]], alpha=0.5)
plt.title('Segmented Signal Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()




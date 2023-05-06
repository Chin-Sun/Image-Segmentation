# RBG image GMM for comparing
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Load color image
# Load image

img = cv2.imread(r'D:\encs6161\git_project\Image\street_view_10_cut.jpg')

# Reshape the image to a large vector
img_vector = img.reshape((-1, 3))

# Initialize means and covariances for GMM
'''
K = 3
init_means = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 50]])  # Reshape means array to have shape (3,3)
init_covars = np.array([[[50, 0, 0], [0, 50, 0], [0, 0, 50]]] * K)
'''
'''
K =4
init_means = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 50], [25, 25, 25]])
init_covars = np.array([[[50, 0, 0], [0, 50, 0], [0, 0, 50]]] * K)
'''

'''
K = 5
init_means = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 50], [25, 25, 25], [50, 50, 50]])
init_covars = np.array([[[50, 0, 0], [0, 50, 0], [0, 0, 50]]] * K)
'''

#'''
K = 10
init_means = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 50], [25, 25, 25], [50, 50, 50], [75, 75, 75], [100, 0, 0], [0, 100, 0], [0, 0, 100], [100, 100, 100]])
init_covars = np.array([[[50, 0, 0], [0, 50, 0], [0, 0, 50]]] * K)
#'''

'''
K = 15
init_means = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 50], [25, 25, 25], [50, 50, 50], [75, 75, 75], [100, 0, 0], [0, 100, 0], [0, 0, 100], [100, 100, 100], [25, 0, 0], [0, 25, 0], [0, 0, 25], [75, 0, 0], [0, 75, 0]])
init_covars = np.array([[[50, 0, 0], [0, 50, 0], [0, 0, 50]]] * K)

'''

# Fit GMM to image vector data
gmm = GaussianMixture(n_components=K, means_init=init_means,
                      covariance_type='full')
gmm.fit(img_vector)

# Assign cluster labels to pixels in the image
labels = gmm.predict(img_vector)

# Map pixel intensities to mean values of clusters
means = gmm.means_.astype(int)
imseg = means[labels].reshape(img.shape)

# Display segmented image
imseg_disp = cv2.convertScaleAbs(imseg)
cv2.imshow('Segmented Image', imseg_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''hanging the image extension will not alter the image content or quality. It will only change the format in which 
the image is stored on your computer. '''
'''This error is raised when trying to display an image with a depth of 16-bit float or 32-bit integer, which are not 
supported by the function cv2.imshow(). To fix this error, you can convert the image to a supported depth before 
displaying it. '''
'''In this example, we first convert the image to an 8-bit depth using cv2.convertScaleAbs(), which is a simple way 
to scale and shift the pixel values to fit into the 0-255 range. We then display the converted image using 
cv2.imshow(). Finally, we wait for a key press and then close the window using cv2.destroyAllWindows(). '''
# Save the segmented image with filename as segmented_image_k={k}.jpg
cv2.imwrite('D:\encs6161\git_project\Segmented Images\Scenario2\Image2_segmented_image_k={}.jpg'.format(K), imseg_disp)
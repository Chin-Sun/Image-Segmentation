# RGB image GMM
import cv2
import numpy as np
from fitfunction import fit_gmm
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from fitfunction import fit_gmm

# Load color image
img = cv2.imread(r'D:\encs6161\git_project\Image\street_view_10_cut.jpg')

# Reshape the image to a large vector
img_vector = img.reshape((-1, 3))

# Initialize means and covariances for GMM
K = 10
init_means = np.array(
    [[50, 0, 0], [0, 50, 0], [0, 0, 50], [25, 25, 25], [50, 50, 50], [75, 75, 75], [100, 0, 0], [0, 100, 0],
     [0, 0, 100], [100, 100, 100]])
init_covars = np.array([[[50, 0, 0], [0, 50, 0], [0, 0, 50]]] * K)

# Fit GMM to image vector data using fit_gmm function
gmm = fit_gmm(img_vector, K, init_means, init_covars)

# Assign cluster labels to pixels in the image
labels = gmm.predict(img_vector)

# Map pixel intensities to mean values of clusters
means = gmm.means.astype(int)
imseg = means[labels].reshape(img.shape)

# Display segmented image
imseg_disp = cv2.convertScaleAbs(imseg)
cv2.imshow('Segmented Image', imseg_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the segmented image with filename as segmented_image_k={k}.jpg
cv2.imwrite('D:\encs6161\git_project\Segmented Images\Scenario3\street_view_10_cut_segmented_image_k={}.jpg'.format(K),
            imseg_disp)

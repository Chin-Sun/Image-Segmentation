#grayscale image GMM
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
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reshape the image to a large vector
gray_img_vector = gray_img.reshape((-1, 1))

# Initialize means and covariances for GMM
K = 4
init_means = np.array([[0], [50], [100], [150]])
init_covars = np.array([[[100]], [[100]], [[100]], [[100]]])

# Fit GMM to image vector data using fit_gmm function
gmm = fit_gmm(gray_img_vector, K, init_means, init_covars)

# Assign cluster labels to pixels in the image
labels = gmm.predict(gray_img_vector)

# Map pixel intensities to mean values of clusters
means = gmm.means.astype(int)
imseg = means[labels].reshape(gray_img.shape)

# Display segmented image
imseg_disp = cv2.convertScaleAbs(imseg)
cv2.imshow('Segmented Image', imseg_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the segmented image with filename as segmented_image_k={k}.jpg
cv2.imwrite('D:\encs6161\git_project\Segmented Images\Scenario4\street_view_10_cut_segmented_greyscale_image_k={}.jpg'.format(K),
            imseg_disp)

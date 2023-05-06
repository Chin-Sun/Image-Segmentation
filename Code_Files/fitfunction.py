import numpy as np
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

'''In this example, I added a small constant value 1e-6 to the diagonal of the covariance matrix cov to obtain a new 
matrix cov_reg. This ensures that cov_reg is always invertible. The constant value can be adjusted based on the scale 
of your data and the desired level of regularization. '''
def gaussian_pdf(data, mean, cov):
    """
    Computes the probability density function of a multivariate Gaussian distribution.

    :param data: numpy array of shape (N, D) containing N data points each of dimension D
    :param mean: numpy array of shape (D,) containing the mean of the Gaussian distribution
    :param cov: numpy array of shape (D, D) containing the covariance matrix of the Gaussian distribution
    :return: numpy array of shape (N,) containing the probability density function of the Gaussian distribution for each data point
    """
    D = data.shape[1]
    cov_det = np.linalg.det(cov)
    cov_reg = cov + 1e-6 * np.identity(D)  # add a small positive constant to the diagonal of the covariance matrix
    cov_inv = np.linalg.pinv(cov_reg)
    norm = np.sqrt(((2 * np.pi) ** D) * (cov_det + 1e-6))
    exp_val = np.exp(-0.5 * np.sum((data - mean) @ cov_inv * (data - mean), axis=1))
    pdf = exp_val / norm
    return np.nan_to_num(pdf)


'''def gaussian_pdf(data, mean, cov):
    """
    Computes the probability density function of a multivariate Gaussian distribution.

    :param data: numpy array of shape (N, D) containing N data points each of dimension D
    :param mean: numpy array of shape (D,) containing the mean of the Gaussian distribution
    :param cov: numpy array of shape (D, D) containing the covariance matrix of the Gaussian distribution
    :return: numpy array of shape (N,) containing the probability density function of the Gaussian distribution for each data point
    """
    mvn = multivariate_normal(mean=mean, cov=cov)
    pdf = mvn.pdf(data)
    return pdf'''

class GMM:
    def __init__(self, n_components, means, covars, weights):
        self.n_components = n_components
        self.means = means
        self.covars = covars
        self.weights = weights

    def predict_proba(self, x):
        # Compute the likelihood of each data point given each component
        likelihoods = np.zeros((x.shape[0], self.n_components))
        for i in range(self.n_components):
            likelihoods[:, i] = gaussian_pdf(x, self.means[i, :], self.covars[i, :, :])
        # Compute the posterior probability of each component given each data point
        responsibilities = likelihoods * self.weights
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

def fit_gmm(data, n_components, init_means=None, init_covars=None, max_iter=100, tol=1e-4):
    """
    Fits a Gaussian mixture model to D-dimensional data.

    :param data: numpy array of shape (N, D) containing N data points each of dimension D
    :param n_components: number of components for the Gaussian mixture model
    :param init_means: numpy array of shape (n_components, D) containing initial mean values for the Gaussian mixture
                       model, default is None
    :param init_covars: numpy array of shape (n_components, D, D) containing initial covariance matrices for the
                        Gaussian mixture model, default is None
    :param max_iter: maximum number of iterations to run the algorithm, default is 100
    :param tol: convergence tolerance, default is 1e-4
    :return: fitted Gaussian mixture model object
    """

    # Step 1: Initialization
    if init_means is None:
        # Initialize means randomly from data points
        init_means = data[np.random.choice(data.shape[0], size=n_components, replace=False)]
    if init_covars is None:
        # Initialize covariances as diagonal matrices with random variances
        init_variances = np.var(data, axis=0)
        init_covars = np.zeros((n_components, data.shape[1], data.shape[1]))
        for i in range(n_components):
            init_covars[i, :, :] = np.diag(np.random.rand(data.shape[1]) * init_variances)

    # Initialize mixing coefficients as uniform probabilities
    mixing_coefs = np.ones(n_components) / n_components

    # Initialize log-likelihood and iteration counter
    log_likelihood = -np.inf
    n_iter = 0

    while n_iter < max_iter:

        # Step 2: E-step - Evaluate responsibilities for each data point
        # Compute the likelihood of each data point given each component
        likelihoods = np.zeros((data.shape[0], n_components))
        for i in range(n_components):
            likelihoods[:, i] = gaussian_pdf(data, init_means[i, :], init_covars[i, :, :])
        # Compute the posterior probability of each component given each data point
        responsibilities = likelihoods * mixing_coefs
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        # Compute the log-likelihood of the data
        log_likelihood_new = np.sum(np.log(np.sum(likelihoods * mixing_coefs, axis=1)))
        # Check for convergence
        if np.abs(log_likelihood_new - log_likelihood) < tol:
            break
        log_likelihood = log_likelihood_new

        # Step 3: M-step - Re-estimate parameters
        # Compute the total responsibility of each component
        total_resp = np.sum(responsibilities, axis=0)
        # Update the mixing coefficients
        mixing_coefs = total_resp / data.shape[0]
        # Update the mean of each component
        for i in range(n_components):
            init_means[i, :] = np.sum(responsibilities[:, i, np.newaxis] * data, axis=0) / total_resp[i]
        # Update the covariance of each component
        for i in range(n_components):
            diff = data - init_means[i, :]
            weighted_diff = (responsibilities[:, i, np.newaxis] * diff).T
            init_covars[i, :, :] = np.dot(weighted_diff, diff) / total_resp[i]

        n_iter += 1

    # Create the Gaussian mixture model object
    gmm = GMM(n_components=n_components, means= init_means, covars= init_covars, weights= mixing_coefs )
    return gmm

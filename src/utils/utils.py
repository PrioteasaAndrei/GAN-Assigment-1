import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import seaborn as sns
import torch
from math import sqrt, log, exp, pi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import seaborn as sns
import torch
from scipy.stats import multivariate_normal
from scipy.stats import norm

sns.set(style='whitegrid')


RANDOM_SEED = 42 # for reproductibility

## set cuda device

# Check if a CUDA GPU is available
if torch.cuda.is_available():
    # Set the default device to the first available GPU
    device = torch.device("cuda")
else:
    # If no GPU is available, use the CPU
    device = torch.device("cpu")


def plot_moons(X,y):
    # Plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='Class 1')

    plt.title("Moon-Shaped Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)

    plt.show()


class Gaussian:
    "Model univariate Gaussian"
    def __init__(self, mu, sigma):
        #mean and standard deviation
        self.mu = mu
        self.sigma = sigma

    #probability density function
    def pdf(self, datum):
        "Probability of a data point given the current parameters"
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y
    
    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)




def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"

        source: https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    if kernel == 'squared_exp':
      bandwidth_range = [0.2, 0.5, 0.9, 1.3]
      for a in bandwidth_range:
        XX += torch.exp(-0.5 * dxx / (a**2))
        YY += torch.exp(-0.5 * dyy / (a**2))
        XY += torch.exp(-0.5 * dxy / (a**2))


    if kernel == "inverse_multi_quadratic":
      bandwidth_range = [0.2, 0.5, 0.9, 1.3]
      for a in bandwidth_range:
          XX += 1.0 / (1.0 + a**2 * dxx)
          YY += 1.0 / (1.0 + a**2 * dyy)
          XY += 1.0 / (1.0 + a**2 * dxy)




    return torch.mean(XX + YY - 2. * XY)



def train_GMM(data,n_components=2,max_iterations=100):

    # Initialize parameters
    n_samples, n_features = data.shape

    # Initialize the parameters for the Gaussian components
    means = np.random.rand(n_components, n_features)
    covariances = np.array([np.identity(n_features) for _ in range(n_components)])
    weights = np.ones(n_components) / n_components

    # EM Algorithm
    for iteration in range(max_iterations):
        # Expectation step
        responsibilities = np.zeros((n_samples, n_components))
        for k in range(n_components):
            responsibilities[:, k] = weights[k] * multivariate_normal.pdf(data, mean=means[k], cov=covariances[k])

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # Maximization step
        Nk = responsibilities.sum(axis=0)
        weights = Nk / n_samples
        means = np.dot(responsibilities.T, data) / Nk[:, np.newaxis]
        covariances = [(np.dot(responsibilities[:, k] * (data - means[k]).T, data - means[k]) / Nk[k]).tolist() for k in range(n_components)]

    # Generate data from the estimated multivariate Gaussian
    estimated_data = np.zeros((n_samples, n_features))
    for k in range(n_components):
        estimated_data += responsibilities[:, k][:, np.newaxis] * np.random.multivariate_normal(means[k], covariances[k], size=n_samples)


    return estimated_data, means, covariances, weights



def plot_estimated_real_two_moons(data,estimated_data):
    # Plot the original data and estimated data
    plt.scatter(data[:, 0], data[:, 1], label="Original Data from two moons", alpha=0.8)
    plt.scatter(estimated_data[:, 0], estimated_data[:, 1], label="Generated Data", alpha=0.3)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()


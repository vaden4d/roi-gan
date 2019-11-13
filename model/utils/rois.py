import numpy as np
import torch
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset

class RoI(Dataset):

    def __init__(self, image_shape, sampling_rule, size):

        self.shape = image_shape
        self.f = sampling_rule
        self.size = size

    def __getitem__(self, idx):
        mask = torch.from_numpy(self.f(self.shape).astype(int)).float()
        return mask

    def __len__(self):

        return self.size

def squared_roi(image_shape=(64, 64), n_squares=1):
    '''Rectangual regions of interest (RoIs) generator

    Parameters
    ----------
    image_shape: tuple, the size of output image
    n_squares: integer, number of rectangles

    ----------
    Returns:
    output: np.ndarray, the output mask
    '''

    output = np.zeros(image_shape)
    for _ in range(n_squares):

        # generate bounds of the rectangle
        low_x, high_x = sorted(np.random.randint(low=0, high=image_shape[0], size=2))
        low_y, high_y = sorted(np.random.randint(low=0, high=image_shape[1], size=2))

        output[low_x:high_x, low_y:high_y] = 1

    return output

def gaussian_roi(image_shape=(200, 200)):
    '''Elliptic regions of interest (RoIs) generator.
    Generated via thresholding of the bivariate
    gaussian distribution.
    Parameters
    ----------
    image_shape: tuple, the size of output image

    ----------
    Returns:
    output: np.ndarray, the output mask'''
    

    # generate output grid
    x, y = np.mgrid[0:image_shape[0], 0:image_shape[1]]
    pos = np.dstack((x, y))

    # set mean prior distribution
    mean_x = image_shape[0] / 2
    mean_y = image_shape[1] / 2
    mean = np.random.multivariate_normal([mean_x, mean_y], cov=np.diag([image_shape[0], image_shape[1]]))

    # set covariance matrix prior distribution
    ro = np.random.uniform(-0.8, 0.8)
    sigma_x = np.random.uniform(image_shape[0] / 10, image_shape[0] / 6)
    sigma_y = np.random.uniform(image_shape[1] / 10, image_shape[1] / 6)
    cov = np.array([[sigma_x**2, ro * sigma_x * sigma_y],
                    [ro * sigma_x * sigma_y, sigma_y**2]])

    # generate distribution on the image
    rv = multivariate_normal(mean, cov)
    z = rv.pdf(pos)

    # get threshold of the distribution - filled ellipse
    output = z > z.max() / 2.0

    return output

def mixture_roi(image_shape=(200, 200), n_gaussians=15):
    '''Complex regions of interest (RoIs) generator.
    Generated via thresholding of the gaussian mixture.
    Parameters
    ----------
    image_shape: tuple, the size of output image
    n_gaussians: integer, the number of gaussians
    in the mixture

    ----------
    Returns:
    output: np.ndarray, the output mask
    '''

    # create coefficients of the mixture
    coeffs = np.random.uniform(-1, 1, size=n_gaussians)
    coeffs = np.exp(coeffs) / np.exp(coeffs).sum()

    # output image
    z = np.zeros(image_shape)
    x, y = np.mgrid[0:image_shape[0], 0:image_shape[1]]
    pos = np.dstack((x, y))

    for coeff in coeffs:

        # set mean prior distribution
        mean_x = image_shape[0] / 2
        mean_y = image_shape[1] / 2
        mean = np.random.multivariate_normal([mean_x, mean_y], cov=np.diag([image_shape[0], image_shape[1]]))

        # set covariance prior distribution
        ro = np.random.uniform(-0.9, 0.9)
        sigma_x = np.random.uniform(image_shape[0] / 10, image_shape[0] / 6)
        sigma_y = np.random.uniform(image_shape[1] / 10, image_shape[1] / 6)
        cov = np.array([[sigma_x**2, ro * sigma_x * sigma_y],
                        [ro * sigma_x * sigma_y, sigma_y**2]])

        # generate probability grid
        rv = multivariate_normal(mean, cov)

        # add result to mixture
        z += coeff * rv.pdf(pos)

    # get final threshold - non-elliptic filled level curve
    output = z > z.max() / 2

    return output


if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    obj = RoI((64, 64), gaussian_roi, device, 'full')
    start = time.time()
    tensors = obj.generate_masks(64)
    print('Time', time.time() - start)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def multivariate_gaussian_kernel(size, sigma_x, sigma_y, angle):
    """
    Generate a 2D Gaussian multivariate kernel.

    Parameters:
    size (tuple): Size of the kernel in the form (height, width).
    sigma_x (float): Standard deviation along the x-axis.
    sigma_y (float): Standard deviation along the y-axis.
    angle (float): Rotation angle of the kernel in degrees.

    Returns:
    np.ndarray: 2D Gaussian multivariate kernel matrix.
    """
    # Create a covariance matrix
    cov = np.array([[sigma_x**2, 0], [0, sigma_y**2]])

    # Rotation matrix
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Generate meshgrid for coordinates
    x = np.arange(-(size[1]-1)/2, (size[1]+1)/2)
    y = np.arange(-(size[0]-1)/2, (size[0]+1)/2)
    X, Y = np.meshgrid(x, y)

    # Rotate and scale coordinates
    coords = np.stack([X, Y], axis=-1)
    coords = np.dot(coords, rotation_matrix.T)

    # Generate the kernel
    kernel = multivariate_normal.pdf(coords, mean=[0, 0], cov=cov)

    return kernel / np.sum(kernel)

# Define parameters for the multivariate Gaussian kernel
size = (100, 100)
sigma_x = 20.0
sigma_y = 10.0
angle = 45.0  # Degrees

# Generate the multivariate Gaussian kernel
kernel = multivariate_gaussian_kernel(size, sigma_x, sigma_y, angle)

# Plot the kernel matrix as a heatmap
plt.imshow(kernel, cmap='hot', interpolation='nearest')
plt.title('Gaussian Multivariate Kernel')
plt.colorbar()
plt.show()

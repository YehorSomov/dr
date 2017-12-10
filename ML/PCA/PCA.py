import numpy as np
from matplotlib import pyplot as plt


def normalize(X):
    '''
      Normalise data before processing
      Return normalized data and normalization parameters
    '''
    num = X.shape[1]
    norm_params = np.zeros((2, num))
    norm_params[0] = X.mean(axis=0)
    norm_params[1] = X.std(axis=0, ddof=1)
    norm_X = (X - norm_params[0]) / norm_params[1]

    return norm_X, norm_params


def transform(X, n_components):
    '''
        Select components with largest variance:
            1) Estimate covariance matrix
            2) Find its eigenvalues and eigenvectors
            3) Check if eigenvalues are complex -> to real space
            4) Sort vals & vectors
            5) Select n components
            5) Project all data on the selected components
    '''

    u, s, v = np.linalg.svd(X)
    e_vect = v.astype(float)
    e_vect_reduced = e_vect[:, :n_components]
    new_X = np.dot(X, e_vect_reduced)

    return new_X, e_vect_reduced

def restore(X_reduced, evect_reduced, norm_params):
    '''
        Restore "original" values:
            1) Restore original size
            2) Rescale
    '''
    return (np.dot(X_reduced, evect_reduced.T) * norm_params[1]) + norm_params[0]
def main():
    points = 10
    X = np.zeros((points, 2))
    x = np.arange(1, points + 1)
    y = 4 * x * x + np.random.randn(points) * 2
    X[:, 1] = y
    X[:, 0] = x
    number_of_components = 1

    # normalization
    X_norm, norm_params = normalize(np.copy(X))

    # dimension reduction
    X_reduced, evect_reduced = transform(X_norm, number_of_components)

    # restoring dimensions
    restored_X = restore(X_reduced, evect_reduced, norm_params)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], color='c', label='Initial')
    plt.scatter(restored_X[:, 0], restored_X[:, 1], color='y', label='Restored')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    main()
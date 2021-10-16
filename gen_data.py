"""Generate simple data."""
import matplotlib.pyplot as plt
import numpy as np


def gen_simple(num_obs=50):
    """Generate simple data."""
    num_obs_per_cluster = int(num_obs / 2)
    X = np.vstack((
        np.random.randn(num_obs_per_cluster, 2) * 0.25 + [[0, 0]],
        np.random.randn(num_obs_per_cluster, 2) * 0.25 + [[1, 0]],
    ))
    Y = np.vstack((
        np.zeros((num_obs_per_cluster, 1)),
        np.ones((num_obs_per_cluster, 1)),
    ))
    return X, Y


def gen_xor(num_obs=50):
    """Generate XOR data."""
    num_obs_per_cluster = int(num_obs / 4)
    np.random.seed(653453)
    X = np.vstack((
        np.random.randn(num_obs_per_cluster, 2) * 0.25 + [[0, 0]],
        np.random.randn(num_obs_per_cluster, 2) * 0.25 + [[1, 1]],
        np.random.randn(num_obs_per_cluster, 2) * 0.25 + [[0, 1]],
        np.random.randn(num_obs_per_cluster, 2) * 0.25 + [[1, 0]],
    ))
    Y = np.hstack((
        np.zeros((2 * num_obs_per_cluster,)),
        np.ones((2 * num_obs_per_cluster,)),
    ))
   
    return X, Y


def main():
    """Plot simple data."""
    X, Y = gen_simple()
    for y in np.unique(Y):
        plt.plot(X[Y[:, 0] == y, 0], X[Y[:, 0] == y, 1], 'o')
    plt.show()


if __name__ == "__main__":
    main()
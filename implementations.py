# Imports
import numpy as np


# Model methods


# Least squares implementation
def compute_mse(e):
    """Calculate the mse for vector e.
    Args:
        e: numpy array of shape (N,), N is the number of samples.

    Returns:
        scalar
    """
    return 1 / 2 * np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the mse loss.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape(D,)

    Returns:
        Scalar
    """
    e = y - tx.dot(w)
    return compute_mse(e)


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

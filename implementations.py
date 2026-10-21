# Model methods

import numpy as np


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


# Compute the gradient
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: a scalar
        tx: numpy array of shape=(,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.

    """

    err = y - tx.dot(w)
    grad = -tx * err
    return grad


# stochastic gradient descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D), D is the number of features
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w_last: numpy array of shape=(D, ) containing the weights of the last iteration
        loss_last : a scalar denoting the loss corresponding to the last iteration, with w_last

    """

    # Define parameters to store w and loss
    w = initial_w
    N = len(y)

    for nb_iters in range(max_iters):
        i = np.random.randint(N)
        x_iter = tx[i, :]
        y_iter = y[i]
        grad = compute_stoch_gradient(y_iter, x_iter, w)
        w = w - gamma * grad

    w_last = w
    loss_last = compute_loss(y, tx, w_last)
    return w_last, loss_last


# Ridge regression


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,)
        loss: scalar denoting the loss with w


    """
    D = tx.shape[1]
    N = tx.shape[0]

    A = np.dot(tx.T, tx) + (lambda_ * np.identity(D)) * 2 * N
    b = np.dot(tx.T, y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    return w, loss

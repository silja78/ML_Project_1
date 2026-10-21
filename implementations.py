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


# Logistic Regression implementation
def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1.0 / (1 + np.exp(-t))


def calculate_negative_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss).item() * (1 / y.shape[0])


def calculate_gradient_negative_log_likelihood(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0])
    return grad


def logistic_regression(y, tx, initial_w, max_iter, gamma):
    """
    Do max_iter steps of gradient descent using logistic regression. Return the loss and w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1)
        max_iter = int
        gamma: float

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """
    w = initial_w
    for iter in range(max_iter):
        # get loss and update w
        grad = calculate_gradient_negative_log_likelihood(y, tx, w)
        w -= gamma * grad
    loss = calculate_negative_log_likelihood(y, tx, w)
    return w, loss

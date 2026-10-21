# Imports
import numpy as np

# Model methods


# Helper functions for MSE
def compute_mse_loss(y, tx, w):
    """Compute the mean squared error loss (with the 1/2 factor).

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        loss: float, the value 1/(2N) * sum((y - tx @ w)^2).
    """
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        grad: numpy array of shape (D,), the gradient -1/N * tx.T @ (y - tx @ w).
    """
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(y)


# Helper functions for Negative Log-Likelihood
def sigmoid(t):
    """Apply the sigmoid function element-wise.

    Args:
        t: scalar or numpy array, the input value(s).

    Returns:
        numpy array (or scalar) of the same shape as t, equal to 1 / (1 + exp(-t)).
    """
    return 1.0 / (1.0 + np.exp(-t))


def calculate_negative_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N,), the labels
        tx: shape=(N, D), the feature matrix
        w:  shape=(D,), the weight vector

    Returns:
        a non-negative loss
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss).item() * (1 / y.shape[0])


def calculate_gradient_negative_log_likelihood(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N,), the labels
        tx: shape=(N, D), the feature matrix
        w:  shape=(D,), the weight vector

    Returns:
        a vector of shape (D,) the gradient 1/N * tx.T @ (sigmoid(tx @ w) - y)
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0])
    return grad


# Least squares implementation
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
    loss = compute_mse_loss(y, tx, w)
    return w, np.float64(loss)


# Logistic Regression implementation
def logistic_regression(y, tx, initial_w, max_iter, gamma):
    """
    Do max_iter steps of gradient descent using logistic regression. Return the loss and w.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        initial_w:  shape=(D,)
        max_iter = int
        gamma: float

    Returns:
        w: shape=(D,)
        loss: scalar number
    """
    w = initial_w
    for iter in range(max_iter):
        # get loss and update w
        grad = calculate_gradient_negative_log_likelihood(y, tx, w)
        w -= gamma * grad
    loss = calculate_negative_log_likelihood(y, tx, w)
    return w, np.float64(loss)


# Linear regression gradient descent implementation
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent on the MSE loss.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weight vector.
        max_iters: int, the number of gradient steps to perform.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the weight vector after the last iteration.
        loss: float, the MSE loss computed with the final w.
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_mse_loss(y, tx, w)
    return w, loss


# Regularized logistic regression implementation
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Minimizes the logistic loss plus the penalty lambda_ * ||w||^2.

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        lambda_: float, the regularization parameter.
        initial_w: numpy array of shape (D,), the initial weight vector.
        max_iters: int, the number of gradient steps to perform.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the weight vector after the last iteration.
        loss: float, the logistic loss computed with the final w,
            without the penalty term.
    """
    w = initial_w
    for _ in range(max_iters):
        # Gradient of the loss plus gradient of the penalty lambda_ * ||w||^2
        grad = calculate_gradient_negative_log_likelihood(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    # Loss at the final w, without the penalty term
    loss = calculate_negative_log_likelihood(y, tx, w)
    return w, loss

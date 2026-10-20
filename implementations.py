# Model methods

<<<<<<< Updated upstream
# Mean_square_error_gd
=======
>>>>>>> Stashed changes
import numpy as np


def compute_mse_loss(y, tx, w):
<<<<<<< Updated upstream

=======
    """Compute the mean squared error loss (with the 1/2 factor).

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        loss: float, the value 1/(2N) * sum((y - tx @ w)^2).
    """
>>>>>>> Stashed changes
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
<<<<<<< Updated upstream
=======
    """Compute the gradient of the MSE loss.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        grad: numpy array of shape (D,), the gradient -1/N * tx.T @ (y - tx @ w).
    """
>>>>>>> Stashed changes
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
<<<<<<< Updated upstream
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma * grad
    # loss calculée avec le w final
    loss = compute_mse_loss(y, tx, w)
    return w, loss
=======
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
    for _ in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_mse_loss(y, tx, w)
    return w, loss


import numpy as np


def sigmoid(t):
    """Apply the sigmoid function element-wise.

    Args:
        t: scalar or numpy array, the input value(s).

    Returns:
        numpy array (or scalar) of the same shape as t, equal to 1 / (1 + exp(-t)).
    """
    return 1.0 / (1.0 + np.exp(-t))


def calculate_logistic_loss(y, tx, w):
    """Compute the logistic (negative log-likelihood) loss.

    Uses the numerically stable form 1/N * sum(log(1 + exp(x_n^T w)) - y_n * x_n^T w).

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        loss: float, the mean logistic loss.
    """
    z = tx.dot(w)
    return np.mean(np.logaddexp(0, z) - y * z)


def calculate_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss.

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        grad: numpy array of shape (D,), the gradient 1/N * tx.T @ (sigmoid(tx @ w) - y).
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)


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
        grad = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    # Loss at the final w, without the penalty term
    loss = calculate_logistic_loss(y, tx, w)
    return w, loss
>>>>>>> Stashed changes

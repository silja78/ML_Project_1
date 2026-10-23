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
       returns optimal weights and mse.

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
def logistic_regression(y, tx, initial_w, max_iters, gamma):
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
    for iter in range(max_iters):
        # get loss and update w
        grad = calculate_gradient_negative_log_likelihood(y, tx, w)
        w = w - gamma * grad
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
    return w, np.float64(loss)


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
    loss_last = compute_mse_loss(y, tx, w_last)
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
    loss = compute_mse_loss(y, tx, w)
    return w, loss

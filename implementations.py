# Model methods
import numpy as np


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def calculate_logistic_loss(y, tx, w):
    z = tx.dot(w)
    return np.mean(np.logaddexp(0, z) - y * z)


def calculate_logistic_gradient(y, tx, w):
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = calculate_logistic_loss(y, tx, w)
    return w, loss

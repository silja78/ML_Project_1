# Model methods

#Mean_square_error_gd
import numpy as np


def compute_mse_loss(y, tx, w):
    
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma * grad
    # loss calculée avec le w final
    loss = compute_mse_loss(y, tx, w)
    return w, loss
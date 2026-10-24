import numpy as np
from implementations import *
import csv


def stratified_split(y, val_ratio=0.2, seed=1):
    """Split indices into train/validation, keeping the class proportions.

    Args:
        y: numpy array of shape (N,), the labels.
        val_ratio: float, fraction of each class that goes to validation.
        seed: int, seed for reproducibility.

    Returns:
        train_idx, val_idx: numpy arrays of indices.
    """
    rng = np.random.default_rng(seed)
    val_parts = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        val_parts.append(idx[: int(len(idx) * val_ratio)])
    val_idx = np.concatenate(val_parts)
    is_val = np.zeros(len(y), dtype=bool)
    is_val[val_idx] = True
    return np.where(~is_val)[0], np.where(is_val)[0]


def f1_score(y_true, y_pred):
    """Compute the F1 score, with labels in {0, 1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def best_threshold(y_true, probs, thresholds):
    """Return the threshold with the highest F1 score, and that score."""
    scores = [f1_score(y_true, (probs >= t).astype(int)) for t in thresholds]
    best = int(np.argmax(scores))
    return thresholds[best], scores[best]

def find_best_logistic_params(y_train, tx_train, y_val, tx_val, gammas, iters_values, thresholds):
    # save threshold, train_loss, val_loss, f1, accuracy, precision, recall
    results = np.zeros((len(gammas), len(iters_values), 7))
    best_f1 = -1

    for i,gamma in enumerate(gammas):
        for j, max_iters in enumerate(iters_values):
            w, train_loss = logistic_regression(y_train, tx_train, np.zeros(tx_train.shape[1]), max_iters, gamma)

            probs_val = sigmoid(tx_val @ w)
            thr, f1 = best_threshold(y_val, probs_val, thresholds)

            y_pred = (probs_val >= thr).astype(int)

            tp = np.sum((y_val == 1) & (y_pred == 1))
            fp = np.sum((y_val == 0) & (y_pred == 1))
            fn = np.sum((y_val == 1) & (y_pred == 0))

            accuracy = np.mean(y_pred == y_val)
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0

            p = np.clip(probs_val, 1e-15, 1 - 1e-15)
            val_loss = -np.mean(
                y_val * np.log(p) + (1 - y_val) * np.log(1 - p)
            )

            results[i, j, :] = [
                thr, train_loss, val_loss, f1,
                accuracy, precision, recall
            ]


            if f1 > best_f1:
                best_f1 = f1
                gamma_best = gamma
                max_iters_best = max_iters
                thr_best = thr

    return gamma_best, max_iters_best, thr_best, results

def save_results_csv(results, gammas, iters_values, name):
    with open(name, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([
            "gamma",
            "max_iters",
            "threshold",
            "train_loss",
            "val_loss",
            "f1",
            "accuracy",
            "precision",
            "recall",
        ])

        for i, gamma in enumerate(gammas):
            for j, max_iters in enumerate(iters_values):
                threshold, train_loss, val_loss, f1, accuracy, precision, recall = results[i, j]

                writer.writerow([
                    gamma,
                    max_iters,
                    threshold,
                    train_loss,
                    val_loss,
                    f1,
                    accuracy,
                    precision,
                    recall,
                ])
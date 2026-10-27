import numpy as np
from implementations import *
import time
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
    """Compute the F1 score, with labels in {0, 1}.

    Args:
        y_true: numpy array of shape (N,), the true labels in {0, 1}.
        y_pred: numpy array of shape (N,), the predicted labels in {0, 1}.

    Returns:
        f1: float, 2 * precision * recall / (precision + recall),
            or 0.0 if there is no true positive.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def best_threshold(y_true, probs, thresholds):
    """Find the decision threshold with the highest F1 score.

    Args:
        y_true: numpy array of shape (N,), the true labels in {0, 1}.
        probs: numpy array of shape (N,), the predicted probabilities of the positive class.
        thresholds: numpy array of shape (T,), the candidate thresholds.

    Returns:
        threshold: float, the threshold with the highest F1 score.
        score: float, the F1 score obtained with this threshold.
    """
    scores = [f1_score(y_true, (probs >= t).astype(int)) for t in thresholds]
    best = int(np.argmax(scores))
    return thresholds[best], scores[best]


def find_best_logistic_params(
    y_train, tx_train, y_val, tx_val, gammas, iters_values, thresholds
):
    """Search the best learning rate and number of iterations for logistic regression.

    For each learning rate, the model is trained once up to the largest number of
    iterations and evaluated at each value of iters_values. For each combination,
    the best threshold for the F1 score is found on the validation data, and the
    combination with the highest validation F1 score is returned.

    Args:
        y_train: numpy array of shape (N_train,), the training labels in {0, 1}.
        tx_train: numpy array of shape (N_train, D), the training features
            (with the bias column).
        y_val: numpy array of shape (N_val,), the validation labels in {0, 1}.
        tx_val: numpy array of shape (N_val, D), the validation features
            (with the bias column).
        gammas: array of shape (G,), the learning rates to test.
        iters_values: list of I ints, the numbers of iterations to test.
        thresholds: numpy array of shape (T,), the candidate thresholds.

    Returns:
        gamma_best: float, the learning rate with the highest validation F1 score.
        max_iters_best: int, the number of iterations with the highest validation F1 score.
        thr_best: float, the threshold with the highest validation F1 score.
        results: numpy array of shape (G, I, 7), containing for each combination
            [threshold, train loss, validation loss, F1, accuracy, precision, recall].
            The entries of the combinations that were stopped because the loss
            increased are NaN.
        gamma_best, max_iters_best and thr_best are None if no combination was evaluated.
    """
    results = np.full((len(gammas), len(iters_values), 7), np.nan)
    best_f1 = -1
    gamma_best = max_iters_best = thr_best = None
    t0 = time.time()

    for i, gamma in enumerate(gammas):
        # One training run for each gamma, saving w at every value in iters_values
        saved = logistic_regression_checkpoints(
            y_train, tx_train, np.zeros(tx_train.shape[1]), iters_values, gamma
        )

        for j, max_iters in enumerate(iters_values):
            if max_iters not in saved:
                print(f"gamma={gamma:.4f}, iters={max_iters}: divergé", flush=True)
                continue

            w, train_loss = saved[max_iters]

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
            val_loss = -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))

            results[i, j, :] = [
                thr,
                train_loss,
                val_loss,
                f1,
                accuracy,
                precision,
                recall,
            ]

            print(
                f"gamma={gamma:.4f}, iters={max_iters}, F1={f1:.4f}, "
                f"val_loss={val_loss:.4f}, {time.time() - t0:.0f}s",
                flush=True,
            )

            if f1 > best_f1:
                best_f1 = f1
                gamma_best, max_iters_best, thr_best = gamma, max_iters, thr

    return gamma_best, max_iters_best, thr_best, results


def logistic_regression_checkpoints(y, tx, initial_w, checkpoints, gamma):
    """Run one gradient descent for logistic regression and save w at several iterations.

    The descent is stopped as soon as the loss increases between two checkpoints
    (or is not finite), which indicates that the learning rate is too large.

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weight vector.
        checkpoints: list of ints, the iterations at which w is saved.
        gamma: float, the learning rate.

    Returns:
        saved: dictionary {iteration: (w, loss)} with w of shape (D,) and loss the
            mean negative log likelihood at this iteration. The checkpoints reached
            after the loss started to increase are missing.
    """
    w = initial_w.copy()
    n = len(y)
    checkpoints = set(checkpoints)
    saved = {}
    prev_loss = np.inf

    for it in range(1, max(checkpoints) + 1):
        w = w - gamma * (tx.T @ (sigmoid(tx @ w) - y)) / n

        if it in checkpoints:
            p = np.clip(sigmoid(tx @ w), 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

            if not np.isfinite(loss) or loss > prev_loss + 1e-6:
                break

            saved[it] = (w.copy(), loss)
            prev_loss = loss

    return saved


def save_results_csv(results, gammas, iters_values, name):
    """Save the hyperparameter search results to a CSV file.

    Args:
        results: NumPy array containing the evaluation metrics for each
            combination of gamma and max_iters.
        gammas: List or NumPy array of learning rates tested.
        iters_values: List or NumPy array of maximum iteration counts tested.
        name: String containing the name or path of the output CSV file.

    Returns:
        None.
    """
    with open(name, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(
            [
                "gamma",
                "max_iters",
                "threshold",
                "train_loss",
                "val_loss",
                "f1",
                "accuracy",
                "precision",
                "recall",
            ]
        )

        for i, gamma in enumerate(gammas):
            for j, max_iters in enumerate(iters_values):
                threshold, train_loss, val_loss, f1, accuracy, precision, recall = (
                    results[i, j]
                )

                writer.writerow(
                    [
                        gamma,
                        max_iters,
                        threshold,
                        train_loss,
                        val_loss,
                        f1,
                        accuracy,
                        precision,
                        recall,
                    ]
                )


def predict_all_negative(n_samples):
    """Baseline model that always predicts the negative class.

    Args:
        n_samples: int, the number of samples to predict.

    Returns:
        y_pred: numpy array of shape (n_samples,), filled with -1
            (meaning "no heart attack").
    """
    return -np.ones(n_samples)

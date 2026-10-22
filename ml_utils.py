import numpy as np


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
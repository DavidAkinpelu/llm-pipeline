"""Calibration metrics for classification predictions.

Expected Calibration Error (ECE): bin predictions by confidence, measure
the gap between average confidence and average accuracy in each bin.

A perfectly-calibrated model has ECE = 0 (its predicted probabilities
match observed frequencies exactly). Most LLMs and classifiers are
overconfident — confidence systematically exceeds accuracy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def expected_calibration_error(
    probs: np.ndarray,                   # [N, num_classes] or [N] for binary
    labels: np.ndarray,                  # [N] int — gold class index
    n_bins: int = 15,
) -> float:
    """Standard ECE.

    Definition: ``ECE = Σᵢ (|Bᵢ| / N) · |acc(Bᵢ) − conf(Bᵢ)|``,
    where bins partition predictions by their max-probability confidence.

    Parameters
    ----------
    probs : np.ndarray
        Predicted probabilities. Shape ``[N, K]`` for K-class or
        ``[N]`` for binary (probability of class 1).
    labels : np.ndarray
        Gold integer class indices, shape ``[N]``.
    n_bins : int
        Number of equal-width confidence bins in [0, 1].
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(np.int64)

    if probs.ndim == 1:
        # Binary: probs[i] is P(class=1). Predicted class = 1 iff > 0.5.
        confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
        predictions = (probs >= 0.5).astype(np.int64)
    else:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)

    accuracies = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if i == 0:
            in_bin = in_bin | (confidences == bin_edges[0])
        if in_bin.sum() == 0:
            continue
        bin_size = in_bin.sum()
        bin_conf = confidences[in_bin].mean()
        bin_acc = accuracies[in_bin].mean()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (centre, accuracy, confidence) for plotting.

    Returns three same-length arrays ``(bin_centers, bin_accuracy,
    bin_confidence)``. Empty bins get NaN entries — the caller can
    ``~np.isnan(...)`` them out before plotting.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(np.int64)

    if probs.ndim == 1:
        confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
        predictions = (probs >= 0.5).astype(np.int64)
    else:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)

    accuracies = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if i == 0:
            in_bin = in_bin | (confidences == bin_edges[0])
        if in_bin.sum() == 0:
            continue
        bin_acc[i] = accuracies[in_bin].mean()
        bin_conf[i] = confidences[in_bin].mean()

    return centers, bin_acc, bin_conf

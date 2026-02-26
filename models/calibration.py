"""Probability calibration for ML model outputs."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """Calibrate predicted probabilities using isotonic regression.

    Raw model probabilities are often poorly calibrated (especially trees).
    This maps raw probabilities to calibrated ones using a held-out set
    so that a predicted 70% confidence actually corresponds to ~70% accuracy.
    """

    def __init__(self):
        self._calibrators: dict[int, IsotonicRegression] = {}
        self._fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Fit calibration mapping from raw probabilities.

        Parameters
        ----------
        y_true : array of true class labels (0, 1, 2)
        y_prob : array of shape (n, 3) with raw predicted probabilities
        """
        n_classes = y_prob.shape[1]
        for cls in range(n_classes):
            # Binary indicator: was the true class == cls?
            binary_true = (y_true == cls).astype(float)
            raw_probs = y_prob[:, cls]

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw_probs, binary_true)
            self._calibrators[cls] = iso

        self._fitted = True

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities.

        Parameters
        ----------
        y_prob : array of shape (n, 3) with raw predicted probabilities

        Returns
        -------
        Calibrated probabilities of shape (n, 3), re-normalized to sum to 1.
        """
        if not self._fitted:
            return y_prob

        n_classes = y_prob.shape[1]
        calibrated = np.zeros_like(y_prob)

        for cls in range(n_classes):
            calibrated[:, cls] = self._calibrators[cls].predict(y_prob[:, cls])

        # Re-normalize so probabilities sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        calibrated = calibrated / row_sums

        return calibrated

    @staticmethod
    def reliability_diagram_data(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
    ) -> dict:
        """Compute data for a reliability (calibration) diagram.

        Returns dict with 'bins', 'accuracy', 'confidence' for plotting.
        """
        # Use the max-class probability as confidence
        confidence = np.max(y_prob, axis=1)
        predicted_class = np.argmax(y_prob, axis=1)
        correct = (predicted_class == y_true).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accuracy = []
        bin_confidence = []
        bin_counts = []

        for i in range(n_bins):
            mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_accuracy.append(correct[mask].mean())
            bin_confidence.append(confidence[mask].mean())
            bin_counts.append(int(mask.sum()))

        return {
            "accuracy": bin_accuracy,
            "confidence": bin_confidence,
            "counts": bin_counts,
        }

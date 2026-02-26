"""Deep learning models: LSTM sequence encoder and attention-based classifier.

These provide an alternative to tree-based models for capturing temporal patterns.
Uses pure numpy for the inference path; training requires PyTorch (optional dependency).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_sequences(
    feature_matrix: pd.DataFrame,
    sequence_length: int = 20,
    target_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list]:
    """Convert a MultiIndex feature matrix into sequences for LSTM input.

    Parameters
    ----------
    feature_matrix : DataFrame with MultiIndex (date, ticker)
    sequence_length : number of days per sequence window
    target_col : if provided, extract target from the last timestep

    Returns
    -------
    X : (n_sequences, sequence_length, n_features)
    y : (n_sequences,) targets or None
    metadata : list of (date, ticker) for each sequence
    """
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
    if target_col and target_col in numeric_cols:
        feature_cols = [c for c in numeric_cols if c != target_col]
    else:
        feature_cols = numeric_cols.tolist()
        target_col = None

    tickers = feature_matrix.index.get_level_values("ticker").unique()
    sequences = []
    targets = []
    metadata = []

    for ticker in tickers:
        ticker_data = feature_matrix.xs(ticker, level="ticker")
        ticker_data = ticker_data.sort_index()

        values = ticker_data[feature_cols].values
        if target_col:
            target_values = ticker_data[target_col].values

        for i in range(sequence_length, len(ticker_data)):
            seq = values[i - sequence_length : i]
            if np.isnan(seq).sum() / seq.size > 0.3:
                continue  # Skip sequences with too many NaNs
            seq = np.nan_to_num(seq, nan=0.0)
            sequences.append(seq)

            if target_col:
                targets.append(target_values[i])

            date = ticker_data.index[i]
            metadata.append((date, ticker))

    X = np.array(sequences) if sequences else np.zeros((0, sequence_length, len(feature_cols)))
    y = np.array(targets) if targets else None

    return X, y, metadata


class LSTMClassifier:
    """LSTM-based sequence classifier wrapper.

    Wraps PyTorch LSTM for training but provides a numpy-compatible
    predict interface. Falls back to a simple average if PyTorch unavailable.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.model = None
        self._pytorch_available = self._check_pytorch()

    @staticmethod
    def _check_pytorch() -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
    ):
        """Train the LSTM model.

        Parameters
        ----------
        X_train : (n, seq_len, features)
        y_train : (n,) class labels
        X_val, y_val : optional validation set for early stopping
        """
        if not self._pytorch_available:
            logger.warning("PyTorch not available. LSTM training skipped.")
            return

        import torch
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0,
                )
                self.attention = nn.Linear(hidden_dim, 1)
                self.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
                # Attention over timesteps
                attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
                context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden)
                return self.fc(context)

        device = torch.device("cpu")
        model = LSTMNet(
            self.input_dim, self.hidden_dim, self.num_layers,
            self.num_classes, self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(X_train).to(device)
        y_t = torch.LongTensor(y_train.astype(int)).to(device)

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_train), batch_size):
                idx = indices[start : start + batch_size]
                batch_x = X_t[idx]
                batch_y = y_t[idx]

                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Early stopping on validation
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_out = model(torch.FloatTensor(X_val).to(device))
                    val_loss = criterion(val_out, torch.LongTensor(y_val.astype(int)).to(device)).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : (n, seq_len, features)

        Returns
        -------
        (n, num_classes) probability array
        """
        if self.model is None or not self._pytorch_available:
            # Fallback: uniform probabilities
            return np.ones((len(X), self.num_classes)) / self.num_classes

        import torch
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)

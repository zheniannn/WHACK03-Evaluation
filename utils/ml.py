"""Stage 11 machine-learning discriminators.

  GBM -- histogram gradient-boosted trees on the fixed track-feature vector
         (sklearn's HistGradientBoostingClassifier, the LightGBM-equivalent).
         The strong tabular baseline; feature importances explain *why*.
  GRU -- a small recurrent net on the per-scan sequence, learning temporal
         motion consistency directly rather than through engineered features.

Both train on the three training days and are evaluated on the held-out day.
Extreme class imbalance (~0.7% true tracks) is handled by class weighting
(GBM) and negative subsampling + positive weighting (GRU).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# ------------------------------------------------------------------ GBM

def train_gbm(X: np.ndarray, y: np.ndarray, seed: int = 0) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight="balanced",
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    model.fit(X, y)
    return model


def predict_gbm(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ GRU

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GRUClassifier(nn.Module):
    def __init__(self, n_ch: int, hidden: int = 48):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1]).squeeze(-1)


def _pad_batch(seqs):
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    T = int(lengths.max())
    n_ch = seqs[0].shape[1]
    out = torch.zeros(len(seqs), T, n_ch, dtype=torch.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.from_numpy(s)
    return out, lengths


def train_gru(seqs, labels, n_ch, epochs: int = 8, batch: int = 512, seed: int = 0):
    """Train the GRU. seqs: list of (T,n_ch) arrays; labels: 0/1 array."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    model = GRUClassifier(n_ch).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)],
                              dtype=torch.float32, device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    idx = np.arange(len(seqs))
    for ep in range(epochs):
        rng.shuffle(idx)
        tot = 0.0
        model.train()
        for b in range(0, len(idx), batch):
            bi = idx[b:b + batch]
            x, lengths = _pad_batch([seqs[i] for i in bi])
            y = torch.tensor(labels[bi], dtype=torch.float32, device=DEVICE)
            x = x.to(DEVICE)
            opt.zero_grad()
            logit = model(x, lengths)
            loss = lossf(logit, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        print(f"    GRU epoch {ep + 1}/{epochs}  loss {tot / len(idx):.4f}")
    return model


@torch.no_grad()
def predict_gru(model, seqs, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = np.empty(len(seqs), dtype=np.float32)
    for b in range(0, len(seqs), batch):
        x, lengths = _pad_batch(seqs[b:b + batch])
        logit = model(x.to(DEVICE), lengths)
        out[b:b + len(logit)] = torch.sigmoid(logit).cpu().numpy()
    return out

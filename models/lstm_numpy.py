import numpy as np

import config


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class LSTMRegressor:
    """Single-layer LSTM with a linear head, batched forward and BPTT, Adam optimiser."""

    def __init__(
        self,
        n_features,
        hidden_size=12,
        seed=config.SEED,
        lr=0.01,
        l2=1e-4,
        epochs=40,
        batch_size=16,
        patience=6,
    ):
        self.n_features, self.H = n_features, hidden_size
        self.lr, self.l2, self.epochs, self.batch_size, self.patience = (
            lr,
            l2,
            epochs,
            batch_size,
            patience,
        )
        self.seed = seed
        rng = np.random.default_rng(seed)
        H, D = hidden_size, n_features
        self.params = {
            "W": rng.normal(0, 1 / np.sqrt(H + D), (4 * H, H + D)),
            "b": np.concatenate([np.ones(H), np.zeros(3 * H)]),
            "Wy": rng.normal(0, 1 / np.sqrt(H), (1, H)),
            "by": np.zeros(1),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    def forward(self, X):
        B, T, _ = X.shape
        H, p = self.H, self.params
        h, c = np.zeros((B, H)), np.zeros((B, H))
        cache = []
        for t in range(T):
            z = np.concatenate([h, X[:, t]], axis=1)
            a = z @ p["W"].T + p["b"]
            f, i = sigmoid(a[:, :H]), sigmoid(a[:, H : 2 * H])
            g, o = np.tanh(a[:, 2 * H : 3 * H]), sigmoid(a[:, 3 * H :])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            cache.append((z, f, i, g, o, c, tc))
            h, c = o * tc, c_new
        return (h @ p["Wy"].T + p["by"])[:, 0], h, cache

    def loss_and_grads(self, X, y):
        pred, h, cache = self.forward(X)
        B, H, p = len(y), self.H, self.params
        err = pred - y
        dy = (err / B)[:, None]
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["Wy"], grads["by"] = dy.T @ h, dy.sum(axis=0)
        dh, dc_next = dy @ p["Wy"], np.zeros((B, H))
        for z, f, i, g, o, c_prev, tc in reversed(cache):
            dc = dh * o * (1 - tc**2) + dc_next
            da = np.concatenate(
                [
                    dc * c_prev * f * (1 - f),
                    dc * g * i * (1 - i),
                    dc * i * (1 - g**2),
                    dh * tc * o * (1 - o),
                ],
                axis=1,
            )
            grads["W"] += da.T @ z
            grads["b"] += da.sum(axis=0)
            dh = (da @ p["W"])[:, :H]
            dc_next = dc * f
        return 0.5 * np.mean(err**2), grads

    def _adam(self, grads):
        self._t += 1
        b1, b2 = 0.9, 0.999
        for k, p in self.params.items():
            g = grads[k] + self.l2 * p
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g**2
            p -= (
                self.lr
                * (self._m[k] / (1 - b1**self._t))
                / (np.sqrt(self._v[k] / (1 - b2**self._t)) + 1e-8)
            )

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        best, best_params, stale = np.inf, None, 0
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(X), self.batch_size):
                idx = order[s : s + self.batch_size]
                self._adam(self.loss_and_grads(X[idx], y[idx])[1])
            if X_val is not None and len(X_val):
                mse = float(np.mean((self.predict(X_val) - y_val) ** 2))
                if mse < best - 1e-9:
                    best, best_params, stale = (
                        mse,
                        {k: v.copy() for k, v in self.params.items()},
                        0,
                    )
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        if best_params is not None:
            for k, v in best_params.items():
                self.params[k][...] = v
        return self

    def predict(self, X):
        return self.forward(X)[0]

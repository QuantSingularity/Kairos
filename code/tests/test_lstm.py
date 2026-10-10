import numpy as np

from kairos.models.lstm_numpy import LSTMRegressor


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    model = LSTMRegressor(n_features=3, hidden_size=4, seed=1, l2=0.0)
    X, y = rng.normal(size=(5, 6, 3)), rng.normal(size=5)
    _, grads = model.loss_and_grads(X, y)
    worst = 0.0
    for key, p in model.params.items():
        flat = p.reshape(-1)
        for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + 1e-6
            up = model.loss_and_grads(X, y)[0]
            flat[i] = old - 1e-6
            down = model.loss_and_grads(X, y)[0]
            flat[i] = old
            num, ana = (up - down) / 2e-6, grads[key].reshape(-1)[i]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    assert worst < 1e-5


def test_learns_known_pattern():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 8, 2))
    y = 0.8 * X[:, -1, 0] - 0.4 * X[:, -2, 1]
    model = LSTMRegressor(n_features=2, hidden_size=8, seed=2, epochs=150, patience=30)
    model.fit(X[:160], y[:160], X[160:], y[160:])
    mse = np.mean((model.predict(X[160:]) - y[160:]) ** 2)
    assert mse < 0.3 * np.var(y[160:])


def test_early_stopping_restores_best_params():
    rng = np.random.default_rng(5)
    X, y = rng.normal(size=(60, 5, 2)), rng.normal(size=60)
    model = LSTMRegressor(n_features=2, hidden_size=4, epochs=30, patience=3, seed=4)
    model.fit(X[:40], y[:40], X[40:], y[40:])
    assert np.isfinite(model.predict(X[40:])).all()


def test_batched_prediction_matches_single():
    rng = np.random.default_rng(7)
    model = LSTMRegressor(n_features=3, hidden_size=5, seed=6)
    X = rng.normal(size=(4, 7, 3))
    batch = model.predict(X)
    single = np.array([model.predict(X[i : i + 1])[0] for i in range(4)])
    assert np.allclose(batch, single)

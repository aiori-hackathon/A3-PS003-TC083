"""
data_pipeline.py  (NumPy only, no PyTorch needed)

Data side of PS-003: federated traffic-flow prediction.

What this file does, and why each step exists:
  1. load_flow          : read the PEMS .npz file into a (time, sensors) flow array.
  2. pick_sensors       : choose which sensors act as simulated clients.
  3. build_clients      : split chronologically, normalize using TRAIN data only,
                          make sliding windows, then partition across clients
                          (IID or non-IID).
  4. metrics            : MAE / RMSE and a naive baseline for sanity checks.
  5. comm_bytes_per_round : communication cost arithmetic.

Run `python data_pipeline.py` to execute the self-tests (synthetic data).
Run `python data_pipeline.py path/to/PEMS08.npz` to inspect the real dataset.
"""

import sys
import numpy as np


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------
def load_flow(path):
    """Load a PEMS .npz file and return flow as float32 array of shape (T, N).

    PEMS04/PEMS08 files store an array under the key "data" shaped
    (time, sensors, features), where feature 0 is traffic flow.
    Why: the problem statement asks for traffic FLOW, not speed or occupancy.
    """
    raw = np.load(path)
    key = "data" if "data" in raw.files else raw.files[0]
    data = raw[key]
    if data.ndim == 3:
        data = data[..., 0]  # keep the flow feature only
    return data.astype(np.float32)


# ----------------------------------------------------------------------------
# 2. Choosing clients
# ----------------------------------------------------------------------------
def pick_sensors(flow, n_clients, train_frac=0.7, mode="spread", seed=0):
    """Pick `n_clients` sensor indices to act as simulated intersections.

    mode="spread": evenly spaced across the ranking of mean flow, so clients have
                   clearly different traffic levels (a natural non-IID setting).
    mode="random": random sensors, for comparison.

    The ranking uses only the training period so test data does not influence
    the choice.
    """
    cut = int(len(flow) * train_frac)
    mean_level = flow[:cut].mean(axis=0)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(flow.shape[1], size=n_clients, replace=False))
    order = np.argsort(mean_level)
    positions = np.linspace(0, len(order) - 1, n_clients).round().astype(int)
    return order[positions]


# ----------------------------------------------------------------------------
# 3. Windowing, scaling, partitioning
# ----------------------------------------------------------------------------
def make_windows(series, input_len, horizon=1):
    """Turn a 1-D series into (X, y) pairs.

    X[i] = series[i : i + input_len]               (the past)
    y[i] = series[i + input_len + horizon - 1]     (the value to predict)
    """
    n = len(series) - input_len - horizon + 1
    if n <= 0:
        raise ValueError("Series too short for this input_len and horizon.")
    starts = np.arange(n)
    X = series[starts[:, None] + np.arange(input_len)[None, :]]
    y = series[starts + input_len + horizon - 1]
    return X.astype(np.float32), y.astype(np.float32)


class Scaler:
    """Z-score scaler. Fit on TRAIN data only.

    Why: fitting on test data would leak information about the future into
    training, which the problem statement explicitly forbids.
    """

    def fit(self, values):
        self.mean = float(np.mean(values))
        self.std = float(np.std(values)) + 1e-8
        return self

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse(self, values):
        return values * self.std + self.mean


def build_clients(flow, sensor_ids, input_len=12, horizon=1,
                  train_frac=0.7, partition="noniid", seed=0):
    """Create the per-client datasets.

    Steps and reasons:
      - Split the raw timeline FIRST (first train_frac for training, the rest
        for testing). Windows are built separately inside each part, so no
        window ever straddles the train/test boundary.
      - Fit one global Scaler on the training part of the chosen sensors.
        (Simplification to mention in your report: a real federated system would
        need to share these two numbers, mean and std, with the server.)
      - partition="noniid": client i keeps sensor i's own data. Clients differ
        in traffic level and pattern.
      - partition="iid": windows from all chosen sensors are pooled, shuffled
        with the seed, and dealt out equally. Every client sees the same mix.
        Total data is identical to the non-IID case, so the comparison is fair.

    Returns (clients, scaler). clients is a list of dicts with keys
    X_train, y_train, X_test, y_test (values are normalized).
    """
    sub = flow[:, sensor_ids]                      # (T, n_clients)
    cut = int(len(sub) * train_frac)
    train_raw, test_raw = sub[:cut], sub[cut:]

    scaler = Scaler().fit(train_raw)

    per_sensor = []
    for s in range(sub.shape[1]):
        Xtr, ytr = make_windows(scaler.transform(train_raw[:, s]), input_len, horizon)
        Xte, yte = make_windows(scaler.transform(test_raw[:, s]), input_len, horizon)
        per_sensor.append((Xtr, ytr, Xte, yte))

    n_clients = len(sensor_ids)
    if partition == "noniid":
        return [dict(X_train=a, y_train=b, X_test=c, y_test=d)
                for a, b, c, d in per_sensor], scaler

    if partition != "iid":
        raise ValueError("partition must be 'iid' or 'noniid'")

    rng = np.random.default_rng(seed)

    def pool_and_deal(i_x, i_y):
        X = np.concatenate([p[i_x] for p in per_sensor])
        y = np.concatenate([p[i_y] for p in per_sensor])
        order = rng.permutation(len(X))
        X, y = X[order], y[order]
        return np.array_split(X, n_clients), np.array_split(y, n_clients)

    Xtr_parts, ytr_parts = pool_and_deal(0, 1)
    Xte_parts, yte_parts = pool_and_deal(2, 3)
    clients = [dict(X_train=Xtr_parts[i], y_train=ytr_parts[i],
                    X_test=Xte_parts[i], y_test=yte_parts[i])
               for i in range(n_clients)]
    return clients, scaler


def pooled(clients, split):
    """Concatenate all clients' data. Use pooled(clients, 'train') as the
    centralised baseline's training set, so it sees exactly the same data as the
    federated clients combined."""
    X = np.concatenate([c[f"X_{split}"] for c in clients])
    y = np.concatenate([c[f"y_{split}"] for c in clients])
    return X, y


# ----------------------------------------------------------------------------
# 4. Metrics
# ----------------------------------------------------------------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def naive_baseline_scores(clients, scaler):
    """Predict 'next value = last observed value'. Any real model should beat
    this. Scores are in ORIGINAL units (vehicles per 5 min), not normalized
    units, because that is what a reader of your report can interpret."""
    X, y = pooled(clients, "test")
    pred = X[:, -1]
    return (mae(scaler.inverse(y), scaler.inverse(pred)),
            rmse(scaler.inverse(y), scaler.inverse(pred)))


# ----------------------------------------------------------------------------
# 5. Communication cost
# ----------------------------------------------------------------------------
def comm_bytes_per_round(num_params, num_clients, bytes_per_param=4):
    """Bytes moved in one FedAvg round, assuming every client participates.

    Each client downloads the global model and uploads its updated model, so
    there are two transfers per client. float32 = 4 bytes per parameter.
    """
    return 2 * num_clients * num_params * bytes_per_param


# ----------------------------------------------------------------------------
# Self-tests and a quick look at real data
# ----------------------------------------------------------------------------
def _self_test():
    # Leakage test: if value == time index, any train value must be smaller than
    # any test value, and the scaler must only know about the training range.
    T, S = 1000, 4
    flow = np.tile(np.arange(T, dtype=np.float32)[:, None], (1, S))
    ids = np.arange(S)
    clients, scaler = build_clients(flow, ids, input_len=12, partition="noniid")
    cut = int(T * 0.7)
    assert abs(scaler.mean - np.arange(cut).mean()) < 1e-3, "scaler saw test data"
    max_train = max(scaler.inverse(c["y_train"]).max() for c in clients)
    min_test = min(scaler.inverse(c["X_test"]).min() for c in clients)
    assert max_train < min_test, "train and test overlap in time"

    # Window shape and alignment.
    X, y = make_windows(np.arange(20, dtype=np.float32), input_len=5, horizon=1)
    assert X.shape == (15, 5) and y.shape == (15,)
    assert X[0].tolist() == [0, 1, 2, 3, 4] and y[0] == 5
    assert y[-1] == 19

    # IID and non-IID hold the same total data.
    rng = np.random.default_rng(0)
    t = np.arange(2000)
    flow = np.stack([(50 * (k + 1)) * (1 + 0.3 * np.sin(2 * np.pi * t / 288))
                     + rng.normal(0, 5, len(t)) for k in range(5)], axis=1)
    flow = flow.astype(np.float32)
    ids = np.arange(5)
    non, sc = build_clients(flow, ids, partition="noniid")
    iid, _ = build_clients(flow, ids, partition="iid", seed=1)
    assert len(pooled(non, "train")[0]) == len(pooled(iid, "train")[0])
    non_means = [c["y_train"].mean() for c in non]
    iid_means = [c["y_train"].mean() for c in iid]
    assert np.ptp(non_means) > 5 * np.ptp(iid_means), "non-IID not clearly different"

    print("All self-tests passed.")
    print("Naive last-value baseline on synthetic data (MAE, RMSE):",
          naive_baseline_scores(non, sc))
    print("Example: 10,000 params, 5 clients ->",
          comm_bytes_per_round(10_000, 5), "bytes per round")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        flow = load_flow(sys.argv[1])
        print("flow shape (time, sensors):", flow.shape)
        print("min / mean / max:", flow.min(), flow.mean(), flow.max())
        ids = pick_sensors(flow, n_clients=4)
        print("chosen sensors:", ids, "mean flow:", flow[:, ids].mean(axis=0))
        clients, scaler = build_clients(flow, ids)
        print("train windows per client:", [len(c["y_train"]) for c in clients])
        print("naive baseline (MAE, RMSE):", naive_baseline_scores(clients, scaler))
    else:
        _self_test()